# -*- coding: utf-8 -*-
"""豆馅医疗 GraphRAG V1.17 / P3-02: 安全增量扩容试运行规划与验收。

plan: 只读扫描现有 Chroma + JSON，核验 P3-01 深度报告，选出小批次候选。
verify: 对比扩容前后完整 Chunk 身份，逐条核验新增记录的 Neo4j 原文坐标。

此模块绝不调用云端 Embedding/LLM，不调用现有批处理写入程序，
不写入 Chroma / Neo4j / SQLite checkpoint；仅创建独立的 JSON 报告。
由于尚未得到 medical_json_bulk_index.py 的源代码，不自动猜测其 CLI 参数。
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ORIGIN = "json_matched_to_neo4j_section"
SCHEMA_ADAPTER_VERSION = "P3-02.1-v117-drugs-dict-and-string-list"
FIELDS = ("indication", "contraindications", "adverse_reactions",
          "warnings", "drug_interactions")
NAME_KEYS = ("drug_name", "name", "drugName", "generic_name",
             "drug_name_cn", "药品名称", "通用名称")
ID_KEYS = ("drug_id", "id", "drugId", "neo4j_id", "药品ID")
BINDING_KEYS = ("drug_id", "record_id", "section_id",
                "source_url", "section_sha256", "quote_start", "quote_end")


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def validate_readiness(report: dict[str, Any]) -> None:
    """只接受包含三个完整检查的 P3-01 结果；不以用户口头确认替代机器记录。"""
    if report.get("stage") != "P3-01" or report.get("project_version") != "V1.17":
        raise ValueError("必须使用 V1.17 的 P3-01 报告")
    gate = report.get("gates") or {}
    if gate.get("status") != "PASS_PRECHECK":
        raise ValueError("P3-01 尚未 PASS_PRECHECK，不得开始批量试运行规划")
    chroma = report.get("chroma") or {}
    if not chroma.get("scan_complete"):
        raise ValueError("P3-01 未完整扫描 Chroma")
    if int(chroma.get("record_count_before", -1)) != int(chroma.get("record_count_after", -2)):
        raise ValueError("P3-01 扫描前后计数不一致")
    if (chroma.get("duplicate_json_quote_coordinates")
            or chroma.get("repeated_chunk_ids_across_pages")
            or chroma.get("json_binding_missing_counts")):
        raise ValueError("P3-01 记录了重复坐标或元数据缺陷")
    checkpoint = report.get("checkpoint")
    neo4j = report.get("neo4j")
    sample = report.get("crossstore_sample")
    if not isinstance(checkpoint, dict) or not checkpoint.get("tables"):
        raise ValueError("P3-01 缺少 SQLite checkpoint 检查")
    if not isinstance(neo4j, dict) or not (neo4j.get("node_counts") or {}).get("Drug"):
        raise ValueError("P3-01 缺少 Neo4j 节点检查")
    if (not isinstance(sample, dict) or not sample.get("complete")
            or int(sample.get("requested", 0)) < 1
            or int(sample.get("verified", -1)) != int(sample.get("requested", -2))
            or int(sample.get("failed", -1)) != 0):
        raise ValueError("P3-01 原文回溯抽样未全部通过")


def _mapped_drug_records(mapping: dict[Any, Any]) -> list[dict[str, Any]]:
    """将 {药品名称: 栏目字典} 转为规划用记录，保留药名键，不修改源 JSON。"""
    result = []
    for name, record in mapping.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(record, dict):
            raise ValueError(
                f"drugs 字典存在无效药名或非字典药品记录：{str(name)[:80]}"
            )
        if "__map_drug_name__" in record:
            raise ValueError("药品记录使用了保留的内部字段 __map_drug_name__")
        result.append({**record, "__map_drug_name__": name.strip()})
    return result


def extract_records(payload: Any) -> list[dict[str, Any]]:
    """支持 V1.17 的 {drugs: {药名: {...}}, interactions: [...]}。

    仅提取 drugs；顶层 interactions、别名、来源元数据均不属于药品记录。
    同时兼容先前的药品列表格式，不对 JSON 做任何写操作。
    """
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("drugs", "records", "items", "data", "drug_list", "药品"):
            if key not in payload:
                continue
            nested = payload[key]
            if isinstance(nested, list):
                return [r for r in nested if isinstance(r, dict)]
            if isinstance(nested, dict):
                return _mapped_drug_records(nested)
            raise ValueError(
                f"JSON 中 {key!r} 应为药品列表或按药名索引的字典，"
                f"实际为 {type(nested).__name__}"
            )
        # 兼容没有顶层 drugs 包装的 {药名: {...}} 结构。
        if payload and all(isinstance(v, dict) for v in payload.values()):
            return _mapped_drug_records(payload)
    raise ValueError("JSON 根结构未识别；需核对来源结构，不能猜测索引范围")


def field_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def source_section_characters(value: Any) -> int | None:
    """仅用于筛选预算的粗略字符数，不模拟原分块器/计费字符。

    str 与 list[str] 都计数；空值视为空栏目。遇到 dict、
    list[dict] 或其他复杂结构时返回 None，防止静默漏算。
    """
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.strip())
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return sum(len(item.strip()) for item in value)
        return None
    return None


def inspect_source(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"药品 JSON 文件不存在：{path}")
    if path.stat().st_size > 600 * 1024 * 1024:
        raise ValueError("药品 JSON 超过 600MB 的单次读取上限")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = extract_records(payload)
    seen_names: set[str] = set()
    duplicates: set[str] = set()
    output: list[dict[str, Any]] = []
    no_name = 0
    missing_fields = 0
    unsupported_records = 0
    unsupported_fields: Counter[str] = Counter()
    for row in rows:
        # 用户 V1.17 的 drugs 字典以键为规范药名；优先使用它。
        name = field_value(row, ("__map_drug_name__",)) or field_value(row, NAME_KEYS)
        if not name:
            no_name += 1
            continue
        if name in seen_names:
            duplicates.add(name)
        seen_names.add(name)
        field_lengths: dict[str, int] = {}
        unhandled = False
        for field in FIELDS:
            length = source_section_characters(row.get(field))
            if length is None:
                unsupported_fields[field] += 1
                unhandled = True
            elif length > 0:
                field_lengths[field] = length
        # 遇到不支持的结构，排除本药而非漏算该字段后继续规划。
        if unhandled:
            unsupported_records += 1
            continue
        if not field_lengths:
            missing_fields += 1
        output.append({
            "drug_name": name,
            "source_drug_id": field_value(row, ID_KEYS),
            "field_characters": field_lengths,
            "source_characters": sum(field_lengths.values()),
        })
    output = [x for x in output if x["drug_name"] not in duplicates]
    return output, {
        "source_records": len(rows),
        "recognized_unique_drugs": len(output),
        "missing_name": no_name,
        "duplicate_drug_names_excluded": len(duplicates),
        "no_target_fields": missing_fields,
        "unsupported_field_records_excluded": unsupported_records,
        "unsupported_field_type_counts": dict(sorted(unsupported_fields.items())),
        "target_fields": list(FIELDS),
        "list_text_handling": "list[str] summed as approximate source character budget",
        "schema_adapter_version": SCHEMA_ADAPTER_VERSION,
    }


def inspect_script(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.suffix.lower() != ".py":
        raise ValueError(f"索引脚本/分块脚本不存在：{path}")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    flags = set()
    function_names = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.append(node.name)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if arg.value.startswith("--"):
                            flags.add(arg.value)
    return {
        "filename": path.name, "sha256": file_sha(path),
        "cli_flags_detected": sorted(flags),
        "function_names": sorted(set(function_names))[:70],
        "note": "静态 AST 检查，不导入/执行索引代码；不能证明程序具备幂等和限额语义",
    }


def identity_digest(metadata: dict[str, Any], content: str) -> str:
    return object_sha({"metadata": metadata, "document": content})


def scan_collection(collection: Any, batch_size: int = 200
                    ) -> tuple[dict[str, str], set[str], set[str], int]:
    """只读 count/get 全量获取：ID → metadata+原文 hash，用于发现旧 Chunk 被改写。"""
    count_before = int(collection.count())
    signatures = {}
    names = set()
    ids = set()
    offset = 0
    while offset < count_before:
        page = collection.get(
            limit=min(batch_size, count_before - offset),
            offset=offset, include=["metadatas", "documents"]
        )
        chunk_ids = page.get("ids") or []
        metas = page.get("metadatas") or []
        docs = page.get("documents") or []
        if not chunk_ids or len(chunk_ids) != len(metas) or len(chunk_ids) != len(docs):
            raise ValueError("Chroma 分页扫描不完整或数据列长度不一致")
        for cid, raw_meta, text in zip(chunk_ids, metas, docs):
            cid = str(cid)
            if cid in signatures:
                raise ValueError("Chroma 同一 Chunk ID 在分页中重复出现")
            meta = raw_meta if isinstance(raw_meta, dict) else {}
            if not isinstance(text, str):
                raise ValueError("Chroma 存在非字符串文档，暂停规划")
            signatures[cid] = identity_digest(meta, text)
            name = str(meta.get("drug_name") or "").strip()
            drug_id = str(meta.get("drug_id") or "").strip()
            if name:
                names.add(name)
            if drug_id:
                ids.add(drug_id)
        offset += len(chunk_ids)
    if int(collection.count()) != count_before or len(signatures) != count_before:
        raise ValueError("Chroma 扫描期间发生数量变更，暂停并重新执行 P3-01")
    return signatures, names, ids, count_before


def choose_pilot(rows: list[dict[str, Any]], existing_names: set[str],
                 existing_ids: set[str], *, max_drugs: int,
                 char_budget: int, max_drug_chars: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = Counter()
    eligible = []
    for row in rows:
        name, did = row["drug_name"], row["source_drug_id"]
        if name in existing_names or (did and did in existing_ids):
            counts["already_indexed"] += 1
            continue
        if not row["field_characters"]:
            counts["no_target_fields"] += 1
            continue
        if row["source_characters"] > max_drug_chars:
            counts["over_per_drug_budget"] += 1
            continue
        eligible.append(row)
    # 优先五栏目较完整的药品，遇到同一覆盖层级再按稳定哈希选取，避免每次受 JSON 顺序影响。
    eligible.sort(key=lambda r: (
        -len(r["field_characters"]),
        hashlib.sha256(("P3-02|" + r["drug_name"]).encode("utf-8")).hexdigest()
    ))
    chosen = []
    total = 0
    for row in eligible:
        if len(chosen) >= max_drugs:
            break
        size = row["source_characters"]
        if total + size > char_budget:
            counts["over_total_budget"] += 1
            continue
        chosen.append(row)
        total += size
    counts["eligible_under_per_drug_budget"] = len(eligible)
    counts["selected"] = len(chosen)
    counts["selected_source_characters"] = total
    return chosen, dict(counts)


def build_plan(*, readiness: dict[str, Any],
               source_rows: list[dict[str, Any]], source_stat: dict[str, Any],
               source_hash: str, indexer_stat: dict[str, Any],
               chunker_stat: dict[str, Any],
               signatures: dict[str, str], existing_names: set[str],
               existing_ids: set[str], count: int,
               chroma_path: str, collection_name: str,
               max_drugs: int, char_budget: int,
               max_drug_chars: int, max_new_chunks: int) -> dict[str, Any]:
    validate_readiness(readiness)
    c = readiness["chroma"]
    if int(c["record_count_after"]) != count:
        raise ValueError("P3-01 与当前 Chroma 数量不同，请先重做 P3-01")
    if str(c.get("collection")) != collection_name:
        raise ValueError("Collection 名称与 P3-01 不匹配")
    if Path(str(c.get("path") or "")).resolve() != Path(chroma_path).resolve():
        raise ValueError("Chroma 路径与 P3-01 不一致")
    if len(existing_names) != int(c.get("unique_drug_names_in_scanned_metadata", -1)):
        raise ValueError("Chroma 药品名数量与 P3-01 不符，请复核数据库快照")
    selected, counts = choose_pilot(source_rows, existing_names, existing_ids,
                                     max_drugs=max_drugs, char_budget=char_budget,
                                     max_drug_chars=max_drug_chars)
    warnings = [
        "本工具没有运行 medical_json_bulk_index.py；不能代替现有索引器的正式 dry-run。",
        "JSON 原文字符预算不是最终 Chunk/embedding 计费字符数；需按原索引器切块后核对。",
        "用户必须确认原始数据的第三方服务发送权限、预算、备份及 checkpoint 恢复方案。",
        "新增文档必须使用既有 Neo4j Section 三元 ID + 原文 offset/hash 校验，不允许跳过。",
        "规划中的 max_drugs/字符预算及 max_new_chunks 不能限制旧索引器真实写入；写入限额必须由原索引器实现并检查。",
    ]
    if source_stat["source_records"] != int((readiness.get("neo4j") or {}).get("node_counts", {}).get("Drug") or 0):
        warnings.append("JSON 记录数与当前 Neo4j Drug 数量不同，可能有去重/数据源差异，必须核对")
    detected = indexer_stat["cli_flags_detected"]
    if not any(f in detected for f in (
        "--limit", "--max-drugs", "--drug-limit", "--max-records",
        "--start", "--end", "--batch-size", "--offset",
    )):
        warnings.append("索引器未发现可识别的限额 CLI 参数：不得猜测或直接全库运行")
    status = "PILOT_PLANNED_NOT_AUTHORIZED" if selected else "NO_ELIGIBLE_PILOT"
    plan = {
        "schema_version": 1, "project_version": "V1.17", "stage": "P3-02",
        "mode": "plan", "created_utc": stamp(), "status": status,
        "readiness_gate": readiness["gates"]["status"],
        "readiness_audit_utc": readiness.get("audit_utc"),
        "source_json_sha256": source_hash, "source": source_stat,
        "indexer": indexer_stat, "chunker": chunker_stat,
        "chroma": {
            "path": str(Path(chroma_path).resolve()), "collection": collection_name,
            "before_count": count, "before_signatures": signatures,
            "before_chunk_ids_sha256": object_sha(sorted(signatures)),
        },
        "pilot_limits": {
            "max_drugs": max_drugs, "source_char_budget": char_budget,
            "max_source_chars_per_drug": max_drug_chars,
            "max_new_chunks": max_new_chunks,
        },
        "candidate_selection": counts, "pilot_drugs": selected,
        "warnings": warnings,
        "write_execution": "NOT_EXECUTED",
        "medical_source_authority": "not_verified_by_this_tool",
    }
    plan["plan_id"] = object_sha({
        "source_json_sha256": source_hash,
        "indexer_sha256": indexer_stat["sha256"],
        "chunker_sha256": chunker_stat["sha256"],
        "chroma_pre_ids": plan["chroma"]["before_chunk_ids_sha256"],
        "pilot": selected,
    })[:20]
    return plan


def check_new_binding(meta: dict[str, Any]) -> str:
    if meta.get("data_origin") != ORIGIN:
        return "wrong_data_origin"
    if any(not isinstance(meta.get(key), str) or not meta[key].strip()
           for key in BINDING_KEYS[:5]):
        return "missing_binding"
    try:
        start = int(meta["quote_start"])
        end = int(meta["quote_end"])
        if start < 0 or end <= start:
            return "invalid_offsets"
    except (KeyError, ValueError, TypeError):
        return "invalid_offsets"
    return ""


def verify_pilot(plan: dict[str, Any], after: dict[str, Any],
                 current: dict[str, tuple[dict[str, Any], str]],
                 *, verify_parent: Any = None) -> dict[str, Any]:
    """不使用 LLM：识别全部新增 ID，并可调用现有 Neo4j Section 原文验证函数。"""
    if plan.get("stage") != "P3-02" or plan.get("mode") != "plan":
        raise ValueError("不是有效的 P3-02 规划文件")
    validate_readiness(after)
    if str(after["chroma"].get("collection")) != str(plan["chroma"]["collection"]):
        raise ValueError("扩容前后 Collection 不一致")
    if Path(str(after["chroma"].get("path"))).resolve() != Path(str(plan["chroma"]["path"])).resolve():
        raise ValueError("扩容前后 Chroma 路径不一致")
    original = plan["chroma"]["before_signatures"]
    missing_old = sorted(set(original) - set(current))
    changed_old = sorted(cid for cid in original.keys() & current.keys()
                         if original[cid] != identity_digest(*current[cid]))
    new_ids = sorted(set(current) - set(original))
    allowed_drugs = {x["drug_name"] for x in plan["pilot_drugs"]}
    bad_names = []
    bad_bindings = []
    lookup_failures = []
    verified = 0
    new_drugs = Counter()
    for cid in new_ids:
        meta, doc = current[cid]
        name = str(meta.get("drug_name") or "").strip()
        new_drugs[name or "<missing>"] += 1
        if name not in allowed_drugs:
            bad_names.append(cid)
        reason = check_new_binding(meta)
        if reason:
            bad_bindings.append({"id": cid, "reason": reason})
            continue
        if verify_parent is not None:
            try:
                parent = verify_parent(meta)
                from medical_evidence_bridge import verify_json_chunk
                ok, reason = verify_json_chunk(meta, doc, parent)
                if ok:
                    verified += 1
                else:
                    lookup_failures.append({"id": cid, "reason": reason})
            except Exception as exc:
                lookup_failures.append({"id": cid, "reason": type(exc).__name__})
    blockers = []
    if missing_old:
        blockers.append("原有 Chunk ID 被删除")
    if changed_old:
        blockers.append("原有 Chunk 内容或 metadata 被改写")
    if not new_ids:
        blockers.append("本次没有新增 Chunk")
    if len(new_ids) > int(plan["pilot_limits"]["max_new_chunks"]):
        blockers.append("新增 Chunk 数超过试运行上限，需核对索引器限额")
    if bad_names:
        blockers.append("出现不属于试运行候选药品的新增 Chunk")
    if bad_bindings:
        blockers.append("新增 Chunk 元数据不完整或 data_origin 不符")
    if lookup_failures:
        blockers.append("新增 Chunk 未能通过 Neo4j 原文回溯")
    if int(after["chroma"]["record_count_after"]) != len(current):
        blockers.append("扩容后 Chroma 完整扫描数量与实际记录数不匹配")
    if int(after["chroma"]["record_count_after"]) != len(original) + len(new_ids) - len(missing_old):
        blockers.append("扩容前后记录数无法对齐")
    if blockers:
        status = "BLOCKED"
    elif verify_parent is None:
        status = "NEEDS_NEO4J_NEW_CHUNK_VERIFICATION"
    else:
        status = "PASS_PILOT_TECHNICAL"
    return {
        "schema_version": 1, "stage": "P3-02", "mode": "verify",
        "created_utc": stamp(), "plan_id": plan.get("plan_id"),
        "status": status, "blockers": blockers,
        "before_chunks": len(original), "after_chunks": len(current),
        "added_chunks": len(new_ids), "missing_old_chunks": missing_old[:25],
        "changed_old_chunks": changed_old[:25], "new_drugs": dict(new_drugs),
        "invalid_drug_ids_count": len(bad_names),
        "invalid_bindings": bad_bindings[:25],
        "neo4j_provenance_verified": verified,
        "neo4j_provenance_failures": lookup_failures[:25],
        "all_new_chunk_ids_sha256": object_sha(new_ids),
        "notes": [
            "PASS_PILOT_TECHNICAL 仅说明此次试运行的技术身份与原文校验通过，不代表来源权威或临床审核。",
            "仍需核对 SQLite checkpoint 状态、调用费用、全部业务日志及备份/恢复记录。",
        ],
    }


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 文件应为对象：{path}")
    return data


def new_report(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def open_existing_collection(chroma_path: Path, collection: str) -> Any:
    if not (chroma_path / "chroma.sqlite3").is_file():
        raise ValueError("找不到 Chroma 已有数据目录，请核对 --chroma-path")
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_path.resolve()))
    return client.get_collection(name=collection, embedding_function=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V1.17 / P3-02 小批次扩容只读规划、回溯验收")
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan", help="规划未覆盖药品，不执行 embedding/写入")
    p.add_argument("--readiness", type=Path, required=True, help="P3-01 完整 PASS_PRECHECK 报告")
    p.add_argument("--raw-json", type=Path, required=True)
    p.add_argument("--indexer", type=Path, required=True)
    p.add_argument("--chunker", type=Path, required=True)
    p.add_argument("--chroma-path", type=Path, required=True)
    p.add_argument("--collection", default="rag")
    p.add_argument("--max-drugs", type=int, default=5)
    p.add_argument("--char-budget", type=int, default=20000)
    p.add_argument("--max-drug-chars", type=int, default=8000)
    p.add_argument("--max-new-chunks", type=int, default=40)
    p.add_argument("--output", type=Path, required=True)

    v = sub.add_parser("verify", help="既有批处理工具运行后，审计是否只新增预定 Chunk")
    v.add_argument("--plan", type=Path, required=True)
    v.add_argument("--after-readiness", type=Path, required=True)
    v.add_argument("--chroma-path", type=Path, required=True)
    v.add_argument("--collection", default="rag")
    v.add_argument("--with-neo4j", action="store_true",
                   help="对全部新 Chunk 调用既有 fetch_parent_section + verify_json_chunk")
    v.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.output.exists():
            raise FileExistsError(f"已存在报告，禁止覆盖：{args.output}")
        collection = open_existing_collection(args.chroma_path, args.collection)
        if args.mode == "plan":
            if not (1 <= args.max_drugs <= 20 and
                    1000 <= args.char_budget <= 200000 and
                    200 <= args.max_drug_chars <= args.char_budget and
                    1 <= args.max_new_chunks <= 500):
                raise ValueError("试运行限额超出安全范围")
            readiness = read_json(args.readiness)
            validate_readiness(readiness)
            source_rows, source_stat = inspect_source(args.raw_json)
            indexer = inspect_script(args.indexer)
            chunker = inspect_script(args.chunker)
            sigs, names, ids, count = scan_collection(collection)
            result = build_plan(
                readiness=readiness, source_rows=source_rows, source_stat=source_stat,
                source_hash=file_sha(args.raw_json),
                indexer_stat=indexer, chunker_stat=chunker,
                signatures=sigs, existing_names=names, existing_ids=ids,
                count=count, chroma_path=str(args.chroma_path),
                collection_name=args.collection, max_drugs=args.max_drugs,
                char_budget=args.char_budget, max_drug_chars=args.max_drug_chars,
                max_new_chunks=args.max_new_chunks
            )
        else:
            plan = read_json(args.plan)
            after = read_json(args.after_readiness)
            if args.collection != str(plan.get("chroma", {}).get("collection")):
                raise ValueError("Collection 与规划不一致")
            if args.chroma_path.resolve() != Path(str(plan.get("chroma", {}).get("path"))).resolve():
                raise ValueError("Chroma 路径与规划不一致")
            sigs, _, _, count = scan_collection(collection)
            # 验收需要真实 metadata 与原文；不读取向量。
            current = {}
            offset = 0
            while offset < count:
                page = collection.get(limit=min(200, count-offset), offset=offset,
                                      include=["metadatas", "documents"])
                for cid, meta, doc in zip(page["ids"], page["metadatas"], page["documents"]):
                    current[str(cid)] = (meta or {}, doc or "")
                offset += len(page["ids"])
            if int(collection.count()) != count or len(current) != count:
                raise ValueError("验收期间 Chroma 数据发生变化")
            if any(sigs[cid] != identity_digest(*data) for cid, data in current.items()):
                raise ValueError("两次扫描的数据不一致，暂停验收")
            if args.with_neo4j:
                from medical_stage3_readiness import _load_local_env
                from drug_graph_retriever import DrugGraphRetriever
                _load_local_env()
                graph = DrugGraphRetriever()
                try:
                    result = verify_pilot(plan, after, current,
                                          verify_parent=graph.fetch_parent_section)
                finally:
                    graph.close()
            else:
                result = verify_pilot(plan, after, current)
        new_report(args.output, result)
        print(f"[P3-02] {result['status']}")
        print(f"报告：{args.output.resolve()}")
        return 0 if result["status"] in (
            "PILOT_PLANNED_NOT_AUTHORIZED", "PASS_PILOT_TECHNICAL"
        ) else 2
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"[P3-02] 检查停止：{exc}\n")


if __name__ == "__main__":
    sys.exit(main())
