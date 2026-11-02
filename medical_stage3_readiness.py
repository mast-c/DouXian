# -*- coding: utf-8 -*-
"""豆馅医疗 GraphRAG V1.17 / 阶段 3 P3-01：扩容前只读盘点。

读取 Chroma collection、可选 SQLite checkpoint、可选 Neo4j，生成一个新 JSON 报告。
不调用 Embedding/Qwen，不写入任何已有 Collection、Neo4j 节点或 checkpoint。
注意：Chroma PersistentClient 的底层 SDK 可能维护内部文件；若要求文件系统级只读，
请在备份副本上运行，且不要与正在写入的索引任务并发执行。

示例（项目根目录内）：
  python medical_stage3_readiness.py --chroma-path ./chroma_db \\
    --collection rag --output ./stage3_reports/p3_01_readiness.json

可选增加：--checkpoint ./path/to/state.sqlite --with-neo4j --sample-size 30
Neo4j 连接信息来自同目录 .env 或环境变量 NEO4J_URI / NEO4J_PASSWORD。
所有错误或不完整状态在报告中标注；PASS_PRECHECK 不等于允许直接启动写入。
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import heapq
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any


JSON_ORIGIN = "json_matched_to_neo4j_section"
REQUIRED_BINDING = ("drug_id", "record_id", "section_id", "source_url", "section_sha256")
GRAPH_LABELS = (
    "Drug", "SourceRecord", "Section", "Chunk", "DrugAlias",
    "DiseaseTerm", "DiseaseCandidate", "ModelAssessment",
)
STATUS_COLUMNS = ("status", "state", "result", "processing_status", "stage")


def _counters(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_binding(meta: dict[str, Any]) -> bool:
    if not all(_nonempty(meta.get(key)) for key in REQUIRED_BINDING):
        return False
    try:
        start, end = int(meta["quote_start"]), int(meta["quote_end"])
    except (KeyError, TypeError, ValueError):
        return False
    return start >= 0 and end > start


def _sample_id(heap: list[tuple[int, str]], item_id: str, size: int) -> None:
    """用 ID 的稳定 hash 挑选分布式样本，避免只检验前 N 个 Chunk。"""
    if size <= 0:
        return
    rank = int.from_bytes(hashlib.sha256(item_id.encode("utf-8")).digest()[:8], "big")
    entry = (-rank, item_id)
    if len(heap) < size:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def audit_collection(collection: Any, *, batch_size: int = 250,
                     scan_limit: int = 0, sample_size: int = 30
                     ) -> tuple[dict[str, Any], list[str]]:
    """仅调用 count/get；按页扫描 metadata，不读取原文或 embedding。"""
    count_before = int(collection.count())
    target = min(count_before, scan_limit) if scan_limit else count_before
    status = Counter()
    fields = Counter()
    origins = Counter()
    missing = Counter()
    seen_drugs: set[str] = set()
    seen_drug_names: set[str] = set()
    identity_keys: set[tuple[str, str, str, str, str]] = set()
    seen_chunk_ids: set[str] = set()
    duplicates = 0
    repeated_ids = 0
    bound_total = 0
    bound_valid = 0
    sample_heap: list[tuple[int, str]] = []
    offset = 0
    started = time.perf_counter()

    while offset < target:
        limit = min(batch_size, target - offset)
        page = collection.get(limit=limit, offset=offset, include=["metadatas"])
        ids = page.get("ids") or []
        metas = page.get("metadatas") or []
        if not ids:
            break
        if len(ids) != len(metas):
            raise RuntimeError("Chroma 返回的 ids / metadatas 数量不一致")
        for chunk_id, raw_meta in zip(ids, metas):
            chunk_id = str(chunk_id)
            if chunk_id in seen_chunk_ids:
                repeated_ids += 1
            seen_chunk_ids.add(chunk_id)
            meta = raw_meta if isinstance(raw_meta, dict) else {}
            origins[str(meta.get("data_origin") or "<missing>")] += 1
            fields[str(meta.get("section_field") or "<missing>")] += 1
            status[str(meta.get("source_status") or "<missing>")] += 1
            if _nonempty(meta.get("drug_id")):
                seen_drugs.add(meta["drug_id"].strip())
            if _nonempty(meta.get("drug_name")):
                seen_drug_names.add(meta["drug_name"].strip())

            if meta.get("data_origin") == JSON_ORIGIN:
                bound_total += 1
                for required in REQUIRED_BINDING:
                    if not _nonempty(meta.get(required)):
                        missing[required] += 1
                if _valid_binding(meta):
                    bound_valid += 1
                else:
                    missing["invalid_or_missing_offsets_or_binding"] += 1
                identity = (
                    str(meta.get("drug_id") or ""),
                    str(meta.get("record_id") or ""),
                    str(meta.get("section_id") or ""),
                    str(meta.get("quote_start") or ""),
                    str(meta.get("quote_end") or ""),
                )
                if identity in identity_keys:
                    duplicates += 1
                else:
                    identity_keys.add(identity)
                _sample_id(sample_heap, chunk_id, sample_size)
        offset += len(ids)

    count_after = int(collection.count())
    report = {
        "record_count_before": count_before,
        "record_count_after": count_after,
        "scanned_records": offset,
        "scan_complete": offset == count_before and count_before == count_after,
        "scan_requested": target,
        "scan_seconds": round(time.perf_counter() - started, 3),
        "unique_drug_ids_in_scanned_metadata": len(seen_drugs),
        "unique_drug_names_in_scanned_metadata": len(seen_drug_names),
        "origin_counts": _counters(origins),
        "section_field_counts": _counters(fields),
        "source_status_counts": _counters(status),
        "json_bound_chunks": bound_total,
        "json_bound_metadata_complete": bound_valid,
        "json_binding_missing_counts": _counters(missing),
        "duplicate_json_quote_coordinates": duplicates,
        "repeated_chunk_ids_across_pages": repeated_ids,
        "sampled_json_chunk_ids_count": len(sample_heap),
    }
    return report, sorted(item_id for _, item_id in sample_heap)


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def audit_checkpoint(path: Path) -> dict[str, Any]:
    """SQLite URI mode=ro + query_only；不创建或更新 checkpoint。"""
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 路径不存在或不是文件：{path}")
    result: dict[str, Any] = {"path": str(path.resolve()), "tables": {}}
    # Path.as_uri 正确转义 Windows 盘符、空格以及路径中的特殊字符。
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as db:
        db.execute("PRAGMA query_only=ON")
        names = [row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        for name in names:
            qname = _quote_ident(name)
            columns = [row[1] for row in db.execute(f"PRAGMA table_info({qname})")]
            count = int(db.execute(f"SELECT COUNT(*) FROM {qname}").fetchone()[0])
            table: dict[str, Any] = {"rows": count, "columns": columns}
            status_column = next((col for col in columns if col.lower() in STATUS_COLUMNS), None)
            if status_column:
                qcol = _quote_ident(status_column)
                values = db.execute(
                    f"SELECT {qcol}, COUNT(*) FROM {qname} GROUP BY {qcol} "
                    "ORDER BY COUNT(*) DESC LIMIT 20"
                ).fetchall()
                table["status_column"] = status_column
                table["status_counts"] = {
                    str(value if value is not None else "<NULL>")[:60]: int(n)
                    for value, n in values
                }
            result["tables"][name] = table
    result["tables_found"] = len(result["tables"])
    return result


def _load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def audit_neo4j_and_sample(collection: Any, sample_ids: list[str],
                           database: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """通过现有检索模块只读读取父级 Section，并复用现有逐字校验函数。"""
    _load_local_env()
    uri = str(os.getenv("NEO4J_URI") or "").strip()
    password = str(os.getenv("NEO4J_PASSWORD") or "")
    user = str(os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j")
    if not uri or not password:
        raise ValueError("缺少 NEO4J_URI / NEO4J_PASSWORD（请在 .env 或环境变量配置）")
    from neo4j import GraphDatabase, READ_ACCESS
    from drug_graph_retriever import DrugGraphRetriever
    from medical_evidence_bridge import verify_json_chunk

    graph_report: dict[str, Any] = {"database": database or os.getenv("NEO4J_DATABASE", "neo4j")}
    sample_report: dict[str, Any] = {"requested": len(sample_ids), "verified": 0, "failed": 0, "reasons": {}}
    started = time.perf_counter()
    with GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5.0) as driver:
        target_db = str(graph_report["database"])
        with driver.session(database=target_db, default_access_mode=READ_ACCESS) as session:
            counts: dict[str, int] = {}
            for label in GRAPH_LABELS:  # 固定白名单，不能被用户构造为任意 Cypher
                cypher = f"MATCH (n:{label}) RETURN count(n) AS n"
                counts[label] = int(session.execute_read(
                    lambda tx, q: tx.run(q).single()["n"], cypher
                ))
        graph_report["node_counts"] = counts
        graph_report["read_seconds"] = round(time.perf_counter() - started, 3)
        fetcher = DrugGraphRetriever(driver=driver)
        fetcher.database = target_db
        if sample_ids:
            retrieved = collection.get(ids=sample_ids, include=["metadatas", "documents"])
            ids, metas, docs = (retrieved.get("ids") or [],
                                 retrieved.get("metadatas") or [], retrieved.get("documents") or [])
            got = {str(i): (m, d) for i, m, d in zip(ids, metas, docs)}
            issues: Counter[str] = Counter()
            for chunk_id in sample_ids:
                if chunk_id not in got:
                    issues["chroma_sample_missing"] += 1
                    continue
                meta, document = got[chunk_id]
                if not isinstance(meta, dict) or not isinstance(document, str):
                    issues["invalid_chroma_record"] += 1
                    continue
                try:
                    parent = fetcher.fetch_parent_section(meta)
                    passed, reason = verify_json_chunk(meta, document, parent)
                except Exception as exc:
                    passed, reason = False, "exception_" + type(exc).__name__
                if passed:
                    sample_report["verified"] += 1
                else:
                    issues[reason] += 1
            sample_report["failed"] = sum(issues.values())
            sample_report["reasons"] = _counters(issues)
        sample_report["complete"] = (
            sample_report["verified"] + sample_report["failed"] == len(sample_ids)
        )
    return graph_report, sample_report


def readiness_gates(chroma: dict[str, Any], checkpoint: dict[str, Any] | None,
                    neo4j: dict[str, Any] | None,
                    sample: dict[str, Any] | None) -> dict[str, Any]:
    """只评价是否获得充分的扩容前诊断证据，不授权实际导入。"""
    blockers: list[str] = []
    missing_checks: list[str] = []
    if chroma["record_count_before"] == 0:
        missing_checks.append("Chroma Collection 为空；先确认是否选择了正确的路径/集合")
    if not chroma["scan_complete"]:
        missing_checks.append("元数据未扫描全部，或扫描期间 Collection 数量发生变化")
    if chroma["duplicate_json_quote_coordinates"]:
        blockers.append("不同 Chunk ID 重复使用同一原文坐标，需人工核对")
    if chroma["repeated_chunk_ids_across_pages"]:
        blockers.append("分页查询存在重复 Chunk ID，请暂停并排查并发写入")
    if chroma["json_binding_missing_counts"]:
        blockers.append("部分 JSON Chunk 来源绑定或原文偏移字段缺失/无效")
    if checkpoint is None:
        missing_checks.append("未读取 SQLite checkpoint，无法确认批处理断点一致性")
    else:
        tables = checkpoint.get("tables") or {}
        stateful_tables = [x for x in tables.values() if x.get("status_column")]
        if not stateful_tables:
            missing_checks.append("checkpoint 未发现已识别的状态列，需按实际 schema 人工复核")
        fail_count = sum(
            int(count)
            for table in stateful_tables
            for status, count in table.get("status_counts", {}).items()
            if status.strip().lower() in {"failed", "failure", "error", "fatal"}
        )
        if fail_count:
            blockers.append(f"checkpoint 包含 {fail_count} 条失败/错误状态，请先检查恢复策略")
    if neo4j is None:
        missing_checks.append("未连接 Neo4j，无法核对知识图谱当前数量和原文关联")
    if chroma["json_bound_chunks"] == 0:
        missing_checks.append("没有检测到可进行 JSON→Neo4j 精确回溯的 Chunk")
    elif sample is None or sample.get("requested", 0) == 0:
        missing_checks.append("尚未执行原文回溯抽样验证")
    elif sample.get("failed", 0):
        blockers.append("Neo4j Section 原文回溯抽样存在失败记录")
    elif not sample.get("complete"):
        missing_checks.append("原文回溯抽样未完成")
    return {
        "status": "BLOCKED" if blockers else (
            "INSUFFICIENT_CHECKS" if missing_checks else "PASS_PRECHECK"
        ),
        "blockers": blockers,
        "missing_checks": missing_checks,
        "interpretation": "即便 PASS_PRECHECK，也仅完成只读数据检查；正式扩容仍需备份、预算、权限和批处理干运行确认",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="豆馅医疗 V1.17 阶段 3：只读知识库扩容前状态盘点")
    parser.add_argument("--chroma-path", type=Path, required=True)
    parser.add_argument("--collection", default="rag")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--with-neo4j", action="store_true")
    parser.add_argument("--neo4j-database", default=None)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--scan-limit", type=int, default=0, help="0 表示扫描全部元数据")
    parser.add_argument("--output", required=True, type=Path, help="新 JSON 文件；拒绝覆盖已有文件")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 1000 or not 0 <= args.sample_size <= 200 or args.scan_limit < 0:
        parser.error("batch-size 必须为1~1000，sample-size 为0~200，scan-limit 不得为负数")
    if args.output.exists():
        parser.error(f"报告路径已经存在，拒绝覆盖：{args.output}")
    path = args.chroma_path.expanduser().resolve()
    if not path.is_dir() or not (path / "chroma.sqlite3").is_file():
        parser.error(f"找不到现有 Chroma PersistentClient 数据目录（含 chroma.sqlite3）：{path}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "project_version": "V1.17",
        "stage": "P3-01",
        "audit_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "read_only_application_calls": True,
        "notes": [
            "不会调用模型、embedding 或批量索引写入接口",
            "本报告仅用于工程扩容前盘点；不等于医学权威来源核验",
            "扫描期间不要并发修改 Chroma；如需文件系统级只读，请先对数据目录做备份副本",
            "只生成新报告，不修改或覆盖已有 checkpoint",
        ],
        "chroma": {"path": str(path), "collection": args.collection},
        "checkpoint": None,
        "neo4j": None,
        "crossstore_sample": None,
    }
    try:
        import chromadb
        report["chroma"]["sdk_version"] = str(getattr(chromadb, "__version__", "unknown"))
        client = chromadb.PersistentClient(path=str(path))
        # 只用 get_collection；禁止任何 get_or_create / add / upsert / delete。
        collection = client.get_collection(name=args.collection, embedding_function=None)
        summary, sample_ids = audit_collection(
            collection, batch_size=args.batch_size,
            scan_limit=args.scan_limit, sample_size=args.sample_size,
        )
        report["chroma"].update(summary)
        if args.checkpoint:
            report["checkpoint"] = audit_checkpoint(args.checkpoint.expanduser())
        if args.with_neo4j:
            graph, sample = audit_neo4j_and_sample(
                collection, sample_ids, args.neo4j_database
            )
            report["neo4j"] = graph
            report["crossstore_sample"] = sample
        report["gates"] = readiness_gates(
            report["chroma"], report["checkpoint"],
            report["neo4j"], report["crossstore_sample"],
        )
        exit_code = 0 if report["gates"]["status"] == "PASS_PRECHECK" else 2
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:600]}
        report["gates"] = {
            "status": "ERROR", "blockers": ["盘点执行失败，不能判断扩容条件"],
            "missing_checks": [],
        }
        exit_code = 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print("[P3-01] status:", report["gates"]["status"])
    if report.get("chroma", {}).get("record_count_before") is not None:
        print("[Chroma] chunks:", report["chroma"]["record_count_before"],
              "scanned:", report["chroma"]["scanned_records"])
        print("[Chroma] unique drug_id:",
              report["chroma"]["unique_drug_ids_in_scanned_metadata"])
    print("[Report]", args.output.resolve())
    if "error" in report:
        print("[Error]", report["error"]["type"], report["error"]["message"], file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
