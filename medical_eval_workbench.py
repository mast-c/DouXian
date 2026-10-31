#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""豆馅 V1.13：医疗知识库只读审计、候选题库、人工标注后评测。

可执行：
  python medical_eval_workbench.py audit
  python medical_eval_workbench.py draft --count 80
  python medical_eval_workbench.py score --reviews medical_gold_review_v113.csv --mode bm25
  python medical_eval_workbench.py score --reviews medical_gold_review_v113.csv --mode hybrid-rerank --allow-cloud-query

不改已有 Chroma / Neo4j / rag.py / knowledge_base.py。
draft 生成的 suggested_section_id 只是标注候选；人工确认为 gold_section_ids
之前，绝不能当成 Recall@5 评测金标准。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ORIGIN = "json_matched_to_neo4j_section"
FIELDS = ("indication", "contraindications", "adverse_reactions",
          "warnings", "drug_interactions")
FIELD_NAMES = {
    "indication": "适应症",
    "contraindications": "禁忌",
    "adverse_reactions": "不良反应",
    "warnings": "注意事项",
    "drug_interactions": "药物相互作用",
}
QUESTION_TEMPLATES = {
    "indication": ("{drug}的适应症是什么？请给出对应原文。",
                   "资料中如何描述{drug}的适用人群？"),
    "contraindications": ("{drug}有哪些禁忌？请给出对应原文。",
                          "根据已有药品资料，{drug}的禁用条件有哪些？"),
    "adverse_reactions": ("{drug}有哪些不良反应？请引用原文。",
                          "{drug}的不良反应栏目记录了哪些信息？"),
    "warnings": ("{drug}的注意事项是什么？请给出对应原文。",
                 "使用{drug}有哪些需要注意的事项？"),
    "drug_interactions": ("{drug}有哪些药物相互作用信息？",
                          "{drug}的相互作用栏目写了什么？"),
}
COLUMNS = (
    "case_id", "question", "target_drugs", "expected_field",
    "suggested_section_id", "suggested_record_id", "suggested_source_url",
    "suggested_evidence_preview", "annotation_status",
    "answerable", "gold_section_ids", "relation_semantics",
    "source_authority_checked", "reviewer", "review_note",
)
STATUSES = {"unreviewed", "reviewed", "excluded"}
ANSWERABILITY = {"yes", "no", "uncertain"}
SEMANTIC_CLASSES = {
    "", "treatment_target", "prevention", "risk_factor", "comorbidity",
    "contraindication", "adverse_event", "background_mention", "other",
}


def check_write_destination(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"文件已存在，已保护不覆盖人工审核成果：{path}。"
            "请换一个 --output 文件名，或明确传 --overwrite。"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_csv_cell(value: Any) -> str:
    value = "" if value is None else str(value)
    # 防止把来源资料中的不可信前缀当 Excel 公式执行。
    return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def _read_csv_cell(value: Any) -> str:
    value = str(value or "").strip()
    return value[1:] if len(value) >= 2 and value[0] == "'" and value[1] in "=+-@\t\r" else value


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def actual_collection() -> tuple[Any, str]:
    """只获取现有 Collection，不创建不存在的新 Collection。"""
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    import config_data as config
    root = Path(config.persist_directory).expanduser().resolve()
    if not (root / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"找不到现有 Chroma：{root / 'chroma.sqlite3'}。"
            "请从原项目目录运行，并检查 config_data.py。"
        )
    import chromadb
    collection = chromadb.PersistentClient(path=str(root)).get_collection(
        name=str(config.collection_name)
    )
    return collection, f"{root} / {config.collection_name}"


def scan_collection(
    collection: Any, *, batch_size: int = 300
) -> list[tuple[str, str, dict[str, Any]]]:
    """分页扫描现有 Chroma；如果扫描期间计数改变，拒绝不一致快照。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    expected = int(collection.count())
    collected: list[tuple[str, str, dict[str, Any]]] = []
    offset = 0
    while offset < expected:
        payload = collection.get(
            limit=min(batch_size, expected - offset),
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = payload.get("ids") or []
        documents = payload.get("documents") or []
        metadata = payload.get("metadatas") or []
        if not ids or len(ids) != len(documents) or len(ids) != len(metadata):
            raise RuntimeError("Chroma 扫描返回数量不一致；请暂停并发写入再重试")
        collected.extend(
            (str(chunk_id), str(content or ""), dict(meta or {}))
            for chunk_id, content, meta in zip(ids, documents, metadata)
        )
        offset += len(ids)
    if int(collection.count()) != expected:
        raise RuntimeError("扫描期间 Chroma 数量发生变化；请停止批量写入再重试")
    return collected


def collection_fingerprint(
    records: Iterable[tuple[str, str, dict[str, Any]]]
) -> str:
    """对 Chunk ID 与真实正文计算稳定指纹，避免不同数据库快照直接比较。"""
    lines = sorted(
        f"{chunk_id}:{_hash(text)}" for chunk_id, text, _ in records
    )
    return _hash("\n".join(lines))


def audit_index(records: Iterable[tuple[str, str, dict[str, Any]]]
                ) -> dict[str, Any]:
    rows = list(records)
    origin_counts: Counter[str] = Counter()
    field_chunks: Counter[str] = Counter()
    field_drugs: dict[str, set[str]] = defaultdict(set)
    drug_names: set[str] = set()
    parent_ids: set[str] = set()
    source_records: set[str] = set()
    source_urls: set[str] = set()
    errors: list[dict[str, str]] = []
    duplicate_ids = 0
    seen_ids: set[str] = set()
    empty_document_count = 0
    for cid, content, m in rows:
        if cid in seen_ids:
            duplicate_ids += 1
        seen_ids.add(cid)
        if not content.strip():
            empty_document_count += 1
        origin = str(m.get("data_origin") or "legacy_or_other")
        origin_counts[origin] += 1
        if origin != ORIGIN:
            continue
        drug = str(m.get("drug_name") or "")
        field = str(m.get("section_field") or "")
        parent = str(m.get("section_id") or "")
        record = str(m.get("record_id") or "")
        source_url = str(m.get("source_url") or "")
        if drug:
            drug_names.add(drug)
            field_drugs[field].add(drug)
        if parent:
            parent_ids.add(parent)
        if record:
            source_records.add(record)
        if source_url:
            source_urls.add(source_url)
        field_chunks[field] += 1
        for name, value in (("drug_name", drug), ("section_id", parent),
                            ("record_id", record), ("section_field", field)):
            if not value:
                errors.append({"chunk_id": cid, "reason": f"缺少 {name}"})
        if str(m.get("chunk_id") or "") != cid:
            errors.append({"chunk_id": cid, "reason": "chunk_id Metadata 不一致"})
        if m.get("content_sha256") and m.get("content_sha256") != _hash(content):
            errors.append({"chunk_id": cid, "reason": "正文 SHA256 不一致"})
    return {
        "indexed_chunks": len(rows),
        "collection_fingerprint": collection_fingerprint(rows),
        "structured_json_chunks": origin_counts[ORIGIN],
        "legacy_or_other_chunks": len(rows) - origin_counts[ORIGIN],
        "drugs_covered": len(drug_names),
        "parent_sections": len(parent_ids),
        "source_records": len(source_records),
        "source_urls": len(source_urls),
        "chunks_by_field": dict(sorted(field_chunks.items())),
        "drugs_by_field": {k: len(v) for k, v in sorted(field_drugs.items())},
        "data_origins": dict(origin_counts),
        "empty_documents": empty_document_count,
        "duplicate_chunk_ids": duplicate_ids,
        "metadata_issues": len(errors),
        "metadata_issue_examples": errors[:12],
        "caution": (
            "统计是索引覆盖和元数据检查，不代表临床适应症有效性或权威来源通过。"
        ),
    }


def checkpoint_summary(path: Path) -> dict[str, Any]:
    """SQLite 只读，检查已处理、无正文和失败药品的实际数量。"""
    if not path.is_file():
        return {"present": False, "path": str(path.resolve())}
    # Path.as_uri 对 Windows 盘符和 Linux 路径都生成正确的 file:// URI。
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        names = {
            str(row[0]) for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "drug_progress" not in names:
            return {
                "present": True, "path": str(path.resolve()),
                "warning": "检查点没有 drug_progress 表",
            }
        statuses = {
            str(state): int(count)
            for state, count in conn.execute(
                "SELECT status, COUNT(*) FROM drug_progress GROUP BY status"
            )
        }
        return {"present": True, "path": str(path.resolve()), "statuses": statuses}
    finally:
        conn.close()


def _drug_sections(
    records: Iterable[tuple[str, str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    sections: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cid, text, meta in records:
        if meta.get("data_origin") != ORIGIN:
            continue
        drug = str(meta.get("drug_name") or "").strip()
        field = str(meta.get("section_field") or "").strip()
        section = str(meta.get("section_id") or "").strip()
        record = str(meta.get("record_id") or "").strip()
        if not all((drug, field in FIELDS, section, record)):
            continue
        key = (drug, field, section)
        if key in sections:
            continue
        preview = text.split("证据原文：", 1)[-1].strip()
        sections[key] = {
            "drug": drug, "field": field, "section_id": section,
            "record_id": record, "source_url": str(meta.get("source_url") or ""),
            "preview": preview[:240],
        }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sections.values():
        grouped[row["field"]].append(row)
    return grouped


def make_review_draft(
    records: list[tuple[str, str, dict[str, Any]]],
    *, count: int = 80,
    seed: int = 20261030,
) -> list[dict[str, str]]:
    """按药品和说明书栏目分层采样，来源候选须由人核对才能成为 gold。"""
    if not 1 <= count <= 500:
        raise ValueError("count 必须在 1～500 之间")
    grouped = _drug_sections(records)
    for field in grouped:
        grouped[field].sort(
            key=lambda row: _hash(
                f"{seed}|{row['drug']}|{row['field']}|{row['section_id']}"
            )
        )
    selections = []
    seen_drugs: set[str] = set()
    used_keys: set[tuple[str, str, str]] = set()
    while len(selections) < count:
        progress = False
        for field in FIELDS:
            options = grouped.get(field, [])
            # 优先跨药品多样性。该字段药品均已用过时允许同药不同栏目。
            candidate = next(
                (row for row in options
                 if row["drug"] not in seen_drugs and
                 (row["drug"], field, row["section_id"]) not in used_keys),
                None,
            )
            if candidate is None:
                candidate = next(
                    (row for row in options
                     if (row["drug"], field, row["section_id"]) not in used_keys),
                    None,
                )
            if candidate is None:
                continue
            key = (candidate["drug"], field, candidate["section_id"])
            used_keys.add(key)
            seen_drugs.add(candidate["drug"])
            idx = len(selections) + 1
            templates = QUESTION_TEMPLATES[field]
            question = templates[(idx - 1) % len(templates)].format(
                drug=candidate["drug"]
            )
            selections.append({
                "case_id": f"MED{idx:03d}",
                "question": question,
                "target_drugs": candidate["drug"],
                "expected_field": field,
                "suggested_section_id": candidate["section_id"],
                "suggested_record_id": candidate["record_id"],
                "suggested_source_url": candidate["source_url"],
                "suggested_evidence_preview": candidate["preview"],
                "annotation_status": "unreviewed",
                "answerable": "",
                "gold_section_ids": "",
                "relation_semantics": "",
                "source_authority_checked": "no",
                "reviewer": "",
                "review_note": "",
            })
            progress = True
            if len(selections) >= count:
                break
        if not progress:
            break
    return selections


def save_review_csv(path: Path, cases: list[dict[str, str]],
                    *, overwrite: bool = False) -> None:
    check_write_destination(path, overwrite=overwrite)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        for case in cases:
            writer.writerow({key: _safe_csv_cell(case.get(key, ""))
                             for key in COLUMNS})


def load_review_csv(path: Path) -> list[dict[str, Any]]:
    """严格区分未审核草案与已确认 gold；从不自动批准 suggestion。"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(COLUMNS).issubset(reader.fieldnames):
            raise ValueError("评测 CSV 缺少必要列，请使用 draft 生成的模板")
        seen: set[str] = set()
        out = []
        for line_no, row in enumerate(reader, 2):
            case = {key: _read_csv_cell(row.get(key)) for key in COLUMNS}
            case_id = case["case_id"]
            if not case_id or case_id in seen:
                raise ValueError(f"第 {line_no} 行 case_id 为空或重复")
            seen.add(case_id)
            status = case["annotation_status"]
            if status not in STATUSES:
                raise ValueError(f"{case_id} 的 annotation_status 无效: {status}")
            if status != "reviewed":
                continue
            answerable = case["answerable"]
            if answerable not in ANSWERABILITY:
                raise ValueError(f"{case_id} 必须明确标注 answerable=yes/no/uncertain")
            if not case["reviewer"]:
                raise ValueError(f"{case_id} 缺少人工审核人 reviewer")
            if case["expected_field"] not in FIELDS:
                raise ValueError(f"{case_id} 的 expected_field 不是已支持栏目")
            if case["relation_semantics"] not in SEMANTIC_CLASSES:
                raise ValueError(f"{case_id} 的 relation_semantics 不在允许的分类中")
            gold = list(dict.fromkeys(
                x.strip() for x in case["gold_section_ids"].split("|")
                if x.strip()
            ))
            if answerable == "yes" and not gold:
                raise ValueError(f"{case_id} 标注为有答案，必须填写 gold_section_ids")
            if answerable == "no" and gold:
                raise ValueError(f"{case_id} 标注为无答案，不应填写 gold_section_ids")
            case["gold_section_ids"] = gold
            case["target_drugs"] = [
                x.strip() for x in case["target_drugs"].split("|") if x.strip()
            ]
            out.append(case)
    return out


def _meta(doc: Any) -> dict[str, Any]:
    m = getattr(doc, "metadata", None)
    return m if isinstance(m, dict) else {}


def score_one(
    case: dict[str, Any],
    documents: Iterable[Any],
    *, k: int = 5,
) -> dict[str, Any]:
    """匹配 Neo4j Section ID，而不是按药品名或相同词汇虚报召回。"""
    from medical_entity_gate import filter_documents_for_drugs, query_drug_names

    pool = list(documents)
    raw = pool[:k]
    names = query_drug_names(
        case["question"], (_meta(doc) for doc in pool),
    )
    scoped = filter_documents_for_drugs(pool, names)[:k]
    gold = set(case["gold_section_ids"])
    expected_drugs = set(case["target_drugs"])

    def summarize(items: list[Any]) -> dict[str, Any]:
        section_ids = [
            str(_meta(doc).get("section_id") or "") for doc in items
        ]
        matched = sorted(set(section_ids) & gold)
        foreign = sum(
            bool(_meta(doc).get("drug_name")) and
            str(_meta(doc)["drug_name"]) not in expected_drugs
            for doc in items
        )
        parent_count = sum(bool(x) for x in section_ids)
        return {
            "matched_gold_section_ids": matched,
            "section_hit": bool(matched),
            "section_recall": round(len(matched) / len(gold), 4) if gold else None,
            "returned_chunks": len(items),
            "wrong_drug_chunks": foreign,
            "traceable_parent_chunks": parent_count,
            "predicted_sections": [
                {
                    "section_id": str(_meta(doc).get("section_id") or ""),
                    "drug_name": str(_meta(doc).get("drug_name") or ""),
                    "section_field": str(_meta(doc).get("section_field") or ""),
                    "record_id": str(_meta(doc).get("record_id") or ""),
                }
                for doc in items
            ],
        }
    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "gold_section_ids": sorted(gold),
        "expected_field": case["expected_field"],
        "target_drugs": case["target_drugs"],
        "detected_query_drugs": list(names),
        "raw": summarize(raw),
        "scoped": summarize(scoped),
    }


def overall(scores: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    if not scores:
        raise ValueError("没有已审核且有正确 Section 的问题，禁止计算 Recall")
    report: dict[str, Any] = {"positive_reviewed_cases": len(scores), "top_k": k}
    for stage in ("raw", "scoped"):
        rows = [s[stage] for s in scores]
        chunks = sum(s["returned_chunks"] for s in rows)
        report[stage] = {
            f"section_hit_at_{k}": round(
                sum(s["section_hit"] for s in rows) / len(rows), 4
            ),
            f"mean_section_recall_at_{k}": round(
                statistics.mean(s["section_recall"] for s in rows), 4
            ),
            "wrong_drug_chunk_rate": (
                round(sum(s["wrong_drug_chunks"] for s in rows) / chunks, 4)
                if chunks else None
            ),
            "average_returned_chunks": round(chunks / len(rows), 2),
            "zero_hit_questions": [
                s["case_id"] for s in scores if not s[stage]["section_hit"]
            ],
        }
    report["interpretation"] = (
        "Recall@K 只针对人工核对的原文 Section ID；"
        "不是药品疗效/医学有效性或最终答案准确率。"
        "scoped 为当前 Entity Gate 的后过滤口径，"
        "不代表检索模型在同一预算内重新补齐结果。"
    )
    return report


def search_knowledge_service(service: Any, *, mode: str, question: str,
                             top_k: int) -> list[Any]:
    pool_k = max(20, top_k)
    if mode == "bm25":
        return service.search_bm25(question)[:pool_k]
    if mode == "hybrid":
        return service.search_hybrid(question, k=pool_k)
    if mode == "hybrid-rerank":
        return service.search_with_rerank(
            question, recall_k=pool_k, top_n=min(pool_k, max(10, top_k))
        )
    raise ValueError(f"不支持的模式: {mode}")


def run_audit(args: Any) -> int:
    collection, label = actual_collection()
    records = scan_collection(collection)
    audit = audit_index(records)
    audit["collection"] = label
    audit["checkpoint"] = checkpoint_summary(Path(args.state))
    output = Path(args.output)
    check_write_destination(output, overwrite=args.overwrite)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("=== Chroma 医疗扩容后真实存量审计 ===")
    for key in ("indexed_chunks", "structured_json_chunks", "drugs_covered",
                "parent_sections", "source_records", "metadata_issues"):
        print(f"{key}: {audit[key]}")
    print("各字段分布:", json.dumps(audit["chunks_by_field"], ensure_ascii=False))
    print("检查点状态:", audit["checkpoint"].get("statuses", "未找到或无记录"))
    print("审计报告:", output.resolve())
    return 0 if not audit["metadata_issues"] else 2


def run_draft(args: Any) -> int:
    collection, label = actual_collection()
    records = scan_collection(collection)
    cases = make_review_draft(records, count=args.count, seed=args.seed)
    if not cases:
        raise RuntimeError("没有找到带完整 Section Metadata 的 JSON 医疗 Chunk")
    output = Path(args.output)
    save_review_csv(output, cases, overwrite=args.overwrite)
    counts = Counter(row["expected_field"] for row in cases)
    drugs = {name for row in cases for name in row["target_drugs"].split("|")}
    print(f"Collection：{label}")
    print(f"待审核问题：{len(cases)} 条，涉及药品：{len(drugs)} 种")
    print("栏目分布：", json.dumps(dict(counts), ensure_ascii=False))
    print("人工标注模板：", output.resolve())
    print("⚠️ 其中 suggested_section_id 是自动提名；"
          "所有 gold_section_ids 仍为空，不会被当作审核完成。")
    return 0


def run_score(args: Any) -> int:
    if args.mode != "bm25" and not args.allow_cloud_query:
        raise PermissionError(
            "Hybrid/重排会将查询或文档传输至云端服务。"
            "需明确加 --allow-cloud-query 并确认许可和费用。"
        )
    cases = load_review_csv(Path(args.reviews))
    positives = [c for c in cases if c["answerable"] == "yes"]
    negatives = [c for c in cases if c["answerable"] == "no"]
    uncertain = [c for c in cases if c["answerable"] == "uncertain"]
    if not positives:
        raise RuntimeError(
            "目前没有人工审核、标注为 answerable=yes 且填写 gold_section_ids "
            "的题目。不能把候选 Section 自动当成 gold。"
        )
    output = Path(args.output or f"medical_score_{args.mode}_v113.json")
    check_write_destination(output, overwrite=args.overwrite)
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    import config_data as config
    collection, _ = actual_collection()
    snapshot = scan_collection(collection)
    fingerprint = collection_fingerprint(snapshot)
    indexed_sections = {
        str(meta.get("section_id"))
        for _, _, meta in snapshot
        if meta.get("data_origin") == ORIGIN and meta.get("section_id")
    }
    # 不把“人工 Gold Section 未出现在 Chroma”误当成检索模型质量问题。
    # 这些 Gold 仍计入召回分母，但在报告中明确显示是索引覆盖缺口。
    # 明确先检查已有 Collection，再创建原 KnowledgeBaseService 的检索对象。
    from knowledge_base import KnowledgeBaseService
    service = KnowledgeBaseService()
    indexed = service.build_bm25_index()
    if indexed == 0:
        raise RuntimeError("当前 BM25 索引为空")
    if args.mode != "bm25":
        service.build_hybrid_retriever()

    results = []
    times = []
    for number, case in enumerate(positives, 1):
        started = time.perf_counter()
        docs = search_knowledge_service(
            service, mode=args.mode, question=case["question"], top_k=args.top_k
        )
        took = time.perf_counter() - started
        row = score_one(case, docs, k=args.top_k)
        row["gold_sections_not_indexed"] = sorted(
            set(case["gold_section_ids"]) - indexed_sections
        )
        row["elapsed_seconds"] = round(took, 3)
        results.append(row)
        times.append(took)
        print(f"[{number}/{len(positives)}] {case['case_id']} "
              f"原始命中={row['raw']['section_hit']} / "
              f"药品约束后命中={row['scoped']['section_hit']} "
              f"/ {took:.2f}s")
    if collection_fingerprint(scan_collection(collection)) != fingerprint:
        raise RuntimeError(
            "评测期间 Chroma 语料发生变化，不能生成可靠的同快照指标。"
            "请停止并发入库后重新运行该模式。"
        )
    summary = overall(results, k=args.top_k)
    summary["positive_cases_with_unindexed_gold_sections"] = sum(
        bool(item["gold_sections_not_indexed"]) for item in results
    )
    summary["unindexed_gold_section_ids"] = sorted({
        section_id
        for item in results for section_id in item["gold_sections_not_indexed"]
    })
    summary["negative_reviewed_cases_not_scored_as_recall"] = len(negatives)
    summary["uncertain_reviewed_cases_excluded"] = len(uncertain)
    summary["average_retrieval_seconds"] = round(statistics.mean(times), 3)
    report = {
        "mode": args.mode,
        "collection": str(config.collection_name),
        "collection_fingerprint": fingerprint,
        "bm25_indexed_chunks": indexed,
        "reviewed_cases": len(cases),
        "summary": summary,
        "results": results,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("=== 基于人工 Section Gold 的检索评测 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("结果报告：", output.resolve())
    return 0



def compare_evaluation_reports(
    left: dict[str, Any], right: dict[str, Any],
) -> dict[str, Any]:
    """只有同一组 Gold、同一 Chroma 快照才允许比较算法差异。"""
    keys = ("collection", "collection_fingerprint")
    for key in keys:
        if not left.get(key) or left.get(key) != right.get(key):
            raise ValueError(
                f"无法直接对比：{key} 不一致或缺失。"
                "请在同一 Chroma 快照上运行 BM25 与 Hybrid。"
            )
    a = left.get("summary") or {}
    b = right.get("summary") or {}
    if a.get("top_k") != b.get("top_k"):
        raise ValueError("无法直接对比：top_k 不相同")

    def gold_map(data: dict[str, Any]) -> dict[str, tuple[str, ...]]:
        return {
            row["case_id"]: tuple(sorted(row["gold_section_ids"]))
            for row in data.get("results", [])
        }
    if not gold_map(left) or gold_map(left) != gold_map(right):
        raise ValueError("无法直接对比：人工审核题目或 Gold Section 不一致")

    a_mode = str(left.get("mode") or "baseline")
    b_mode = str(right.get("mode") or "candidate")
    comparisons = {}
    for scope in ("raw", "scoped"):
        stats_a = a[scope]
        stats_b = b[scope]
        k = int(a["top_k"])
        keys_to_compare = (
            f"section_hit_at_{k}",
            f"mean_section_recall_at_{k}",
            "wrong_drug_chunk_rate",
            "average_returned_chunks",
        )
        comparisons[scope] = {
            name: {
                a_mode: stats_a.get(name),
                b_mode: stats_b.get(name),
                "difference": (
                    round(stats_b[name] - stats_a[name], 4)
                    if isinstance(stats_a.get(name), (int, float)) and
                    isinstance(stats_b.get(name), (int, float))
                    else None
                ),
            }
            for name in keys_to_compare
        }
    return {
        "baseline_mode": a_mode, "candidate_mode": b_mode,
        "collection": left["collection"],
        "fingerprint": left["collection_fingerprint"],
        "top_k": a["top_k"],
        "gold_cases": len(gold_map(left)),
        "comparisons": comparisons,
        "caution": (
            "仅同一快照和人工 Section Gold 的检索对照，"
            "不衡量医学事实正确性，也不是最终生成回答的准确率。"
        ),
    }


def run_compare(args: Any) -> int:
    left = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    right = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    report = compare_evaluation_reports(left, right)
    path = Path(args.output)
    check_write_destination(path, overwrite=args.overwrite)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("=== 同题、同 Gold、同语料快照的检索对照 ===")
    print(f"{report['baseline_mode']} → {report['candidate_mode']}")
    print(f"已审核问题：{report['gold_cases']} 条；Top-{report['top_k']}")
    for scope, metrics in report["comparisons"].items():
        print(scope, ":", json.dumps(metrics, ensure_ascii=False))
    print("比较报告：", path.resolve())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="医疗知识库 V1.13：扩容存量审计、人工题库及 Recall@K 评测"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="只读统计扩容后实际药品和 Section 覆盖")
    audit.add_argument("--state", default="medical_bulk_state.sqlite3")
    audit.add_argument("--output", default="medical_kb_audit_v113.json")
    audit.add_argument("--overwrite", action="store_true")

    draft = sub.add_parser("draft", help="从真实已入库 Section 生成待人工审核问题")
    draft.add_argument("--count", type=int, default=80)
    draft.add_argument("--seed", type=int, default=20261030)
    draft.add_argument("--output", default="medical_gold_review_v113.csv")
    draft.add_argument("--overwrite", action="store_true")

    score = sub.add_parser("score", help="仅对人工 reviewed + gold 题计算 Recall@K")
    score.add_argument("--reviews", required=True)
    score.add_argument("--mode", choices=("bm25", "hybrid", "hybrid-rerank"),
                       default="bm25")
    score.add_argument("--allow-cloud-query", action="store_true")
    score.add_argument("--top-k", type=int, default=5)
    score.add_argument("--output", default=None)
    score.add_argument("--overwrite", action="store_true")

    compare = sub.add_parser(
        "compare", help="对同一 Chroma 快照的 BM25/Hybrid 结果作公平对照"
    )
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", default="medical_eval_comparison_v113.json")
    compare.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "draft" and not 1 <= args.count <= 500:
        parser.error("--count 需在 1～500 之间")
    if args.command == "score" and not 1 <= args.top_k <= 20:
        parser.error("--top-k 需在 1～20 之间")
    if args.command == "audit":
        return run_audit(args)
    if args.command == "draft":
        return run_draft(args)
    if args.command == "compare":
        return run_compare(args)
    return run_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
