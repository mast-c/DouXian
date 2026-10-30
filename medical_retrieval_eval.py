# -*- coding: utf-8 -*-
"""医疗 Hybrid RAG 基线：读取现有 Chroma，衡量药品/栏目检索命中与跨药品污染。

默认 BM25，不调用云端，不写数据库；结果报告写到指定 JSON 文件。
Hybrid+Rerank 需要显式 --allow-cloud-query，可能产生费用。
这只是检索层的“药名/栏目命中”基线，绝不声称是医学事实准确率/Recall@K。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from medical_entity_gate import filter_documents_for_drugs, query_drug_names


def read_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到评测样例：{path}")
    cases = []
    seen = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"第 {line_no} 行不是有效 JSON：{exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"第 {line_no} 行不是对象")
        query = item.get("question")
        case_id = str(item.get("id") or f"case_{line_no}")
        targets = item.get("target_drugs")
        field = item.get("expected_field")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"第 {line_no} 行缺少 question")
        if not isinstance(targets, list) or not all(isinstance(x, str) for x in targets):
            raise ValueError(f"第 {line_no} 行 target_drugs 必须是字符串列表")
        if field is not None and not isinstance(field, str):
            raise ValueError(f"第 {line_no} 行 expected_field 必须为字符串或 null")
        if case_id in seen:
            raise ValueError(f"重复评测 ID：{case_id}")
        seen.add(case_id)
        cases.append({
            "id": case_id,
            "question": query.strip(),
            "target_drugs": [x.strip() for x in targets if x.strip()],
            "expected_field": field,
            "annotation_status": str(item.get("annotation_status") or "unreviewed"),
        })
    if not cases:
        raise ValueError("评测样例为空")
    return cases


def _doc_meta(doc: Any) -> dict[str, Any]:
    meta = getattr(doc, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def summarize_case(case: dict[str, Any], documents: list[Any], *,
                   top_k: int = 5, elapsed_seconds: float = 0.0) -> dict[str, Any]:
    raw = list(documents[:top_k])
    known = query_drug_names(case["question"], (_doc_meta(doc) for doc in documents))
    restricted = filter_documents_for_drugs(documents, known)[:top_k]
    expected_drugs = set(case["target_drugs"])
    field = case.get("expected_field")

    def hit(doc: Any) -> bool:
        return bool(expected_drugs and _doc_meta(doc).get("drug_name") in expected_drugs)

    def offtarget(doc: Any) -> bool:
        name = str(_doc_meta(doc).get("drug_name") or "")
        return bool(name and expected_drugs and name not in expected_drugs)

    def field_hit(doc: Any) -> bool:
        return bool(hit(doc) and (not field or _doc_meta(doc).get("section_field") == field))

    def info(doc: Any) -> dict[str, str]:
        meta = _doc_meta(doc)
        return {
            "drug_name": str(meta.get("drug_name") or ""),
            "section_field": str(meta.get("section_field") or ""),
            "section_id": str(meta.get("section_id") or ""),
            "chunk_id": str(meta.get("chunk_id") or ""),
            "source_status": str(meta.get("source_status") or ""),
        }

    return {
        "id": case["id"],
        "question": case["question"],
        "target_drugs": sorted(expected_drugs),
        "expected_field": field,
        "detected_drugs": list(known),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "raw": {
            "hits": [info(doc) for doc in raw],
            "target_drug_hit": any(hit(doc) for doc in raw) if expected_drugs else None,
            "target_drug_field_hit": any(field_hit(doc) for doc in raw) if expected_drugs else None,
            "offtarget_chunks": sum(offtarget(doc) for doc in raw),
        },
        "scoped": {
            "hits": [info(doc) for doc in restricted],
            "target_drug_hit": any(hit(doc) for doc in restricted) if expected_drugs else None,
            "target_drug_field_hit": any(field_hit(doc) for doc in restricted) if expected_drugs else None,
            "offtarget_chunks": sum(offtarget(doc) for doc in restricted),
        },
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    targeted = [row for row in results if row["target_drugs"]]
    if not results:
        return {"cases": 0, "targeted_cases": 0}
    total_raw = sum(len(row["raw"]["hits"]) for row in targeted)
    total_scoped = sum(len(row["scoped"]["hits"]) for row in targeted)
    raw_offtarget = sum(row["raw"]["offtarget_chunks"] for row in targeted)
    scoped_offtarget = sum(row["scoped"]["offtarget_chunks"] for row in targeted)
    return {
        "cases": len(results),
        "targeted_cases": len(targeted),
        "raw_target_drug_hit_rate": round(
            sum(bool(row["raw"]["target_drug_hit"]) for row in targeted) / len(targeted), 4
        ) if targeted else None,
        "scoped_target_drug_hit_rate": round(
            sum(bool(row["scoped"]["target_drug_hit"]) for row in targeted) / len(targeted), 4
        ) if targeted else None,
        "scoped_drug_field_hit_rate": round(
            sum(bool(row["scoped"]["target_drug_field_hit"]) for row in targeted) / len(targeted), 4
        ) if targeted else None,
        "offtarget_chunk_rate_raw": round(raw_offtarget / total_raw, 4) if total_raw else None,
        "offtarget_chunk_rate_scoped": round(scoped_offtarget / total_scoped, 4)
        if total_scoped else None,
        "total_retrieval_seconds": round(sum(r["elapsed_seconds"] for r in results), 3),
        "notes": [
            "指标仅检查目标药品与目标说明书栏目，不能当作医学事实准确率。",
            "没有人工标注完整 gold_section_ids，不能将命中率称为 Recall@5。",
            "来源文本与 Neo4j 一致仅表示技术溯源通过，并非权威批准核验。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="医疗文档检索基线：药品/栏目命中及跨药品污染")
    parser.add_argument("--cases", type=Path,
                        default=Path(__file__).with_name("retrieval_eval_seed.jsonl"))
    parser.add_argument("--mode", choices=("bm25", "hybrid-rerank"), default="bm25")
    parser.add_argument("--allow-cloud-query", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report", type=Path, default=Path("medical_retrieval_baseline_report.json"))
    args = parser.parse_args()
    if not 1 <= args.top_k <= 15:
        parser.error("--top-k 必须在 1～15 之间")
    if args.mode != "bm25" and not args.allow_cloud_query:
        parser.error("Hybrid+Reranker 需加 --allow-cloud-query（有可能产生费用）")

    cases = read_cases(args.cases)
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    import config_data as config
    db = Path(config.persist_directory).expanduser().resolve() / "chroma.sqlite3"
    if not db.is_file():
        raise FileNotFoundError(f"未找到 Chroma 数据库：{db}")
    import chromadb
    chromadb.PersistentClient(path=str(db.parent)).get_collection(name=config.collection_name)

    from knowledge_base import KnowledgeBaseService
    service = KnowledgeBaseService()
    count = service.build_bm25_index()
    if not count:
        raise RuntimeError("Chroma BM25 没有读取到文档")
    if args.mode == "hybrid-rerank":
        service.build_hybrid_retriever()

    results = []
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        if args.mode == "bm25":
            docs = service.search_bm25(case["question"])[:args.top_k]
        else:
            docs = service.search_with_rerank(
                case["question"], recall_k=max(10, args.top_k),
                top_n=args.top_k,
            )
        took = time.perf_counter() - started
        result = summarize_case(case, docs, top_k=args.top_k, elapsed_seconds=took)
        results.append(result)
        print(
            f"[{index}/{len(cases)}] {case['id']} "
            f"目标药品命中={result['scoped']['target_drug_hit']} "
            f"栏目命中={result['scoped']['target_drug_field_hit']} "
            f"无关药品={result['raw']['offtarget_chunks']}→"
            f"{result['scoped']['offtarget_chunks']} "
            f"耗时={took:.2f}s"
        )

    report = {
        "mode": args.mode, "top_k": args.top_k,
        "collection": str(config.collection_name), "indexed_chunks": count,
        "summary": aggregate_results(results),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 评测统计 ===")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("报告：", args.report.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
