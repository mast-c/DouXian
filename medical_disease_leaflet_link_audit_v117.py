# -*- coding: utf-8 -*-
"""豆馅医疗 GraphRAG V1.17 / P3-03A：疾病候选与药品说明书跨库关联审计。

只读 Neo4j & Chroma；可输出新的证据级关联 JSONL。
不会创建 Neo4j 关系、修改 Chroma/SQLite、调用云模型或把候选疾病升级为 TREATS。
代码要求本地存在原项目的 medical_evidence_bridge.py。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Callable
from urllib.parse import quote as url_quote

DATA_ORIGIN = "json_matched_to_neo4j_section"
ASSOCIATION_TYPE = "unverified_disease_mention"
ALLOWED_FIELDS = {
    "indication", "contraindications", "warnings",
    "adverse_reactions", "drug_interactions", "dosage", "composition"
}
# 候选与父级 Section 的两个原始关联是 Neo4j 已有关系：
# (d)-[:HAS_CANDIDATE]->(c)-[:MENTIONS_TERM]->(t)
# (c)-[:FROM_RECORD]->(r), (c)-[:SUPPORTED_BY]->(s)
# 必须通过 (d)-[:HAS_RECORD]->(r)-[:HAS_SECTION]->(s) 验证它们属于同一个药品。
# 分页先于 OPTIONAL MATCH：每页最多 batch 个候选-术语对。
CYPHER_PAGE = """
MATCH (d:Drug)-[:HAS_CANDIDATE]->(c:DiseaseCandidate)-[:MENTIONS_TERM]->(t:DiseaseTerm)
WITH d, c, t
ORDER BY c.id, t.id, d.id
SKIP $skip LIMIT $limit
OPTIONAL MATCH (d)-[:HAS_RECORD]->(r:SourceRecord)<-[:FROM_RECORD]-(c)
OPTIONAL MATCH (r)-[:HAS_SECTION]->(s:Section)<-[:SUPPORTED_BY]-(c)
WITH d, c, t, collect(DISTINCT {
    record_id: r.id, source_url: r.source_url,
    section_id: s.id, section_field: s.field, section_text: s.text
}) AS bindings
RETURN d.id AS drug_id, d.name AS drug_name,
       c.id AS candidate_id, c.evidence_quote AS evidence_quote,
       c.candidate_status AS candidate_status,
       c.clinical_status AS clinical_status,
       c.source_status AS source_status,
       c.context_flag AS context_flag,
       t.id AS disease_term_id, t.name AS disease_name,
       bindings AS bindings
ORDER BY candidate_id, disease_term_id, drug_id
"""


def new_file(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"报告文件已存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def normalized(s: Any) -> str:
    return s if isinstance(s, str) else ""


def nonempty(s: Any) -> bool:
    return isinstance(s, str) and bool(s.strip())


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_url(s: Any) -> str:
    from medical_json_chunker import safe_url as chunker_safe_url
    return chunker_safe_url(s)


def checkpoint_status(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 路径不存在：{path}")
    uri = "file:" + url_quote(path.resolve().as_posix(), safe="/:") + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=10) as conn:
        counts = {
            str(status): int(count)
            for status, count in conn.execute(
                "SELECT status, count(*) FROM drug_progress GROUP BY status"
            )
        }
    return counts


def scan_chroma(collection: Any, *, batch_size: int = 400,
                max_rows: int = 300000
                ) -> tuple[dict[tuple[str, str, str], list[dict]], dict]:
    """一次扫描已入库 Chunk，按来源三 ID 建立内存字典，不调用 Embedding。"""
    before = int(collection.count())
    if not 0 <= before <= max_rows:
        raise RuntimeError(f"Chroma {before} 条超过扫描安全上限 {max_rows}")
    seen = set()
    index: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    skipped = Counter()
    offset = 0
    while offset < before:
        page = collection.get(
            limit=min(batch_size, before - offset), offset=offset,
            include=["documents", "metadatas"]
        )
        ids = page.get("ids") or []
        metas = page.get("metadatas") or []
        docs = page.get("documents") or []
        if not ids or not (len(ids) == len(metas) == len(docs)):
            raise RuntimeError("Chroma 分页返回空页或 metadata/document 列不一致")
        for cid, meta, doc in zip(ids, metas, docs):
            if not isinstance(cid, str) or cid in seen:
                raise RuntimeError("Chroma Chunk ID 非字符串或重复（可能有并发写入）")
            seen.add(cid)
            if not isinstance(meta, dict) or not isinstance(doc, str):
                skipped["invalid_document_or_metadata"] += 1
                continue
            if meta.get("data_origin") != DATA_ORIGIN:
                skipped["non_json_origin"] += 1
                continue
            values = tuple(normalized(meta.get(k)) for k in (
                "drug_id", "record_id", "section_id"
            ))
            if not all(values):
                skipped["missing_source_ids"] += 1
                continue
            index[values].append({"id": cid, "metadata": meta, "text": doc})
        offset += len(ids)
    if offset != before or len(seen) != before or int(collection.count()) != before:
        raise RuntimeError("Chroma 扫描时数量变化；请停止并发写入后重跑")
    return dict(index), {
        "chroma_count_before": before,
        "chroma_count_after": int(collection.count()),
        "indexed_chunks": sum(len(v) for v in index.values()),
        "indexed_parent_sections": len(index),
        "skipped": dict(sorted(skipped.items()))
    }


def exact_occurrences(text: str, quote: str, *, max_count: int = 2) -> list[int]:
    """为避免重复短文本导致错误偏移，最多找两处，出现两处即歧义。"""
    result = []
    if not quote:
        return result
    start = text.find(quote)
    while start >= 0 and len(result) < max_count:
        result.append(start)
        start = text.find(quote, start + 1)
    return result


def evaluate_candidate(
    row: dict, chunks_by_section: dict[tuple[str, str, str], list[dict]]
) -> dict:
    """仅验证关联路径、Section 证据以及 Chroma Chunk 的精确原文坐标。"""
    from medical_evidence_bridge import verify_json_chunk

    drug_id = normalized(row.get("drug_id"))
    drug_name = normalized(row.get("drug_name"))
    candidate_id = normalized(row.get("candidate_id"))
    term_id = normalized(row.get("disease_term_id"))
    term_name = normalized(row.get("disease_name"))
    quote = normalized(row.get("evidence_quote"))
    bindings = row.get("bindings")
    output = {
        "association_type": ASSOCIATION_TYPE,
        "candidate_id": candidate_id,
        "disease_term_id": term_id,
        "disease_name": term_name,
        "drug_id": drug_id,
        "drug_name": drug_name,
        "candidate_status": normalized(row.get("candidate_status")),
        "clinical_status": normalized(row.get("clinical_status")),
        "source_status": normalized(row.get("source_status")),
        "context_flag": normalized(row.get("context_flag")),
        "record_id": "", "section_id": "", "section_field": "",
        "source_url": "", "section_sha256": "",
        "evidence_quote_sha256": sha256(quote) if quote else "",
        "quote_start": None, "quote_end": None,
        "chunk_ids": [], "status": ""
    }
    if not all((drug_id, drug_name, candidate_id, term_id, term_name, quote)):
        output["status"] = "INVALID_CANDIDATE"
        return output
    if not isinstance(bindings, list):
        output["status"] = "INVALID_GRAPH_BINDINGS"
        return output
    parent_rows = [
        binding for binding in bindings
        if isinstance(binding, dict)
        and all(nonempty(binding.get(k)) for k in ("record_id", "section_id"))
        and isinstance(binding.get("section_text"), str)
    ]
    unique_keys = {
        (b["record_id"], b["section_id"]) for b in parent_rows
    }
    if not unique_keys:
        output["status"] = "MISSING_PARENT_CHAIN"
        return output
    if len(unique_keys) != 1:
        output["status"] = "MULTIPLE_PARENT_CHAINS"
        return output
    chosen = next(
        x for x in parent_rows
        if (x["record_id"], x["section_id"]) in unique_keys
    )
    record_id, section_id = str(chosen["record_id"]), str(chosen["section_id"])
    field = normalized(chosen.get("section_field")).lower()
    text = chosen["section_text"]
    source_url = safe_url(chosen.get("source_url"))
    output.update({
        "record_id": record_id, "section_id": section_id,
        "section_field": field, "source_url": source_url,
        "section_sha256": sha256(text),
    })
    if field not in ALLOWED_FIELDS:
        output["status"] = "UNKNOWN_SECTION_FIELD"
        return output
    if not source_url:
        output["status"] = "MISSING_VALID_SOURCE_URL"
        return output
    occurrences = exact_occurrences(text, quote)
    if not occurrences:
        output["status"] = "QUOTE_NOT_IN_SECTION"
        return output
    if len(occurrences) > 1:
        output["status"] = "QUOTE_AMBIGUOUS_IN_SECTION"
        return output
    start = occurrences[0]
    end = start + len(quote)
    output["quote_start"] = start
    output["quote_end"] = end
    records = chunks_by_section.get((drug_id, record_id, section_id), [])
    if not records:
        output["status"] = "NO_INDEXED_SECTION"
        return output
    parent = {
        "drug_name": drug_name, "field": field,
        "source_url": source_url, "full_section": text,
    }
    covering = []
    bad_provenance = 0
    for record in records:
        meta = record["metadata"]
        try:
            chunk_start = int(meta["quote_start"])
            chunk_end = int(meta["quote_end"])
        except (ValueError, TypeError, KeyError):
            bad_provenance += 1
            continue
        if not (chunk_start <= start and end <= chunk_end):
            continue
        verified, _ = verify_json_chunk(meta, record["text"], parent)
        if verified:
            covering.append(record["id"])
        else:
            bad_provenance += 1
    if bad_provenance:
        # 不会忽略同一原文覆盖范围里的任何失败的记录。
        output["status"] = "COVERED_CHUNK_PROVENANCE_FAILED"
        return output
    if not covering:
        output["status"] = "QUOTE_NOT_COVERED_BY_ONE_CHUNK"
        return output
    output["chunk_ids"] = sorted(set(covering))
    output["status"] = "LINKED_EXACT_CHUNK"
    return output


def run_audit(
    collection: Any, fetch_page: Callable[[int, int], list[dict]], *,
    batch_size: int = 400, max_pairs: int = 100000,
    rows_output: Path | None = None, state: Path | None = None
) -> dict:
    index, chroma_stats = scan_chroma(collection, batch_size=batch_size)
    checkpoint = checkpoint_status(state)
    if checkpoint and (checkpoint.get("in_progress", 0)
                       or checkpoint.get("write_failed", 0)
                       or checkpoint.get("chroma_conflict", 0)):
        raise RuntimeError(
            "checkpoint 存在 in_progress/write_failed/chroma_conflict，"
            "请先完成或核对入库，避免跨库不一致"
        )
    counts = Counter()
    fields = Counter()
    candidate_ids = set()
    disease_ids = set()
    drug_ids = set()
    examples = []
    seen = set()
    offset = 0
    if rows_output is not None:
        new_file(rows_output)
    out = None
    try:
        if rows_output is not None:
            out = rows_output.open("x", encoding="utf-8")
        while True:
            if offset >= max_pairs:
                # 不能将限额处被截断的输出误判为全量成功。
                if len(fetch_page(offset, 1)) > 0:
                    raise RuntimeError(
                        f"候选数超出 max_pairs={max_pairs}；"
                        "已有局部 JSONL，请调大上限后使用新路径重跑"
                    )
                break
            limit = min(batch_size, max_pairs - offset)
            rows = fetch_page(offset, limit)
            if not rows:
                break
            # 每页的 Neo4j 查询在 OPTIONAL MATCH 后已经合并 parent bindings；
            # 必须保证每个候选-术语-药品组合只返回一行。
            if len(rows) > limit:
                raise RuntimeError("Neo4j 分页返回超过限制的行")
            for row in rows:
                key = (
                    normalized(row.get("drug_id")),
                    normalized(row.get("candidate_id")),
                    normalized(row.get("disease_term_id")),
                )
                if key in seen:
                    raise RuntimeError(
                        "Neo4j 候选重复，可能发生并发修改或 ID 冲突；停止"
                    )
                seen.add(key)
                result = evaluate_candidate(row, index)
                counts[result["status"]] += 1
                if result["section_field"]:
                    fields[result["section_field"]] += 1
                if result["candidate_id"]:
                    candidate_ids.add(result["candidate_id"])
                if result["disease_term_id"]:
                    disease_ids.add(result["disease_term_id"])
                if result["drug_id"]:
                    drug_ids.add(result["drug_id"])
                if out is not None:
                    out.write(json.dumps(
                        result, ensure_ascii=False, separators=(",", ":")
                    ) + "\n")
                if result["status"] != "LINKED_EXACT_CHUNK" and len(examples) < 30:
                    examples.append({
                        "candidate_id": result["candidate_id"],
                        "drug_name": result["drug_name"][:100],
                        "disease_name": result["disease_name"][:100],
                        "status": result["status"]
                    })
            offset += len(rows)
            print(
                f"[P3-03A] 已检查 {offset} 组疾病候选，"
                f"直接关联 Chroma={counts['LINKED_EXACT_CHUNK']}",
                flush=True
            )
            if len(rows) < limit:
                break
    finally:
        if out is not None:
            out.close()
    if int(collection.count()) != chroma_stats["chroma_count_before"]:
        raise RuntimeError("扫描之后 Chroma 发生数量变化；结果不可作为冻结基线")
    total = sum(counts.values())
    success = counts["LINKED_EXACT_CHUNK"]
    if total == 0:
        status = "NO_CANDIDATES"
    elif success == total:
        status = "PASS_ALL_LINKED"
    elif success > 0:
        status = "PARTIAL_WITH_UNRESOLVED_LINKS"
    else:
        status = "NO_EXACT_CHUNK_LINKS"
    return {
        "schema_version": 1,
        "stage": "P3-03A",
        "mode": "read_only_cross_store_evidence_association",
        "audit_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "association_type": ASSOCIATION_TYPE,
        "drug_ids": len(drug_ids),
        "disease_term_ids": len(disease_ids),
        "disease_candidate_ids": len(candidate_ids),
        "candidate_drug_term_pairs": total,
        "direct_chroma_links": success,
        "status_counts": dict(sorted(counts.items())),
        "section_fields": dict(sorted(fields.items())),
        "chroma": chroma_stats,
        "checkpoint_status": checkpoint,
        "unresolved_examples": examples,
        "rows_output": str(rows_output.resolve()) if rows_output else None,
        "medical_claim": (
            "所有关联仅表示未经医学核验的疾病术语提及；"
            "不得自动生成 Drug-TREATS-Disease 或获批适应症结论"
        ),
    }


def neo4j_fetcher(driver: Any, database: str) -> Callable[[int, int], list[dict]]:
    def fetch(skip: int, limit: int) -> list[dict]:
        with driver.session(database=database) as session:
            return session.execute_read(
                lambda tx: [
                    r.data() for r in tx.run(CYPHER_PAGE, skip=skip, limit=limit)
                ]
            )
    return fetch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="只读验证疾病候选→Neo4j说明书Section→Chroma Chunk精确关联"
    )
    parser.add_argument("--chroma-path", type=Path, default=Path("./chroma_db"))
    parser.add_argument("--collection", default="rag")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--links-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--max-pairs", type=int, default=100000)
    args = parser.parse_args(argv)
    if not 20 <= args.batch_size <= 1000:
        parser.error("--batch-size 应在 20～1000")
    if not 100 <= args.max_pairs <= 1000000:
        parser.error("--max-pairs 应在 100～1000000")
    for f in (args.report, args.links_output):
        if f:
            if f.exists():
                parser.error(f"输出已存在，拒绝覆盖：{f}")
    if args.links_output and args.links_output.resolve() == args.report.resolve():
        parser.error("--report 与 --links-output 不能相同")
    db_path = args.chroma_path.resolve()
    if not (db_path / "chroma.sqlite3").is_file():
        parser.error(f"Chroma 已有数据目录不存在：{db_path}")
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    import chromadb
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        parser.error("缺少 NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(name=args.collection, embedding_function=None)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        result = run_audit(
            collection, neo4j_fetcher(driver, database),
            batch_size=args.batch_size,
            max_pairs=args.max_pairs,
            rows_output=args.links_output,
            state=args.checkpoint
        )
    finally:
        driver.close()
    new_file(args.report)
    with args.report.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(
        f"[P3-03A] {result['status']}: "
        f"候选-术语对 {result['candidate_drug_term_pairs']}，"
        f"直接对齐 Chunk {result['direct_chroma_links']}，"
        f"报告 {args.report.resolve()}",
        flush=True
    )
    return 0 if result["status"] == "PASS_ALL_LINKED" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"[P3-03A] BLOCKED: {exc}", file=sys.stderr)
        sys.exit(2)
