# -*- coding: utf-8 -*-
"""验证原有 BM25 / Hybrid+Rerank 检索和 Neo4j 父级 Section 精确回溯。

BM25 模式不调用云端；--mode hybrid-rerank 必须显式允许云端查询。
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

_PARENT_CYPHER = """
MATCH (d:Drug {id: $drug_id})-[:HAS_RECORD]->(r:SourceRecord {id: $record_id})
      -[:HAS_SECTION]->(s:Section {id: $section_id})
RETURN d.name AS drug_name, r.source_url AS source_url,
       s.text AS full_section, s.field AS field
LIMIT 2
"""


def verify_hit(metadata: dict, page_content: str, source_record: dict | None) -> dict:
    """只确认文本证据是否能被准确定位；不判断医学真实性。"""
    if not source_record:
        return {"ok": False, "reason": "找不到对应 Neo4j 来源记录"}
    full_text = source_record.get("full_section")
    if not isinstance(full_text, str):
        return {"ok": False, "reason": "Neo4j Section.text 为空"}
    try:
        start, end = int(metadata["quote_start"]), int(metadata["quote_end"])
    except (KeyError, ValueError, TypeError):
        return {"ok": False, "reason": "原文字符定位缺失或错误"}
    if not 0 <= start < end <= len(full_text):
        return {"ok": False, "reason": "原文坐标越界"}
    if hashlib.sha256(full_text.encode("utf-8")).hexdigest() != metadata.get("section_sha256"):
        return {"ok": False, "reason": "Section 原文已与入库时的快照不同"}
    quote = page_content.split("\n证据原文：", 1)
    if len(quote) != 2 or quote[1] != full_text[start:end]:
        return {"ok": False, "reason": "Chunk 原文与 Section 对应位置不一致"}
    if source_record.get("drug_name") != metadata.get("drug_name"):
        return {"ok": False, "reason": "药品名称与图谱不一致"}
    if str(source_record.get("field") or "").lower() != metadata.get("section_field"):
        return {"ok": False, "reason": "章节类型与图谱不一致"}
    if source_record.get("source_url") != metadata.get("source_url"):
        return {"ok": False, "reason": "来源 URL 与图谱不一致"}
    return {
        "ok": True, "reason": "原文与来源路径精确匹配",
        "quote": quote[1], "full_section": full_text,
        "source_url": metadata.get("source_url") or "",
    }


def fetch_parent(driver, database: str, metadata: dict):
    with driver.session(database=database) as session:
        rows = session.execute_read(lambda tx: [
            row.data() for row in tx.run(
                _PARENT_CYPHER,
                drug_id=metadata["drug_id"],
                record_id=metadata["record_id"],
                section_id=metadata["section_id"],
            )
        ])
    return rows[0] if len(rows) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 / Hybrid 检索 + Neo4j 原文章节回溯")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=("bm25", "hybrid-rerank"), default="bm25")
    parser.add_argument("--allow-cloud-query", action="store_true")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.top <= 10:
        parser.error("--top 需在 1～10 之间")
    if args.mode == "hybrid-rerank" and not args.allow_cloud_query:
        parser.error("Hybrid+Rerank 会发送检索问题及候选文本给模型，须加 --allow-cloud-query")

    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    from knowledge_base import KnowledgeBaseService

    service = KnowledgeBaseService()
    count = service.build_bm25_index()
    print("Chroma 可索引 Chunk 数：", count)
    if not count:
        print("没有文档：先执行 medical_json_index.py 的 --apply 试点导入。")
        return

    if args.mode == "bm25":
        docs = service.search_bm25(args.query)[:args.top]
    else:
        service.build_hybrid_retriever()
        docs = service.search_with_rerank(
            args.query, recall_k=10, top_n=args.top
        )

    print(f"本次 {args.mode} 检索命中：{len(docs)}")
    if not docs:
        return

    # 全部采用已有 Neo4j 只读事务；不修改 Chroma。
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("缺少 NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        for index, doc in enumerate(docs, 1):
            meta = dict(doc.metadata or {})
            print(f"\n[{index}] {meta.get('drug_name') or meta.get('source')} · "
                  f"{meta.get('section_path')}")
            print("Embedding/Rerank 分数：", meta.get("rerank_score", "未执行"))
            if meta.get("data_origin") != "json_matched_to_neo4j_section":
                print("来源：其他 Chroma 文档；没有本试点所需的精确 Section 绑定")
                continue
            parent = fetch_parent(driver, database, meta)
            verdict = verify_hit(meta, doc.page_content, parent)
            print("可追溯性：", "验证通过" if verdict["ok"] else "验证失败")
            print("详情：", verdict["reason"])
            if verdict["ok"]:
                quote = verdict["quote"]
                print("原文片段：", quote[:450] + ("…" if len(quote) > 450 else ""))
                print("父级 Section 全文（节选）：", verdict["full_section"][:700])
                print("来源：", verdict["source_url"])
            else:
                print("证据引用应暂缓：需要检查原文或 ID 是否发生变化。")


if __name__ == "__main__":
    main()
