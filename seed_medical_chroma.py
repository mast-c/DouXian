#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neo4j 医疗药品原文 -> 现有 Chroma Collection 的小规模、可复核增量导入。

默认仅预览；仅 --apply --allow-cloud-embedding 时调用 DashScope 并写入 Chroma。
不会修改 Neo4j，不删除其他 Chroma 文档，不导入疾病候选作为已确认事实。
需在含 knowledge_base.py、config_data.py、.env 的项目目录中执行。
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

FIELD_NAMES = {
    "indication": "适应症",
    "contraindications": "禁忌",
    "warnings": "注意事项",
    "adverse_reactions": "不良反应",
    "drug_interactions": "药物相互作用",
    "composition": "药品成分",
    "dosage": "用法用量",
}
DEFAULT_FIELDS = ("indication", "contraindications", "adverse_reactions")

CYPHER = """
MATCH (d:Drug)
WHERE $drug_name IS NULL OR d.name = $drug_name
WITH d
ORDER BY d.name
LIMIT $limit_drugs
MATCH (d)-[:HAS_RECORD]->(r:SourceRecord)-[:HAS_SECTION]->(s:Section)
WHERE toLower(coalesce(s.field, '')) IN $fields
  AND s.text IS NOT NULL AND trim(s.text) <> ''
RETURN d.id AS drug_id, d.name AS drug_name,
       r.id AS record_id, r.source_url AS source_url,
       r.authority_status AS authority_status,
       s.id AS section_id, toLower(s.field) AS field,
       s.text AS text
ORDER BY drug_name, record_id, field, section_id
LIMIT $limit_sections
"""


def build_documents(rows: list[dict[str, Any]], *,
                    max_records: int = 20, max_chars_per_field: int = 15000
                    ) -> list[dict[str, Any]]:
    """组装为原有 KnowledgeBaseService 能识别的 Markdown 文档。

    每个来源记录生成一个稳定文件名；额外标记其第三方、未核验性质。
    """
    grouped = OrderedDict()
    for row in rows:
        drug_id = str(row.get("drug_id") or "").strip()
        record_id = str(row.get("record_id") or "").strip()
        drug_name = str(row.get("drug_name") or "").strip()
        field = str(row.get("field") or "").strip().lower()
        text = str(row.get("text") or "").strip()
        if not all((drug_id, record_id, drug_name, text)) or field not in FIELD_NAMES:
            continue
        key = (drug_id, record_id)
        if key not in grouped:
            if len(grouped) >= max_records:
                continue
            grouped[key] = {
                "drug": drug_name,
                "source_url": str(row.get("source_url") or "").strip(),
                "authority_status": str(row.get("authority_status") or "").strip(),
                "sections": OrderedDict(),
            }
        grouped[key]["sections"].setdefault(field, [])
        group = grouped[key]["sections"][field]
        section_id = str(row.get("section_id") or "").strip()
        # 原文可重复提及，但同一 Section ID 不应重复入库。
        if not any(existing_id == section_id for existing_id, _ in group):
            group.append((section_id, text))

    results = []
    for (drug_id, record_id), item in grouped.items():
        digest = hashlib.sha256(f"{drug_id}|{record_id}".encode("utf-8")).hexdigest()[:24]
        filename = f"medical_{digest}.md"
        parts = [f"# {item['drug']}", ""]
        for field, paragraphs in item["sections"].items():
            section_body = "\n\n".join(text for _, text in paragraphs)
            original_length = len(section_body)
            if original_length > max_chars_per_field:
                section_body = (section_body[:max_chars_per_field] +
                                "\n\n（提示：本次试验仅导入了该字段的前一部分原文。）")
            parts.extend([
                f"## {FIELD_NAMES[field]}",
                "资料来源性质：数据集中保存的第三方药品资料，尚未经医学权威核验；"
                "以下仅为资料原文，不能视作官方批准适应症或个体治疗建议。",
                f"来源记录ID：{record_id}",
                f"原始来源网址：{item['source_url'] or '未记录'}",
                f"章节ID：{', '.join(sid for sid, _ in paragraphs if sid)}",
                "",
                section_body,
                "",
            ])
        results.append({
            "filename": filename,
            "drug": item["drug"],
            "record_id": record_id,
            "source_url": item["source_url"],
            "markdown": "\n".join(parts),
            "sections": sum(len(parts) for parts in item["sections"].values()),
        })
    return results


def fetch_rows(drug_name: str | None, max_drugs: int, max_sections: int,
               fields: tuple[str, ...]) -> list[dict[str, Any]]:
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not password:
        raise RuntimeError("缺少 NEO4J_PASSWORD；请检查 .env")
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            records = session.execute_read(
                lambda tx: list(tx.run(
                    CYPHER,
                    drug_name=drug_name,
                    limit_drugs=max_drugs,
                    limit_sections=max_sections,
                    fields=list(fields),
                ))
            )
    return [record.data() for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Neo4j -> Chroma 医疗原文小规模导入")
    parser.add_argument("--drug", help="精确药品名（推荐第一轮指定）")
    parser.add_argument("--limit-drugs", type=int, default=10)
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--max-sections", type=int, default=160)
    parser.add_argument("--max-chars-per-field", type=int, default=15000)
    parser.add_argument("--fields", nargs="+", choices=list(FIELD_NAMES),
                        default=list(DEFAULT_FIELDS))
    parser.add_argument("--expect-collection", default="rag")
    parser.add_argument("--apply", action="store_true",
                        help="明确要求写入 Chroma；不加此参数只预览")
    parser.add_argument("--allow-cloud-embedding", action="store_true",
                        help="确认有权将选中的第三方资料发送给外部 Embedding 服务")
    args = parser.parse_args()
    if not (1 <= args.limit_drugs <= 100):
        parser.error("--limit-drugs 需在 1～100 之间")
    if not (1 <= args.max_records <= 100):
        parser.error("--max-records 需在 1～100 之间")
    if not (1 <= args.max_sections <= 2000):
        parser.error("--max-sections 需在 1～2000 之间")
    if not (200 <= args.max_chars_per_field <= 50000):
        parser.error("--max-chars-per-field 需在 200～50000 之间")
    if args.apply and not args.allow_cloud_embedding:
        parser.error("实际写入时必须增加 --allow-cloud-embedding")

    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    import config_data as config

    target = str(config.collection_name)
    print("=== 医疗原文导入预检 ===")
    print("Chroma persist_directory:", Path(config.persist_directory).resolve())
    print("Chroma Collection:", target)
    if target != args.expect_collection:
        raise RuntimeError(
            f"配置中的 Collection 是 {target!r}，与预期的 "
            f"{args.expect_collection!r} 不符。停止操作。"
        )
    rows = fetch_rows(args.drug, args.limit_drugs,
                      args.max_sections, tuple(args.fields))
    docs = build_documents(rows, max_records=args.max_records,
                           max_chars_per_field=args.max_chars_per_field)
    if not docs:
        print("没有获取到可导入的原文章节。检查药名或 Neo4j Section.field/text/关系名称。")
        return

    print(f"读取 Section 行数：{len(rows)}")
    print(f"待处理文档数：{len(docs)}")
    print(f"待处理字符数：{sum(len(d['markdown']) for d in docs):,}")
    for item in docs[:10]:
        print(f"- {item['drug']} / {item['filename']} / "
              f"{item['sections']} 段 / {len(item['markdown'])} 字")
    if not args.apply:
        print("\n【预览完成】未写入 Neo4j/Chroma，也未发起 Embedding 请求。")
        print("确认数据来源使用许可、模型费用和内容后，再使用 "
              "--apply --allow-cloud-embedding。")
        return

    if not os.getenv("DASHSCOPE_API_KEY") or not os.getenv("DASHSCOPE_EMBEDDING_MODEL"):
        raise RuntimeError("缺少 DASHSCOPE_API_KEY 或 DASHSCOPE_EMBEDDING_MODEL")
    from knowledge_base import KnowledgeBaseService
    service = KnowledgeBaseService()
    before = service.chroma._collection.count()
    print(f"开始写入；原有 Chroma 记录数：{before}")
    success = skipped = failed = 0
    for item in docs:
        try:
            existing = service.chroma.get(
                where={"source": item["filename"]},
                include=["metadatas"],
            )
            if existing.get("ids"):
                skipped += 1
                print("[跳过已存在]", item["filename"])
                continue
            result = service.upload_by_str(item["markdown"], item["filename"])
            success += 1
            print("[已处理]", item["filename"], "=>", result.splitlines()[0])
        except Exception as exc:
            failed += 1
            print("[失败]", item["filename"], type(exc).__name__, str(exc)[:160])
            # 部分 Embedding 调用失败时不继续发请求，可按需修复后重试。
            break
    after = service.chroma._collection.count()
    print(f"\n完成：新处理文档 {success}、已存在跳过 {skipped}、失败 {failed}")
    print(f"Chroma Chunk 数量：{before} → {after}")
    print("请重启 Streamlit，使现有 BM25 索引重新加载并执行 Hybrid 检索。")


if __name__ == "__main__":
    main()
