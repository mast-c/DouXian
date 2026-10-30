# -*- coding: utf-8 -*-
"""使用原始 JSON 字段切分后写入现有 Chroma；默认仅预览，严格校验来源。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from medical_json_chunker import (
    DEFAULT_FIELDS, FIELD_LABELS, align_to_neo4j, extract_raw_chunks,
    make_chroma_record,
)

_FETCH_SECTIONS = """
MATCH (d:Drug {name: $drug_name})-[:HAS_RECORD]->(r:SourceRecord)
      -[:HAS_SECTION]->(s:Section)
WHERE toLower(coalesce(s.field, '')) IN $fields
  AND s.text IS NOT NULL
RETURN d.id AS drug_id, d.name AS drug_name,
       r.id AS record_id, r.source_url AS source_url,
       coalesce(r.authority_status, r.source_status, 'not_verified') AS source_status,
       s.id AS section_id, toLower(s.field) AS field, s.text AS section_text
ORDER BY r.id, s.id
LIMIT $limit
"""


def read_json_drug(filepath: Path, name: str) -> dict:
    with filepath.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("drugs"), dict):
        raise ValueError("JSON 必须有对象格式的 drugs 字段")
    drug = data["drugs"].get(name)
    if not isinstance(drug, dict):
        raise ValueError(f"JSON 中找不到精确药品名称：{name}")
    return drug


def read_neo4j_sections(name: str, fields: tuple[str, ...],
                        *, limit: int = 200) -> list[dict]:
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD 未配置。")
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            rows = session.execute_read(
                lambda tx: [r.data() for r in tx.run(
                    _FETCH_SECTIONS, drug_name=name, fields=list(fields), limit=limit
                )]
            )
    if len(rows) >= limit:
        raise RuntimeError(
            "Neo4j Section 结果达到上限，拒绝使用可能被截断的数据匹配；"
            "请先检查该药品关联的来源记录数量。"
        )
    return rows


def add_to_chroma(records: list[dict], *,
                  expected_collection: str, batch_size: int = 8) -> tuple[int, int, int]:
    # import 仅发生在 --apply 模式：普通预览不会初始化任何 Chroma 集合。
    import config_data as config
    if str(config.collection_name) != expected_collection:
        raise RuntimeError(
            f"当前 config.collection_name={config.collection_name!r}，"
            f"与指定 {expected_collection!r} 不同，已停止。"
        )
    if not os.getenv("DASHSCOPE_API_KEY") or not os.getenv("DASHSCOPE_EMBEDDING_MODEL"):
        raise RuntimeError("缺少 DASHSCOPE_API_KEY 或 DASHSCOPE_EMBEDDING_MODEL")
    if len({row["id"] for row in records}) != len(records):
        raise RuntimeError("导入计划出现重复 Chunk ID，请先修复源数据。")

    from knowledge_base import KnowledgeBaseService
    service = KnowledgeBaseService()
    chroma = service.chroma
    count_before = chroma._collection.count()

    # 先全部检查冲突，再开始第一笔写入。
    ids = [row["id"] for row in records]
    old = chroma.get(ids=ids, include=["documents", "metadatas"])
    exists = {}
    for cid, content, meta in zip(
        old.get("ids", []),
        old.get("documents", []) or [],
        old.get("metadatas", []) or [],
    ):
        exists[cid] = (content, meta or {})

    pending = []
    skipped = 0
    for row in records:
        prior = exists.get(row["id"])
        if prior is None:
            pending.append(row)
        elif (prior[0] == row["text"] and
              prior[1].get("content_sha256") ==
              row["metadata"]["content_sha256"]):
            skipped += 1
        else:
            raise RuntimeError(
                f"原有 Chroma Chunk ID {row['id']} 内容冲突，未开始写入；"
                "请审查 JSON、Neo4j 或版本变更，不能静默覆盖。"
            )
    print(f"[PRECHECK] Collection={expected_collection} 当前={count_before}，"
          f"本批新 Chunk={len(pending)}，原有同内容跳过={skipped}")
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        # 保留 KnowledgeBaseService 原有 Chroma 配置和 DashScope Embedding；
        # 这里故意绕过 upload_by_str，避免 JSON 条目被再次按 Markdown 拆开。
        chroma.add_texts(
            texts=[row["text"] for row in batch],
            metadatas=[row["metadata"] for row in batch],
            ids=[row["id"] for row in batch],
        )
        print(f"[WRITE] Chunk {offset + len(batch)}/{len(pending)}")
    count_after = chroma._collection.count()
    return count_before, count_after, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JSON 字段感知切分 → Neo4j 精确匹配 → Chroma 原检索库"
    )
    parser.add_argument("--json", required=True, type=Path, help="已清洗药品原始 JSON 的真实路径")
    parser.add_argument("--drug", required=True, help="精确药品名；第一轮只做一款药品")
    parser.add_argument("--fields", nargs="+", choices=tuple(FIELD_LABELS),
                        default=list(DEFAULT_FIELDS))
    parser.add_argument("--verify-neo4j", action="store_true",
                        help="只读检查全部 Chunk 能否精确回溯到 Neo4j Section")
    parser.add_argument("--apply", action="store_true",
                        help="写入现有 Chroma Collection；必须同时确认云端 Embedding")
    parser.add_argument("--allow-cloud-embedding", action="store_true",
                        help="确认有权把选中的原文提交到 DashScope，并接受相应费用")
    parser.add_argument("--expect-collection", default="rag")
    parser.add_argument("--max-chunks", type=int, default=100,
                        help="为防误操作，试点最多允许 100 Chunk")
    args = parser.parse_args()

    if args.apply and not args.allow_cloud_embedding:
        parser.error("实际写入必须同时提供 --allow-cloud-embedding")
    if not 1 <= args.max_chunks <= 100:
        parser.error("本试点工具 --max-chunks 必须为 1～100")
    if not args.json.is_file():
        parser.error(f"文件不存在：{args.json}")

    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    import config_data as config

    # config_data.py 是用户真实配置；不读取脚本内部臆测的数值。
    chunk_size = int(config.chunk_size)
    overlap = int(config.chunk_overlap)
    target = str(config.collection_name)
    if target != args.expect_collection:
        raise RuntimeError(
            f"配置 Collection={target!r} 与预期={args.expect_collection!r} 不一致。"
        )

    drug = read_json_drug(args.json, args.drug)
    chunks = extract_raw_chunks(
        args.drug, drug, tuple(args.fields),
        chunk_size=chunk_size, overlap=overlap,
    )
    print("\n=== JSON 医疗切分预览 ===")
    print("JSON:", args.json.resolve())
    print("药品:", args.drug)
    print(f"Chunk 参数: size={chunk_size}, overlap={overlap}")
    print("栏目:", "、".join(FIELD_LABELS[field] for field in args.fields))
    print("待检查原文 Chunk:", len(chunks))
    print("是否强制截断长句:", sum(1 for c in chunks if c.forced_cut))
    for i, item in enumerate(chunks[:8], 1):
        print(f" [{i}] {item.json_path} [{item.item_start}:{item.item_end}] "
              f"{len(item.quote)} 字 | {item.quote[:110]!r}")
    if not chunks:
        print("当前所选字段没有有效内容，无需写入。")
        return
    if len(chunks) > args.max_chunks:
        raise RuntimeError(
            f"本次生成 {len(chunks)} 条，超出试点上限 {args.max_chunks}；停止。"
        )

    if not args.verify_neo4j and not args.apply:
        print("\n[DRY RUN] 只读取 JSON，没有连接 Neo4j/Chroma，没有云端费用。")
        print("下一步可增加 --verify-neo4j 核对 Section ID 与原文坐标。")
        return

    rows = read_neo4j_sections(args.drug, tuple(args.fields))
    links, errors = align_to_neo4j(chunks, rows, drug.get("source", ""))
    print("\n=== Neo4j 来源回溯校验 ===")
    print(f"图谱 Section 行数：{len(rows)}；精确匹配：{len(links)}/{len(chunks)}")
    if errors:
        for issue in errors[:12]:
            print(" [不能自动入库]", issue)
        print(f"合计有 {len(errors)} 条未能唯一定位，未通过精确校验。")
        if args.apply:
            raise RuntimeError(
                "部分 JSON 片段无法精确定位到 Neo4j，整批拒绝写入，避免虚假证据引用。"
            )
        return

    records = [
        make_chroma_record(
            item, chunk_index=i, chunk_total=len(links),
            json_filename=args.json.name,
        ) for i, item in enumerate(links)
    ]
    for row in records[:3]:
        meta = row["metadata"]
        print(f" [OK] {meta['section_id']} [{meta['quote_start']}:{meta['quote_end']}] "
              f"← {meta['json_path']}")

    if not args.apply:
        print("\n[VERIFIED DRY RUN] 全部定位成功；没有写入任何数据库。")
        return

    print("\n=== 受控写入 Chroma（会发送文本给配置的云端 Embedding 服务） ===")
    before, after, skipped = add_to_chroma(
        records, expected_collection=args.expect_collection
    )
    print(f"Chroma Chunk 数量：{before} → {after}")
    print(f"本批原内容复用、未重复 Embedding 的 Chunk：{skipped}")
    print("完成后请重启 Streamlit，使内存中的 BM25 索引从新的 Chroma 数据重新构建。")


if __name__ == "__main__":
    main()
