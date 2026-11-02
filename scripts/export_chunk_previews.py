"""Export actual Neo4j and Chroma chunks for inspection, using read-only queries."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
FIELDS = {
    "indication": "适应症", "contraindications": "禁忌",
    "adverse_reactions": "不良反应", "warnings": "注意事项",
    "drug_interactions": "药物相互作用",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_record(db, eid, chunk_id):
    metadata = {}
    document = None
    for key, sv, iv, fv, bv in db.execute(
        "SELECT key,string_value,int_value,float_value,bool_value "
        "FROM embedding_metadata WHERE id=?", (eid,)
    ):
        value = (sv if sv is not None else iv if iv is not None else
                 fv if fv is not None else bool(bv) if bv is not None else None)
        if key == "chroma:document":
            document = value
        else:
            metadata[key] = value
    if not isinstance(document, str):
        raise ValueError(f"Missing Chroma document: {chunk_id}")
    return {"id": chunk_id, "document": document, "metadata": metadata}


def write_md(path, title, samples, total):
    with path.open("x", encoding="utf-8") as out:
        out.write(f"# {title}\n\n总块数：{total:,}；本预览展示 {len(samples)} 块，正文完整保留。\n\n")
        out.write("样例偏向分块索引较大的长文栏目，不是随机抽样。两套预览优先展示相同药品和栏目，但块不要求一一对应。\n\n")
        for number, row in enumerate(samples, 1):
            meta = row["metadata"]
            field = meta.get("section_field", "")
            out.write(f"## {number}. {meta.get('drug_name', '')} · {FIELDS.get(field, field)}\n\n")
            for label, value in (
                ("Chunk ID", row["id"]), ("Section ID", meta.get("section_id")),
                ("原文起点", meta.get("quote_start")), ("原文终点（不含）", meta.get("quote_end")),
                ("JSON 路径", meta.get("json_path", "图谱按栏目全文切分")),
                ("来源网址", meta.get("source_url")),
            ):
                out.write(f"- {label}：`{value}`\n")
            out.write("\n### 存储正文\n\n" + row["document"] + "\n\n---\n\n")


def export(output: Path):
    from dotenv import load_dotenv
    from neo4j import GraphDatabase, READ_ACCESS

    load_dotenv(PROJECT / ".env", override=False)
    uri = os.getenv("NEO4J_URI", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "")
    if not uri or not password:
        raise ValueError("Missing configured Neo4j connection")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    dbpath = PROJECT / "chroma_db/chroma.sqlite3"
    if not dbpath.is_file():
        raise FileNotFoundError(dbpath)
    if output.exists():
        raise FileExistsError(f"Use a new output folder; refusing overwrite: {output}")
    with GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5,
                              max_transaction_retry_time=5) as driver:
        driver.verify_connectivity()
        with closing(sqlite3.connect(dbpath.as_uri() + "?mode=ro", uri=True, timeout=15)) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            collection_id = db.execute("SELECT id FROM collections WHERE name='rag'").fetchone()
            if collection_id is None:
                raise ValueError("Chroma collection rag does not exist")
            collection_id = collection_id[0]
            expected_chroma = db.execute(
                "SELECT count(*) FROM embeddings e JOIN segments s ON s.id=e.segment_id "
                "WHERE s.collection=?", (collection_id,)
            ).fetchone()[0]
            samples = []
            for field in FIELDS:
                rows = db.execute(
                    "SELECT e.id,e.embedding_id FROM embeddings e "
                    "JOIN segments s ON s.id=e.segment_id "
                    "JOIN embedding_metadata f ON f.id=e.id AND f.key='section_field' "
                    "LEFT JOIN embedding_metadata ix ON ix.id=e.id AND ix.key='chunk_index' "
                    "WHERE s.collection=? AND f.string_value=? "
                    "ORDER BY coalesce(ix.int_value,0) DESC,e.embedding_id LIMIT 6",
                    (collection_id, field),
                ).fetchall()
                samples.extend(metadata_record(db, eid, chunkid) for eid, chunkid in rows)
            output.mkdir(parents=True)
            # All output files are new; partial failures never replace an existing preview.
            chroma_file = output / "chroma_chunks.jsonl"
            chroma_count = 0
            distribution = Counter()
            with chroma_file.open("x", encoding="utf-8") as out:
                current_id, current = None, None
                query = (
                    "SELECT e.id,e.embedding_id,m.key,m.string_value,m.int_value,m.float_value,m.bool_value "
                    "FROM embeddings e JOIN segments s ON s.id=e.segment_id "
                    "LEFT JOIN embedding_metadata m ON m.id=e.id "
                    "WHERE s.collection=? ORDER BY e.id,m.key"
                )

                def flush(row):
                    nonlocal chroma_count
                    if not isinstance(row.get("document"), str):
                        raise ValueError(f"Missing Chroma document: {row['id']}")
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    chroma_count += 1
                    distribution[row["metadata"].get("section_field", "<missing>")] += 1

                for eid, chunkid, key, sv, iv, fv, bv in db.execute(query, (collection_id,)):
                    if eid != current_id:
                        if current is not None:
                            flush(current)
                        current_id = eid
                        current = {"id": chunkid, "document": None, "metadata": {}}
                    value = (sv if sv is not None else iv if iv is not None else
                             fv if fv is not None else bool(bv) if bv is not None else None)
                    if key == "chroma:document":
                        current["document"] = value
                    elif key is not None:
                        current["metadata"][key] = value
                if current is not None:
                    flush(current)
            if chroma_count != expected_chroma:
                raise ValueError("Chroma export count mismatch")

        graph_file = output / "neo4j_chunks.jsonl"
        graph_samples = []
        sample_sections = {row["metadata"].get("section_id") for row in samples}
        graph_distribution = Counter()
        graph_count = 0
        with driver.session(database=database, default_access_mode=READ_ACCESS, fetch_size=1000) as session:
            with session.begin_transaction(timeout=180) as tx:
                expected_graph = tx.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
                with graph_file.open("x", encoding="utf-8") as out:
                    rows = tx.run(
                        "MATCH (c:Chunk) "
                        "OPTIONAL MATCH (s:Section)-[:HAS_CHUNK]->(c) "
                        "OPTIONAL MATCH (r:SourceRecord)-[:HAS_SECTION]->(s) "
                        "OPTIONAL MATCH (d:Drug)-[:HAS_RECORD]->(r) "
                        "RETURN c.id AS id, c.text AS document, properties(c) AS chunk_properties, "
                        "d.id AS drug_id,d.name AS drug_name,r.id AS record_id,r.source_url AS source_url, "
                        "s.id AS section_id,s.field AS section_field,c.start AS quote_start,c.end AS quote_end "
                        "ORDER BY c.id"
                    )
                    seen = set()
                    for record in rows:
                        value = record.data()
                        chunkid = value.pop("id")
                        document = value.pop("document")
                        if not chunkid or chunkid in seen or not isinstance(document, str):
                            raise ValueError("Duplicate/missing Neo4j chunk or invalid document")
                        seen.add(chunkid)
                        row = {"id": chunkid, "document": document, "metadata": value}
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        graph_count += 1
                        graph_distribution[value.get("section_field", "<missing>")] += 1
                        if value.get("section_id") in sample_sections:
                            graph_samples.append(row)
                if graph_count != expected_graph:
                    raise ValueError("Neo4j export count mismatch")
                tx.commit()
    # Order graph samples by the corresponding Chroma sample sections.
    section_order = {row["metadata"].get("section_id"): i for i, row in enumerate(samples)}
    graph_samples.sort(key=lambda r: (section_order[r["metadata"]["section_id"]], r["metadata"]["quote_start"]))
    graph_samples = graph_samples[:40]
    write_md(output / "neo4j_preview.md", "Neo4j 图谱 Chunk 预览", graph_samples, graph_count)
    write_md(output / "chroma_preview.md", "Chroma 检索 Chunk 预览", samples, chroma_count)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in output.iterdir() if p.is_file()}
    manifest = {
        "status": "completed", "exported_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "neo4j": {"database": database, "source": "live_read_query", "chunks": graph_count,
                   "sample_chunks": len(graph_samples), "field_counts": dict(graph_distribution)},
        "chroma": {"database_file": str(dbpath), "collection": "rag", "source": "sqlite_read_only_snapshot",
                   "chunks": chroma_count, "sample_chunks": len(samples), "field_counts": dict(distribution)},
        "files": files,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as out:
        json.dump(manifest, out, ensure_ascii=False, indent=2)
        out.write("\n")
    readme = f"""# Chunk 切分预览目录

导出时间：{manifest['exported_at']}。此目录集中保存两套实际数据库切块的查阅快照。

| 文件 | 用途 | 数量 |
| --- | --- | ---: |
| [neo4j_preview.md](neo4j_preview.md) | 图谱切块的完整正文样例 | {len(graph_samples)} |
| [chroma_preview.md](chroma_preview.md) | 检索切块的完整正文样例 | {len(samples)} |
| [neo4j_chunks.jsonl](neo4j_chunks.jsonl) | 实际 Neo4j Chunk 全量导出，附药名、栏目及来源 | {graph_count:,} |
| [chroma_chunks.jsonl](chroma_chunks.jsonl) | Chroma rag 集合全量导出，保留存储正文与元数据 | {chroma_count:,} |
| [manifest.json](manifest.json) | 数量、栏目统计、来源与文件校验值 | — |

JSONL 每行是一块，统一外层结构为 `id`、`document`、`metadata`；Neo4j 的原始 Chunk 属性在 `metadata.chunk_properties` 中。搜索 `drug_name` 可以定位药品，搜索 `section_id` 可以对照两套切块。

Neo4j 与 Chroma 使用不同切分器、参数和栏目范围。Chroma 正文增加了药品与栏目等前缀，两者通过 Section ID 和原文字符位置对应，不要求 Chunk ID 或块数相同。这里的正文都是药品说明书；疾病百科待入库切块不计入本目录。

本目录是导出快照，业务程序不从这里读取。原 `kg_output_clean_v2/chunks.jsonl` 保留在原位置，图谱构建、导入和维护脚本无需改路径。Chroma 数据目录也保持原位置。未更改数据库内容；文件导出不调用模型或 Embedding。

脚本所有输入默认根据脚本所在目录定位，可从任意工作目录调用。刷新时指定新的目录，已有目录拒绝覆盖。例如在 PowerShell 执行：

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
python 'E:\\VsCode_Project\\DockerTest_Agent\\RAG_project\\data\\scripts\\export_chunk_previews.py' --output "E:\\VsCode_Project\\DockerTest_Agent\\RAG_project\\data\\chunk_previews_$stamp"
```

生成脚本：[export_chunk_previews.py](../scripts/export_chunk_previews.py)。导出期间请避免同时运行数据库入库任务；两个数据库分别读取，导出不构成跨库的同一时刻事务快照。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(output), "neo4j_chunks": graph_count, "chroma_chunks": chroma_count}, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "chunk_previews")
    args = parser.parse_args()
    export(args.output.expanduser().resolve())
