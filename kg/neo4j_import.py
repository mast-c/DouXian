"""Idempotent Neo4j importer. Default mode is dry-run; --apply writes to Neo4j.

No direct verified DRUG_INTERACTS_WITH edges are inferred from raw text.
"""
import argparse
import json
import os
from pathlib import Path

CONSTRAINTS = [
    "CREATE CONSTRAINT drug_kg_id IF NOT EXISTS FOR (n:Drug) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT source_record_kg_id IF NOT EXISTS FOR (n:SourceRecord) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT section_kg_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_kg_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT mention_kg_id IF NOT EXISTS FOR (n:NameMention) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT interaction_kg_id IF NOT EXISTS FOR (n:InteractionCandidate) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT disease_term_kg_id IF NOT EXISTS FOR (n:DiseaseTerm) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT indication_claim_kg_id IF NOT EXISTS FOR (n:IndicationClaim) REQUIRE n.id IS UNIQUE",
]

QUERIES = {
"drugs": """
    UNWIND $rows AS r
    MERGE (d:Drug {id: r.id})
    SET d.name = r.name,
        d.aliases = coalesce(r.aliases, []),
        d.original_names = coalesce(r.original_names, [r.name]),
        d.entity_scope = r.entity_scope
""",
"source_records": """
    UNWIND $rows AS r
    MATCH (d:Drug {id:r.drug_id})
    MERGE (s:SourceRecord {id:r.id})
    SET s.source_url=r.source_url,
        s.record_sha256=r.record_sha256,
        s.snapshot_sha256=r.snapshot_sha256,
        s.dataset=r.dataset,
        s.version_status=r.version_status,
        s.authority_status=r.authority_status,
        s.license_status=r.license_status
    MERGE (d)-[:HAS_RECORD]->(s)
""",
"sections": """
    UNWIND $rows AS r
    MATCH (s:SourceRecord {id:r.record_id})
    MERGE (x:Section {id:r.id})
    SET x.field=r.field, x.title=r.title, x.text=r.text,
        x.placeholder=r.placeholder
    MERGE (s)-[:HAS_SECTION]->(x)
""",
"chunks": """
    UNWIND $rows AS r
    MATCH (s:Section {id:r.section_id})
    MERGE (c:Chunk {id:r.id})
    SET c.field=r.field, c.start=r.start,
        c.end=r.end, c.text=r.text
    MERGE (s)-[:HAS_CHUNK]->(c)
""",
"mentions": """
    UNWIND $rows AS r
    MERGE (m:NameMention {id:r.id})
    SET m.name=r.name, m.semantic_status=r.semantic_status
""",
"interaction_candidates": """
    UNWIND $rows AS r
    MATCH (sourceMention:NameMention {id:r.source_mention_id})
    MATCH (targetMention:NameMention {id:r.target_mention_id})
    MERGE (c:InteractionCandidate {id:r.id})
    SET c.note=r.note,
        c.risk_label_in_source=r.risk_label_in_source,
        c.evidence_status=r.evidence_status,
        c.evidence_field=r.evidence_field,
        c.evidence_char_start=r.evidence_char_start,
        c.evidence_char_end=r.evidence_char_end,
        c.clinical_status=r.clinical_status,
        c.source_name=r.source_name,
        c.target_name=r.target_name,
        c.original_pair=r.original_pair
    MERGE (c)-[:SOURCE_MENTION]->(sourceMention)
    MERGE (c)-[:TARGET_MENTION]->(targetMention)
    WITH c, r
    OPTIONAL MATCH (drug1:Drug {id:r.source_drug_id})
    FOREACH (_ IN CASE WHEN drug1 IS NULL THEN [] ELSE [1] END |
        MERGE (c)-[:SOURCE_DRUG]->(drug1))
    WITH c, r
    OPTIONAL MATCH (drug2:Drug {id:r.target_drug_id})
    FOREACH (_ IN CASE WHEN drug2 IS NULL THEN [] ELSE [1] END |
        MERGE (c)-[:TARGET_DRUG]->(drug2))
    WITH c, r
    OPTIONAL MATCH (s:Section {id:r.evidence_section_id})
    FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
        MERGE (c)-[:SUPPORTED_BY]->(s))
""",
"disease_terms": """
    UNWIND $rows AS r
    MERGE (d:DiseaseTerm {id:r.id})
    SET d.name=r.name, d.normalization_status=r.normalization_status
""",
"indication_claims": """
    UNWIND $rows AS r
    MATCH (drug:Drug {id:r.drug_id})
    MATCH (disease:DiseaseTerm {id:r.disease_id})
    MATCH (section:Section {id:r.section_id})
    MERGE (c:IndicationClaim {id:r.id})
    SET c.predicate=r.predicate,
        c.evidence_quote=r.evidence_quote,
        c.char_start=r.char_start, c.char_end=r.char_end,
        c.review_status=r.review_status,
        c.clinical_status=r.clinical_status
    MERGE (c)-[:SUBJECT]->(drug)
    MERGE (c)-[:OBJECT]->(disease)
    MERGE (c)-[:SUPPORTED_BY]->(section)
"""
}

def iter_jsonl(file_path):
    if not Path(file_path).exists():
        return
    with Path(file_path).open("r", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as ex:
                    raise ValueError(f"{file_path}:{number}: JSON 格式不合法") from ex

def batches(rows, batch_size=300):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def import_kg(output_dir, apply=False, batch_size=300, uri=None,
              user=None, password=None, database="neo4j"):
    output_dir = Path(output_dir)
    if not (output_dir / "build_report.json").is_file():
        raise FileNotFoundError(f"{output_dir}/build_report.json 未找到，请先运行 kg.build")
    counts = {
        kind: sum(1 for _ in iter_jsonl(output_dir / f"{kind}.jsonl"))
        for kind in QUERIES
    }
    if not apply:
        print(json.dumps({"mode": "DRY RUN (数据库未修改)", "row_counts": counts},
                         ensure_ascii=False, indent=2))
        return counts

    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise RuntimeError("需要安装 Neo4j Python 驱动：pip install neo4j") from e

    uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.environ.get("NEO4J_USER", "neo4j")
    password = password or os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("缺少 NEO4J_PASSWORD，未进行数据库操作")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            for query in CONSTRAINTS:
                session.run(query).consume()
            for kind, query in QUERIES.items():
                for batch in batches(iter_jsonl(output_dir / f"{kind}.jsonl"),
                                     batch_size):
                    def run_batch(tx, cypher, rows):
                        return tx.run(cypher, rows=rows).consume()
                    session.execute_write(run_batch, query, batch)
            # Index is optional for full text browsing; later RAG stages
            # will use Chroma separately without touching existing indices.
            session.run(
                "CREATE FULLTEXT INDEX kg_section_text IF NOT EXISTS "
                "FOR (s:Section) ON EACH [s.text]"
            ).consume()
        print(json.dumps({"mode": "Neo4j import completed",
                          "row_counts": counts}, ensure_ascii=False, indent=2))
    finally:
        driver.close()
    return counts

def main():
    p = argparse.ArgumentParser(description="Import KG JSONL into Neo4j")
    p.add_argument("--in-dir", default="data/kg_output")
    p.add_argument("--batch-size", type=int, default=300)
    p.add_argument("--apply", action="store_true",
                   help="必须显式提供该参数才真正写入数据库")
    p.add_argument("--uri", default=None)
    p.add_argument("--user", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("batch-size 必须 >= 1")
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass
    import_kg(a.in_dir, a.apply, a.batch_size,
              a.uri, a.user, a.password, a.database)

if __name__ == "__main__":
    main()
