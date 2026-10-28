"""Import candidate drug aliases from the cleaned v2 snapshot.

Run only after importing Drug entities with kg.neo4j_import.
No direct Drug SAME_AS relations are created; alias mappings are marked candidate.
"""
import argparse
import json
import os
from pathlib import Path

DATASET="MedGraphRAG-Drug-DB drug_db_v2.json"

def iter_jsonl(path):
    with Path(path).open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def batch(rows,size=250):
    group=[]
    for row in rows:
        group.append(row)
        if len(group)>=size:
            yield group
            group=[]
    if group:
        yield group

def import_aliases(directory, apply=False, *, driver=None, database="neo4j",
                   uri=None,user=None,password=None):
    directory=Path(directory)
    alias_path=directory/"drug_aliases.jsonl"
    if not alias_path.is_file():
        raise FileNotFoundError(alias_path)
    aliases=list(iter_jsonl(alias_path))
    supplemental_ids = [
        row["id"] for row in
        iter_jsonl(directory / "source_records.jsonl")
        if row.get("source_variant") == "additional_merged_source"
    ]
    if len(set(x["id"] for x in aliases))!=len(aliases):
        raise ValueError("duplicate alias IDs")
    info={"alias_candidates":len(aliases),
          "additional_source_records":len(supplemental_ids),
          "mode":"apply" if apply else "dry_run",
          "dataset":DATASET}
    if not apply:
        print(json.dumps(info,ensure_ascii=False,indent=2))
        return info
    if driver is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except ImportError:
            pass
        from neo4j import GraphDatabase
        uri=uri or os.getenv("NEO4J_URI","bolt://localhost:7687")
        user=user or os.getenv("NEO4J_USER","neo4j")
        password=password or os.getenv("NEO4J_PASSWORD")
        if not password:
            raise RuntimeError("NEO4J_PASSWORD is missing")
        driver=GraphDatabase.driver(uri,auth=(user,password))
        own=True
    else:
        own=False
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            session.run(
                "CREATE CONSTRAINT drug_alias_cleaned_id IF NOT EXISTS "
                "FOR (a:DrugAlias) REQUIRE a.id IS UNIQUE"
            ).consume()
            for ids in batch(supplemental_ids):
                def variant_callback(tx, record_ids):
                    return tx.run("""
                       UNWIND $ids AS id
                       MATCH (r:SourceRecord {id:id})
                       SET r.source_variant='additional_merged_source'
                    """, ids=record_ids).consume()
                session.execute_write(variant_callback,ids)
            for group in batch(aliases):
                def callback(tx, rows):
                    return tx.run("""
                        UNWIND $rows AS row
                        MATCH (d:Drug {id:row.drug_id})
                        MERGE (a:DrugAlias {id:row.id})
                        SET a.name=row.name,
                            a.canonical_name=row.canonical_name,
                            a.dataset=row.dataset,
                            a.review_status=row.review_status,
                            a.snapshot_sha256=row.snapshot_sha256
                        MERGE (a)-[:ALIAS_CANDIDATE_OF]->(d)
                    """,rows=rows).consume()
                session.execute_write(callback,group)
            # In a repeated run, delete only stale alias nodes that belong
            # to the v2 dataset. Other labels/databases are not touched.
            new_ids={x["id"] for x in aliases}
            old_ids=[
                row["id"] for row in session.run(
                    "MATCH (a:DrugAlias {dataset:$dataset}) RETURN a.id AS id",
                    dataset=DATASET
                )
            ]
            stale=sorted(set(old_ids)-new_ids)
            for ids in batch(stale):
                def delete_callback(tx, rows):
                    return tx.run("""
                       UNWIND $rows AS id
                       MATCH (a:DrugAlias {id:id})
                       WHERE a.dataset=$dataset AND NOT EXISTS {
                           MATCH (a)-[rel]-()
                           WHERE type(rel)<>'ALIAS_CANDIDATE_OF'
                       }
                       DETACH DELETE a
                    """,rows=rows,dataset=DATASET).consume()
                session.execute_write(delete_callback,ids)
            info["stale_alias_ids_attempted_to_remove"]=len(stale)
        print(json.dumps(info,ensure_ascii=False,indent=2))
        return info
    finally:
        if own:
            driver.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--in-dir",default="data/kg_output_clean_v2")
    p.add_argument("--apply",action="store_true")
    p.add_argument("--database",default="neo4j")
    a=p.parse_args()
    import_aliases(a.in_dir,a.apply,database=a.database)

if __name__=="__main__":
    main()
