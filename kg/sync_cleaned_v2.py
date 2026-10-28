"""Reconcile cleaned drug_db_v2.json with an existing Neo4j graph.

Depends on the prior RAG_project `kg.build` and `kg.neo4j_import` modules.
Default operation: dry-run. Requires --apply --backup-confirmed to mutate DB.
Does not modify original JSON, ChromaDB, or unrelated node labels.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

DATASET = "MedGraphRAG-Drug-DB drug_db_v2.json"

def jlines(path):
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} invalid JSON") from e

def batched(seq, size=150):
    seq = list(seq)
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def query(driver, database, cypher, **params):
    with driver.session(database=database) as session:
        return [dict(x) for x in session.run(cypher, **params)]

def execute(driver, database, cypher, **params):
    with driver.session(database=database) as session:
        def cb(tx):
            return tx.run(cypher, **params).consume()
        session.execute_write(cb)

def get_state(driver, database):
    # Dataset property from the existing kg.neo4j_import.py
    rows = query(driver, database, """
        MATCH (r:SourceRecord)
        WHERE r.dataset = $dataset
        OPTIONAL MATCH (d:Drug)-[:HAS_RECORD]->(r)
        RETURN r.id AS id, d.id AS drug_id, d.name AS drug_name
    """, dataset=DATASET)
    return {r["id"]: r for r in rows if r.get("id")}

def check_reviewed_claims(driver, database, stale_ids):
    count = 0
    for part in batched(stale_ids, 200):
        if not part:
            continue
        rows = query(driver, database, """
          UNWIND $ids AS rid
          MATCH (r:SourceRecord {id:rid})-[:HAS_SECTION]->(s:Section)
          MATCH (c)-[:SUPPORTED_BY]->(s)
          WHERE NOT c:InteractionCandidate
          RETURN count(DISTINCT c) AS count
        """, ids=part)
        count += rows[0]["count"] if rows else 0
    return count

def build_plan(new_dir, old_dir, state, claim_count):
    new_dir = Path(new_dir)
    old_dir = Path(old_dir)
    new_report = json.loads((new_dir / "build_report.json").read_text(encoding="utf-8"))
    if new_report.get("sample_limit") != 0:
        raise RuntimeError("New graph must be built with --limit 0 for full sync.")

    new_drugs = {x["id"] for x in jlines(new_dir / "drugs.jsonl")}
    new_sources = {x["id"] for x in jlines(new_dir / "source_records.jsonl")}
    new_candidate_ids = {
        x["id"] for x in jlines(new_dir / "interaction_candidates.jsonl")
    }
    old_candidates_file = old_dir / "interaction_candidates.jsonl"
    old_mentions_file = old_dir / "mentions.jsonl"
    old_candidate_ids = (
        {x["id"] for x in jlines(old_candidates_file)}
        if old_candidates_file.is_file() else set()
    )
    old_mention_ids = (
        {x["id"] for x in jlines(old_mentions_file)}
        if old_mentions_file.is_file() else set()
    )
    new_mention_ids = {x["id"] for x in jlines(new_dir / "mentions.jsonl")}
    if not new_sources or not new_drugs:
        raise RuntimeError("New graph output is empty. Refusing to synchronize.")
    db_old = set(state)
    stale_sources = sorted(db_old - new_sources)
    stale_candidates = sorted(old_candidate_ids - new_candidate_ids)

    return {
        "new_drug_count": len(new_drugs),
        "new_source_count": len(new_sources),
        "new_candidate_count": len(new_candidate_ids),
        "db_old_source_count": len(db_old),
        "sources_to_add": len(new_sources - db_old),
        "sources_to_remove": len(stale_sources),
        "old_manifest_found": old_candidates_file.is_file(),
        "old_manifest_candidates": len(old_candidate_ids),
        "candidate_ids_to_remove": len(stale_candidates),
        "obsolete_mentions_to_check": len(old_mention_ids - new_mention_ids),
        "reviewed_claims_blocking_sync": claim_count,
        "removed_drug_names_preview": [
            state[r].get("drug_name") for r in stale_sources[:20]
        ],
        "stale_source_ids": stale_sources,
        "stale_candidate_ids": stale_candidates,
        "stale_mention_ids": sorted(old_mention_ids - new_mention_ids),
        "old_drug_ids": sorted({
            x["drug_id"] for x in state.values() if x.get("drug_id")
        }),
    }

def plan_preview(plan):
    return {
        k: v for k, v in plan.items()
        if k not in {"stale_source_ids", "stale_candidate_ids", "stale_mention_ids", "old_drug_ids"}
    }

def fix_candidate_links(driver, database, new_dir, size=200):
    """MERGE never removes old SOURCE_DRUG/TARGET_DRUG/SUPPORTED_BY edges."""
    cypher = """
        UNWIND $rows AS row
        MATCH (c:InteractionCandidate {id:row.id})
        MATCH (c)-[edge:SOURCE_DRUG|TARGET_DRUG|SUPPORTED_BY]->(dst)
        WHERE (type(edge)='SOURCE_DRUG' AND dst.id <> row.source_drug_id)
           OR (type(edge)='TARGET_DRUG' AND dst.id <> row.target_drug_id)
           OR (type(edge)='SUPPORTED_BY' AND dst.id <> row.evidence_section_id)
        DELETE edge
    """
    buf = []
    for x in jlines(Path(new_dir) / "interaction_candidates.jsonl"):
        buf.append(x)
        if len(buf) >= size:
            execute(driver, database, cypher, rows=buf)
            buf = []
    if buf:
        execute(driver, database, cypher, rows=buf)

def purge_stale_candidates(driver, database, ids):
    cypher = """
      UNWIND $ids AS cid
      MATCH (c:InteractionCandidate {id:cid})
      WHERE c.clinical_status = 'not_validated'
      DETACH DELETE c
    """
    for group in batched(ids, 150):
        execute(driver, database, cypher, ids=group)

def purge_stale_records(driver, database, ids):
    cypher = """
      UNWIND $ids AS rid
      MATCH (r:SourceRecord {id:rid})
      WHERE r.dataset=$dataset
      OPTIONAL MATCH (r)-[:HAS_SECTION]->(s:Section)
      OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
      WITH r, collect(DISTINCT s) AS sections, collect(DISTINCT c) AS chunks
      FOREACH (x IN chunks | DETACH DELETE x)
      FOREACH (x IN sections | DETACH DELETE x)
      DETACH DELETE r
    """
    for group in batched(ids, 60):
        execute(driver, database, cypher, ids=group, dataset=DATASET)

def purge_orphan_old_drugs(driver, database, old_ids):
    cypher = """
      UNWIND $ids AS did
      MATCH (d:Drug {id:did})
      WHERE NOT EXISTS { MATCH (d)--() }
      DELETE d
    """
    for group in batched(old_ids, 200):
        execute(driver, database, cypher, ids=group)

def purge_obsolete_orphan_mentions(driver, database, mention_ids):
    cypher = """
      UNWIND $ids AS mid
      MATCH (m:NameMention {id:mid})
      WHERE NOT EXISTS { MATCH (m)--() }
      DELETE m
    """
    for group in batched(mention_ids, 200):
        execute(driver, database, cypher, ids=group)

def run(args):
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Run: python -m pip install -r requirements-kg.txt") from exc

    from kg.prepare_cleaned_v2 import prepare_cleaned
    from kg.import_cleaned_aliases import import_aliases
    from kg.neo4j_import import import_kg

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.out)
    old_dir = Path(args.old_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing cleaned input: {input_path}")
    if out_dir.resolve() == old_dir.resolve():
        raise ValueError("--out must differ from --old-dir: keep prior manifest!")
    # Always generate an entirely NEW full export, never reuse old node hashes.
    prepare_cleaned(input_path, out_dir, limit=0)

    password = args.password or os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD missing in .env or --password")
    user = args.user or os.environ.get("NEO4J_USER", "neo4j")
    uri = args.uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        state = get_state(driver, args.database)
        report = json.loads(
            (out_dir / "build_report.json").read_text(encoding="utf-8")
        )
        fresh_sources = {
            row["id"] for row in jlines(out_dir / "source_records.jsonl")
        }
        stale_ids = sorted(set(state) - fresh_sources)
        blockers = check_reviewed_claims(driver, args.database, stale_ids)
        plan = build_plan(out_dir, old_dir, state, blockers)
        public_plan = plan_preview(plan)
        public_plan.update({
            "mode": "PENDING" if args.apply else "READ_ONLY_PLAN",
            "source_file": str(input_path),
            "new_dataset_sha256": report.get("input_sha256"),
            "cleaned_snapshot": report.get("cleaned_snapshot", {}),
            "warnings": [
                "This project uses a name-derived Drug.id; renaming can create a new ID.",
                "SourceRecord.id also depends on source content; changes need stale-record cleanup.",
                "Requires database backup before --apply.",
                "Third-party medication data remains clinically unverified.",
            ]
        })
        plan_file = out_dir / "sync_plan.json"
        plan_file.write_text(
            json.dumps(public_plan, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(json.dumps(public_plan, ensure_ascii=False, indent=2))
        if not args.apply:
            return public_plan

        if not args.backup_confirmed:
            raise RuntimeError("Refusing changes until --backup-confirmed.")
        if blockers:
            raise RuntimeError(
                f"{blockers} reviewed claims refer to obsolete evidence; "
                "re-review those claims before synchronizing."
            )
        if plan["db_old_source_count"] and (
            plan["new_source_count"] < plan["db_old_source_count"] * 0.70
        ) and not args.allow_large_drop:
            raise RuntimeError(
                "New source has <70% of old source records. "
                "Aborting; verify that your input isn't a subset."
            )
        if not plan["old_manifest_found"]:
            raise RuntimeError(
                "Previous interaction_candidates.jsonl is missing from --old-dir. "
                "Cannot guarantee removal of stale candidates; refusing to apply."
            )

        # Upsert first. If it fails, existing graph is still intact.
        import_kg(out_dir, apply=True, uri=uri, user=user,
                  password=password, database=args.database)
        # Candidate alias data is separate from old ontology. A dedicated
        # ALIAS_CANDIDATE_OF relation preserves weak terminology evidence.
        import_aliases(out_dir, apply=True, database=args.database,
                       uri=uri, user=user, password=password)

        # Rectify stale links that the MERGE-only importer cannot retract.
        fix_candidate_links(driver, args.database, out_dir)
        purge_stale_candidates(
            driver, args.database, plan["stale_candidate_ids"]
        )
        purge_stale_records(driver, args.database, plan["stale_source_ids"])
        purge_orphan_old_drugs(driver, args.database, plan["old_drug_ids"])
        purge_obsolete_orphan_mentions(
            driver, args.database, plan["stale_mention_ids"]
        )

        after = get_state(driver, args.database)
        remaining = sorted(set(after) - fresh_sources)
        missing = sorted(fresh_sources - set(after))
        result = {
            "status": "SUCCESS" if not remaining and not missing else "NEEDS_REVIEW",
            "source_records_expected": len(fresh_sources),
            "source_records_actual": len(after),
            "old_source_records_remaining": len(remaining),
            "new_source_records_missing": len(missing),
            "note": (
                "New source output lives in --out. Use that directory "
                "for subsequent disease extraction and RAG indexing."
            ),
        }
        (out_dir / "sync_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "SUCCESS":
            raise RuntimeError("Reconciliation incomplete; inspect sync_result.json")
        return result
    finally:
        driver.close()

def main():
    p = argparse.ArgumentParser(
        description="Sync drug_db_v2_cleaned.json and audited alias/extra source data to Neo4j"
    )
    p.add_argument("--input", required=True,
                   help="Path to your *locally cleaned* drug_db_v2_cleaned.json")
    p.add_argument("--out", default="data/kg_output_clean_v2")
    p.add_argument("--old-dir", default="data/kg_output",
                   help="Previous JSONL manifest, helps remove obsolete interactions")
    p.add_argument("--uri")
    p.add_argument("--user")
    p.add_argument("--password")
    p.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    p.add_argument("--apply", action="store_true",
                   help="Actually upsert and selectively delete obsolete nodes")
    p.add_argument("--backup-confirmed", action="store_true",
                   help="Confirm you have saved a Neo4j DB backup")
    p.add_argument("--allow-large-drop", action="store_true",
                   help="Permit >30% reduction in record count")
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()
