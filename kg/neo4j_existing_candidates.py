"""Attach unreviewed DrugKG disease candidates to an existing cleaned V2 graph.

This script does NOT import/rewrite Drug, SourceRecord, Section, Chunk or DrugAlias.
It works with old Neo4j nodes lacking `kg_snapshot` and validates graph IDs and
exact Section.text evidence before any write. It only MERGEs:
DiseaseTerm, DiseaseCandidate, ModelAssessment and their evidence links.
No TREATS/INDICATED_FOR edges. No clinical approval. No delete.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from kg import neo4j_drug_graph as source

EXPECTED_BASE = {
    "Drug": ("drugs", 11953),
    "SourceRecord": ("source_records", 11960),
    "Section": ("sections", 106445),
    "Chunk": ("chunks", 107787),
    "DrugAlias": ("drug_aliases", 343),
}
EXISTING_CANDIDATE_Q = (
    "MATCH (x:DiseaseCandidate) "
    "RETURN x.id AS id, x.kg_snapshot AS snapshot"
)
EXISTING_ASSESSMENT_Q = (
    "MATCH (x:ModelAssessment) "
    "RETURN x.id AS id, x.kg_snapshot AS snapshot"
)
EXISTING_TERMS_Q = (
    "MATCH (x:DiseaseTerm) "
    "RETURN x.id AS id, x.name AS name"
)
READ_EVIDENCE = """
UNWIND $rows AS r
OPTIONAL MATCH (d:Drug {id:r.drug_id})
  -[:HAS_RECORD]->(rec:SourceRecord {id:r.source_record_id})
  -[:HAS_SECTION]->(sec:Section {id:r.section_id})
WITH r, count(sec) AS paths,
     sum(CASE WHEN
         d.name = r.drug_name
         AND rec.source_url = r.source_url
         AND rec.snapshot_sha256 = $snapshot
         AND sec.field = 'indication'
         AND substring(sec.text,r.quote_start,
                       r.quote_end-r.quote_start) = r.evidence_quote
         AND substring(sec.text,r.disease_start,
                       r.disease_end-r.disease_start) = r.disease_name
         THEN 1 ELSE 0 END) AS valid_paths
RETURN r.id AS id, paths, valid_paths
"""
WRITE_CANDIDATES = """
UNWIND $rows AS r
MATCH (d:Drug {id:r.drug_id})
  -[:HAS_RECORD]->(rec:SourceRecord {id:r.source_record_id})
  -[:HAS_SECTION]->(sec:Section {id:r.section_id})
WHERE d.name = r.drug_name
  AND rec.source_url = r.source_url
  AND rec.snapshot_sha256 = $snapshot
  AND sec.field = 'indication'
  AND substring(sec.text,r.quote_start,
                r.quote_end-r.quote_start) = r.evidence_quote
  AND substring(sec.text,r.disease_start,
                r.disease_end-r.disease_start) = r.disease_name
MERGE (term:DiseaseTerm {id:r.disease_id})
  ON CREATE SET term.name=r.disease_name,
                term.normalization_status='dataset_exact_match_not_ontology_mapped',
                term.term_classification='unverified_disease_or_symptom',
                term.kg_snapshot=$snapshot
MERGE (c:DiseaseCandidate {id:r.id})
  ON CREATE SET c.drug_name=r.drug_name,
                c.disease_name=r.disease_name,
                c.evidence_quote=r.evidence_quote,
                c.quote_start=r.quote_start,
                c.quote_end=r.quote_end,
                c.disease_start=r.disease_start,
                c.disease_end=r.disease_end,
                c.context_flag=r.context_flag,
                c.candidate_status='pending_human_review',
                c.clinical_status='not_validated',
                c.source_status='third_party_unverified',
                c.kg_snapshot=$snapshot
MERGE (d)-[:HAS_CANDIDATE]->(c)
MERGE (c)-[:MENTIONS_TERM]->(term)
MERGE (c)-[:SUPPORTED_BY]->(sec)
MERGE (c)-[:FROM_RECORD]->(rec)
RETURN count(c) AS written
"""
WRITE_ASSESSMENTS = """
UNWIND $rows AS r
MATCH (c:DiseaseCandidate {id:r.candidate_id,kg_snapshot:$snapshot})
MERGE (a:ModelAssessment {id:r.id})
  ON CREATE SET a.candidate_decision=r.candidate_decision,
                a.text_relation_status=r.text_relation_status,
                a.proposed_relation=r.proposed_relation,
                a.llm_status=r.llm_status,
                a.evidence_validated=r.evidence_validated,
                a.specific_target_surface=r.specific_target_surface,
                a.model=r.model,
                a.prompt_sha256=r.prompt_sha256,
                a.experiment_version=r.experiment_version,
                a.qualifiers_json=r.qualifiers_json,
                a.override_reasons_json=r.override_reasons_json,
                a.validation_error=r.validation_error,
                a.review_status='machine_proposal_not_clinically_validated',
                a.kg_snapshot=$snapshot
MERGE (a)-[:ASSESSES]->(c)
RETURN count(a) AS written
"""
CHECK_CANDIDATE_LINKS = """
MATCH (c:DiseaseCandidate {kg_snapshot:$snapshot})
OPTIONAL MATCH (d:Drug)-[:HAS_CANDIDATE]->(c)
OPTIONAL MATCH (c)-[:MENTIONS_TERM]->(t:DiseaseTerm)
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(sec:Section)
OPTIONAL MATCH (c)-[:FROM_RECORD]->(rec:SourceRecord)
WITH c,
     count(DISTINCT d) AS dc,
     count(DISTINCT t) AS tc,
     count(DISTINCT sec) AS sc,
     count(DISTINCT rec) AS rc,
     max(CASE WHEN
        substring(sec.text,c.quote_start,c.quote_end-c.quote_start)
         = c.evidence_quote
        THEN 1 ELSE 0 END) AS has_quote
RETURN count(c) AS candidates,
       sum(CASE WHEN dc=1 AND tc=1 AND sc=1 AND rc=1
                         AND has_quote=1 THEN 0 ELSE 1 END) AS invalid
"""

def candidate_rows(in_dir,candidates,pilot,snapshot):
    urls={
        x["id"]: x.get("source_url","")
        for x in source.each_jsonl(Path(in_dir)/"source_records.jsonl")
    }
    for row in source.each_payload(
        "disease_candidates",in_dir,candidates,pilot,snapshot
    ):
        row["source_url"]=urls[row["source_record_id"]]
        yield row

def assessment_rows(in_dir,candidates,pilot,snapshot):
    return source.each_payload(
        "model_assessments",in_dir,candidates,pilot,snapshot
    )

def database_id_check(session,in_dir,progress_every=2):
    """Require full ID sets identical; count equality alone is insufficient."""
    report={}
    for index,(label,(filename,expected)) in enumerate(EXPECTED_BASE.items(),1):
        local_ids=set()
        for row in source.each_jsonl(Path(in_dir)/(filename+".jsonl")):
            if row["id"] in local_ids:
                raise ValueError(f"Duplicate local {label} ID: {row['id']}")
            local_ids.add(row["id"])
        if len(local_ids)!=expected:
            raise ValueError(
                f"Wrong local {label} count {len(local_ids)} != {expected}"
            )
        db_ids=set()
        db_n=0
        for record in session.run(f"MATCH (n:{label}) RETURN n.id AS id"):
            value=record["id"]
            if value is None:
                raise RuntimeError(f"Existing Neo4j {label} has node without id")
            if value in db_ids:
                raise RuntimeError(f"Duplicate Neo4j {label} id: {value}")
            db_n+=1
            db_ids.add(value)
        if db_ids!=local_ids:
            example_missing=sorted(local_ids-db_ids)[:3]
            example_extra=sorted(db_ids-local_ids)[:3]
            raise RuntimeError(
                f"Existing Neo4j {label} IDs differ from local clean V2. "
                f"db={db_n}, local={expected}; "
                f"missing_ids_sample={example_missing}; "
                f"unexpected_ids_sample={example_extra}. "
                "Abort safely; do not reimport the full graph."
            )
        report[label]=db_n
        if index%progress_every==0 or index==len(EXPECTED_BASE):
            print(f"[BASE] ID 完全匹配：{label} {db_n:,}",flush=True)
    return report

def check_existing_layer(session,in_dir,candidates,pilot,snapshot,expected):
    candidate_ids={
        x["id"] for x in candidate_rows(in_dir,candidates,pilot,snapshot)
    }
    expected_cands=expected["disease_candidates"]
    if len(candidate_ids)!=expected_cands:
        raise ValueError("Duplicate local DiseaseCandidate IDs")

    for name,cypher,allow in [
        ("DiseaseCandidate",EXISTING_CANDIDATE_Q,candidate_ids),
        ("ModelAssessment",EXISTING_ASSESSMENT_Q,{
            r["id"] for r in assessment_rows(
                in_dir,candidates,pilot,snapshot
            )
        })
    ]:
        ids=set()
        for record in session.run(cypher):
            value=record["id"]
            state=record["snapshot"]
            if value not in allow or value in ids or state!=snapshot:
                raise RuntimeError(
                    f"Existing {name} incompatible with current snapshot. "
                    f"Problem node={value}, snapshot={state}. "
                    "No writes have been performed."
                )
            ids.add(value)
        print(f"[PENDING] 现有{name}: {len(ids)}（可安全续跑）",flush=True)
    valid_names={
        x["id"]:x["name"] for x in source.each_payload(
            "disease_terms",in_dir,candidates,pilot,snapshot
        )
    }
    for x in session.run(EXISTING_TERMS_Q):
        if x["id"] in valid_names and x["name"]!=valid_names[x["id"]]:
            raise RuntimeError(
                f"DiseaseTerm existing ID/name collision: {x['id']}"
            )
    return {
        "candidate_total":len(candidate_ids),
        "terms_total":len(valid_names),
        "model_assessments_total":expected["model_assessments"],
    }

def validate_paths(session,rows,snapshot,batch_size,progress_every):
    total=0
    for batch in source.batched(rows,batch_size):
        results=list(session.run(READ_EVIDENCE,rows=batch,snapshot=snapshot))
        result_by_id={}
        for item in results:
            cid=item["id"]
            if cid in result_by_id:
                raise RuntimeError("Duplicate evidence check result: "+cid)
            result_by_id[cid]=(item["paths"],item["valid_paths"])
        for r in batch:
            paths,valid=result_by_id.get(r["id"],(-1,-1))
            if paths!=1 or valid!=1:
                raise RuntimeError(
                    f"Existing graph evidence path mismatch: {r['id']} "
                    f"drug={r['drug_name']} disease={r['disease_name']} "
                    f"paths={paths} matched={valid}. "
                    "No import was started; inspect the existing Section.text, "
                    "source_record.sha256, and relationships first."
                )
        total+=len(batch)
        if total%progress_every<batch_size:
            print(f"[EVIDENCE] 已验证 {total:,} 个来源路径",flush=True)
    print(f"[EVIDENCE] {total:,} 条候选原文验证通过",flush=True)
    return total

def tx_add(tx,cypher,rows,snapshot):
    rec=tx.run(cypher,rows=rows,snapshot=snapshot).single()
    written=rec["written"] if rec else -1
    if written!=len(rows):
        raise RuntimeError(
            f"Transaction refused partial batch: {written} != {len(rows)}"
        )
    return written

def apply_batches(session,rows,cypher,snapshot,batch_size,kind):
    total=0
    for batch in source.batched(rows,batch_size):
        written=session.execute_write(tx_add,cypher,batch,snapshot)
        total+=written
        if total%5000<batch_size:
            print(f"[WRITE] {kind}: {total:,}",flush=True)
    print(f"[WRITE] {kind}: 已提交 {total:,}",flush=True)
    return total

def final_verify(session,snapshot,expected):
    candidates=session.run(
        "MATCH (c:DiseaseCandidate {kg_snapshot:$snap}) "
        "RETURN count(c) AS n",snap=snapshot
    ).single()["n"]
    assessments=session.run(
        "MATCH (a:ModelAssessment {kg_snapshot:$snap}) "
        "RETURN count(a) AS n",snap=snapshot
    ).single()["n"]
    rel=session.run(CHECK_CANDIDATE_LINKS,snapshot=snapshot).single()
    if (
        candidates!=expected["disease_candidates"]
        or assessments!=expected["model_assessments"]
        or rel["candidates"]!=expected["disease_candidates"]
        or rel["invalid"]!=0
    ):
        raise RuntimeError(
            f"POST-IMPORT VERIFICATION FAIL: "
            f"candidates={candidates}, assessments={assessments}, "
            f"links={rel['candidates']}, invalid={rel['invalid']}"
        )
    return {
        "status":"VERIFIED",
        "disease_candidates":candidates,
        "model_assessments":assessments,
        "invalid_links_or_evidence":rel["invalid"],
        "verified_clinical_indications":0,
    }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["attach","inspect"])
    parser.add_argument("--in-dir",default="kg_output_clean_v2_import_fixed")
    parser.add_argument(
        "--candidates",default="kg_output_clean_v2/disease_candidates_full.csv"
    )
    parser.add_argument(
        "--pilot",default="kg_output_clean_v2/pilot_v31_merged60.csv"
    )
    parser.add_argument("--uri",default="bolt://localhost:7687")
    parser.add_argument("--user",default="neo4j")
    parser.add_argument("--database",default="neo4j")
    parser.add_argument("--batch-size",type=int,default=400)
    parser.add_argument("--apply",action="store_true")
    parser.add_argument("--report")
    args=parser.parse_args()
    if not 1<=args.batch_size<=2000:
        parser.error("--batch-size must be 1..2000")
    if args.command!="attach" and args.apply:
        parser.error("Only 'attach' supports --apply")
    if args.command=="attach" and not args.apply:
        parser.error("'attach' requires explicit --apply")
    try:
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except ImportError:
            pass
        print("[LOCAL] 校验 cleaned V2 数据和原文引用...",flush=True)
        pre=source.preflight(args.in_dir,args.candidates,args.pilot)
        snapshot=pre["snapshot_sha256"]
        expected=pre["source_graph_counts"]
        driver=source.connect(args.uri,args.user,args.database)
        try:
            with driver.session(database=args.database) as session:
                ids=database_id_check(session,args.in_dir)
                layer=check_existing_layer(
                    session,args.in_dir,args.candidates,args.pilot,
                    snapshot,expected
                )
                validate_paths(
                    session,candidate_rows(
                        args.in_dir,args.candidates,args.pilot,snapshot
                    ),snapshot,args.batch_size,5000
                )
                if args.command=="inspect":
                    result={
                        "status":"SAFE_TO_ATTACH",
                        "base_ids":ids,
                        "candidate_provenance":layer,
                        "graph_writes":0
                    }
                else:
                    for label in ("DiseaseTerm","DiseaseCandidate","ModelAssessment"):
                        session.run(
                            f"CREATE CONSTRAINT drugkg_attach_{label.lower()}_id_v1 "
                            f"IF NOT EXISTS FOR (n:{label}) "
                            "REQUIRE n.id IS UNIQUE"
                        ).consume()
                    n1=apply_batches(
                        session,candidate_rows(
                            args.in_dir,args.candidates,args.pilot,snapshot
                        ),WRITE_CANDIDATES,snapshot,args.batch_size,
                        "DiseaseCandidate"
                    )
                    n2=apply_batches(
                        session,assessment_rows(
                            args.in_dir,args.candidates,args.pilot,snapshot
                        ),WRITE_ASSESSMENTS,snapshot,args.batch_size,
                        "ModelAssessment"
                    )
                    if n1!=expected["disease_candidates"]:
                        raise RuntimeError("Incomplete candidate batches")
                    if n2!=expected["model_assessments"]:
                        raise RuntimeError("Incomplete assessment batches")
                    result=final_verify(session,snapshot,expected)
                    result["base_ids_confirmed"]=ids
                    result["graph_writes_performed"]=True
        finally:
            driver.close()
        if args.report:
            source.atomic_json(args.report,result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        print(f"[ERROR] {exc}",file=sys.stderr,flush=True)
        sys.exit(2)

if __name__=="__main__":
    main()
