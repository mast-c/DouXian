"""Phase-4 qualified review of approved drug-disease *text claims*.

Drop into existing RAG_project/kg/ without replacing disease_stage3.py.

Commands:
  python -m kg.review_context_v2 export --csv ... --in-dir ... --out-dir ...
  python -m kg.review_context_v2 import --out-dir ...
  python -m kg.review_context_v2 import --out-dir ... --apply

Only imports a research claim with pending expert validation; never creates
unqualified Drug-[:TREATS]->Disease links or clinical approval assertions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

from kg.disease_stage3 import (
    hash_id, RELATION_TYPES, each_jsonl, connect_driver, in_batches,
    graph_snapshot_sha256,
)

REQUIRED_FIELDS={
    "candidate_id","drug_name","disease_name","source_record_id",
    "section_id","source_url","evidence_quote",
    "quote_start","quote_end","disease_start","disease_end",
    "context_flag","review_status","relation_type","reviewer","review_note"
}
ALLOWED_LABELS={"approved","rejected","pending","uncertain"}
SCHEMA_VERSION="qualified_review_v2_2026_10"

def sha256_file(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()

def save_jsonl(path, entries):
    with Path(path).open("w",encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e,ensure_ascii=False,sort_keys=True)+"\n")

def load_sources(in_dir):
    path=Path(in_dir)
    drugs={r["id"]:r for r in each_jsonl(path/"drugs.jsonl")}
    records={r["id"]:r for r in each_jsonl(path/"source_records.jsonl")}
    sections={r["id"]:r for r in each_jsonl(path/"sections.jsonl")}
    if not drugs or not records or not sections:
        raise ValueError("缺少 Drug / SourceRecord / Section 数据")
    return drugs,records,sections

def check_candidate(row, drugs, records, sections):
    """Strictly validate provenance, no implicit normalization."""
    disease=(row.get("disease_name") or "").strip()
    section=sections.get(row.get("section_id"))
    record=records.get(row.get("source_record_id"))
    if not section or not record:
        raise ValueError("Section or SourceRecord not found")
    if section["record_id"]!=record["id"] or section.get("field")!="indication":
        raise ValueError("Section does not match the indicated source record")
    drug=drugs.get(record["drug_id"])
    if not drug or drug.get("name")!=row.get("drug_name"):
        raise ValueError("Drug identity mismatch")
    if row.get("source_url","")!=record.get("source_url",""):
        raise ValueError("Source URL differs from the graph manifest")
    try:
        qs=int(row["quote_start"]); qe=int(row["quote_end"])
        ds=int(row["disease_start"]); de=int(row["disease_end"])
    except (TypeError,ValueError,KeyError) as e:
        raise ValueError("Offsets are invalid") from e
    source=section["text"]
    if not (0 <= qs <= ds < de <= qe <= len(source)):
        raise ValueError("Offsets out of bounds or not nested")
    if source[qs:qe]!=row["evidence_quote"] or source[ds:de]!=disease:
        raise ValueError("Evidence text is not an exact match")
    expected=hash_id("DiseaseCandidate",section["id"],disease,ds,de)
    if row.get("candidate_id")!=expected:
        raise ValueError("candidate_id does not match source evidence")
    return drug,record,section,(qs,qe,ds,de)

def annotation_method(reviewer):
    # Name is a clue, not sufficient to claim expert human verification.
    return ("AI_assisted_not_expert_verified"
            if "codex" in reviewer.casefold() else
            "reviewer_identity_unverified")

def export_reviewed(csv_path, in_dir, out_dir, *, require_complete=True):
    csv_path,in_dir,out_dir=Path(csv_path),Path(in_dir),Path(out_dir)
    drugs, records, sections=load_sources(in_dir)
    disease_nodes={}
    claims={}
    seen_candidates=set()
    decisions=Counter()
    flag_matrix=Counter()
    errors=[]
    sample_sources=[]
    with csv_path.open("r",encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if not REQUIRED_FIELDS.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV 缺少字段："+str(sorted(REQUIRED_FIELDS-set(reader.fieldnames or [])))
            )
        for i,row in enumerate(reader,2):
            try:
                drug,record,section,offsets=check_candidate(
                    row,drugs,records,sections
                )
                cid=row["candidate_id"]
                if cid in seen_candidates:
                    raise ValueError("duplicate candidate_id")
                seen_candidates.add(cid)
                status=(row.get("review_status") or "").strip().lower()
                flag=(row.get("context_flag") or "").strip()
                if status not in ALLOWED_LABELS:
                    raise ValueError("unexpected review_status")
                note=(row.get("review_note") or "").strip()
                reviewer=(row.get("reviewer") or "").strip()
                relation=(row.get("relation_type") or "").strip().lower()
                decisions[status]+=1
                flag_matrix[(flag,status)]+=1

                if status=="rejected":
                    if not note:
                        raise ValueError("rejection reason is required")
                    continue
                if status in {"pending","uncertain"}:
                    if require_complete:
                        raise ValueError(
                            "Incomplete review: pending or uncertain candidate"
                        )
                    continue
                if not reviewer or not note or relation not in RELATION_TYPES:
                    raise ValueError(
                        "approved candidate needs reviewer, note and relation_type"
                    )
                qs,qe,ds,de=offsets
                disease=row["disease_name"].strip()
                disease_id=hash_id("DiseaseTerm",disease)
                disease_nodes[disease_id]={
                    "id":disease_id,"name":disease,
                    "normalization_status":"dataset_exact_match_not_ontology_mapped"
                }
                claim_id=hash_id(
                    "IndicationClaim",section["id"],disease_id,ds,de,relation
                )

                context_sensitive=(flag=="negation_or_restriction_check")
                annotation=annotation_method(reviewer)
                claims[claim_id]={
                    "id":claim_id,
                    "drug_id":drug["id"],
                    "disease_id":disease_id,
                    "section_id":section["id"],
                    "source_record_id":record["id"],
                    "candidate_id":cid,
                    "predicate":"INDICATION_TEXT_MENTIONS",
                    "relation_type":relation,
                    "evidence_quote":row["evidence_quote"],
                    "disease_name":disease,
                    "char_start":qs,"char_end":qe,
                    "disease_char_start":ds,"disease_char_end":de,
                    "review_status":"provisional_text_relation",
                    "source_annotation_status":"approved",
                    "annotation_method":annotation,
                    "annotator_id":reviewer,
                    "review_note":note,
                    "context_flag":flag,
                    "requires_condition_check":context_sensitive,
                    "qualification_status":(
                        "flagged_context_requires_manual_interpretation"
                        if context_sensitive else "not_fully_assessed_for_conditions"
                    ),
                    "qualification_note":note if context_sensitive else "",
                    "expert_review_status":"pending",
                    "clinical_status":"not_validated",
                    "regulatory_status":"source_version_unverified",
                    "schema_version":SCHEMA_VERSION,
                }
            except Exception as exc:
                errors.append({"csv_line":i,"error":str(exc)})
    if errors:
        raise ValueError(
            "候选证据或人工标注校验失败，未导出任何 Claim。首条错误："+
            json.dumps(errors[0],ensure_ascii=False)
        )
    if require_complete and decisions["approved"]+decisions["rejected"]!=len(seen_candidates):
        raise ValueError("Not all candidates are approved or rejected")

    out_dir.mkdir(parents=True,exist_ok=True)
    claim_file=out_dir/"pilot_indication_claims.jsonl"
    disease_file=out_dir/"pilot_disease_terms.jsonl"
    save_jsonl(claim_file,sorted(claims.values(),key=lambda x:x["id"]))
    save_jsonl(disease_file,sorted(disease_nodes.values(),key=lambda x:x["id"]))
    by_relation=Counter(x["relation_type"] for x in claims.values())
    by_method=Counter(x["annotation_method"] for x in claims.values())
    report={
        "status":"exported_provisional_text_claims_not_expert_validated",
        "candidates_checked":len(seen_candidates),
        "decisions":dict(decisions),
        "accepted_claims":len(claims),
        "disease_terms":len(disease_nodes),
        "relation_types":dict(by_relation),
        "context_sensitive_approved":sum(
            x["requires_condition_check"] for x in claims.values()
        ),
        "annotation_methods":dict(by_method),
        "context_flag_decision_matrix":{
            f"{a}|{b}":count for (a,b),count in sorted(flag_matrix.items())
        },
        "source_csv_sha256":sha256_file(csv_path),
        "graph_snapshot_sha256":graph_snapshot_sha256(in_dir),
        "output_claim_sha256":sha256_file(claim_file),
        "output_disease_sha256":sha256_file(disease_file),
        "schema_version":SCHEMA_VERSION,
        "warning":"This is a text-supported provisional graph, NOT NMPA-approved indications."
    }
    (out_dir/"pilot_review_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    return report

# Must validate exact evidence again against Neo4j before constructing nodes.
PREFLIGHT = """
UNWIND $rows AS r
MATCH (drug:Drug {id:r.drug_id})-[:HAS_RECORD]->
      (src:SourceRecord {id:r.source_record_id})-[:HAS_SECTION]->
      (s:Section {id:r.section_id})
WHERE substring(s.text, r.char_start, r.char_end-r.char_start) = r.evidence_quote
  AND substring(s.text,
                r.disease_char_start,
                r.disease_char_end-r.disease_char_start) = r.disease_name
RETURN count(*) AS n
"""
DISEASES = """
UNWIND $rows AS r
MERGE (d:DiseaseTerm {id:r.id})
SET d.name=r.name,
    d.normalization_status=r.normalization_status
"""
CLAIMS = """
UNWIND $rows AS r
MATCH (drug:Drug {id:r.drug_id})
MATCH (s:Section {id:r.section_id})
MATCH (d:DiseaseTerm {id:r.disease_id})
MATCH (src:SourceRecord {id:r.source_record_id})-[:HAS_SECTION]->(s)
MATCH (drug)-[:HAS_RECORD]->(src)
WHERE substring(s.text, r.char_start, r.char_end-r.char_start) = r.evidence_quote
  AND substring(s.text,
                r.disease_char_start,
                r.disease_char_end-r.disease_char_start) = d.name
MERGE (c:IndicationClaim {id:r.id})
SET c.predicate=r.predicate,
    c.relation_type=r.relation_type,
    c.evidence_quote=r.evidence_quote,
    c.char_start=r.char_start,
    c.char_end=r.char_end,
    c.disease_char_start=r.disease_char_start,
    c.disease_char_end=r.disease_char_end,
    c.review_status=r.review_status,
    c.source_annotation_status=r.source_annotation_status,
    c.annotation_method=r.annotation_method,
    c.annotator_id=r.annotator_id,
    c.review_note=r.review_note,
    c.context_flag=r.context_flag,
    c.requires_condition_check=r.requires_condition_check,
    c.qualification_status=r.qualification_status,
    c.qualification_note=r.qualification_note,
    c.expert_review_status=r.expert_review_status,
    c.clinical_status=r.clinical_status,
    c.regulatory_status=r.regulatory_status,
    c.schema_version=r.schema_version
MERGE (c)-[:SUBJECT]->(drug)
MERGE (c)-[:OBJECT]->(d)
MERGE (c)-[:SUPPORTED_BY]->(s)
"""
def import_reviewed(out_dir, *, apply=False, database="neo4j"):
    out_dir=Path(out_dir)
    report_path=out_dir/"pilot_review_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(str(report_path))
    report=json.loads(report_path.read_text(encoding="utf-8"))
    terms=list(each_jsonl(out_dir/"pilot_disease_terms.jsonl"))
    claims=list(each_jsonl(out_dir/"pilot_indication_claims.jsonl"))
    summary={
        "mode":"READ_ONLY" if not apply else "APPLY_REQUESTED",
        "database":database,
        "disease_term_count":len(terms),
        "provisional_claim_count":len(claims),
        "context_sensitive_claim_count":sum(
            bool(x.get("requires_condition_check")) for x in claims
        ),
        "expert_validation":"pending",
    }
    if report.get("output_claim_sha256")!=sha256_file(
            out_dir/"pilot_indication_claims.jsonl"):
        raise RuntimeError("Claim file has changed since reviewed export")
    if report.get("output_disease_sha256")!=sha256_file(
            out_dir/"pilot_disease_terms.jsonl"):
        raise RuntimeError("DiseaseTerm file has changed since reviewed export")
    if report.get("accepted_claims")!=len(claims):
        raise RuntimeError("Claim count does not match review report")
    if report.get("schema_version")!=SCHEMA_VERSION:
        raise RuntimeError("Different review schema version")
    if not apply:
        return summary
    if not claims:
        raise RuntimeError("No provisional claims to import")
    if any(
        c.get("expert_review_status")!="pending"
        or c.get("clinical_status")!="not_validated"
        or c.get("source_annotation_status")!="approved"
        or c.get("review_status")!="provisional_text_relation"
        for c in claims
    ):
        raise RuntimeError("Unexpected claim validation status")
    disease_by_id={d["id"]:d["name"] for d in terms}
    if any(
        c["disease_id"] not in disease_by_id
        or c.get("disease_name")!=disease_by_id[c["disease_id"]]
        for c in claims
    ):
        raise RuntimeError("DiseaseTerm links do not match claim names")
    driver=connect_driver()
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            exists=0
            for items in in_batches(claims):
                exists+=session.run(PREFLIGHT,rows=items).single()["n"]
            if exists!=len(claims):
                raise RuntimeError(
                    f"Neo4j evidence preflight mismatch {exists}/{len(claims)}. "
                    "No writes made."
                )
            # Do not overwrite a previously expert-approved claim.
            for items in in_batches(claims):
                protected=session.run("""
                    UNWIND $ids AS id
                    MATCH (c:IndicationClaim {id:id})
                    WHERE c.expert_review_status IN ['approved','rejected']
                    RETURN count(c) AS n
                """,ids=[item["id"] for item in items]).single()["n"]
                if protected:
                    raise RuntimeError(
                        "Existing expert-reviewed claims found. "
                        "Refusing to overwrite them."
                    )
            session.run(
                "CREATE CONSTRAINT disease_term_v2_id IF NOT EXISTS "
                "FOR (d:DiseaseTerm) REQUIRE d.id IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT indication_claim_v2_id IF NOT EXISTS "
                "FOR (c:IndicationClaim) REQUIRE c.id IS UNIQUE"
            ).consume()
            for batch in in_batches(terms):
                session.execute_write(
                    lambda tx,chunk: tx.run(DISEASES,rows=chunk).consume(),batch
                )
            for batch in in_batches(claims):
                session.execute_write(
                    lambda tx,chunk: tx.run(CLAIMS,rows=chunk).consume(),batch
                )
            summary["actual_claims_present"]=sum(
                session.run("""
                    UNWIND $ids AS id
                    MATCH (c:IndicationClaim {id:id})
                    RETURN count(c) AS n
                """,ids=[x["id"] for x in items]).single()["n"]
                for items in in_batches(claims)
            )
            if summary["actual_claims_present"]!=len(claims):
                raise RuntimeError(
                    "Some provisional claims did not import, inspect Neo4j graph"
                )
            summary["mode"]="APPLIED"
            return summary
    finally:
        driver.close()

def main():
    parser=argparse.ArgumentParser(
        description="支持条件性适应症的保守审核模块（不会创建无条件 TREATS 边）"
    )
    sub=parser.add_subparsers(dest="cmd",required=True)
    exp=sub.add_parser("export")
    exp.add_argument("--csv",required=True)
    exp.add_argument("--in-dir",default="data/kg_output_clean_v2")
    exp.add_argument("--out-dir",default="data/kg_output_clean_v2/pilot_review_v2")
    exp.add_argument("--allow-pending",action="store_true")
    imp=sub.add_parser("import")
    imp.add_argument("--out-dir",default="data/kg_output_clean_v2/pilot_review_v2")
    imp.add_argument("--database",default="neo4j")
    imp.add_argument("--apply",action="store_true")
    a=parser.parse_args()
    if a.cmd=="export":
        result=export_reviewed(a.csv,a.in_dir,a.out_dir,
                              require_complete=not a.allow_pending)
    else:
        result=import_reviewed(a.out_dir,apply=a.apply,database=a.database)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
