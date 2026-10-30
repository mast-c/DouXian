"""Safe, repeatable Neo4j import of cleaned DrugKG V2 + disease mentions.

No LLM calls. No source-text claim is considered clinically validated.
Each raw disease match is a :DiseaseCandidate, never a confirmed :INDICATED_FOR.

Example (from RAG_project/data):
  python -m kg.neo4j_drug_graph preflight --in-dir kg_output_clean_v2 --candidates kg_output_clean_v2/disease_candidates_full.csv --pilot kg_output_clean_v2/pilot_v31_merged60.csv
  python -m kg.neo4j_drug_graph apply --in-dir kg_output_clean_v2 --candidates kg_output_clean_v2/disease_candidates_full.csv --pilot kg_output_clean_v2/pilot_v31_merged60.csv --apply
  python -m kg.neo4j_drug_graph verify --in-dir kg_output_clean_v2 --candidates kg_output_clean_v2/disease_candidates_full.csv --pilot kg_output_clean_v2/pilot_v31_merged60.csv
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterator

VERSION="drugkg_clean_v2_candidates_only_import_v1"
REQUIRED_CANDIDATE_COLUMNS={
    "candidate_id","drug_name","disease_name",
    "source_record_id","section_id","source_url",
    "evidence_quote","quote_start","quote_end",
    "disease_start","disease_end",
    "context_flag","review_status","relation_type"
}
BASE_KINDS=("drugs","source_records","sections","chunks","drug_aliases")
BATCH_KINDS=BASE_KINDS+("disease_terms","disease_candidates","model_assessments")

CONSTRAINTS={
    "Drug":"drugkg_drug_id_v1",
    "SourceRecord":"drugkg_record_id_v1",
    "Section":"drugkg_section_id_v1",
    "Chunk":"drugkg_chunk_id_v1",
    "DiseaseTerm":"drugkg_term_id_v1",
    "DiseaseCandidate":"drugkg_diseasecandidate_id_v1",
    "ModelAssessment":"drugkg_modelassessment_id_v1",
    "DrugAlias":"drugkg_alias_id_v1",
}
INDEXES=(
    "CREATE INDEX drugkg_drug_name_v1 IF NOT EXISTS FOR (n:Drug) ON (n.name)",
    "CREATE INDEX drugkg_term_name_v1 IF NOT EXISTS FOR (n:DiseaseTerm) ON (n.name)",
    "CREATE INDEX drugkg_candidate_snapshot_v1 IF NOT EXISTS FOR (n:DiseaseCandidate) ON (n.kg_snapshot)",
    "CREATE INDEX drugkg_alias_name_v1 IF NOT EXISTS FOR (n:DrugAlias) ON (n.name)",
    "CREATE FULLTEXT INDEX drugkg_names_fulltext_v1 IF NOT EXISTS FOR (n:Drug|DiseaseTerm) ON EACH [n.name]",
    "CREATE FULLTEXT INDEX drugkg_section_text_v1 IF NOT EXISTS FOR (n:Section) ON EACH [n.text]",
)
QUERIES={
"drugs": """
    UNWIND $rows AS r
    MERGE (d:Drug {id:r.id})
    ON CREATE SET d.dataset_authority='third_party_unverified'
    SET d.name=r.name,
        d.entity_scope=r.entity_scope,
        d.kg_snapshot=$snapshot,
        d.dataset_authority=coalesce(
            d.dataset_authority,'third_party_unverified'
        )
""",
"source_records": """
    UNWIND $rows AS r
    MATCH (d:Drug {id:r.drug_id,kg_snapshot:$snapshot})
    MERGE (s:SourceRecord {id:r.id})
    ON CREATE SET s.version_status=r.version_status,
                  s.authority_status=r.authority_status,
                  s.license_status=r.license_status
    SET s.source_url=r.source_url,
        s.record_sha256=r.record_sha256,
        s.snapshot_sha256=r.snapshot_sha256,
        s.dataset=r.dataset,
        s.source_variant=coalesce(r.source_variant,s.source_variant),
        s.kg_snapshot=$snapshot
    MERGE (d)-[:HAS_RECORD]->(s)
""",
"sections": """
    UNWIND $rows AS r
    MATCH (s:SourceRecord {id:r.record_id,kg_snapshot:$snapshot})
    MERGE (sec:Section {id:r.id})
    SET sec.field=r.field, sec.title=r.title,
        sec.text=r.text, sec.placeholder=r.placeholder,
        sec.kg_snapshot=$snapshot
    MERGE (s)-[:HAS_SECTION]->(sec)
""",
"chunks": """
    UNWIND $rows AS r
    MATCH (s:Section {id:r.section_id,kg_snapshot:$snapshot})
    MERGE (c:Chunk {id:r.id})
    SET c.field=r.field, c.start=r.start,
        c.end=r.end, c.text=r.text,
        c.kg_snapshot=$snapshot
    MERGE (s)-[:HAS_CHUNK]->(c)
""",
"drug_aliases": """
    UNWIND $rows AS r
    MATCH (d:Drug {id:r.drug_id,kg_snapshot:$snapshot})
    MERGE (a:DrugAlias {id:r.id})
    ON CREATE SET a.review_status='candidate',
                  a.authority_status='unverified_alias'
    SET a.name=r.name,
        a.canonical_name=r.canonical_name,
        a.kg_snapshot=$snapshot
    MERGE (a)-[:ALIAS_CANDIDATE_OF]->(d)
""",
"disease_terms": """
    UNWIND $rows AS r
    MERGE (t:DiseaseTerm {id:r.id})
    ON CREATE SET t.normalization_status='dataset_exact_match_not_ontology_mapped',
                  t.term_classification='unverified_disease_or_symptom'
    SET t.name=r.name,
        t.kg_snapshot=$snapshot
""",
"disease_candidates": """
    UNWIND $rows AS r
    MATCH (d:Drug {id:r.drug_id,kg_snapshot:$snapshot})
    MATCH (term:DiseaseTerm {id:r.disease_id,kg_snapshot:$snapshot})
    MATCH (sec:Section {id:r.section_id,kg_snapshot:$snapshot})
    MATCH (record:SourceRecord {id:r.source_record_id,kg_snapshot:$snapshot})
    MERGE (c:DiseaseCandidate {id:r.id})
    ON CREATE SET c.candidate_status='pending_human_review',
                  c.clinical_status='not_validated',
                  c.source_status='third_party_unverified'
    SET c.disease_name=r.disease_name,
        c.drug_name=r.drug_name,
        c.evidence_quote=r.evidence_quote,
        c.quote_start=r.quote_start,
        c.quote_end=r.quote_end,
        c.disease_start=r.disease_start,
        c.disease_end=r.disease_end,
        c.context_flag=r.context_flag,
        c.kg_snapshot=$snapshot
    MERGE (d)-[:HAS_CANDIDATE]->(c)
    MERGE (c)-[:MENTIONS_TERM]->(term)
    MERGE (c)-[:SUPPORTED_BY]->(sec)
    MERGE (c)-[:FROM_RECORD]->(record)
""",
"model_assessments": """
    UNWIND $rows AS r
    MATCH (c:DiseaseCandidate {id:r.candidate_id,kg_snapshot:$snapshot})
    MERGE (a:ModelAssessment {id:r.id})
    ON CREATE SET a.review_status='machine_proposal_not_clinically_validated'
    SET a.candidate_decision=r.candidate_decision,
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
        a.kg_snapshot=$snapshot
    MERGE (a)-[:ASSESSES]->(c)
"""
}

def hash_id(kind,*parts):
    encoded=json.dumps(
        [kind,*parts],ensure_ascii=False,sort_keys=True,
        separators=(",",":")
    ).encode("utf-8")
    return kind.lower()+"_"+hashlib.sha256(encoded).hexdigest()[:24]

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()

def each_jsonl(path) -> Iterator[dict]:
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    with path.open("r",encoding="utf-8-sig") as stream:
        for number,line in enumerate(stream,1):
            if not line.strip():
                continue
            try:
                row=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}") from exc
            if not isinstance(row,dict):
                raise ValueError(f"Expected JSON object in {path}:{number}")
            yield row

def each_csv(path,required_columns) -> Iterator[dict]:
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV missing: {path}")
    with path.open("r",encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        fields=set(reader.fieldnames or [])
        missing=required_columns-fields
        if missing:
            raise ValueError(f"CSV missing fields {sorted(missing)} in {path}")
        for index,row in enumerate(reader,2):
            if not row.get("candidate_id"):
                raise ValueError(f"Missing candidate_id at {path}:{index}")
            yield row

def integer(row,key):
    try:
        n=int(row[key])
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError(
            f"Invalid numeric field {key} in {row.get('candidate_id','?')}"
        ) from exc
    return n

def load_pilot(pilot_path,snapshot,full_candidates):
    if pilot_path is None:
        return None,[]
    pilot=Path(pilot_path)
    report_path=pilot.with_suffix(".v31_report.json")
    if not report_path.is_file():
        raise ValueError(f"Pilot experiment report missing: {report_path}")
    report=json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("complete") is not True or report.get("selected_rows")!=60:
        raise ValueError("Pilot must be the completed 60-row merged dev set")
    if report.get("source_graph_sha256")!=snapshot:
        raise ValueError("Pilot source graph hash differs from cleaned data")
    if report.get("output_sha256")!=sha(pilot):
        raise ValueError("Pilot CSV SHA differs from recorded report")
    required=REQUIRED_CANDIDATE_COLUMNS|{
        "v31_llm_status","v31_candidate_decision","v31_text_relation_status",
        "v31_model","v31_version","v31_evidence_validated",
        "v31_proposed_relation","v31_specific_target_surface"
    }
    rows=[]
    seen=set()
    for r in each_csv(pilot,required):
        cid=r["candidate_id"]
        if cid in seen:
            raise ValueError("Duplicated candidate_id in pilot")
        seen.add(cid)
        original=full_candidates.get(cid)
        if original is None:
            raise ValueError(f"Pilot candidate is not in full candidate set: {cid}")
        for key in REQUIRED_CANDIDATE_COLUMNS-{"review_status","relation_type"}:
            if r.get(key)!=original.get(key):
                raise ValueError(f"Pilot evidence/source changed: {cid} {key}")
        rows.append(r)
    if len(rows)!=60:
        raise ValueError(f"Expected 60 pilot assessments, found {len(rows)}")
    return report,rows

def preflight(in_dir,candidates,pilot=None):
    """Read and verify all source links, exact evidence spans, IDs, snapshot."""
    started=time.monotonic()
    in_dir=Path(in_dir)
    candidate_path=Path(candidates)
    report_path=in_dir/"build_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing build report: {report_path}")
    build=json.loads(report_path.read_text(encoding="utf-8-sig"))
    snapshot=build.get("input_sha256","")
    if not isinstance(snapshot,str) or len(snapshot)!=64:
        raise ValueError("Invalid cleaned source snapshot hash")
    if build.get("source_drug_count")!=11953 and build.get("counts",{}).get("drugs")!=11953:
        # The importer is deliberately scoped to the user's cleaned snapshot.
        raise ValueError(
            "Expected cleaned V2 graph of 11,953 drugs. "
            "You may have passed an old 12,043-drug graph."
        )
    expected_counts=build.get("counts",{})
    counts=Counter()
    drugs={}
    for drug in each_jsonl(in_dir/"drugs.jsonl"):
        k=drug["id"]
        if k in drugs or not drug.get("name"):
            raise ValueError(f"Duplicate/empty drug: {k}")
        drugs[k]=drug
        counts["drugs"]+=1
    records={}
    for record in each_jsonl(in_dir/"source_records.jsonl"):
        k=record["id"]
        if k in records or record.get("drug_id") not in drugs:
            raise ValueError(f"Duplicate or orphan source record: {k}")
        if record.get("snapshot_sha256")!=snapshot:
            raise ValueError(f"Source record snapshot mismatch: {k}")
        records[k]=record
        counts["source_records"]+=1
    sections={}
    for section in each_jsonl(in_dir/"sections.jsonl"):
        k=section["id"]
        if k in sections or section.get("record_id") not in records:
            raise ValueError(f"Duplicate or orphan Section: {k}")
        if not isinstance(section.get("text"),str):
            raise ValueError(f"Section text must be a string: {k}")
        sections[k]=section
        counts["sections"]+=1
    chunks_seen=set()
    for c in each_jsonl(in_dir/"chunks.jsonl"):
        k=c["id"]
        if k in chunks_seen or c.get("section_id") not in sections:
            raise ValueError(f"Duplicate/orphan Chunk: {k}")
        chunks_seen.add(k)
        parent=sections[c["section_id"]]
        start,end=c.get("start"),c.get("end")
        if (not isinstance(start,int) or not isinstance(end,int) or
            not (0<=start<=end<=len(parent["text"])) or
            parent["text"][start:end]!=c.get("text")):
            raise ValueError(f"Chunk span/text mismatch: {k}")
        counts["chunks"]+=1
    del chunks_seen
    # The current kg.build (schema v0.2) may incorporate all seven
    # additional_source_records itself. In that case prepare_cleaned_v2
    # correctly reports 0 *newly appended* records, even though 11,960
    # SourceRecord rows and all seven extra sources are already present.
    # Never interpret `additional_source_records` here as the *total*
    # additional sources. Validate actual row counts and provenance links.
    cleaned_info=build.get("cleaned_snapshot",{})
    new_records=cleaned_info.get("additional_source_records",None)
    if new_records is not None and (
        not isinstance(new_records,int) or new_records<0
    ):
        raise ValueError("Invalid cleaned_snapshot.additional_source_records")
    expected_cleaned={
        "drugs":11953,"source_records":11960,
        "sections":106445,"chunks":107787,
        "drug_aliases":343
    }
    for kind,value in expected_cleaned.items():
        if counts.get(kind,0)!=value and kind!="drug_aliases":
            raise ValueError(
                f"Cleaned V2 {kind} count {counts.get(kind)} != {value}."
            )
    alias_ids=set()
    for alias in each_jsonl(in_dir/"drug_aliases.jsonl"):
        if alias["id"] in alias_ids or alias.get("drug_id") not in drugs:
            raise ValueError(f"Duplicate/orphan DrugAlias: {alias['id']}")
        if alias.get("snapshot_sha256")!=snapshot:
            raise ValueError(f"DrugAlias snapshot mismatch: {alias['id']}")
        alias_ids.add(alias["id"])
        counts["drug_aliases"]+=1
    if counts["drug_aliases"]!=expected_cleaned["drug_aliases"]:
        raise ValueError(
            "Missing 343 cleaned V2 alias candidates; use kg.prepare_cleaned_v2"
        )
    for kind in BASE_KINDS:
        if kind=="drug_aliases":
            continue  # build_report stores alias count in cleaned_snapshot
        if counts[kind]!=expected_counts.get(kind):
            raise ValueError(
                f"Build count mismatch for {kind}: "
                f"{counts[kind]} != {expected_counts.get(kind)}"
            )

    ids=set()
    names=set()
    candidate_refs={}
    for r in each_csv(candidate_path,REQUIRED_CANDIDATE_COLUMNS):
        cid=r["candidate_id"]
        if cid in ids:
            raise ValueError(f"Duplicate disease candidate: {cid}")
        ids.add(cid)
        if r["review_status"]!="pending" or r.get("relation_type"):
            raise ValueError(
                f"Full candidate CSV must be unreviewed: {cid}. "
                "Do not import AI-assisted labels as clinical gold."
            )
        disease=r.get("disease_name","")
        section=sections.get(r["section_id"])
        record=records.get(r["source_record_id"])
        if section is None or record is None:
            raise ValueError(f"Candidate references missing source: {cid}")
        if section.get("field")!="indication":
            raise ValueError(f"Candidate not in indication section: {cid}")
        if record["id"]!=section["record_id"]:
            raise ValueError(f"Candidate Section→SourceRecord mismatch: {cid}")
        drug=drugs[record["drug_id"]]
        if drug["name"]!=r["drug_name"]:
            raise ValueError(f"Candidate Drug name mismatch: {cid}")
        if record.get("source_url")!=r.get("source_url"):
            raise ValueError(f"Candidate URL mismatch: {cid}")
        qs,qe,ds,de=[
            integer(r,key) for key in (
                "quote_start","quote_end","disease_start","disease_end"
            )
        ]
        text=section["text"]
        if not (0<=qs<=ds<de<=qe<=len(text)):
            raise ValueError(f"Invalid candidate evidence span: {cid}")
        if text[qs:qe]!=r["evidence_quote"] or text[ds:de]!=disease:
            raise ValueError(f"Candidate evidence text mismatch: {cid}")
        if cid!=hash_id("DiseaseCandidate",r["section_id"],disease,ds,de):
            raise ValueError(f"Candidate provenance id mismatch: {cid}")
        names.add(disease)
        candidate_refs[cid]=r
        counts["disease_candidates"]+=1
    counts["disease_terms"]=len(names)
    if not names:
        raise ValueError("No disease candidate terms found")

    candidate_report_file=candidate_path.with_suffix(".report.json")
    candidate_report_found=candidate_report_file.is_file()
    if candidate_report_found:
        cand_report=json.loads(
            candidate_report_file.read_text(encoding="utf-8-sig")
        )
        if cand_report.get("graph_source_sha256")!=snapshot:
            raise ValueError("Candidate CSV from a different KG snapshot")
        if cand_report.get("candidate_mentions")!=counts["disease_candidates"]:
            raise ValueError("Candidate report count mismatch")

    pilot_report,assessment_rows=load_pilot(pilot,snapshot,candidate_refs)
    counts["model_assessments"]=len(assessment_rows)
    if pilot_report is not None and pilot_report.get("counts",{}).get("llm_valid")!=58:
        # Additional experiments can exist, but this version corresponds to
        # the specific 58/60 merged development run of the current project.
        raise ValueError("Pilot results do not match the provided 58/60 dev version")

    result={
        "status":"PREFLIGHT_PASS",
        "snapshot_sha256":snapshot,
        "build_report_sha256":sha(report_path),
        "source_graph_counts":dict(counts),
        "candidate_count":counts["disease_candidates"],
        "source_url_authority":"third_party_unverified",
        "candidate_relationship_status":"unvalidated_mentions_only",
        "candidate_csv_sha256":sha(candidate_path),
        "candidate_report_present":candidate_report_found,
        "pilot_assessment_sha256":sha(pilot) if pilot else None,
        "pilot_report_sha256":sha(
            Path(pilot).with_suffix(".v31_report.json")
        ) if pilot else None,
        "out_of_scope":["FDA/NMPA label approval","clinical validation",
                        "ontology normalization","therapy claims"],
        "elapsed_seconds":round(time.monotonic()-started,2),
        "graph_writes":0
    }
    return result

def each_payload(kind,in_dir,candidates,pilot,snapshot):
    if kind in BASE_KINDS:
        for r in each_jsonl(Path(in_dir)/f"{kind}.jsonl"):
            yield r
    elif kind=="disease_terms":
        names={r["disease_name"] for r in each_csv(
            candidates,REQUIRED_CANDIDATE_COLUMNS
        )}
        for name in sorted(names):
            yield {"id":hash_id("DiseaseTerm",name),"name":name}
    elif kind=="disease_candidates":
        records={
            row["id"]:row["drug_id"]
            for row in each_jsonl(Path(in_dir)/"source_records.jsonl")
        }
        for r in each_csv(candidates,REQUIRED_CANDIDATE_COLUMNS):
            yield {
                "id":r["candidate_id"],
                "drug_id":records[r["source_record_id"]],
                "disease_id":hash_id("DiseaseTerm",r["disease_name"]),
                "source_record_id":r["source_record_id"],
                "section_id":r["section_id"],
                "drug_name":r["drug_name"],
                "disease_name":r["disease_name"],
                "evidence_quote":r["evidence_quote"],
                "quote_start":int(r["quote_start"]),
                "quote_end":int(r["quote_end"]),
                "disease_start":int(r["disease_start"]),
                "disease_end":int(r["disease_end"]),
                "context_flag":r["context_flag"]
            }
    elif kind=="model_assessments":
        if pilot is not None:
            pilot_report=json.loads(
                Path(pilot).with_suffix(".v31_report.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            for r in each_csv(pilot,REQUIRED_CANDIDATE_COLUMNS):
                yield {
                    "id":hash_id(
                        "ModelAssessment",r["candidate_id"],
                        r.get("v31_version",""),
                        pilot_report["prompt_sha256"],
                        pilot_report["model"],
                        pilot_report["output_sha256"]
                    ),
                    "candidate_id":r["candidate_id"],
                    "candidate_decision":r.get("v31_candidate_decision",""),
                    "text_relation_status":r.get("v31_text_relation_status",""),
                    "proposed_relation":r.get("v31_proposed_relation",""),
                    "llm_status":r.get("v31_llm_status",""),
                    "evidence_validated":r.get(
                        "v31_evidence_validated","false"
                    ).lower()=="true",
                    "specific_target_surface":r.get(
                        "v31_specific_target_surface",""
                    ),
                    "model":r.get("v31_model",""),
                    "prompt_sha256":pilot_report["prompt_sha256"],
                    "experiment_version":r.get("v31_version",""),
                    "qualifiers_json":r.get("v31_qualifiers_json",""),
                    "override_reasons_json":r.get("v31_override_reasons_json",""),
                    "validation_error":r.get("v31_validation_error","")
                }
    else:
        raise ValueError(f"Invalid import stage: {kind}")

def batched(iterator,size):
    batch=[]
    for row in iterator:
        batch.append(row)
        if len(batch)>=size:
            yield batch
            batch=[]
    if batch:
        yield batch

def connect(uri,username,database):
    key=os.environ.get("NEO4J_PASSWORD","")
    if not key:
        raise RuntimeError(
            "NEO4J_PASSWORD not set; set it locally in PowerShell. "
            "Do not send passwords in chat or command-line arguments."
        )
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "Neo4j Python driver missing. Install with: pip install neo4j"
        ) from exc
    driver=GraphDatabase.driver(uri,auth=(username,key))
    driver.verify_connectivity()
    return driver

def write_batch(tx,query,rows,snapshot):
    return tx.run(query,rows=rows,snapshot=snapshot).consume()

def run_stage(session,kind,in_dir,candidates,pilot,snapshot,batch_size,progress):
    total=0
    started=time.monotonic()
    query=QUERIES[kind]
    for batch in batched(
        each_payload(kind,in_dir,candidates,pilot,snapshot),batch_size
    ):
        session.execute_write(write_batch,query,batch,snapshot)
        total+=len(batch)
        if total%progress<batch_size:
            print(
                f"  [{kind}] 已提交 {total:,} 条；"
                f"耗时 {time.monotonic()-started:.1f} 秒",
                flush=True
            )
    print(
        f"  [{kind}] DONE {total:,} 条，"
        f"耗时 {time.monotonic()-started:.1f} 秒",flush=True
    )
    return total

def check_baseline(session,expected,snapshot):
    if not expected: return
    for label,kind in [
        ("Drug","drugs"),("SourceRecord","source_records"),
        ("Section","sections"),("Chunk","chunks"),
    ]:
        got=session.run(
            f"MATCH (n:{label} {{kg_snapshot:$snap}}) RETURN count(n) AS n",
            snap=snapshot
        ).single()["n"]
        if got!=expected[kind]:
            raise RuntimeError(
                f"Cannot import candidates without matching base: "
                f"{label} count {got} != {expected[kind]} for cleaned snapshot"
            )

VERIFY_LABELS=(
    ("Drug","drugs"),("SourceRecord","source_records"),
    ("Section","sections"),("Chunk","chunks"),
    ("DrugAlias","drug_aliases"),
    ("DiseaseTerm","disease_terms"),
    ("DiseaseCandidate","disease_candidates"),
    ("ModelAssessment","model_assessments"),
)
VERIFY_EDGES=(
    ("SourceRecord","HAS_RECORD","Drug","in"),
    ("Section","HAS_SECTION","SourceRecord","in"),
    ("Chunk","HAS_CHUNK","Section","in"),
    ("DrugAlias","ALIAS_CANDIDATE_OF","Drug","out"),
    ("DiseaseCandidate","HAS_CANDIDATE","Drug","in"),
    ("DiseaseCandidate","MENTIONS_TERM","DiseaseTerm","out"),
    ("DiseaseCandidate","SUPPORTED_BY","Section","out"),
    ("DiseaseCandidate","FROM_RECORD","SourceRecord","out"),
    ("ModelAssessment","ASSESSES","DiseaseCandidate","out"),
)

def verify_graph(session,snapshot,counts):
    actual={}
    problems=[]
    for label,kind in VERIFY_LABELS:
        n=session.run(
            f"MATCH (n:{label} {{kg_snapshot:$snap}}) RETURN count(n) AS n",
            snap=snapshot
        ).single()["n"]
        actual[kind]=n
        if n!=counts[kind]:
            problems.append(f"Node count {label}: {n} != {counts[kind]}")
    # Validate that each node of a type has exactly one required link.
    # All queries are scoped to the cleaned snapshot on both endpoints.
    edge_counts={}
    for src_label,rel,dest_label,direction in VERIFY_EDGES:
        if direction=="in":
            patt=f"(other:{dest_label})-[:{rel}]->(n)"
        else:
            patt=f"(n)-[:{rel}]->(other:{dest_label})"
        query=(
            f"MATCH (n:{src_label} {{kg_snapshot:$snap}}) "
            f"OPTIONAL MATCH {patt} "
            f"WHERE other.kg_snapshot=$snap "
            "WITH n,count(other) AS links "
            "RETURN count(n) AS nodes, "
            "sum(CASE WHEN links=1 THEN 0 ELSE 1 END) AS bad"
        )
        r=session.run(query,snap=snapshot).single()
        failed=r["bad"] or 0
        edge_counts[f"{src_label}.{rel}"]=failed
        if failed:
            problems.append(f"Missing/duplicate {src_label}.{rel}: {failed}")

    evidence_query=(
        "MATCH (c:DiseaseCandidate {kg_snapshot:$snap}) "
        "MATCH (c)-[:SUPPORTED_BY]->(s:Section {kg_snapshot:$snap}) "
        "WHERE substring(s.text,c.quote_start,"
        "c.quote_end-c.quote_start) <> c.evidence_quote "
        "RETURN count(c) AS n"
    )
    mismatch=session.run(evidence_query,snap=snapshot).single()["n"]
    if mismatch:
        problems.append(f"Evidence substring mismatch: {mismatch}")
    return {
        "status":"VERIFIED" if not problems else "FAILED",
        "expected":counts,"actual":actual,
        "bad_links_by_type":edge_counts,
        "bad_evidence_quotes":mismatch,
        "problems":problems
    }

def ensure_isolated_dataset(session,snapshot):
    """Prevent silently mixing an old 12,043-drug graph with cleaned V2."""
    existing=session.run(
        "MATCH (d:Drug) "
        "WHERE d.kg_snapshot IS NULL OR d.kg_snapshot <> $snapshot "
        "RETURN count(d) AS old_drugs",
        snapshot=snapshot
    ).single()["old_drugs"]
    if existing:
        raise RuntimeError(
            f"Found {existing:,} Drug nodes outside cleaned V2 snapshot. "
            "Importer is ADDITIVE and never deletes old entries; "
            "do not mix old and cleaned graphs. "
            "Use the supplied isolated Docker Neo4j instance on port 7688, "
            "or separately run your established backed-up cleanup/sync procedure."
        )

def do_import(args,pre):
    if not args.apply:
        raise ValueError(
            "Refusing to change Neo4j without explicit --apply."
        )
    snap=pre["snapshot_sha256"]
    source_counts=pre["source_graph_counts"]
    uri=args.uri or os.environ.get("NEO4J_URI","bolt://localhost:7687")
    user=args.user or os.environ.get("NEO4J_USER","neo4j")
    database=args.database or os.environ.get("NEO4J_DATABASE","neo4j")
    kinds=(
        BASE_KINDS+("disease_terms","disease_candidates")+
        (("model_assessments",) if args.pilot else ())
        if args.stages=="all" else
        BASE_KINDS if args.stages=="base" else
        ("disease_terms","disease_candidates") if args.stages=="candidates" else
        ("model_assessments",)
    )
    if args.stages=="assessments" and not args.pilot:
        raise ValueError("Assessment-only stage requires --pilot")
    driver=connect(uri,user,database)
    imported={}
    try:
        with driver.session(database=database) as session:
            ensure_isolated_dataset(session,snap)
            for label,name in CONSTRAINTS.items():
                c=(
                    f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                )
                session.run(c).consume()
            for index in INDEXES:
                session.run(index).consume()
            if args.stages in ("candidates","assessments"):
                check_baseline(session,source_counts,snap)
            if args.stages=="assessments":
                counts=source_counts
                for label,kind in [
                    ("DiseaseCandidate","disease_candidates"),
                    ("DiseaseTerm","disease_terms")
                ]:
                    actual=session.run(
                        f"MATCH (x:{label} {{kg_snapshot:$snap}}) "
                        "RETURN count(x) AS n",snap=snap
                    ).single()["n"]
                    if actual!=counts[kind]:
                        raise RuntimeError(
                            f"Cannot add assessments before {label} import"
                        )
            for kind in kinds:
                print(f"[IMPORT] {kind} 开始...",flush=True)
                imported[kind]=run_stage(
                    session,kind,args.in_dir,args.candidates,args.pilot,
                    snap,args.batch_size,args.progress_every
                )
                if imported[kind]!=source_counts[kind]:
                    raise RuntimeError(
                        f"{kind} imported {imported[kind]} rows, "
                        f"expected {source_counts[kind]}. Stopping."
                    )
            if args.stages=="all":
                verified=verify_graph(session,snap,source_counts)
            else:
                verified={"status":"PARTIAL_STAGE_COMPLETE",
                          "note":"Run 'verify' after all base and candidate stages"}
    finally:
        driver.close()
    return {
        "mode":"NEO4J_WRITES_PERFORMED",
        "snapshot_sha256":snap,
        "database":database,"uri":uri,
        "stages":args.stages,
        "imported_rows":imported,
        "post_import_check":verified,
        "medical_claims_promoted":0,
        "clinical_review":False
    }

def do_verify(args,pre):
    uri=args.uri or os.environ.get("NEO4J_URI","bolt://localhost:7687")
    user=args.user or os.environ.get("NEO4J_USER","neo4j")
    database=args.database or os.environ.get("NEO4J_DATABASE","neo4j")
    driver=connect(uri,user,database)
    try:
        with driver.session(database=database) as session:
            outcome=verify_graph(
                session,pre["snapshot_sha256"],pre["source_graph_counts"]
            )
    finally:
        driver.close()
    return outcome

def atomic_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    try:
        with tmp.open("w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "command",choices=["preflight","apply","verify"],
        help="preflight never touches Neo4j; apply writes only with --apply; "
             "verify checks Neo4j"
    )
    ap.add_argument("--in-dir",default="kg_output_clean_v2")
    ap.add_argument("--candidates",default="kg_output_clean_v2/disease_candidates_full.csv")
    ap.add_argument("--pilot",default=None,
                    help="optional verified merged 60-row dev CSV, stored ONLY as ModelAssessment")
    ap.add_argument("--uri",default=None)
    ap.add_argument("--user",default=None)
    ap.add_argument("--database",default=None)
    ap.add_argument("--apply",action="store_true")
    ap.add_argument("--batch-size",type=int,default=400)
    ap.add_argument("--progress-every",type=int,default=5000)
    ap.add_argument(
        "--stages",choices=["all","base","candidates","assessments"],
        default="all"
    )
    ap.add_argument("--report",default=None)
    args=ap.parse_args()
    if args.batch_size<1 or args.batch_size>5000:
        ap.error("batch-size must be 1..5000")
    if args.progress_every<1:
        ap.error("progress-every must be >=1")
    if args.command!="apply" and args.apply:
        ap.error("--apply is only permitted with the apply command")
    try:
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except ImportError:
            pass
        print("[PREFLIGHT] 验证 cleaned V2 数据快照与 19,685 条候选原文...",flush=True)
        pre=preflight(args.in_dir,args.candidates,args.pilot)
        if args.command=="preflight":
            result=pre
        elif args.command=="apply":
            result=do_import(args,pre)
        else:
            result=do_verify(args,pre)
        if args.report:
            atomic_json(args.report,result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        if args.command=="verify" and result["status"]!="VERIFIED":
            sys.exit(1)
        if args.command=="apply" and args.stages=="all" and result["post_import_check"]["status"]!="VERIFIED":
            sys.exit(1)
    except (OSError,RuntimeError,ValueError,KeyError) as exc:
        print(f"[ERROR] {exc}",file=sys.stderr,flush=True)
        sys.exit(2)

if __name__=="__main__":
    main()
