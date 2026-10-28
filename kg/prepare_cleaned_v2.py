"""Prepare an already-cleaned MedGraphRAG-Drug-DB v2 snapshot for Neo4j.

This is an additive module: imports from the previously installed kg.build.
Keeps ALL canonical source records and `additional_source_records` (7 in
the uploaded snapshot), and emits a *candidate* alias mapping.

Does not turn malformed historical fragments from drug_name_redirects into
searchable pharmaceutical aliases.
"""
import argparse
import hashlib
import json
from pathlib import Path

from kg.build import (
    prepare as base_prepare, read_input, entity_id, text_of,
    chunk_sections, TEXT_FIELDS, NO_INFORMATION,
)

DATASET = "MedGraphRAG-Drug-DB drug_db_v2.json"

def hash_record(record):
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def write_rows(out_dir, kind, rows, mode="w"):
    path = Path(out_dir) / (kind + ".jsonl")
    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True)+"\n")
            count += 1
    return count

def _original_metadata(raw, canonical):
    for name in sorted(raw.get("drug_original_names", {})):
        if name in canonical:
            yield {"canonical_drug":name,
                   "original_names":raw["drug_original_names"][name]}

def prepare_cleaned(input_path, out_dir, *, limit=0, seed=42,
                    chunk_size=800, overlap=90):
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    raw = read_input(input_path)
    for required in ("additional_source_records","drug_aliases",
                     "drug_name_redirects","drug_original_names"):
        if not isinstance(raw.get(required),dict):
            raise ValueError(f"cleaned snapshot missing dictionary field {required}")

    # Base import preserves the original ID recipe, so unchanged drug records
    # retain their existing Neo4j Drug and SourceRecord identities.
    result = base_prepare(input_path, out_dir, limit=limit, seed=seed,
                          chunk_size=chunk_size, overlap=overlap)
    names = set(raw["drugs"])
    selected = names if limit == 0 else {
        r["name"] for r in iter_jsonl(out_dir/"drugs.jsonl")
    }

    # Require canonical targets to exist before writing anything else.
    invalid_targets = (
        [name for name in raw["additional_source_records"] if name not in names]
        + [name for name in raw["drug_aliases"] if name not in names]
        + [v for v in raw["drug_name_redirects"].values() if v not in names]
    )
    if invalid_targets:
        raise ValueError(f"Found unresolved canonical target: {invalid_targets[:10]}")

    stored_records = {r["id"] for r in iter_jsonl(out_dir/"source_records.jsonl")}
    source_rows, section_rows, chunk_rows = [], [], []
    sha = result["input_sha256"]

    for canonical, records in sorted(raw["additional_source_records"].items()):
        if canonical not in selected:
            continue
        if not isinstance(records, list):
            raise ValueError(f"additional_source_records[{canonical}] must be a list")
        drug_id = entity_id("Drug", canonical)
        for record in records:
            if not isinstance(record,dict):
                raise ValueError(f"additional record for {canonical} must be object")
            source_url = text_of(record.get("source"))
            rec_hash = hash_record(record)
            record_id = entity_id("SourceRecord",drug_id,source_url,rec_hash)
            if record_id in stored_records:
                continue
            stored_records.add(record_id)
            source_rows.append({
                "id":record_id, "drug_id":drug_id,
                "source_url":source_url, "record_sha256":rec_hash,
                "snapshot_sha256":sha,
                "dataset":DATASET,
                "version_status":"unknown",
                "authority_status":"third_party_unverified",
                "license_status":"not_assessed_for_redistribution",
                "source_variant":"additional_merged_source",
            })
            for field,title in TEXT_FIELDS.items():
                text = text_of(record.get(field))
                if not text:
                    continue
                section_id = entity_id("Section",record_id,field)
                section_rows.append({
                    "id":section_id,"record_id":record_id,
                    "field":field,"title":title,"text":text,
                    "placeholder":text.strip("。．；,，; ") in NO_INFORMATION
                })
                for start,end,chunk in chunk_sections(text,chunk_size,overlap):
                    chunk_rows.append({
                        "id":entity_id("Chunk",section_id,start,end),
                        "section_id":section_id,"field":field,
                        "start":start,"end":end,"text":chunk,
                    })

    # Extend the *existing* graph JSONL inputs so Neo4j importer will create
    # these supplementary SourceRecord -> Section -> Chunk paths.
    counts = {}
    for kind,rows in (("source_records",source_rows),
                      ("sections",section_rows),("chunks",chunk_rows)):
        counts[kind] = write_rows(out_dir,kind,rows,mode="a")
        result["counts"][kind] = result["counts"].get(kind,0)+counts[kind]

    # Canonical aliases are *unverified name mappings*, not clinical claims
    # and not necessarily approved interchangeable drug products.
    alias_rows=[]
    alias_names=set()
    alias_conflicts=[]
    for canonical,vals in sorted(raw["drug_aliases"].items()):
        if canonical not in selected:
            continue
        if not isinstance(vals,list):
            raise ValueError(f"drug_aliases[{canonical}] must be a list")
        for alias in sorted(set(vals)):
            if not isinstance(alias,str) or not alias.strip():
                continue
            if alias in alias_names:
                alias_conflicts.append(alias)
                continue
            alias_names.add(alias)
            alias_rows.append({
                "id":entity_id("DrugAlias",canonical,alias),
                "name":alias,"drug_id":entity_id("Drug",canonical),
                "canonical_name":canonical,
                "dataset":DATASET,"review_status":"candidate",
                "snapshot_sha256":sha,
            })
    if alias_conflicts:
        raise ValueError(f"Ambiguous aliases mapped to multiple entries: {alias_conflicts[:10]}")
    write_rows(out_dir,"drug_aliases",alias_rows)

    # Keep potentially malformed redirected names for audit ONLY.
    write_rows(out_dir,"name_redirects_audit",(
        {"old_name":old,"canonical_name":target,
         "canonical_id":entity_id("Drug",target),
         "graph_imported":False}
        for old,target in sorted(raw["drug_name_redirects"].items())
        if target in selected
    ))
    write_rows(out_dir,"name_provenance_audit",_original_metadata(raw,selected))

    result["cleaned_snapshot"] = {
        "name_cleaning_version":raw.get("name_cleaning_version"),
        "additional_source_records":counts["source_records"],
        "additional_sections":counts["sections"],
        "additional_chunks":counts["chunks"],
        "candidate_alias_nodes":len(alias_rows),
        "redirects_audit_count":sum(1 for x in raw["drug_name_redirects"].values() if x in selected),
        "provenance_audit_count":sum(1 for x in raw["drug_original_names"] if x in selected),
    }
    (out_dir/"build_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),encoding="utf-8")
    return result

def iter_jsonl(path):
    with Path(path).open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    parser=argparse.ArgumentParser(
        description="Build the v2-cleaned drug graph including aliases and extra sources"
    )
    parser.add_argument("--input",required=True)
    parser.add_argument("--out",default="data/kg_output_clean_v2")
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    result=prepare_cleaned(args.input,args.out,limit=args.limit,seed=args.seed)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
