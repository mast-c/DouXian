"""Import ONLY annotated disease mentions that have exact support in indication text.

Important: text-checked is not approval by a regulator or clinical validation.
Annotation format:
{"drug_name": "...", "disease_name": "...",
 "evidence_quote": "...", "review_status":"human_checked"}
"""
import argparse
import json
from pathlib import Path
from .build import entity_id
from .neo4j_import import iter_jsonl

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def review(in_dir, annotations):
    in_dir = Path(in_dir)
    drugs = {d["name"]: d for d in iter_jsonl(in_dir / "drugs.jsonl")}
    record_to_drug = {
        r["id"]: r["drug_id"]
        for r in iter_jsonl(in_dir / "source_records.jsonl")
    }
    sections = {}
    for s in iter_jsonl(in_dir / "sections.jsonl"):
        if s["field"] == "indication":
            sections.setdefault(record_to_drug[s["record_id"]], []).append(s)

    accepted, failed = {}, []
    diseases = {}
    for lineno, item in enumerate(iter_jsonl(annotations), 1):
        if item.get("review_status") != "human_checked":
            continue
        name = (item.get("drug_name") or "").strip()
        disease = (item.get("disease_name") or "").strip()
        quote = (item.get("evidence_quote") or "").strip()
        drug = drugs.get(name)
        if not drug or not disease or not quote or disease not in quote:
            failed.append({"line": lineno, "reason": "药品缺失、疾病/原文为空，或疾病名不在引文中"})
            continue
        found = [s for s in sections.get(drug["id"], []) if quote in s["text"]]
        if len(found) != 1:
            failed.append({"line": lineno, "reason": "无法唯一匹配适应症原文"})
            continue
        section = found[0]
        offset = section["text"].find(quote)
        disease_id = entity_id("DiseaseTerm", disease)
        diseases[disease_id] = {
            "id": disease_id, "name": disease,
            "normalization_status": "not_mapped_to_standard_terminology"
        }
        claim_id = entity_id("IndicationClaim", section["id"], disease, quote)
        accepted[claim_id] = {
            "id": claim_id, "drug_id": drug["id"], "disease_id": disease_id,
            "section_id": section["id"], "predicate": "INDICATION_TEXT_MENTIONS",
            "evidence_quote": quote,
            "char_start": offset, "char_end": offset + len(quote),
            "review_status": "human_checked_text_match",
            "clinical_status": "not_validated",
        }
    write_jsonl(in_dir / "disease_terms.jsonl", diseases.values())
    write_jsonl(in_dir / "indication_claims.jsonl", accepted.values())
    result = {"disease_term_count": len(diseases),
              "evidence_claim_count": len(accepted),
              "rejected": failed,
              "note": "药品说明书中出现疾病不等于批准适应症已获临床验证"}
    (in_dir / "review_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result

def main():
    p = argparse.ArgumentParser(description="Artificially reviewed indication evidence")
    p.add_argument("--in-dir", default="data/kg_output")
    p.add_argument("--annotations", required=True)
    a = p.parse_args()
    print(json.dumps(review(a.in_dir, a.annotations),
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
