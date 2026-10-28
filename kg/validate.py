"""Validate JSONL identity, foreign keys, provenance hashes and exact offsets."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .build import entity_id, file_sha256, read_input
from .clean_names import clean_dataset, fingerprint, name_errors
from .neo4j_import import iter_jsonl, QUERIES
import hashlib


def validate(directory, source_path=None):
    directory = Path(directory)
    tables, errors = {}, []
    for kind in QUERIES:
        rows = list(iter_jsonl(directory / f"{kind}.jsonl"))
        by_id = {}
        for row in rows:
            if row.get("id") in by_id or not row.get("id"):
                errors.append({"kind": kind, "error": "duplicate_or_empty_id", "id": row.get("id")})
            by_id[row.get("id")] = row
        tables[kind] = by_id

    def fail(kind, row, reason):
        errors.append({"kind": kind, "id": row.get("id"), "error": reason})

    def reference(kind, field, target, optional=False):
        for row in tables[kind].values():
            value = row.get(field)
            if optional and not value:
                continue
            if value not in tables[target]:
                fail(kind, row, "missing_" + field)

    for name, field, target, optional in [
        ("source_records", "drug_id", "drugs", False),
        ("sections", "record_id", "source_records", False),
        ("chunks", "section_id", "sections", False),
        ("interaction_candidates", "source_mention_id", "mentions", False),
        ("interaction_candidates", "target_mention_id", "mentions", False),
        ("interaction_candidates", "source_drug_id", "drugs", True),
        ("interaction_candidates", "target_drug_id", "drugs", True),
        ("interaction_candidates", "evidence_section_id", "sections", True),
        ("indication_claims", "drug_id", "drugs", False),
        ("indication_claims", "disease_id", "disease_terms", False),
        ("indication_claims", "section_id", "sections", False),
    ]:
        reference(name, field, target, optional)
    names = set()
    for row in tables["drugs"].values():
        issues = name_errors(row["name"])
        if issues:
            fail("drugs", row, ",".join(issues))
        if row["name"] in names:
            fail("drugs", row, "duplicate_name")
        names.add(row["name"])
        if row["id"] != entity_id("Drug", row["name"]):
            fail("drugs", row, "name_id_mismatch")
    for row in tables["chunks"].values():
        section = tables["sections"].get(row["section_id"])
        if section and not (0 <= row["start"] < row["end"] <= len(section["text"])
                            and row["text"] == section["text"][row["start"]:row["end"]]
                            and row["field"] == section["field"]):
            fail("chunks", row, "invalid_text_offsets_or_field")
    for row in tables["interaction_candidates"].values():
        for role in ("source", "target"):
            mention = tables["mentions"].get(row[role + "_mention_id"])
            if mention and mention["name"] != row[role + "_name"]:
                fail("interaction_candidates", row, "mention_name_mismatch")
            drug = tables["drugs"].get(row[role + "_drug_id"])
            if drug and drug["name"] != row[role + "_name"]:
                fail("interaction_candidates", row, "drug_name_mismatch")
        section = tables["sections"].get(row["evidence_section_id"])
        if section:
            start, end = row["evidence_char_start"], row["evidence_char_end"]
            record = tables["source_records"].get(section["record_id"], {})
            if not (0 <= start < end <= len(section["text"])
                    and section["text"][start:end] == row["note"]
                    and section["field"] == row["evidence_field"]
                    and record.get("drug_id") == row["source_drug_id"]
                    and row["evidence_status"] == "exact_quote_found"):
                fail("interaction_candidates", row, "invalid_evidence")
        elif not (row["evidence_status"] == "unverified"
                  and row["evidence_char_start"] == row["evidence_char_end"] == -1
                  and row["evidence_field"] == ""):
            fail("interaction_candidates", row, "invalid_missing_evidence_status")
    report = json.loads((directory / "build_report.json").read_text(encoding="utf-8"))
    for kind, count in report["counts"].items():
        if count != len(tables[kind]):
            errors.append({"error": "build_count_mismatch", "kind": kind})
    if source_path:
        raw, _ = clean_dataset(read_input(source_path))
        sha = file_sha256(source_path)
        if sha != report["input_sha256"]:
            errors.append({"error": "build_snapshot_mismatch"})
        record_keys = set()
        for name, record in raw["drugs"].items():
            for value in [record] + raw.get("additional_source_records", {}).get(name, []):
                content_hash = hashlib.sha256(fingerprint(value).encode("utf-8")).hexdigest()
                record_keys.add((entity_id("Drug", name), str(value.get("source", "")).strip(), content_hash))
        for row in tables["source_records"].values():
            if row["snapshot_sha256"] != sha:
                fail("source_records", row, "snapshot_mismatch")
            if (row["drug_id"], row["source_url"], row["record_sha256"]) not in record_keys:
                fail("source_records", row, "record_hash_mismatch")
    return {"counts": {k: len(v) for k, v in tables.items()},
            "error_count": len(errors), "errors": errors,
            "mention_status_counts": dict(Counter(r["semantic_status"] for r in tables["mentions"].values()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--source")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = validate(args.in_dir, args.source)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, ensure_ascii=False, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
