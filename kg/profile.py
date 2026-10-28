"""Full source dataset profiling; no original medication texts are exported."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .build import TEXT_FIELDS, NO_INFORMATION, file_sha256, read_input, text_of

APPROVAL_RE = re.compile(r"国药准字\s*[A-Za-z][0-9]{8}")

def profile(path):
    raw = read_input(path)
    drugs = raw["drugs"]
    interaction_rows = raw["interactions"]
    fields_nonempty, placeholders = Counter(), Counter()
    approval_valid = 0
    alerts = Counter()

    for name, value in drugs.items():
        if not isinstance(value, dict):
            alerts["non_dict_drug_records"] += 1
            continue
        for field in TEXT_FIELDS:
            text = text_of(value.get(field))
            if text:
                fields_nonempty[field] += 1
                if text.strip("。．；,，; ") in NO_INFORMATION:
                    placeholders[field] += 1
        raw_approvals = value.get("approval_numbers") or []
        if isinstance(raw_approvals, list) and any(
            APPROVAL_RE.search(str(item)) for item in raw_approvals
        ):
            approval_valid += 1
        for item in value.get("contraindications", []):
            if any(k in str(item) for k in ("慎用", "谨慎", "不宜")):
                alerts["contraindication_contains_caution_phrase"] += 1
                break

    source_urls = [v.get("source") for v in drugs.values() if isinstance(v, dict)]
    risk = Counter()
    resolution = Counter()
    evidence = Counter()
    invalid = 0
    pairs = set()
    for row in interaction_rows:
        pair = row.get("pair") if isinstance(row, dict) else None
        if not (isinstance(pair, list) and len(pair) == 2):
            invalid += 1
            continue
        a, b = pair
        if not isinstance(a, str) or not isinstance(b, str):
            invalid += 1
            continue
        pairs.add((a, b))
        risk[row.get("risk", "missing")] += 1
        resolution[f"source_{'known' if a in drugs else 'unknown'}__target_{'known' if b in drugs else 'unknown'}"] += 1
        note = text_of(row.get("note"))
        if a in drugs and note:
            src = drugs[a]
            if note in text_of(src.get("drug_interactions")):
                evidence["exact_note_in_interaction_text"] += 1
            elif note in text_of(src.get("warnings")):
                evidence["exact_note_in_warnings_text"] += 1
            else:
                evidence["source_note_not_exactly_found"] += 1
        else:
            evidence["source_unresolved_or_note_empty"] += 1

    return {
        "profile_generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": Path(path).name,
        "input_sha256": file_sha256(path),
        "drug_records": len(drugs),
        "interaction_candidates": len(interaction_rows),
        "field_nonempty_counts": dict(fields_nonempty),
        "field_nonempty_percent": {
            k: round(v * 100 / max(1, len(drugs)), 2)
            for k, v in fields_nonempty.items()
        },
        "placeholder_text_counts": dict(placeholders),
        "unique_source_urls": len(set(source_urls)),
        "records_with_approval_code_pattern": approval_valid,
        "contraindication_caution_record_count": alerts["contraindication_contains_caution_phrase"],
        "risk_labels": dict(risk),
        "endpoint_exact_name_resolution": dict(resolution),
        "interaction_quote_evidence": dict(evidence),
        "invalid_interaction_records": invalid,
        "duplicate_interaction_pairs": len(interaction_rows) - len(pairs) - invalid,
        "caveats": [
            "此报告仅统计结构和字符串匹配，未审核临床正确性",
            "risk 是数据源标签，不是权威临床风险等级",
            "nonempty 字段可能包含 尚不明确、慎用等非严格事实",
            "数据是一份按药名汇总的第三方记录，不能代表已批准的单一产品说明书版本",
        ],
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="data/kg_output/profile.json")
    a = p.parse_args()
    result = profile(a.input)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
