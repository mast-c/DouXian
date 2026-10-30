"""Convert MedGraphRAG-Drug-DB drug_db_v2.json into provenance-aware graph JSONL.

No source text is silently interpreted as a verified medical fact.
CLI: python -m kg.build --input PATH --out data/kg_output --limit 300
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re

TEXT_FIELDS = {
    "indication": "适应症",
    "contraindications": "禁忌",
    "warnings": "注意事项",
    "composition": "成分",
    "dosage": "用法用量",
    "adverse_reactions": "不良反应",
    "drug_interactions": "药物相互作用",
    "storage": "贮藏",
    "validity": "有效期",
    "approval_numbers": "批准文号原始记录",
    "manufacturers": "生产企业原始记录",
}
NO_INFORMATION = {"尚不明确", "尚不清楚", "不详", "暂无", "无资料", "未见报道", "未提供"}

def digest(*parts):
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

def entity_id(kind, *parts):
    return kind.lower() + "_" + digest(kind, *parts)

def text_of(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(text_of(x) for x in value if text_of(x))
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

def read_input(path):
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict) or not isinstance(raw.get("drugs"), dict):
        raise ValueError("预期顶层为 {'drugs': {name: fields}, 'interactions': [...]}")
    if not isinstance(raw.get("interactions", []), list):
        raise ValueError("interactions 必须为 list")
    return raw

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def chunk_sections(text, chunk_size=800, overlap=90):
    """Preserve offsets within the normalized Section.text field."""
    if chunk_size < 80 or not (0 <= overlap < chunk_size // 2):
        raise ValueError("chunk_size 至少 80；overlap 需小于 chunk_size/2")
    start = 0
    separators = ("。", "；", "！", "？", "\n")
    while start < len(text):
        stop = min(start + chunk_size, len(text))
        if stop < len(text):
            min_stop = start + int(chunk_size * 0.6)
            best = max((text.rfind(x, min_stop, stop) for x in separators), default=-1)
            if best >= min_stop:
                stop = best + 1
        yield start, stop, text[start:stop]
        if stop == len(text):
            break
        start = max(start + 1, stop - overlap)

class JsonlWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.kinds = ("drugs", "source_records", "sections", "chunks",
                      "mentions", "interaction_candidates")
        # Truncate all outputs at the start, avoiding stale rows on reruns.
        self.handles = {
            kind: (self.directory / f"{kind}.jsonl").open("w", encoding="utf-8")
            for kind in self.kinds
        }
        self.counts = Counter()

    def add(self, kind, obj):
        self.handles[kind].write(
            json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self.counts[kind] += 1

    def close(self):
        for fh in self.handles.values():
            fh.close()

def select_names(drugs, limit, seed):
    names = sorted(drugs)
    if limit < 0:
        raise ValueError("--limit 不得为负数，0 表示全部")
    if limit and limit < len(names):
        return sorted(random.Random(seed).sample(names, limit))
    return names

def prepare(input_path, out_dir, limit=300, seed=42,
            chunk_size=800, overlap=90):
    input_path, out_dir = Path(input_path), Path(out_dir)
    source = read_input(input_path)
    all_drugs = source["drugs"]
    chosen = select_names(all_drugs, limit, seed)
    chosen_set = set(chosen)
    sha = file_sha256(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Rebuilding base records invalidates any prior reviewed claims.
    for stale_name in ("disease_terms.jsonl", "indication_claims.jsonl",
                       "review_report.json"):
        (out_dir / stale_name).unlink(missing_ok=True)
    writer = JsonlWriter(out_dir)
    ids_by_name = {}
    sec_by_name_field = {}
    kept = Counter()
    invalid = []

    try:
        for name in chosen:
            record = all_drugs[name]
            if not isinstance(record, dict):
                invalid.append(name)
                continue

            source_url = text_of(record.get("source"))
            drug_id = entity_id("Drug", name)
            content_hash = hashlib.sha256(
                json.dumps(record, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            record_id = entity_id("SourceRecord", drug_id, source_url, content_hash)
            ids_by_name[name] = drug_id
            writer.add("drugs", {
                "id": drug_id, "name": name,
                "entity_scope": "dataset_drug_name_not_approved_product",
            })
            writer.add("source_records", {
                "id": record_id, "drug_id": drug_id,
                "source_url": source_url, "record_sha256": content_hash,
                "snapshot_sha256": sha,
                "dataset": "MedGraphRAG-Drug-DB drug_db_v2.json",
                "version_status": "unknown",
                "authority_status": "third_party_unverified",
                "license_status": "not_assessed_for_redistribution",
            })
            for field, title in TEXT_FIELDS.items():
                text = text_of(record.get(field))
                if not text:
                    continue
                sec_id = entity_id("Section", record_id, field)
                sec_by_name_field[(name, field)] = (sec_id, text)
                is_placeholder = text.strip("。．；,，; ") in NO_INFORMATION
                writer.add("sections", {
                    "id": sec_id, "record_id": record_id,
                    "field": field, "title": title, "text": text,
                    "placeholder": is_placeholder,
                })
                kept[f"section_{field}"] += 1
                for start, end, chunk in chunk_sections(text, chunk_size, overlap):
                    writer.add("chunks", {
                        "id": entity_id("Chunk", sec_id, start, end),
                        "section_id": sec_id, "field": field,
                        "start": start, "end": end, "text": chunk,
                    })

        mentions_written = set()
        for entry in source["interactions"]:
            if not isinstance(entry, dict):
                kept["invalid_interaction"] += 1
                continue
            pair = entry.get("pair")
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or not all(isinstance(x, str) and x.strip() for x in pair)):
                kept["invalid_interaction"] += 1
                continue
            src_name, target_name = (x.strip() for x in pair)
            if limit != 0 and src_name not in chosen_set and target_name not in chosen_set:
                continue
            note, risk = text_of(entry.get("note")), text_of(entry.get("risk"))
            src_id = ids_by_name.get(src_name, "")
            target_id = ids_by_name.get(target_name, "")
            evidence_id, start, end = "", -1, -1
            evidence_field = ""
            for field in ("drug_interactions", "warnings"):
                section = sec_by_name_field.get((src_name, field))
                if section and note and note in section[1]:
                    evidence_id = section[0]
                    start = section[1].find(note)
                    end = start + len(note)
                    evidence_field = field
                    break
            for mention in (src_name, target_name):
                mention_id = entity_id("Mention", mention)
                if mention_id not in mentions_written:
                    writer.add("mentions", {
                        "id": mention_id, "name": mention,
                        "semantic_status": "unresolved_name_or_category",
                    })
                    mentions_written.add(mention_id)
            writer.add("interaction_candidates", {
                "id": entity_id("InteractionCandidate", src_name, target_name, note),
                "source_name": src_name, "target_name": target_name,
                "source_mention_id": entity_id("Mention", src_name),
                "target_mention_id": entity_id("Mention", target_name),
                "source_drug_id": src_id, "target_drug_id": target_id,
                "risk_label_in_source": risk or "unknown",
                "note": note,
                "evidence_section_id": evidence_id,
                "evidence_field": evidence_field,
                "evidence_char_start": start, "evidence_char_end": end,
                "evidence_status": "exact_quote_found" if evidence_id else "unverified",
                "clinical_status": "not_validated",
            })
            if evidence_id:
                kept["interaction_exact_evidence"] += 1
            else:
                kept["interaction_without_exact_evidence"] += 1
            if src_id and target_id:
                kept["both_drug_names_resolved"] += 1
    finally:
        writer.close()

    report = {
        "schema_version": "0.1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_name": input_path.name,
        "input_sha256": sha,
        "source_drug_count": len(all_drugs),
        "source_interaction_count": len(source["interactions"]),
        "sample_limit": limit,
        "sample_seed": seed,
        "counts": dict(writer.counts),
        "field_counts": {k: v for k, v in kept.items() if k.startswith("section_")},
        "interaction_metrics": {
            k: v for k, v in kept.items() if not k.startswith("section_")
        },
        "invalid_drug_name_samples": invalid[:20],
        "notes": [
            "Drug 节点是数据源的名称级实体，不等于真实注册药品产品",
            "InteractionCandidate 是未经医学验证的候选相互作用",
            "不会自动从适应症文本推断药品-疾病获批关系",
            "保留原文字段和字符偏移以便后续人工校验与向量检索",
        ],
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report

def main():
    p = argparse.ArgumentParser(description="药品数据 -> 可溯源 Neo4j 图节点/关系 JSONL")
    p.add_argument("--input", required=True, help="drug_db_v2.json 本地路径")
    p.add_argument("--out", default="data/kg_output")
    p.add_argument("--limit", type=int, default=300, help="抽样药品数；0 = 全量")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=90)
    a = p.parse_args()
    report = prepare(a.input, a.out, a.limit, a.seed, a.chunk_size, a.overlap)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
