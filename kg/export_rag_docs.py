"""Export graph Chunks as standalone RAG document JSONL, without touching ChromaDB.

Each row: {"id":..., "text":..., "metadata":{...}}.
"""
import argparse
import json
from pathlib import Path

from .neo4j_import import iter_jsonl

def export_rag(in_dir, output, fields=None, skip_placeholders=True):
    in_dir, output = Path(in_dir), Path(output)
    drugs = {d["id"]: d["name"] for d in iter_jsonl(in_dir / "drugs.jsonl")}
    records = {
        r["id"]: r
        for r in iter_jsonl(in_dir / "source_records.jsonl")
    }
    sections = {
        s["id"]: s
        for s in iter_jsonl(in_dir / "sections.jsonl")
        if (fields is None or s["field"] in fields)
        and (not skip_placeholders or not s["placeholder"])
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for chunk in iter_jsonl(in_dir / "chunks.jsonl"):
            section = sections.get(chunk["section_id"])
            if section is None:
                continue
            record = records[section["record_id"]]
            row = {
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": {
                    "drug_name": drugs[record["drug_id"]],
                    "field": chunk["field"],
                    "section_id": section["id"],
                    "source_record_id": record["id"],
                    "source_url": record["source_url"],
                    "char_start": chunk["start"],
                    "char_end": chunk["end"],
                    "version_status": record["version_status"],
                }
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"exported_rag_documents": count,
                      "output_file": str(output)}, ensure_ascii=False, indent=2))
    return count

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="data/kg_output")
    p.add_argument("--out", default="data/kg_output/rag_documents.jsonl")
    p.add_argument("--fields", default="indication,contraindications,warnings,"
                   "composition,adverse_reactions,drug_interactions")
    a = p.parse_args()
    fields = {f.strip() for f in a.fields.split(",")} if a.fields else None
    export_rag(a.in_dir, a.out, fields=fields)

if __name__ == "__main__":
    main()
