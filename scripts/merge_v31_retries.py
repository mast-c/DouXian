#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely consolidate V3.1 retry outputs into a new 60-row CSV.

Only *validated* retry rows may replace previously failed rows.
All unchanged and failed records remain in the original order. 
No Neo4j writes; no external dependencies or network calls.

PowerShell:
python scripts/merge_v31_retries.py --base kg_output_clean_v2/pilot_v31_checkpoint60.csv --retry kg_output_clean_v2/pilot_v31_timeout29_retry.csv --retry kg_output_clean_v2/pilot_v31_format2_retry.csv --out kg_output_clean_v2/pilot_v31_merged60.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_run(path: Path):
    report_path = path.with_suffix(".v31_report.json")
    if not path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            f"Need both {path} and {report_path}. A partial or missing run cannot be merged."
        )
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("complete") is not True:
        raise ValueError(f"Incomplete run: {path}")
    actual_sha = sha256(path)
    if actual_sha != report.get("output_sha256"):
        raise ValueError(f"CSV SHA-256 does not match report: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        names = list(reader.fieldnames or [])
        records = list(reader)
    if not records or "candidate_id" not in names:
        raise ValueError(f"Empty CSV or missing candidate_id: {path}")
    if len(records) != report.get("selected_rows"):
        raise ValueError(f"CSV/report row-count mismatch: {path}")
    ids = [r["candidate_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate candidate IDs: {path}")
    return names, records, report


def verify_same_experiment(baseline: dict, retry: dict, path: Path):
    for name in (
        "version", "mode", "model", "input_sha256",
        "source_graph_sha256", "prompt_sha256"
    ):
        if baseline.get(name) != retry.get(name):
            raise ValueError(
                f"Source/model/prompt differs in {path}: {name}. "
                "Do not silently merge results from different prompt versions."
            )


def count_status(records):
    counts = Counter()
    for r in records:
        counts["rows"] += 1
        counts["decision_" + r["v31_candidate_decision"]] += 1
        counts["text_" + r["v31_text_relation_status"]] += 1
        status = r["v31_llm_status"]
        if status == "validated_model_output":
            if r["v31_evidence_validated"].lower() != "true":
                raise ValueError(
                    f"Inconsistent evidence validation for {r['candidate_id']}"
                )
            counts["llm_valid"] += 1
        elif status == "failed_needs_manual_review":
            counts["llm_errors"] += 1
        else:
            raise ValueError(
                f"Unexpected row status '{status}' for {r['candidate_id']}"
            )
        if json.loads(r.get("v31_override_reasons_json") or "[]"):
            counts["overridden"] += 1
    return counts


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--retry", required=True, action="append")
    p.add_argument("--out", required=True)
    args=p.parse_args()

    base=Path(args.base)
    retries=[Path(x) for x in args.retry]
    out=Path(args.out)
    source_paths=[base,*retries]
    if len(set(x.resolve() for x in source_paths)) != len(source_paths):
        raise ValueError("Base/retry input paths must be distinct")
    if out.resolve() in {x.resolve() for x in source_paths}:
        raise ValueError("Output must be a new file, never overwrite any input")
    if out.exists() or out.with_suffix(".v31_report.json").exists():
        raise FileExistsError("Output or its report already exists; choose a new name")

    fields, base_rows, base_report=read_run(base)
    base_by_id={r["candidate_id"]:r for r in base_rows}
    original_fields=[n for n in fields if not n.startswith("v31_")]
    merged_by_id=dict(base_by_id)
    seen_ids=set()
    replaced=[]
    failed_retry=[]
    runs=[(base,base_report)]
    for path in retries:
        other_fields, other_rows, other_report=read_run(path)
        verify_same_experiment(base_report,other_report,path)
        if fields!=other_fields:
            raise ValueError(f"CSV column schema differs: {path}")
        runs.append((path,other_report))
        for r in other_rows:
            cid=r["candidate_id"]
            if cid not in base_by_id:
                raise ValueError(f"Retry candidate not in original: {cid}")
            if cid in seen_ids:
                raise ValueError(f"Candidate appears in multiple retry files: {cid}")
            seen_ids.add(cid)
            original=base_by_id[cid]
            if original["v31_llm_status"]!="failed_needs_manual_review":
                raise ValueError(
                    f"Retry target is already validated in original: {cid}"
                )
            mismatch=[
                k for k in original_fields if r.get(k)!=original.get(k)
            ]
            if mismatch:
                raise ValueError(
                    f"Retry source text or metadata changed for {cid}: {mismatch[:5]}"
                )
            if r["v31_llm_status"]=="validated_model_output":
                if (r["v31_evidence_validated"].lower()!="true" or
                    r["v31_validation_error"].strip()):
                    raise ValueError(
                        f"Retry says 'validated' but its evidence isn't verified: {cid}"
                    )
                merged_by_id[cid]=r
                replaced.append({"candidate_id":cid,"source":str(path)})
            elif r["v31_llm_status"]=="failed_needs_manual_review":
                failed_retry.append({
                    "candidate_id":cid,
                    "last_retry_error":r.get("v31_validation_error",""),
                    "source":str(path)
                })
            else:
                raise ValueError(f"Invalid retry status on {cid}")
    merged=[merged_by_id[r["candidate_id"]] for r in base_rows]
    counts=count_status(merged)
    recorded_requests=sum(
        int(rep.get("counts",{}).get("api_requests",0))
        for _,rep in runs
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    temp=out.with_name(out.name+".tmp")
    try:
        with temp.open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields)
            w.writeheader()
            w.writerows(merged)
            f.flush()
            os.fsync(f.fileno())
        output_sha=sha256(temp)
        combined_report={
            "version":base_report["version"],
            "mode":base_report["mode"],
            "model":base_report["model"],
            "complete":True,
            "selected_rows":len(merged),
            "completed_rows":len(merged),
            "remaining_rows":0,
            "input_sha256":base_report["input_sha256"],
            "source_graph_sha256":base_report.get("source_graph_sha256",""),
            "prompt_sha256":base_report["prompt_sha256"],
            "output_sha256":output_sha,
            "counts":dict(counts),
            "validated_retry_replacements":len(replaced),
            "retry_candidates_still_failed":failed_retry,
            "original_validated_rows_preserved":sum(
                r["v31_llm_status"]=="validated_model_output"
                for r in base_rows
            ),
            "source_runs":[{
                "csv":str(path),
                "csv_sha256":sha256(path),
                "report_sha256":sha256(path.with_suffix(".v31_report.json")),
                "recorded_api_requests":int(
                    rep.get("counts",{}).get("api_requests",0)
                )
            } for path,rep in runs],
            "recorded_api_requests_across_all_runs":recorded_requests,
            "replaced_candidates":replaced,
            "manual_review_status":"all_proposals_pending_human_review",
            "graph_writes":0,
            "warning":"A merged development set is not a prospective independent test. "
                      "Original labels are AI-assisted, not clinical expert gold. "
                      "Retry failures are still uncertain."
        }
        report_path=out.with_suffix(".v31_report.json")
        rtemp=report_path.with_name(report_path.name+".tmp")
        with rtemp.open("w",encoding="utf-8") as f:
            json.dump(combined_report,f,ensure_ascii=False,indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp,out)
        os.replace(rtemp,report_path)
    except BaseException:
        if temp.exists():
            temp.unlink()
        raise
    print(json.dumps({
        "output":str(out),
        "report":str(report_path),
        "rows":len(merged),
        "original_valid_preserved":combined_report["original_validated_rows_preserved"],
        "validated_retry_replacements":len(replaced),
        "still_failed":counts["llm_errors"],
        "accepted":counts["decision_accept"],
        "uncertain":counts["decision_uncertain"],
        "recorded_api_requests_across_runs":recorded_requests
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
