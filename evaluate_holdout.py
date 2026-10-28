#!/usr/bin/env python3
"""Compute stratified candidate-level precision from a *fully annotated* holdout.

Workflow: mark review_status=approved/rejected (or uncertain) in the
independent_evaluation_300.csv; don't tune your model on this split.
Only reports precision if every row has a definitive review label.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import math

def evaluate(csv_path, manifest_path):
    manifest=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    population=manifest["source_strata"]
    expected=manifest["holdout_counts"]
    groups=defaultdict(list)
    with Path(csv_path).open("r",encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        required={"candidate_id","context_flag","review_status","relation_type","reviewer"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("缺少必须列："+str(required-set(reader.fieldnames or [])))
        for row in reader:
            groups[row["context_flag"]].append(row)

    if set(groups)!=set(population):
        raise ValueError("样本层不一致，检查是否误删/合并 CSV 行")
    if any(len(groups[k])!=expected[k] for k in population):
        raise ValueError("抽样层的行数不一致；不能直接使用旧权重")
    ids=[row["candidate_id"] for g in groups.values() for row in g]
    if len(ids)!=len(set(ids)):
        raise ValueError("candidate_id 重复")

    summary={}
    unfinished=0
    for flag,rows in groups.items():
        statuses=Counter(row["review_status"].strip().lower() for row in rows)
        reviewed=[r for r in rows if r["review_status"].strip().lower() in ("approved","rejected")]
        approved=[r for r in rows if r["review_status"].strip().lower()=="approved"]
        missing_details=sum(
            not r["reviewer"].strip() or
            r["relation_type"].strip().lower() not in
                {"treatment","prevention","symptom_relief"}
            for r in approved
        )
        unfinished+=len(rows)-len(reviewed)+missing_details
        summary[flag]={
            "population":population[flag],
            "sampled":len(rows),
            "statuses":dict(statuses),
            "approved":len(approved),
            "missing_review_details_in_approvals":missing_details,
        }

    result={
        "evaluation_unit":"candidate row (text-supported positive relation)",
        "per_stratum":summary,
        "completed":unfinished==0,
        "remaining_decisions_or_approval_details":unfinished,
        "warning":"This measures candidate precision, not extraction recall or clinical validity."
    }
    if unfinished:
        result["next_step"]="Complete manual review before computing a precision estimate."
        return result

    total=sum(population.values())
    weighted=sum(
        (population[k]/total) *
        (summary[k]["approved"]/summary[k]["sampled"])
        for k in population
    )
    # Approximate variance of stratified SRS with finite population correction,
    # not accounting for reviewer agreement or correlated sources.
    variance=0
    for k in population:
        Nh=population[k]
        nh=summary[k]["sampled"]
        ph=summary[k]["approved"]/nh
        variance+=(
            (Nh/total)**2 * (1-nh/Nh) *
            ph*(1-ph)/(nh-1)
        )
    se=math.sqrt(max(variance,0))
    result.update({
        "weighted_candidate_precision":round(weighted,4),
        "approximate_95pct_interval":[
            round(max(0,weighted-1.96*se),4),
            round(min(1,weighted+1.96*se),4)
        ],
        "important":"Weighted for oversampled negation cases; not an F1 score."
    })
    return result

def main():
    parser=argparse.ArgumentParser(description="分层抽样候选关系精准率评估")
    parser.add_argument("--csv",default="independent_evaluation_300.csv")
    parser.add_argument("--manifest",default="sampling_manifest.json")
    args=parser.parse_args()
    print(json.dumps(evaluate(args.csv,args.manifest),
                     ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
