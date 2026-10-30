#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DrugKG V3.1 research-stage freeze, blind review and holdout evaluation.

Only Python stdlib. Never calls an LLM and never writes to Neo4j.
DO NOT use the AI-assisted pilot tags as clinical gold.
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

SCHEMA_VERSION = "drugkg_v31_study_gate_1"
CLASSIFICATIONS = {
    "SUPPORTED_UNQUALIFIED",
    "SUPPORTED_QUALIFIED",
    "NOT_SUPPORTED",
    "NON_TARGET_CONTEXT",
    "INSUFFICIENT_EVIDENCE",
}
RELATIONS = {"treatment", "prevention", "symptom_relief"}
TYPES = {
    "Disease", "Symptom", "ClinicalFinding",
    "PatientCondition", "AdverseEvent", "Other"
}
POLARITIES = {"positive", "negative", "conditional", "uncertain"}
REQUIRED_SOURCE_FIELDS = [
    "candidate_id", "drug_name", "disease_name", "source_record_id",
    "section_id", "source_url", "evidence_quote",
    "quote_start", "quote_end", "disease_start", "disease_end",
    "context_flag"
]
INFERENCE_FIELDS = REQUIRED_SOURCE_FIELDS + [
    "review_status", "relation_type", "reviewer", "review_note"
]
GOLD_FIELDS = [
    "candidate_id","drug_name","disease_name",
    "evidence_quote","source_url",
    "gold_classification","gold_exact_target","gold_target_type",
    "gold_target_role","gold_relation_type","gold_polarity",
    "gold_qualifiers_json","source_identity_verified",
    "regulatory_source_url","regulatory_label_version",
    "annotator_name","annotator_role","adjudication_notes"
]
REVIEW_COMPARISON = [
    "gold_classification","gold_exact_target","gold_target_type",
    "gold_target_role","gold_relation_type","gold_polarity",
    "gold_qualifiers_json","source_identity_verified",
]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def load_csv(path):
    path=Path(path)
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        cols=list(reader.fieldnames or [])
        rows=list(reader)
    if not cols or not rows:
        raise ValueError(f"Empty CSV or header: {path}")
    if "candidate_id" not in cols:
        raise ValueError(f"Missing candidate_id: {path}")
    ids=[r["candidate_id"] for r in rows]
    if any(not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError(f"Missing or duplicate candidate_id in {path}")
    return cols,rows

def write_csv_new(path,fields,rows):
    path=Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def write_json_new(path,obj):
    path=Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    try:
        with tmp.open("w",encoding="utf-8") as f:
            json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()

def as_json(text, *, context=""):
    try:
        return json.loads(text)
    except (json.JSONDecodeError,TypeError) as e:
        raise ValueError(f"Invalid JSON field {context}: {e}") from e

def make_review_forms(args):
    _, source=load_csv(args.priority)
    if len(source)!=21:
        raise ValueError(f"Expected exactly 21 priority records, found {len(source)}")
    output=Path(args.out_dir)
    for who in ("A","B"):
        newrows=[]
        for r in source:
            row={
                "candidate_id":r["candidate_id"],
                "drug_name":r["drug_name"],
                "disease_name":r["disease_name"],
                "evidence_quote":r["evidence_quote"],
                "source_url":r["source_url"],
            }
            row.update({k:"" for k in GOLD_FIELDS if k not in row})
            newrows.append(row)
        p=output/f"reviewer_{who}_21_blind.csv"
        write_csv_new(p,GOLD_FIELDS,newrows)
        print(f"[盲审] {p} ({len(newrows)} 条)")
    print("表格不含 Qwen 预测与旧 AI 审核标签；源出处也尚未验证。")

def validate_gold(r):
    cid=r["candidate_id"]
    value=(r.get("gold_classification") or "").strip()
    if not value:
        return ["未填写 gold_classification"]
    errors=[]
    if value not in CLASSIFICATIONS:
        errors.append(f"unknown gold_classification: {value}")
        return errors
    for field in ("annotator_name","annotator_role"):
        if not (r.get(field) or "").strip():
            errors.append(f"missing {field}")
    if (r.get("source_identity_verified") or "").lower() not in {"yes","no","unknown"}:
        errors.append("source_identity_verified must be yes/no/unknown")
    if (r.get("source_identity_verified") or "").lower()=="yes":
        if not (r.get("regulatory_source_url") or "").strip():
            errors.append("verified source requires regulatory_source_url")
        if not (r.get("regulatory_label_version") or "").strip():
            errors.append("verified source requires regulatory_label_version")
    if value.startswith("SUPPORTED_"):
        exact=(r.get("gold_exact_target") or "").strip()
        if not exact:
            errors.append("supported relationship needs exact target")
        elif exact not in (r.get("evidence_quote") or ""):
            errors.append("gold_exact_target must occur verbatim in evidence_quote")
        if r.get("gold_target_type") not in TYPES:
            errors.append("supported relationship needs target type")
        if r.get("gold_relation_type") not in RELATIONS:
            errors.append("supported relationship needs relation type")
        if r.get("gold_polarity") not in POLARITIES:
            errors.append("supported relationship needs polarity")
    quals=(r.get("gold_qualifiers_json") or "").strip()
    if value=="SUPPORTED_QUALIFIED" and not quals:
        errors.append("qualified support requires nonempty qualifiers JSON")
    if quals:
        try:
            obj=as_json(quals,context=f"{cid}.gold_qualifiers_json")
            if not isinstance(obj,list):
                errors.append("gold_qualifiers_json must be a JSON array")
            elif value=="SUPPORTED_QUALIFIED" and not obj:
                errors.append("qualified support requires at least one qualifier")
        except ValueError as e:
            errors.append(str(e))
    return errors

def review_status(args):
    _,a=load_csv(args.reviewer_a)
    _,b=load_csv(args.reviewer_b)
    ai={r["candidate_id"]:r for r in a}
    bi={r["candidate_id"]:r for r in b}
    if set(ai)!=set(bi) or len(a)!=21 or len(b)!=21:
        raise ValueError("Expected identical sets of 21 priority IDs in two files")
    all_errors=[]
    disagreement=[]
    agreed=0
    for cid in ai:
        first,second=ai[cid],bi[cid]
        errors_a=validate_gold(first)
        errors_b=validate_gold(second)
        if errors_a or errors_b:
            all_errors.append({
                "candidate_id":cid,
                "reviewer_A": "; ".join(errors_a),
                "reviewer_B": "; ".join(errors_b)
            })
        elif any(first.get(k,"")!=second.get(k,"") for k in REVIEW_COMPARISON):
            disagreement.append({
                "candidate_id":cid, "drug_name":first["drug_name"],
                "disease_name":first["disease_name"],
                "reviewer_A_classification":first["gold_classification"],
                "reviewer_B_classification":second["gold_classification"],
                "fields_in_disagreement":"|".join(
                    k for k in REVIEW_COMPARISON if first.get(k,"")!=second.get(k,"")
                ),
            })
        else:
            agreed+=1
    state={
        "total":21,
        "complete_in_both_forms":21-len(all_errors),
        "agreed_on_core_fields":agreed,
        "disagreements":len(disagreement),
        "incomplete_or_invalid":len(all_errors),
        "ready_for_adjudication":len(all_errors)==0,
        "not_medically_validated_by_this_tool":True,
        "missing_or_invalid_first_12":all_errors[:12]
    }
    if args.disagreements and disagreement:
        write_csv_new(args.disagreements,list(disagreement[0]),disagreement)
    print(json.dumps(state,ensure_ascii=False,indent=2))
    return state

def freeze(args):
    code=Path(args.code)
    prompt=Path(args.prompt)
    dev=Path(args.dev_csv)
    dev_report=Path(args.dev_report)
    for f in (code,prompt,dev,dev_report):
        if not f.is_file():
            raise FileNotFoundError(f"Freeze input missing: {f}")
    dr=as_json(dev_report.read_text(encoding="utf-8-sig"),context="dev_report")
    _, devrows=load_csv(dev)
    if (len(devrows)!=60 or dr.get("complete") is not True or
        dr.get("selected_rows")!=60 or dr.get("completed_rows")!=60):
        raise ValueError("Development run must be a completed, 60-row baseline")
    if sha(dev)!=dr.get("output_sha256"):
        raise ValueError("Developer CSV does not match report output_sha256")
    if sha(prompt)!=dr.get("prompt_sha256"):
        raise ValueError("Prompt file differs from merged development report")
    if args.model!=dr.get("model"):
        raise ValueError("Requested model differs from development report")
    version_line=f'VERSION = "{dr.get("version", "")}"'
    if version_line not in code.read_text(encoding="utf-8"):
        raise ValueError(
            "Source script VERSION does not match merged development report. "
            "A new algorithm version requires a separate documented baseline."
        )
    manifest={
        "schema_version":SCHEMA_VERSION,
        "status":"engineering_baseline_only",
        "clinical_release_ready":False,
        "never_claims_clinical_approval":True,
        "model":args.model,
        "endpoint":args.endpoint,
        "timeout":args.timeout,
        "max_retries":args.max_retries,
        "code_path":str(code),
        "code_sha256":sha(code),
        "prompt_path":str(prompt),
        "prompt_sha256":sha(prompt),
        "dev_csv_path":str(dev),
        "dev_csv_sha256":sha(dev),
        "dev_report_path":str(dev_report),
        "dev_report_sha256":sha(dev_report),
        "dev_candidate_ids_sha256":hashlib.sha256(
            "\n".join(sorted(r["candidate_id"] for r in devrows)).encode()
        ).hexdigest(),
        "source_graph_sha256":dr.get("source_graph_sha256",""),
        "dev_input_sha256":dr.get("input_sha256",""),
        "dev_eval_size":60,
        "dev_output_valid":dr.get("counts",{}).get("llm_valid",None),
        "metadata":{
            "reference_labels":"AI-assisted, NOT independently expert verified",
            "holdout_accessed_during_freeze":False,
            "frozen_before_holdout":True,
            "no_ne04j_writes":True,
            "memo":"Freeze does not validate drug indications or authorization."
        }
    }
    write_json_new(args.out,manifest)
    print(json.dumps({
        "frozen_manifest":str(args.out),
        "code_sha256":manifest["code_sha256"],
        "prompt_sha256":manifest["prompt_sha256"],
        "status":manifest["status"],
        "holdout_evaluated":False
    },ensure_ascii=False,indent=2))
    return manifest

def verify_freeze(path):
    manifest=as_json(Path(path).read_text(encoding="utf-8-sig"),
                     context="freeze_manifest")
    if manifest.get("schema_version")!=SCHEMA_VERSION:
        raise ValueError("Freeze manifest version mismatch")
    for name in ("code","prompt","dev_csv","dev_report"):
        p=Path(manifest[f"{name}_path"])
        if not p.is_file():
            raise FileNotFoundError(f"Frozen file missing: {p}")
        if sha(p)!=manifest[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} file has changed: {p}")
    return manifest

def prepare_holdout(args):
    freeze_meta=verify_freeze(args.freeze)
    _,dev=load_csv(freeze_meta["dev_csv_path"])
    cols,raw=load_csv(args.input)
    missing=[x for x in REQUIRED_SOURCE_FIELDS if x not in cols]
    if missing:
        raise ValueError(
            "Holdout candidates need Stage3 source evidence fields: "
            + ", ".join(missing)
        )
    if len(raw)!=args.expect:
        raise ValueError(f"Expected {args.expect} holdout candidates, got {len(raw)}")
    dev_ids={r["candidate_id"] for r in dev}
    raw_ids={r["candidate_id"] for r in raw}
    overlap=dev_ids & raw_ids
    if overlap:
        raise ValueError(
            f"Holdout overlaps development set by {len(overlap)} candidate IDs; "
            "do not silently remove rows or contaminate held-out evaluation"
        )
    inf=[]
    blind=[]
    for i,r in enumerate(raw,1):
        try:
            qs,qe,ds,de=(int(r[k]) for k in (
                "quote_start","quote_end","disease_start","disease_end"
            ))
        except (TypeError,ValueError) as exc:
            raise ValueError(f"Invalid offsets at holdout row {i}") from exc
        quote=r["evidence_quote"]
        if not (0<=qs<=ds<de<=qe and qe-qs==len(quote)):
            raise ValueError(f"Evidence positions invalid at holdout row {i}")
        if quote[ds-qs:de-qs]!=r["disease_name"]:
            raise ValueError(f"Disease mention does not match at holdout row {i}")
        source={k:r[k] for k in REQUIRED_SOURCE_FIELDS}
        source.update({
            "review_status":"unlabeled", "relation_type":"",
            "reviewer":"", "review_note":""
        })
        inf.append(source)
        g={k:source.get(k,"") for k in GOLD_FIELDS}
        g.update({k:"" for k in GOLD_FIELDS if k not in g})
        blind.append(g)
    out=Path(args.out)
    if out.exists() or Path(args.annotation_out).exists():
        raise FileExistsError("Refusing to overwrite holdout inference/annotation files")
    write_csv_new(out,INFERENCE_FIELDS,inf)
    write_csv_new(args.annotation_out,GOLD_FIELDS,blind)
    state={
        "freeze_sha256":sha(args.freeze),
        "source_holdout_path":str(Path(args.input)),
        "source_holdout_sha256":sha(args.input),
        "sanitized_inference_csv":str(out),
        "sanitized_inference_sha256":sha(out),
        "blind_annotation_csv":str(Path(args.annotation_out)),
        "blind_annotation_sha256":sha(args.annotation_out),
        "candidate_count":len(inf),
        "dev_candidate_id_overlap":0,
        "source_record_overlap_count":len(
            {x["source_record_id"] for x in inf} &
            {x["source_record_id"] for x in dev}
        ),
        "section_overlap_count":len(
            {x["section_id"] for x in inf} &
            {x["section_id"] for x in dev}
        ),
        "source_quote_overlap_count":len(
            {x["evidence_quote"] for x in inf} &
            {x["evidence_quote"] for x in dev}
        ),
        "independence_scope":"candidate_id_only; source/document overlap may remain",
        "model":freeze_meta["model"],
        "timeout":freeze_meta["timeout"],
        "max_retries":freeze_meta["max_retries"],
        "label_columns_in_inference_csv":False
    }
    write_json_new(out.with_suffix(".holdout_preflight.json"),state)
    print(json.dumps(state,ensure_ascii=False,indent=2))
    return state

def score_holdout(args):
    frozen=verify_freeze(args.freeze)
    prep=as_json(Path(args.preflight).read_text(encoding="utf-8-sig"),
                 context="holdout_preflight")
    if prep["freeze_sha256"]!=sha(args.freeze):
        raise ValueError("This holdout belongs to a different freeze manifest")
    infer=Path(prep["sanitized_inference_csv"])
    if sha(infer)!=prep["sanitized_inference_sha256"]:
        raise ValueError("Frozen holdout inference CSV changed")
    pr=as_json(Path(args.pred_report).read_text(encoding="utf-8-sig"),
               context="prediction_report")
    if not pr.get("complete") or pr.get("selected_rows")!=prep["candidate_count"]:
        raise ValueError("Holdout prediction is incomplete")
    if sha(args.pred)!=pr.get("output_sha256"):
        raise ValueError("Holdout predictions do not match run report")
    for k,v in (
        ("model",frozen["model"]),
        ("prompt_sha256",frozen["prompt_sha256"]),
        ("input_sha256",prep["sanitized_inference_sha256"])
    ):
        if pr.get(k)!=v:
            raise ValueError(f"Prediction report does not match freeze: {k}")
    pcols,pred=load_csv(args.pred)
    _,gold=load_csv(args.gold)
    gmap={r["candidate_id"]:r for r in gold}
    pm={r["candidate_id"]:r for r in pred}
    if len(gold)!=len(pred) or set(gmap)!=set(pm):
        raise ValueError("Gold/predicted candidates differ or incomplete")
    errs=[]
    counts=Counter()
    per_class=Counter()
    issues=[]
    for cid, goldrow in gmap.items():
        invalid=validate_gold(goldrow)
        if invalid:
            errs.append(f"{cid}: {'; '.join(invalid)}")
            continue
        r=pm[cid]
        cls=goldrow["gold_classification"]
        decision=r.get("v31_candidate_decision","")
        counts["total"]+=1
        counts["pred_"+decision]+=1
        per_class[cls]+=1
        if r.get("v31_llm_status")=="failed_needs_manual_review":
            counts["model_failed"]+=1
        if decision=="accept":
            counts["accepted"]+=1
            label_matches=(
                cls=="SUPPORTED_UNQUALIFIED" and
                goldrow.get("gold_relation_type")==
                    r.get("v31_proposed_relation") and
                goldrow.get("gold_exact_target")==r.get("v31_specific_target_surface")
            )
            if label_matches:
                counts["accepted_matches_gold_unqualified"]+=1
            else:
                issues.append({
                    "candidate_id":cid,
                    "gold_classification":cls,
                    "gold_relation":goldrow.get("gold_relation_type",""),
                    "model_relation":r.get("v31_proposed_relation",""),
                    "reason":"Accepted but does not match unqualified expert gold"
                })
            if cls in {"NOT_SUPPORTED","NON_TARGET_CONTEXT","SUPPORTED_QUALIFIED"}:
                counts["unsafe_or_unqualified_accept"]+=1
            if goldrow["source_identity_verified"].lower()!="yes":
                counts["accepted_without_verified_source_identity"]+=1
        elif decision=="uncertain":
            counts["abstained"]+=1
        elif decision=="reject":
            counts["rejected"]+=1
        else:
            errs.append(f"{cid}: invalid predicted decision {decision}")
    if errs:
        raise ValueError(
            f"Invalid or incomplete adjudication, first errors: {errs[:8]}"
        )
    n=counts["accepted"]
    result={
        "assessment":"Independent holdout study after engineering freeze; "
                     "not evidence of regulatory approval or clinical safety",
        "freeze_sha256":sha(args.freeze),
        "prediction_csv_sha256":sha(args.pred),
        "gold_csv_sha256":sha(args.gold),
        "holdout_size":len(pred),
        "source_record_overlap_count":prep.get("source_record_overlap_count",None),
        "section_overlap_count":prep.get("section_overlap_count",None),
        "source_quote_overlap_count":prep.get("source_quote_overlap_count",None),
        "independence_scope":prep.get("independence_scope","unknown"),
        "counts":dict(counts),
        "gold_distribution":dict(per_class),
        "accepted_unqualified_gold_match":{
            "numerator":counts["accepted_matches_gold_unqualified"],
            "denominator":n,
            "ratio":counts["accepted_matches_gold_unqualified"]/n if n else None
        },
        "unsafe_or_unqualified_accept_count":counts["unsafe_or_unqualified_accept"],
        "source_identity_not_verified_among_accepts":counts[
            "accepted_without_verified_source_identity"
        ],
        "manual_review_flag":"Gold requires independent human adjudication; "
                             "this script cannot authenticate reviewers",
        "issues_first_100":issues[:100],
        "note":"No inference was performed in this score command."
    }
    write_json_new(args.out,result)
    print(json.dumps({
        k:result[k] for k in (
            "holdout_size","counts","accepted_unqualified_gold_match",
            "unsafe_or_unqualified_accept_count",
            "source_identity_not_verified_among_accepts"
        )
    },ensure_ascii=False,indent=2))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("make-review-forms")
    p.add_argument("--priority",required=True)
    p.add_argument("--out-dir",required=True)
    p.set_defaults(func=make_review_forms)
    p=commands.add_parser("review-status")
    p.add_argument("--reviewer-a",required=True)
    p.add_argument("--reviewer-b",required=True)
    p.add_argument("--disagreements")
    p.set_defaults(func=review_status)
    p=commands.add_parser("freeze")
    p.add_argument("--code",required=True)
    p.add_argument("--prompt",required=True)
    p.add_argument("--dev-csv",required=True)
    p.add_argument("--dev-report",required=True)
    p.add_argument("--model",default="qwen3.7-flash")
    p.add_argument("--endpoint",default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    p.add_argument("--timeout",type=int,default=90)
    p.add_argument("--max-retries",type=int,default=0)
    p.add_argument("--out",required=True)
    p.set_defaults(func=freeze)
    p=commands.add_parser("verify-freeze")
    p.add_argument("--freeze",required=True)
    p.set_defaults(func=lambda a: print(json.dumps(
        {"freeze_valid":True,
         "status":verify_freeze(a.freeze)["status"]},
        ensure_ascii=False)))
    p=commands.add_parser("prepare-holdout")
    p.add_argument("--freeze",required=True)
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--annotation-out",required=True)
    p.add_argument("--expect",type=int,default=300)
    p.set_defaults(func=prepare_holdout)
    p=commands.add_parser("score-holdout")
    p.add_argument("--freeze",required=True)
    p.add_argument("--preflight",required=True)
    p.add_argument("--pred",required=True)
    p.add_argument("--pred-report",required=True)
    p.add_argument("--gold",required=True)
    p.add_argument("--out",required=True)
    p.set_defaults(func=score_holdout)
    args=parser.parse_args()
    try:
        args.func(args)
    except (OSError,ValueError,KeyError) as error:
        print(f"ERROR: {error}",file=sys.stderr)
        sys.exit(2)

if __name__=="__main__":
    main()
