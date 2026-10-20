#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Directly A/B test V1 vs V2 guard prompts on the SAME retrieved evidence.

Requires your existing working LangChain/Model Studio environment.
Example:
    python guard_ab_probe.py rag_baseline_results.json --runs 3

This script DOES incur real model API usage. It does not retrieve from Chroma.
"""
import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path


def summarize_decision(report):
    checks = report.get("validation_diagnostics") or []
    bad = [
        {"question":item.get("question",""),
         "model_status":item.get("model_status"),
         "final_status":item.get("final_status"),
         "failed_quotes":item.get("invalid_citations", [])}
        for item in checks if item.get("invalid_citations")
    ]
    return {
        "decision":report.get("decision"),
        "accepted_evidence_ids":report.get("evidence_ids", []),
        "supported_count":sum(
            a.get("status") in ("supported","refuted")
            for a in report.get("aspects",[])
        ),
        "insufficient_count":sum(
            a.get("status")=="insufficient"
            for a in report.get("aspects",[])
        ),
        "invalid_quotes":bad,
        "message": report.get("message",""),
    }


def main():
    parser=argparse.ArgumentParser(
        description="使用相同原文评估 V1 与 V2 审核质量和延迟"
    )
    parser.add_argument("baseline", nargs="?",default="rag_baseline_results.json")
    parser.add_argument("--sample",type=int,default=2,
                        help="选取哪次样本的原文证据（默认第2次，五条）")
    parser.add_argument("--runs",type=int,default=3,
                        help="每个 prompt 重复的次数（每次是实际模型请求）")
    parser.add_argument("--output",default="guard_ab_results.json")
    args=parser.parse_args()
    if args.runs<1:
        parser.error("--runs 必须大于零")
    data=json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    records=data.get("samples") or []
    if not 1<=args.sample<=len(records):
        parser.error("--sample 超过基线样本数")
    baseline=records[args.sample-1]
    excerpts=[
        {
            "id":index+1,
            "source":item.get("source",""),
            "section_path":item.get("section_path",""),
            "content":item.get("content",""),
        }
        for index,item in enumerate(
            baseline.get("trace",{}).get("knowledge_evidence",[])
        ) if isinstance(item,dict)
    ]
    if not excerpts:
        parser.error("选中样本没有原文证据")
    queries=[]
    for detail in baseline.get("trace",{}).get("timing_details",[]):
        if detail.get("stage")=="kb.hybrid_search":
            q=(detail.get("details") or {}).get("query")
            if isinstance(q,str) and q not in queries:
                queries.append(q)

    # Do not create or change project model config; reuse the existing client.
    from rag import RagService
    from evidence_guard import assess_evidence
    model=RagService().chat_model
    results=[]
    current=os.environ.get("RAG_GUARD_PROMPT_VERSION")
    try:
        for iteration in range(args.runs):
            for version in ("v1","v2"):
                os.environ["RAG_GUARD_PROMPT_VERSION"]=version
                t0=time.perf_counter()
                try:
                    report=assess_evidence(
                        model=model, question=data["question"],
                        excerpts=excerpts,retrieval_queries=queries
                    )
                    error=None
                except Exception as exc:
                    report={"decision":"error","message":str(exc)}
                    error=f"{type(exc).__name__}: {exc}"
                elapsed=round(time.perf_counter()-t0,4)
                result={
                    "prompt":version,"iteration":iteration+1,
                    "seconds":elapsed,"error":error,
                    **summarize_decision(report)
                }
                results.append(result)
                print(
                    f"{version} round {iteration+1}: "
                    f"{elapsed:.2f}s, "
                    f"decision={result['decision']}, "
                    f"supported={result['supported_count']}, "
                    f"invalid_quote_aspects={len(result['invalid_quotes'])}"
                )
    finally:
        if current is None:
            os.environ.pop("RAG_GUARD_PROMPT_VERSION",None)
        else:
            os.environ["RAG_GUARD_PROMPT_VERSION"]=current

    output={
        "question":data.get("question",""),
        "sample":args.sample,
        "excerpts_count":len(excerpts),
        "run_time":datetime.now().astimezone().isoformat(),
        "options": {
            "guard_model":os.getenv("RAG_GUARD_MODEL") or "inherit",
            "disable_thinking":os.getenv("RAG_GUARD_DISABLE_THINKING","0"),
        },
        "results":results
    }
    for version in ("v1","v2"):
        values=[r["seconds"] for r in results
                if r["prompt"]==version and r["error"] is None]
        if values:
            output[f"{version}_average_seconds"]=round(statistics.mean(values),3)
    Path(args.output).write_text(
        json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print("已保存 A/B 对照结果:",Path(args.output).resolve())


if __name__=="__main__":
    main()
