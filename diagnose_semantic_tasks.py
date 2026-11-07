# -*- coding: utf-8 -*-
"""Diagnose per-subquestion GraphRAG coverage, without printing API keys.

Example from RAG_project/data:
python diagnose_semantic_tasks.py --env-file .env --root . --question "布地奈德鼻喷雾剂会导致荨麻疹吗？它是什么类型的疾病？"

This script calls the configured LLM and reads the local source graph.
It DOES NOT retrieve disease classification documents on behalf of 豆馅.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description="GraphRAG semantic-task planner diagnostics")
    ap.add_argument("--root",type=Path,default=Path("."))
    ap.add_argument("--env-file",type=Path,default=Path(".env"))
    ap.add_argument("--question",default=(
        "布地奈德鼻喷雾剂会导致荨麻疹吗？它是什么类型的疾病？"
    ))
    ap.add_argument("--show-queries",action="store_true",
                    help="Include query text; avoid for real patient data")
    args=ap.parse_args()
    os.environ["MEDICAL_QUERY_ENV_FILE"]=str(args.env_file.resolve())
    from medical_semantic_retrieval import search_semantic_graph

    evidence, report = search_semantic_graph(
        args.question, project_root=args.root.resolve()
    )
    llm=report.get("query_understanding_llm") or {}
    summary={
        "status":report.get("status"),
        "reason":report.get("reason"),
        "mode":report.get("query_understanding_mode"),
        "model_source":report.get("query_understanding_model_source"),
        "graph_backend":report.get("storage"),
        "task_refinement":report.get("semantic_task_refinement"),
        "llm_status":llm.get("status"),
        "llm_error_category":llm.get("error_category"),
        "verified_graph_paths":len(evidence),
        "direct_relation_paths":report.get("direct_relation_paths"),
        "multi_hop_paths":report.get("multi_hop_paths"),
        "multi_relation_paths":report.get("multi_relation_paths"),
        "all_subtasks_answered":report.get("all_subtasks_answered"),
        "subtasks":[
            {
                "id":task.get("id"),
                "kind":task.get("kind"),
                "status":task.get("status"),
                "requested_entities":task.get("requested_entities"),
                "verified_graph_paths":task.get("verified_graph_paths"),
                "verified_direct_paths":task.get("verified_direct_paths"),
                "requires_external_text_search":task.get("text_retrieval_required"),
                "text_evidence_status":task.get("text_evidence_status"),
                **({"query":task.get("query")} if args.show_queries else {}),
            }
            for task in report.get("task_coverage") or []
        ],
        "text_retrieval_requests":[
            {"task_id":req.get("task_id"),"kind":req.get("kind"),
             "suggested_source":req.get("suggested_source"),
             "executed":req.get("executed"),
             **({"query":req.get("query")} if args.show_queries else {})}
            for req in report.get("text_retrieval_requests") or []
        ],
    }
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
    if summary["mode"]!="llm_first":
        print("WARNING: LLM query understanding was not confirmed on this request.")
        return 2
    if not summary["subtasks"]:
        print("NOTE: the question may have a single aspect; standard graph routing is still used.")
    if any(r["kind"]=="disease_classification"
           for r in summary["text_retrieval_requests"]):
        print("NOTE: disease taxonomy awaits a separate source-text retrieval step.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
