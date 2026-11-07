# -*- coding: utf-8 -*-
"""Inspect path disappearance across graph proof, merge, review and UI trace.

Usage:
    python diagnose_graph_pipeline_v53.py --trace medical_trace_082.json

Only prints counts/status fields. Do not upload full patient traces publicly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical_graph_visibility import graph_visibility_report


def inspect_trace(obj: dict) -> dict:
    trace = obj.get("trace") if isinstance(obj.get("trace"), dict) else obj
    graph = trace.get("semantic_graph") or {}
    evidence = trace.get("knowledge_evidence") or []
    verified = [item for item in evidence if isinstance(item, dict)
                and item.get("graph_verified")
                and (item.get("graph_path") or {}).get("verified_against_original")]
    inventory = graph_visibility_report(graph, verified)
    flow = trace.get("evidence_flow") or {}
    stages = []
    for stage in (flow.get("stages") or []):
        rows = stage.get("evidence") or []
        stages.append({
            "name": stage.get("name"),
            "evidence_count": len(rows),
            "bound_target_count": len({binding.get("target_id")
                for row in rows for binding in (row.get("bindings") or [])
                if binding.get("target_id")}),
            "uncovered_targets": len(stage.get("uncovered_targets") or []),
        })
    return {
        "graph_status": graph.get("status"),
        "graph_backend": graph.get("storage"),
        "graph_reason_type": (
            "source_verification_blocked" if graph.get("status") == "source_verification_blocked"
            else "normal_or_nonintegrity_status"
        ),
        "paths": inventory["retrieved_verified_paths"],
        "paths_present_in_knowledge_evidence": len({item["graph_path"]["id"]
            for item in verified if (item.get("graph_path") or {}).get("id")}),
        "neo4j_raw_candidates": inventory["candidate_counts"]["raw_neo4j_candidates"],
        "proof_rejections": inventory["candidate_counts"]["source_proof_rejections"],
        "task_mismatch_rejections": inventory["candidate_counts"]["semantic_task_mismatch_rejections"],
        "candidate_truncated": inventory["truncated"],
        "stages": stages,
        "ui_widget_needed": "Cannot verify actual Streamlit rendering from a trace alone",
        "scope": "data-flow diagnostics only; not clinical evidence validation",
    }


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--trace",type=Path,required=True)
    args=parser.parse_args()
    with args.trace.open(encoding="utf-8") as f:
        obj=json.load(f)
    print(json.dumps(inspect_trace(obj), ensure_ascii=False,indent=2,default=str))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
