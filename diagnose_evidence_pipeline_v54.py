# -*- coding: utf-8 -*-
"""Privacy-safe GraphRAG v5.4 audit/retry pipeline diagnostics.

Usage: python diagnose_evidence_pipeline_v54.py --trace medical_trace_017.json
This script never prints user questions, prompt text, raw document excerpts,
API credentials or model reasoning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(raw):
    trace = raw.get("trace", raw) if isinstance(raw, dict) else {}
    graph = trace.get("semantic_graph") or {}
    flow = trace.get("evidence_flow") or {}
    coverage = trace.get("evidence_coverage") or {}
    retry = trace.get("self_rag_retry") or {}
    stages = []
    for stage in flow.get("stages") or ():
        if not isinstance(stage, dict):
            continue
        rows = stage.get("evidence") or ()
        supported = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            for binding in row.get("bindings") or ():
                if isinstance(binding, dict):
                    supported.add(binding.get("target_id"))
        stages.append({"name": stage.get("name"),
                       "items": len(rows),
                       "target_ids_with_candidates": sorted(x for x in supported if x),
                       "uncovered_target_ids": stage.get("uncovered_targets") or []})
    return {
        "graph_backend": graph.get("storage"),
        "graph_status": graph.get("status"),
        "verified_graph_paths": len(graph.get("paths") or []),
        "graph_relation_types_multiple": graph.get("multi_relation_paths"),
        "answer_route": graph.get("answer_route"),
        "targets": [{"id": t.get("id"), "field": t.get("field"),
                     "interaction_scope": t.get("scope")}
                    for t in flow.get("targets") or [] if isinstance(t, dict)],
        "stage_counts": stages,
        "role_transport": trace.get("evidence_role_transport") or {},
        "retry_integrity": trace.get("retry_evidence_integrity") or {},
        "targeted_retry_routing": trace.get("targeted_retry_routing") or {},
        "retry_status": retry.get("status"),
        "new_chunks": retry.get("new_chunks"),
        "evidence_decision": coverage.get("decision"),
        "accepted_evidence_count": len(coverage.get("evidence_ids") or []),
        "aspect_statuses": [a.get("status") for a in coverage.get("aspects") or []
                            if isinstance(a, dict)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    args = ap.parse_args()
    report = summarize(json.loads(args.trace.read_text(encoding="utf-8")))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
