# -*- coding: utf-8 -*-
"""Read-only verified-path inventory independent from Evidence Guard selection.

This module ONLY reports source-verified paths; it never creates medical
assertions, suppresses reviews, or changes final generation evidence.
"""
from __future__ import annotations


def _count_paths(paths):
    unique = {p["id"]: p for p in paths if isinstance(p, dict) and p.get("id")}
    return {
        "total": len(unique),
        "direct": sum(p.get("hops") == 1 for p in unique.values()),
        "multi_hop": sum(int(p.get("hops") or 0) > 1 for p in unique.values()),
        "multi_relation": sum(len(set(p.get("relations") or ())) > 1
                              for p in unique.values()),
        "single_relation_multi_hop": sum(int(p.get("hops") or 0) > 1
                                         and len(set(p.get("relations") or ())) == 1
                                         for p in unique.values()),
    }


def graph_visibility_report(report: dict, verified_excerpts: list[dict],
                            generation_excerpts: list[dict] | None = None) -> dict:
    """Explain WHERE verified graph paths went (including after evidence cap).

    ``report['paths']`` is the frontend visualization source of truth. These
    are structural source-verified mentions, NOT clinical causality proof.
    """
    graph_paths = {
        item["graph_path"]["id"]: item["graph_path"]
        for item in (verified_excerpts or [])
        if (isinstance(item, dict) and item.get("graph_verified")
            and (item.get("graph_path") or {}).get("verified_against_original")
            and (item.get("graph_path") or {}).get("id"))
    }
    report_paths = {
        path["id"]: path for path in report.get("paths") or []
        if isinstance(path, dict) and path.get("id")
        and path.get("verified_against_original")
    }
    all_valid = {**report_paths, **graph_paths}
    included = set()
    if generation_excerpts is not None:
        included = {
            item["graph_path"]["id"]
            for item in generation_excerpts
            if (isinstance(item, dict) and item.get("graph_verified")
                and (item.get("graph_path") or {}).get("verified_against_original")
                and (item.get("graph_path") or {}).get("id"))
        }
    other = sorted(set(all_valid) - included) if generation_excerpts is not None else []
    return {
        "schema_version": 1,
        "graph_status": report.get("status"),
        "graph_storage": report.get("storage"),
        "visualization_source": "semantic_graph.paths",
        "source_verified_only": True,
        "medical_causality_established": False,
        "retrieved_verified_paths": _count_paths(list(all_valid.values())),
        "accepted_into_generation": (
            _count_paths([all_valid[k] for k in included if k in all_valid])
            if generation_excerpts is not None else None
        ),
        "not_selected_for_generation": len(other),
        "path_ids_not_selected_for_generation": other[:24],
        "graph_paths_visible_in_trace": len(report_paths),
        "candidate_counts": {
            "raw_neo4j_candidates": (report.get("stats") or {}).get("raw_candidate_rows"),
            "source_proof_rejections": (
                (report.get("stats") or {}).get("rejected_invalid_local_edge", 0)
                + (report.get("stats") or {}).get("rejected_malformed_candidate", 0)
            ),
            "semantic_task_mismatch_rejections": (report.get("stats") or {}).get(
                "rejected_irrelevant_task_connection", 0
            ),
        },
        "truncated": bool(report.get("truncated")),
        "display_note": (
            "Verified source mentions may be visualized, including direct edges; "
            "only separately reviewed evidence may support medical answers."
        ),
    }
