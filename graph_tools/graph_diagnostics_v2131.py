"""V2.13.1: Pure read-only graph evidence diagnostics.

Exact ID overlap and content-equivalent overlap are deliberately disjoint.
This module never changes Document, Chroma, ranking, or answer selection.
"""
from collections import defaultdict

def doc_id(doc):
    metadata = getattr(doc, "metadata", None)
    return str(metadata.get("chunk_id") or "").strip() if isinstance(metadata, dict) else ""

def unique(values):
    return list(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))

def finish_run(run, reranked_docs, final_docs):
    """Update only a copied diagnostics dict, not retrieval results."""
    result = dict(run)
    candidates = set(unique(result.get("graph_candidate_ids") or []))
    injected = set(unique(result.get("injected_chunk_ids") or []))
    reranked_ids = {cid for doc in reranked_docs if (cid := doc_id(doc))}
    final_ids = {cid for doc in final_docs if (cid := doc_id(doc))}
    exact_final = sorted(candidates & final_ids)
    injected_final = sorted(injected & final_ids)
    result["graph_evidence_final_ids"] = exact_final
    result["graph_evidence_in_final"] = len(exact_final)
    result["graph_incremental_final_ids"] = injected_final
    result["graph_incremental_final"] = len(injected_final)
    result["survived_rerank_ids"] = sorted(injected & reranked_ids)
    result["survived_rerank"] = len(result["survived_rerank_ids"])
    result["survived_final"] = len(injected_final)  # legacy contract
    # Content equivalence must NOT be counted as exact ID evidence coverage.
    content_map = result.get("matched_by_content_map") or {}
    final_equiv = {}
    for graph_id, hybrid_ids in content_map.items():
        overlaps = sorted(set(hybrid_ids) & final_ids)
        if overlaps and graph_id not in final_ids:
            final_equiv[graph_id] = overlaps
    result["content_equivalent_final_map"] = final_equiv
    result["content_equivalent_in_final"] = len(final_equiv)
    return result

def aggregate(runs, final_docs):
    final_ids = {cid for doc in final_docs if (cid := doc_id(doc))}
    union_candidates = set()
    union_injected = set()
    union_reranked = set()
    union_exact_hybrid = set()
    union_content = set()
    for run in runs:
        union_candidates.update(unique(run.get("graph_candidate_ids") or []))
        union_injected.update(unique(run.get("injected_chunk_ids") or []))
        union_reranked.update(unique(run.get("survived_rerank_ids") or []))
        union_exact_hybrid.update(unique(run.get("already_in_hybrid_ids") or []))
        union_content.update(unique(run.get("matched_by_content_ids") or []))
    exact_final = sorted(union_candidates & final_ids)
    incremental_final = sorted(union_injected & final_ids)
    content_final_map = {}
    for run in runs:
        for cid, others in (run.get("content_equivalent_final_map") or {}).items():
            content_final_map.setdefault(cid, set()).update(others)
    return {
        "graph_candidate_ids": sorted(union_candidates),
        "already_in_hybrid_ids": sorted(union_exact_hybrid),
        "matched_by_content_ids": sorted(union_content),
        "graph_evidence_final_ids": exact_final,
        "graph_evidence_in_final": len(exact_final),
        "graph_incremental_final_ids": incremental_final,
        "graph_incremental_final": len(incremental_final),
        "content_equivalent_final_map": {k:sorted(v) for k,v in sorted(content_final_map.items())},
        "content_equivalent_in_final": len(content_final_map),
        "survived_rerank": len(union_reranked),
        "survived_final": len(incremental_final),
    }
