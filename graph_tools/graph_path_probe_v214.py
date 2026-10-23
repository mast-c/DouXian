# -*- coding: utf-8 -*-
"""V2.14 OFFLINE relation-path candidate probe for DouXian.

READ-ONLY by design:
- Loads promoted V2.12 graph/verified bindings/alias registry only.
- No Chroma calls, no reranker, no LLM, no edits to rag.py/knowledge_base.py.
- Exact candidate IDs and relation-edge provenance only.
- Offline *candidate novelty* is NOT evidence relevance or improved answers.

Run: python -m graph_tools.graph_path_probe_v214 --out graph_reports/probe.json
"""
from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

try:
    from .graph_paths import CONTENT_DIR, BINDINGS_DIR, REPORTS_DIR
except ImportError:
    from graph_paths import CONTENT_DIR, BINDINGS_DIR, REPORTS_DIR

SEMANTIC = frozenset({
    "IS_A", "SPECIAL_CASE_OF", "HAS_PART", "USES", "BASED_ON",
    "SUBFIELD_OF", "COMPUTES", "ALSO_CALLED",
})
TRAVERSABLE = SEMANTIC - {"ALSO_CALLED"}
MAX_QUERY_LENGTH = 500
FIXED_QUERIES = (
    "残差单元是什么？",
    "马尔可夫随机场是什么？",
    "吉布斯采样的理论依据是什么？",
    "卷积神经网络包含哪些层，这些层怎样进行特征计算？",
    "卷积神经网络由哪些层构成？",
    "汇聚层如何使用最大汇聚进行下采样？",
    "残差网络由哪些结构组成，又通过什么连接方式提高信息传播效率？",
)

class OfflineGraphError(ValueError):
    """Untrusted or mismatched graph files; abort without changing data."""

def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise OfflineGraphError(msg)

def _read_json(p: Path) -> dict:
    obj = json.loads(p.read_text(encoding="utf-8-sig"))
    _require(isinstance(obj, dict), "Invalid JSON object: " + str(p))
    return obj

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def _lexical(x: str) -> str:
    return re.sub(r"\s+", "", str(x or "")).casefold()

def _has_term(text: str, term: str) -> bool:
    """Match the V2.13 token boundaries exactly, including Chinese substrings."""
    if not isinstance(text, str) or not isinstance(term, str) or not term.strip():
        return False
    if any(ord(c) >= 128 for c in term):
        return term in text
    return re.search(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])",
                     text, re.IGNORECASE) is not None

@dataclass
class OfflineBundle:
    concepts: dict[str, dict]
    edges: list[dict]
    claim_by_edge: dict[str, dict]
    linked_ids: dict[str, str]
    canonical_by_node: dict[str, str]
    term_to_nodes: dict[str, set[str]]
    file_hashes: dict[str, str]

    def canonical(self, node_id: str) -> str:
        return self.canonical_by_node.get(node_id, node_id)

def load_offline_bundle(
    graph_path: Path | None = None,
    links_path: Path | None = None,
    aliases_path: Path | None = None
) -> OfflineBundle:
    """Validate cross-file integrity and exact verified Claim/Edge consistency.

    Does NOT independently verify source quote against original Markdown or
    live Chroma. Those inputs are intentionally not part of this offline probe.
    """
    graph_path = Path(graph_path or CONTENT_DIR / "knowledge_graph_v212.json")
    links_path = Path(links_path or BINDINGS_DIR / "graph_chunk_links_v212.json")
    aliases_path = Path(aliases_path or CONTENT_DIR / "graph_alias_registry_v212.json")
    for p in (graph_path, links_path, aliases_path):
        _require(p.is_file(), "Missing required formal graph artifact: " + str(p))
    hashes = {
        "graph": _sha(graph_path),
        "links": _sha(links_path),
        "aliases": _sha(aliases_path),
    }
    graph, links, aliases = map(
        _read_json, (graph_path, links_path, aliases_path)
    )
    _require(graph.get("metadata", {}).get("v212", {}).get("phase") == "promoted",
             "Graph is not promoted V2.12")
    _require(links.get("metadata", {}).get("graph_sha256") == hashes["graph"],
             "Links do not refer to this graph SHA256")
    _require(aliases.get("graph_sha256") == hashes["graph"],
             "Aliases do not refer to this graph SHA256")
    _require(aliases.get("links_sha256") == hashes["links"],
             "Aliases do not refer to this links SHA256")
    _require(links["metadata"].get("source_sha256") ==
             graph.get("metadata", {}).get("source_sha256"),
             "Graph/source SHA reference differs from links")
    _require(links["metadata"].get("semantic_claims") == 29,
             "Expected the frozen V2.12 29-claim baseline")

    concepts = {n["id"]: n for n in graph.get("nodes", [])
                if isinstance(n, dict) and n.get("type") == "Concept"}
    semantic = [e for e in graph.get("edges", [])
                if isinstance(e, dict) and e.get("relation") in SEMANTIC]
    by_id = {e["id"]: e for e in semantic}
    claims = links.get("claims", [])
    _require(isinstance(claims, list) and len(claims) == 29 and len(by_id) == 29,
             "Expected 29 semantic graph edges and 29 verified claims")
    claim_by_edge = {c["claim_id"]: c for c in claims}
    _require(len(claim_by_edge) == len(claims) and len(by_id) == len(semantic),
             "Duplicate graph claim or edge ID")
    _require(set(claim_by_edge) == set(by_id), "Graph and bindings have different edges")
    for eid, claim in claim_by_edge.items():
        edge = by_id[eid]
        _require(claim.get("status") == "verified" and
                 isinstance(claim.get("chunk_ids"), list) and
                 len(claim["chunk_ids"]) == 1 and
                 isinstance(claim["chunk_ids"][0], str) and
                 bool(claim["chunk_ids"][0].strip()),
                 "Invalid / ambiguous online evidence binding: " + eid)
        _require(edge.get("source") in concepts and edge.get("target") in concepts,
                 "Missing concept node: " + eid)
        _require((claim.get("subject_id"), claim.get("predicate"),
                  claim.get("object_id")) ==
                 (edge["source"], edge["relation"], edge["target"]),
                 "Edge direction/relation mismatch: " + eid)
        ev = edge.get("evidence") or {}
        _require((claim.get("source_file"), claim.get("section_path"),
                  claim.get("source_line"), claim.get("evidence_text")) ==
                 (ev.get("source_file"), ev.get("section_path"),
                  ev.get("line"), ev.get("text")),
                 "Evidence metadata differs between graph and binding: " + eid)
    canonical = {nid: nid for nid in concepts}
    used_groups = set()
    synonym_groups = aliases.get("verified_synonym_groups", [])
    _require(len(synonym_groups) == 8, "Unexpected verified synonym group count")
    for grp in synonym_groups:
        leader = grp.get("canonical_id")
        members = grp.get("member_ids", [])
        _require(leader in concepts and isinstance(members, list)
                 and leader in members and len(set(members)) == len(members),
                 "Invalid canonical synonym group")
        for nid in members:
            _require(nid in concepts and
                     (canonical[nid] == nid or canonical[nid] == leader),
                     "Conflicting/unknown synonym concept")
            canonical[nid] = leader
        for eid in grp.get("verified_claim_ids", []):
            _require(eid in by_id and by_id[eid]["relation"] == "ALSO_CALLED",
                     "Synonym group has no verified ALSO_CALLED claim")
            used_groups.add(eid)
    # Search-only hints remain term matching signals, NEVER proof of equivalence.
    term_to_nodes = defaultdict(set)
    ambiguous = {
        x.get("lexical_key") for x in aliases.get("ambiguous_terms", [])
        if isinstance(x, dict)
    }
    def remember(term: Any, node_id: str) -> None:
        if isinstance(term, str) and term.strip() and _lexical(term) not in ambiguous:
            term_to_nodes[term.strip()].add(node_id)
    for nid, node in concepts.items():
        remember(node.get("label"), nid)
        for hint in node.get("aliases_for_search", []):
            remember(hint, nid)
    for hint in aliases.get("search_only_hints", []):
        if isinstance(hint, dict) and hint.get("concept_id") in concepts:
            remember(hint.get("term"), hint["concept_id"])
    for grp in synonym_groups:
        for label in grp.get("verified_labels", []):
            for nid in grp["member_ids"]:
                remember(label, nid)
    return OfflineBundle(
        concepts=concepts, edges=semantic, claim_by_edge=claim_by_edge,
        linked_ids={eid: c["chunk_ids"][0] for eid, c in claim_by_edge.items()},
        canonical_by_node=canonical, term_to_nodes=dict(term_to_nodes),
        file_hashes=hashes,
    )

def detect_query_concepts(query: str, bundle: OfflineBundle) -> tuple[set[str], set[str], list[str]]:
    detected = set()
    for term, nodes in bundle.term_to_nodes.items():
        if _has_term(query, term):
            detected.update(nodes)
    roots = {bundle.canonical(n) for n in detected}
    labels = sorted({bundle.concepts[n]["label"] for n in detected})
    return detected, roots, labels

def v213_candidates(query: str, bundle: OfflineBundle,
                    limit: int = 2) -> tuple[list[str], dict]:
    """Exact V2.13 candidate scoring/selection copied into this offline reference.

    For live installations, compare against graph_retriever_v213.candidate_chunk_ids.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH:
        return [], {"matched_labels": [], "matched_edges": 0}
    detected, roots, labels = detect_query_concepts(query, bundle)
    if not detected:
        return [], {"matched_labels": [], "matched_edges": 0}
    scored = []
    for edge in bundle.edges:
        subject, target = edge["source"], edge["target"]
        src_root, dst_root = bundle.canonical(subject), bundle.canonical(target)
        if src_root not in roots and dst_root not in roots:
            continue
        chunk_id = bundle.linked_ids[edge["id"]]
        direct = int(subject in detected) + int(target in detected)
        double = int(src_root in roots and dst_root in roots)
        evidence_score = sum(
            _has_term(query, word) for word in
            (bundle.concepts[subject]["label"], bundle.concepts[target]["label"])
        )
        priority = double*20 + direct*5 + evidence_score*2 + int(edge["relation"] == "ALSO_CALLED")
        scored.append((-priority, edge["id"], chunk_id))
    scored.sort()
    selected = []
    seen = set()
    for _, _, cid in scored:
        if cid in seen:
            continue
        selected.append(cid)
        seen.add(cid)
        if len(selected) >= max(1, min(3, int(limit))):
            break
    return selected, {"matched_labels": labels[:12], "matched_edges": len(scored)}

def classify_intents(query: str) -> list[str]:
    """Conservative, explicit query intent gates; do not expand definition queries."""
    intents = []
    if not isinstance(query, str):
        return intents
    # Structure + mechanism, NOT plain definition or plain part listing.
    if (re.search(r"有哪些层|包含哪些层|组成|构成|哪些部分|哪些结构", query)
        and re.search(r"计算|运算|处理|工作|机制|实现|怎样|如何|怎么", query)):
        intents.append("composition_mechanism")
    if re.search(r"理论依据|理论基础|基本依据|基于什么|依赖什么理论", query):
        intents.append("theoretical_basis")
    if re.search(r"属于什么方法|属于哪种方法|方法层级|方法类别|是哪一类", query):
        intents.append("method_hierarchy")
    if ("汇聚" in query or "池化" in query) and \
            re.search(r"下采样|汇聚函数|池化函数", query):
        intents.append("pooling_operation")
    return intents

# Allowed directed edge *patterns*. No traversal via MENTIONED_IN/PART_OF.
# ALSO_CALLED is resolved only through the verified synonym registry.
PATTERNS = {
    "composition_mechanism": {("HAS_PART", "USES"), ("HAS_PART", "COMPUTES")},
    "theoretical_basis": {("IS_A", "BASED_ON"), ("SPECIAL_CASE_OF", "BASED_ON")},
    "method_hierarchy": {("IS_A", "IS_A")},
    "pooling_operation": {("USES", "IS_A")},
}

def find_two_hop_paths(query: str, bundle: OfflineBundle) -> list[dict]:
    """Return evidence paths without synthesizing an A->C graph fact."""
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH:
        return []
    intents = classify_intents(query)
    if not intents:
        return []
    _, roots, _ = detect_query_concepts(query, bundle)
    directed = [edge for edge in bundle.edges if edge["relation"] in TRAVERSABLE]
    outgoing = defaultdict(list)
    for e in directed:
        outgoing[bundle.canonical(e["source"])].append(e)
    paths = []
    used = set()
    for start_root in sorted(roots):
        for e1 in outgoing[start_root]:
            mid = bundle.canonical(e1["target"])
            for e2 in outgoing[mid]:
                end = bundle.canonical(e2["target"])
                if e1["id"] == e2["id"] or end == start_root:
                    continue
                allowed_by = [
                    intent for intent in intents if
                    (e1["relation"], e2["relation"]) in PATTERNS[intent]
                ]
                if not allowed_by:
                    continue
                key = (e1["id"], e2["id"])
                if key in used:
                    continue
                used.add(key)
                ids = [bundle.linked_ids[e1["id"]], bundle.linked_ids[e2["id"]]]
                edges = []
                for edge in (e1, e2):
                    claim = bundle.claim_by_edge[edge["id"]]
                    edges.append({
                        "claim_id": edge["id"], "predicate": edge["relation"],
                        "subject": bundle.concepts[edge["source"]]["label"],
                        "object": bundle.concepts[edge["target"]]["label"],
                        "chunk_id": bundle.linked_ids[edge["id"]],
                        "source_file": claim["source_file"],
                        "section_path": claim["section_path"],
                        "evidence_text": claim["evidence_text"],
                    })
                paths.append({
                    "start_concept": bundle.concepts[start_root]["label"],
                    "intents": allowed_by,
                    "edges": edges,
                    "path_chunk_ids": list(dict.fromkeys(ids)),
                    "second_hop_chunk_id": ids[-1],
                    "note": "Path gives linked evidence only; it is not a new direct A-to-C claim.",
                })
    return paths

def probe_query(query: str, bundle: OfflineBundle,
                max_graph_candidates: int = 2) -> dict:
    """Budget-matched candidate probe; never rerank or inject anything."""
    _require(1 <= max_graph_candidates <= 3, "Graph budget must be 1..3")
    old, info = v213_candidates(query, bundle, limit=max_graph_candidates)
    paths = find_two_hop_paths(query, bundle)
    discovered = list(dict.fromkeys(
        p["second_hop_chunk_id"] for p in paths
    ))
    newly_discovered = [cid for cid in discovered if cid not in old]

    # For relation-intent queries only: preserve top V2.13 evidence and allow
    # at most one genuinely different 2-hop candidate into a fixed-size pool.
    # Candidate routing is not a forced Reranker/final-evidence slot.
    if newly_discovered and max_graph_candidates >= 2 and old:
        selected = old[:1] + newly_discovered[:1]
        for cid in old[1:]:
            if len(selected) >= max_graph_candidates:
                break
            if cid not in selected:
                selected.append(cid)
    else:
        selected = list(old)
        if paths:
            for cid in newly_discovered:
                if len(selected) >= max_graph_candidates:
                    break
                if cid not in selected:
                    selected.append(cid)

    return {
        "query": query,
        "intent_gates": classify_intents(query),
        "v213_candidate_ids": old,
        "v213_matched_edges": info["matched_edges"],
        "v213_matched_labels": info["matched_labels"],
        "two_hop_paths": paths,
        "two_hop_second_edge_ids": discovered,
        "new_two_hop_ids_before_budget": newly_discovered,
        "v214_budgeted_candidate_ids": selected,
        "new_ids_in_budgeted_v214": [cid for cid in selected if cid not in old],
        "v213_ids_displaced_by_budget": [cid for cid in old if cid not in selected],
        "budget": max_graph_candidates,
        "not_measured": [
            "live_chroma_id_content_validation", "hybrid_overlap",
            "reranker_relevance", "final_evidence", "answer_quality",
            "token_usage", "runtime_latency"
        ],
    }

def graph_inventory(bundle: OfflineBundle) -> dict:
    edges = [e for e in bundle.edges if e["relation"] in TRAVERSABLE]
    paths = []
    for e1 in edges:
        for e2 in edges:
            if (bundle.canonical(e1["target"]) == bundle.canonical(e2["source"])
                and bundle.canonical(e1["source"]) != bundle.canonical(e2["target"])):
                paths.append((e1["id"], e2["id"]))
    return {
        "semantic_relation_count": len(bundle.edges),
        "relation_type_counts": dict(sorted(Counter(
            e["relation"] for e in bundle.edges).items())),
        "distinct_verified_bound_chunk_ids": len(set(bundle.linked_ids.values())),
        "directed_two_hop_path_count_excluding_synonyms": len(paths),
        "directed_two_hop_edge_paths": [list(p) for p in paths],
    }

def make_offline_report(bundle: OfflineBundle, questions=FIXED_QUERIES,
                        budget: int = 2) -> dict:
    return {
        "experiment": "DouXian_V2.14_Offline_Relation_Path_Probe",
        "phase": "offline_candidates_only",
        "source_artifact_sha256": bundle.file_hashes,
        "inventory": graph_inventory(bundle),
        "comparison": "V2.13 lexical-edge candidate selection vs. V2.14 controlled two-hop; same candidate budget",
        "no_live_qa_claim": True,
        "queries": [probe_query(q, bundle, budget) for q in questions],
    }

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None,
                   help="Optional NEW JSON path under graph_reports; never overwritten.")
    p.add_argument("--query", action="append", default=None,
                   help="Repeat this flag for multiple custom Chinese queries.")
    p.add_argument("--budget", type=int, default=2, choices=(1, 2, 3))
    args = p.parse_args(argv)
    bundle = load_offline_bundle()
    report = make_offline_report(bundle, questions=args.query or FIXED_QUERIES,
                                 budget=args.budget)
    print("[V2.14 OFFLINE] relation edges:",
          report["inventory"]["semantic_relation_count"])
    print("[V2.14 OFFLINE] distinct verified bound Chunk IDs:",
          report["inventory"]["distinct_verified_bound_chunk_ids"])
    for run in report["queries"]:
        print("Q:", run["query"])
        print("  V2.13:", len(run["v213_candidate_ids"]),
              "V2.14:", len(run["v214_budgeted_candidate_ids"]),
              "new vs V2.13:", len(run["new_ids_in_budgeted_v214"]),
              "two-hop paths:", len(run["two_hop_paths"]))
    if args.out is not None:
        out = args.out.resolve()
        reports_root = REPORTS_DIR.resolve()
        _require(out.parent == reports_root and out.suffix.lower() == ".json",
                 "--out must be a .json directly inside graph_reports/")
        out.parent.mkdir(parents=True, exist_ok=True)
        # x = exclusive create; old historical reports are never overwritten.
        with out.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("[V2.14 OFFLINE] report saved:", out)
    else:
        print("[V2.14 OFFLINE] no output written; use --out for new report")

if __name__ == "__main__":
    main()
