# -*- coding: utf-8 -*-
"""Lossless, source-bound evidence transport across GraphRAG review and retry.

No LLM inference and no clinical decisions are made here. This module is used
by rag.py to keep per-path provenance, roles, stable identities, and source
verification flags intact as excerpts are re-numbered for Evidence Guard.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from medical_evidence_coverage import evidence_key

# Do not blindly copy complete input dictionaries into the model-facing JSON:
# only fields used by verifiers, classifiers and diagnostic trace are carried.
TRANSPORT_FIELDS = (
    "discovery_role", "product", "partner", "task_id", "task_kind",
    "source_ids", "source_urls", "source_file", "source_id",
    "candidate_names", "retrieval_task_ids", "_retrieval_tool",
    "graph_verified", "graph_path", "record_id", "section_id", "chunk_id",
    "candidate_id", "section_field", "parent_verified",
    "source_status", "clinical_status", "evidence_type", "drug_name",
    "disease_term", "evidence_quote", "source_url", "source",
    "section_path", "link_retrieval_route", "linked_candidate_ids",
    "disease_link_verified", "association_type", "source_verification",
)

# The current medical_targeted_retry implementation handles leaflet section
# targets, NOT a JSON-pointer dataset_category or discovered-source target.
SUPPORTED_SECTION_RETRY_FIELDS = frozenset({
    "indication", "composition", "dosage", "warnings", "contraindications",
    "adverse_reactions", "drug_interactions", "mechanism",
})


def retain_provenance(destination: dict, source: dict) -> dict:
    """Copy the explicitly allowed lossless evidence metadata and stable ID."""
    if not isinstance(destination, dict) or not isinstance(source, dict):
        return destination
    for key in TRANSPORT_FIELDS:
        if key in source:
            destination[key] = source[key]
    destination["stable_evidence_key"] = evidence_key(source)
    if "id" in source:
        destination["original_evidence_id"] = source["id"]
    return destination


def split_retry_targets(targets: list[dict] | None) -> tuple[list[dict], list[str]]:
    """Keep category and other non-leaflet tasks out of the section-only API."""
    eligible = []
    skipped = []
    for target in targets or ():
        field = target.get("field") if isinstance(target, dict) else None
        if field in SUPPORTED_SECTION_RETRY_FIELDS:
            eligible.append(target)
        else:
            skipped.append(str(field or "unknown"))
    return eligible, sorted(set(skipped))


def verify_retry_integrity(before: list[dict], old_coverage: dict,
                           retry_result: Any) -> tuple[dict, dict]:
    """Fail closed on dropped/renumbered/mutated original evidence.

    A retry may append new evidence; it must not silently remove or rewrite
    source-validated evidence, which would invalidate source bindings and
    citation IDs. On corruption, use the initial reviewer decision unchanged.
    This does NOT turn unaccepted evidence into approved generation evidence.
    """
    def fallback(reason):
        return ({
            "coverage": deepcopy(old_coverage), "excerpts": deepcopy(before),
            "retry_queries": [], "new_chunks": 0,
            "status": "retry_integrity_fallback", "execution": {},
        }, {"status": "restored_initial_review", "reason": reason,
            "original_evidence_count": len(before), "retained_evidence_count": len(before)})

    if not isinstance(retry_result, dict):
        return fallback("retry_response_is_not_a_dict")
    later = retry_result.get("excerpts")
    coverage = retry_result.get("coverage")
    if not isinstance(later, list) or not isinstance(coverage, dict):
        return fallback("retry_response_missing_excerpts_or_coverage")

    identities = {}
    for item in before:
        if not isinstance(item, dict):
            continue
        key = evidence_key(item)
        if key in identities and identities[key] != item.get("id"):
            return fallback("duplicate_initial_source_identity")
        identities[key] = item.get("id")
    observed = {}
    observed_ids = set()
    for item in later:
        if not isinstance(item, dict):
            return fallback("retry_contains_invalid_excerpt")
        identity = evidence_key(item)
        evid_id = item.get("id")
        if not isinstance(evid_id, int) or isinstance(evid_id, bool) or evid_id < 1:
            return fallback("retry_contains_invalid_id")
        if evid_id in observed_ids:
            return fallback("retry_contains_duplicate_ids")
        observed_ids.add(evid_id)
        observed[identity] = evid_id
    for key, evid_id in identities.items():
        if observed.get(key) != evid_id:
            return fallback("prior_evidence_dropped_or_renumbered")

    kept = {item.get("id") for item in later}
    raw_ids = coverage.get("evidence_ids") or []
    if not isinstance(raw_ids, list) or any(
        not isinstance(x, int) or isinstance(x, bool) or x not in kept
        for x in raw_ids
    ):
        return fallback("review_references_unknown_evidence_id")
    if not isinstance(coverage.get("aspects"), list):
        return fallback("retry_review_missing_aspects")
    for aspect in coverage["aspects"]:
        if not isinstance(aspect, dict):
            return fallback("retry_review_invalid_aspect")
        if any(eid not in kept for eid in (aspect.get("evidence_ids") or [])):
            return fallback("retry_aspect_references_unknown_evidence")
    return retry_result, {
        "status": "verified", "reason": "source_identity_and_citation_ids_preserved",
        "original_evidence_count": len(before),
        "retained_evidence_count": len(later),
        "new_evidence_count": len(later)-len(before),
    }


def evidence_role_counts(items: list[dict] | None) -> dict:
    """Privacy-safe stage diagnostics: categories only, no question or quotes."""
    roles = {}
    verified = 0
    for item in items or ():
        if not isinstance(item, dict):
            continue
        role = str(item.get("discovery_role") or "ordinary")
        roles[role] = roles.get(role, 0) + 1
        if item.get("graph_verified") and (item.get("graph_path") or {}).get("verified_against_original"):
            verified += 1
    return {"count": sum(roles.values()), "roles": roles, "source_verified_graph_items": verified}
