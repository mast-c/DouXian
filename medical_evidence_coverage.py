"""Track source coverage separately from clinical support and answer citations."""
from __future__ import annotations

import hashlib
import json
import re

from medical_query_plan import FIELD_LABELS, FIELD_RELATIONS

GRAPH_TYPE = "evidence_backed_semantic_graph_path"
SECTION_TYPES = {"json_neo4j_text_verified", "unverified_drug_source_section"}


def evidence_key(item):
    # IDs change when evidence is merged; source/content identity does not.
    identity = [item.get("evidence_type"), item.get("record_id"), item.get("section_id"),
                (item.get("graph_path") or {}).get("id"), item.get("source"), item.get("content")]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:24]


def build_answer_targets(question, understanding, plan):
    """Retrieval targets are NOT new user questions or proof of sufficiency."""
    medicines = understanding.get("medicines", [])
    if not medicines or not plan or plan.get("status") != "planned":
        return []
    discovery = plan.get("category_discovery")
    if discovery:
        # The requested subjects are discovered products, not the seed drug.
        candidate_names = list(discovery.get("candidate_names", []))
        return [{"id": "target_category", "field": "dataset_category", "label": "起始药物分类归属", "medicines": [], "pair": False,
                 "discovery_roles": ["seed_category", "seed_composition"]},
                {"id": "target_discovered_sources", "field": "discovered_sources",
                 "label": discovery["level_label"]+"候选药物的独立原文及约束核对", "medicines": [], "pair": False,
                 "discovery_roles": ["member", "source_field", "interaction_example"],
                 "candidate_names": candidate_names}]
    targets = []
    interaction = understanding.get("intent", {}).get("interaction_question") and len(medicines) == 2
    for field in plan.get("fields", []):
        if field not in FIELD_LABELS:
            continue
        groups = [medicines] if interaction and field == "drug_interactions" else [[m] for m in medicines]
        for group in groups:
            label = "与".join(m["surface"] for m in group) + "的" + FIELD_LABELS[field]
            targets.append({"id": f"target_{len(targets)+1}", "field": field, "label": label,
                            "medicines": [{"surface": m["surface"],
                                           "names": m.get("candidate_drug_names", []) or [m["surface"]],
                                           "product_known": bool(m.get("actual_patient_product_known"))} for m in group],
                            "pair": len(group) == 2})
    return targets


def required_source_aspects(question, understanding, targets):
    """Fix a checklist only when it exhausts a simple, explicit source-field ask.

    Patient symptoms, efficacy, causality, advice, comparison and implicit
    retrieval goals stay with dynamic review; do not turn them into field lookups.
    """
    if any(t.get("discovery_roles") for t in targets) or not targets or not 1 <= len(targets) <= 5 or len(question) > 180:
        return []
    if any(not m.get("actual_patient_product_known") for m in understanding.get("medicines", [])):
        return []
    tail = question
    medicine_ends = []
    for medicine in sorted(understanding.get("medicines", []), key=lambda m: -len(m["surface"])):
        medicine_ends.extend(match.end() for match in re.finditer(re.escape(medicine["surface"]), question))
        tail = tail.replace(medicine["surface"], " " * len(medicine["surface"]))
    aliases = {"indication": r"适应症|适应证|用途", "composition": r"组成成分|主要成分|活性成分|成分|组成",
               "dosage": r"用法用量|用法|用量", "warnings": r"注意事项|警告",
               "contraindications": r"禁忌", "adverse_reactions": r"不良反应|副作用",
               "drug_interactions": r"药物相互作用|相互作用"}
    first_field = min((m.start() for pattern in aliases.values() for m in re.finditer(pattern, tail)), default=-1)
    # "A的成分和B的禁忌" assigns different fields to different drugs. It is
    # not the shared-subject form "A和B的成分和禁忌"; retain dynamic review.
    if not medicine_ends or first_field < max(medicine_ends):
        return []
    explicit = set()
    for field, pattern in aliases.items():
        if re.search(pattern, tail):
            explicit.add(field)
            tail = re.sub(pattern, "", tail)
    tail = re.sub(r"请问|请介绍一下|请介绍|介绍一下|帮我查一下|请说明|分别|各自|有哪些|有何|有什么|是什么|如何描述|如何记载|原文|资料|记载|说明|介绍|列出|以及|还有|的|和|与|及|是|什么", "", tail)
    tail = re.sub(r"[\s、，,。？！?；;：:（）()]+", "", tail)
    # Warnings are often auxiliary retrieval targets for interaction queries.
    # They must not become extra mandatory answer aspects.
    if tail or not explicit:
        return []
    return [t["label"] + "原文如何描述？" for t in targets if t["field"] in explicit]


def original_text(item):
    text = str(item.get("content") or "")
    for marker in ("\n证据原文：", "\n原文："):
        if marker in text:
            text = text.split(marker, 1)[1]
            for stop in ("\n重要：", "\n资料状态：", "\n来源核验状态："):
                text = text.split(stop, 1)[0]
            return text.strip()
    return str(item.get("evidence_quote") or "")


def _matches(name, medicine):
    return name in medicine["names"]


def evidence_bindings(item, targets):
    """An available section/mention is a candidate, never clinical approval."""
    bindings = []
    graph = item.get("evidence_type") == GRAPH_TYPE
    path = item.get("graph_path") or {}
    if item.get("evidence_type") == "category_discovery_source_path":
        if not (item.get("graph_verified") and path.get("verified_against_original")):
            return []
        bindings = []
        for target in targets:
            if target.get("id") == "target_category" and item.get("discovery_role") in target.get("discovery_roles", []):
                bindings.append({"target_id": target["id"], "source_verified": True,
                                 "scope": "dataset_selection_proof_not_clinical_conclusion",
                                 "edge_ids": [e["id"] for e in path.get("edges", [])]})
            elif target.get("id") == "target_discovered_sources":
                role = item.get("discovery_role")
                if role not in target.get("discovery_roles", []):
                    continue
                # Candidate membership alone is not the requested medical
                # source fact. Mixed paths carry the verified source edge.
                if role not in ("interaction_example", "source_field"):
                    continue
                bindings.append({"target_id": target["id"], "source_verified": True,
                                 "scope": "literal_source_mention_not_clinical_conclusion",
                                 "edge_ids": [e["id"] for e in path.get("edges", [])]})
        return bindings
    if item.get("evidence_type") in SECTION_TYPES and any(t.get("id") == "target_discovered_sources" for t in targets):
        owner = str(item.get("drug_name") or "")
        if owner:
            return [{"target_id": t["id"], "source_verified": bool(item.get("parent_verified")),
                     "scope": "candidate_product_source_section_pending_semantic_review", "edge_ids": []}
                    for t in targets if t.get("id") == "target_discovered_sources"
                    and owner in t.get("candidate_names", [])]
    if graph and not (item.get("graph_verified") and path.get("verified_against_original")):
        return []
    for target in targets:
        medicines, field = target["medicines"], target["field"]
        if graph:
            edges = path.get("edges", [])
            relevant = [e for e in edges if e.get("relation") == FIELD_RELATIONS.get(field)
                        and e.get("section_field") == field and e.get("quote")
                        and e["quote"] in str(item.get("content") or "")
                        and any(_matches(e.get("src", {}).get("name"), m) for m in medicines)]
            if target["pair"]:
                # Reject shared symptoms, independent one-drug facts and third-drug joins.
                drug_names = {n.get("name") for n in path.get("nodes", []) if n.get("kind") == "Drug"}
                connected = all(any(_matches(name, m) for name in drug_names) for m in medicines)
                if not connected:
                    continue
            if relevant:
                bindings.append({"target_id": target["id"], "source_verified": True,
                                 "scope": "source_mention_not_clinical_conclusion" if field != "composition" else "composition_source",
                                 "edge_ids": [e.get("id") for e in relevant]})
        elif item.get("evidence_type") in SECTION_TYPES and item.get("section_field") == field:
            owner = str(item.get("drug_name") or "")
            if not any(_matches(owner, m) for m in medicines):
                continue
            text = original_text(item)
            if not text:
                continue
            if target["pair"] and not all(
                _matches(owner, m) or any(name in text for name in [m["surface"], *m["names"]])
                for m in medicines
            ):
                continue
            bindings.append({"target_id": target["id"], "source_verified": bool(item.get("parent_verified")),
                             "scope": "source_section_pending_semantic_review", "edge_ids": []})
    return bindings


def select_target_evidence(ordinary, graph, targets, cap):
    """Reserve unique target coverage before redundant paths or source quotas."""
    remaining, seen = [], set()
    for item in [*graph, *ordinary]:
        key = evidence_key(item)
        if key not in seen:
            remaining.append(item)
            seen.add(key)
    selected, covered, lanes = [], set(), set()
    while remaining and len(selected) < cap:
        def gain(pair):
            i, item = pair
            bindings = evidence_bindings(item, targets)
            fresh = [b for b in bindings if b["target_id"] not in covered]
            return (len(fresh) * 100 + sum(b["source_verified"] for b in fresh) * 10
                    + int(item.get("_retrieval_tool") not in lanes), -i)
        i, item = max(enumerate(remaining), key=gain)
        remaining.pop(i)
        selected.append(item)
        covered.update(b["target_id"] for b in evidence_bindings(item, targets))
        lanes.add(item.get("_retrieval_tool"))
    return [{**item, "id": i} for i, item in enumerate(selected, 1)]


def start_evidence_flow(targets):
    return {"version": 1, "scope": "source_coverage_not_clinical_sufficiency", "targets": targets,
            "stages": [], "transitions": [], "review": [], "answer": []}


def record_evidence_stage(flow, name, items, *, reason=""):
    rows = [{"key": evidence_key(item), "id": item.get("id"),
             "bindings": evidence_bindings(item, flow["targets"])} for item in items]
    if flow["stages"]:
        previous = flow["stages"][-1]
        present = {r["key"] for r in rows}
        flow["transitions"].append({"from": previous["name"], "to": name, "excluded": [
            {"key": row["key"], "previous_id": row["id"], "target_ids": [b["target_id"] for b in row["bindings"]],
             "reason": reason} for row in previous["evidence"] if row["key"] not in present]})
    flow["stages"].append({"name": name, "evidence": rows, "uncovered_targets": [
        t["id"] for t in flow["targets"] if not any(b["target_id"] == t["id"] for row in rows for b in row["bindings"])]})


def record_review(flow, excerpts, coverage):
    accepted = set(coverage.get("evidence_ids", []))
    by_id = {item["id"]: item for item in excerpts}
    flow["review"] = []
    for target in flow["targets"]:
        candidates = [i for i, item in by_id.items() if any(b["target_id"] == target["id"] for b in evidence_bindings(item, [target]))]
        flow["review"].append({"target_id": target["id"], "candidate_ids": candidates,
                               "generation_evidence_ids": sorted(accepted.intersection(candidates)),
                               "status": "unchecked" if coverage.get("decision") == "unchecked" else "reviewed",
                               "note": "证据进入生成不代表该证据涉及的所有目标均获审核支持"})
    flow["aspects"] = [{k: a.get(k) for k in ("question", "status", "evidence_ids", "reason", "validation_method")}
                       for a in coverage.get("aspects", [])]


def record_answer_usage(flow, answer):
    # A path citation may support only part of the path; never label all targets answered.
    body = re.split(r"\n(?:#{1,6}\s*)?(?:\*\*)?参考资料", answer, maxsplit=1)[0]
    cited = {int(x) for x in re.findall(r"(?<!\[)\[(\d+)\](?!\()", body)}
    flow["answer"] = [{"target_id": r["target_id"],
                       "cited_candidate_ids": sorted(cited.intersection(r["generation_evidence_ids"])),
                       "scope": "citation_presence_only_not_answer_correctness"} for r in flow.get("review", [])]


def coverage_hint(targets, items):
    """Small reviewer index; the original full excerpts remain authoritative."""
    return [{"target": t["label"], "evidence_ids": [i["id"] for i in items if evidence_bindings(i, [t])],
             "scope": "source_candidates_only_requires_semantic_review"} for t in targets]
