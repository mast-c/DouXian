"""Track source coverage separately from clinical support and answer citations."""
from __future__ import annotations

import hashlib
import json
import re

from medical_query_plan import (
    FIELD_LABELS, FIELD_RELATIONS, MECHANISM_RELATIONS,
    _asks_disease_symptoms,
)

GRAPH_TYPE = "evidence_backed_semantic_graph_path"
SECTION_TYPES = {"json_neo4j_text_verified", "unverified_drug_source_section"}


def evidence_key(item):
    # IDs change when evidence is merged; source/content identity does not.
    identity = [item.get("evidence_type"), item.get("record_id"), item.get("section_id"),
                (item.get("graph_path") or {}).get("id"), item.get("source"), item.get("content")]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:24]


def _build_answer_targets_legacy(question, understanding, plan):
    """Retrieval targets are NOT new user questions or proof of sufficiency."""
    medicines = understanding.get("medicines", [])
    if not plan or plan.get("status") != "planned":
        return []
    diseases = [c for c in understanding.get("patient_context", [])
                if c.get("kind") == "disease_mention"
                and c.get("assertion") not in ("negated", "other_person")]
    symptoms = [c for c in understanding.get("patient_context", [])
                if c.get("kind") == "symptom_word"
                and c.get("assertion") not in ("negated", "other_person")]
    if not medicines and not diseases and not symptoms:
        return []
    discovery = plan.get("category_discovery")
    if discovery:
        # A category membership fact, a candidate list and a seed-specific
        # interaction are three DIFFERENT questions; one path that happens to
        # mention another drug's interaction must never satisfy all three.
        candidate_names = list(discovery.get("candidate_names", []))
        operation = discovery.get("operation")
        targets = [{"id": "target_category", "field": "dataset_category",
                    "label": "起始药物在本地数据集的分类归属", "medicines": [], "pair": False,
                    "discovery_roles": ["seed_category"]},
                   {"id": "target_discovered_sources", "field": "discovered_sources",
                    "label": discovery["level_label"]+"下已核实的分类候选", "medicines": [], "pair": False,
                    "discovery_roles": ["member", "source_field", "interaction_example"],
                    "candidate_names": candidate_names,
                    "scope": "dataset_membership_candidates_not_pair_interaction"}]
        if operation == "interaction" and discovery.get("interaction_scope") == "seed_pair":
            targets.append({
                "id": "target_seed_interaction", "field": "seed_pair_interaction",
                "label": "同类候选与起始药物之间的具体相互作用原文",
                "medicines": [], "pair": True,
                "discovery_roles": ["interaction_example"],
                "seed_surface": discovery.get("seed_surface") or "",
                "candidate_names": candidate_names,
                "scope": "requires_exact_both_endpoints_verified",
            })
        return targets
    targets = []
    intent = understanding.get("intent", {})
    for disease in diseases:
        if intent.get("classification_question"):
            targets.append({"id": f"target_{len(targets)+1}", "field": "disease_classification",
                            "label": disease["surface"] + "的疾病分类/科别",
                            "disease": disease["surface"],
                            "medicines": [], "pair": False,
                            "claim_scope": "source_explicit_classification_not_authoritative_taxonomy"})
        if _asks_disease_symptoms(question, intent):
            targets.append({"id": f"target_{len(targets)+1}", "field": "symptoms",
                            "label": disease["surface"] + "的症状", "disease": disease["surface"],
                            "medicines": [], "pair": False})
        if intent.get("drug_discovery_question"):
            targets.append({"id": f"target_{len(targets)+1}", "field": "disease_drug_candidates",
                            "label": disease["surface"] + "相关药品资料", "disease": disease["surface"],
                            "medicines": [], "pair": False})
        if intent.get("interaction_question"):
            targets.append({"id": f"target_{len(targets)+1}", "field": "disease_interactions",
                            "label": disease["surface"] + "相关药品的明确相互作用", "disease": disease["surface"],
                            "medicines": [], "pair": False})
    for symptom in symptoms:
        if intent.get("drug_discovery_question"):
            targets.append({"id": f"target_{len(targets)+1}", "field": "symptom_drug_candidates",
                            "label": symptom["surface"] + "相关药品资料候选", "symptom": symptom["surface"],
                            "medicines": [], "pair": False,
                            "claim_scope": "source_candidates_not_patient_specific_recommendation"})
        if intent.get("risk_question"):
            targets.append({"id": f"target_{len(targets)+1}", "field": "patient_context_safety",
                            "label": "用户背景相关的说明书禁忌及注意事项", "symptom": symptom["surface"],
                            "medicines": [], "pair": False,
                            "claim_scope": "source_candidates_not_patient_specific_recommendation"})
    interaction = understanding.get("intent", {}).get("interaction_question") and len(medicines) == 2
    for field in plan.get("fields", []) if medicines else []:
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
    if understanding.get("intent", {}).get("mechanism_question"):
        targets.append({"id": f"target_{len(targets)+1}", "field": "mechanism",
                        "label": "与".join(m["surface"] for m in medicines) + "的药理机制",
                        "medicines": [{"surface": m["surface"],
                                       "names": m.get("candidate_drug_names", []) or [m["surface"]],
                                       "product_known": bool(m.get("actual_patient_product_known"))} for m in medicines],
                        "pair": len(medicines) == 2})
    return targets



_SEMANTIC_KIND_FIELD = {
    "drug_adverse_event": "adverse_reactions",
    "drug_interaction": "drug_interactions",
    "disease_classification": "disease_classification",
    "unresolved_classification": "unresolved_classification",
    "drug_classification": "dataset_category",
    "drug_safety": "warnings",
    "disease_symptoms": "symptoms",
    "treatment_discovery": "indication",
    "mechanism": "mechanism",
    "dosage": "dosage",
    "efficacy": "indication",
}


def build_answer_targets(question, understanding, plan, semantic_task_plan=None):
    """Unify legacy targets with LLM subtask identities without inventing facts.

    This creates retrieval COVERAGE GOALS, not medical conclusions. Each
    graph edge/quote still has to pass original-source and Evidence Guard checks.
    The existing positional 3-argument API remains compatible.
    """
    targets = _build_answer_targets_legacy(question, understanding, plan)
    task_plan = semantic_task_plan or (plan or {}).get("semantic_task_plan") or {}
    tasks = task_plan.get("tasks") if isinstance(task_plan, dict) else []
    if not isinstance(tasks, list):
        return targets
    targets = [dict(t) for t in targets]
    for task in tasks[:8]:
        if not isinstance(task, dict):
            continue
        task_id, kind = task.get("id"), task.get("kind")
        expected_field = _SEMANTIC_KIND_FIELD.get(kind)
        anchors = task.get("anchor_surfaces") or []
        if (not isinstance(task_id, str) or not task_id or expected_field is None
                or not isinstance(anchors, list)):
            continue
        # Do not turn an unknown/ambiguous drug into a verified patient drug.
        medicine_anchors = [name for name in anchors
                            if isinstance(name, str) and any(
                                m.get("surface") == name
                                for m in understanding.get("medicines") or []
                            )]
        disease_anchors = [name for name in anchors if isinstance(name, str)
                           and name not in medicine_anchors]
        # Existing targets with the same semantic meaning remain valid;
        # annotate only when the target is an unambiguous match.
        matches = [t for t in targets if t.get("field") == expected_field and (
            (expected_field == "disease_classification" and t.get("disease") in disease_anchors)
            or (medicine_anchors and len(medicine_anchors) == 1
                and len(t.get("medicines") or []) == 1
                and t["medicines"][0].get("surface") == medicine_anchors[0])
            or (kind == "drug_interaction" and len(medicine_anchors) == 2
                and t.get("pair")
                and {m.get("surface") for m in t.get("medicines") or []} == set(medicine_anchors))
        )]
        if len(matches) == 1 and not matches[0].get("semantic_task_id"):
            target = matches[0]
        else:
            target = {
                "id": "target_semantic_" + task_id,
                "field": expected_field,
                "label": task.get("query") or kind,
                "medicines": [{
                    "surface": name, "names": [name], "product_known": False,
                } for name in medicine_anchors],
                "pair": kind == "drug_interaction" and len(medicine_anchors) == 2,
                "disease": (disease_anchors[0] if disease_anchors else None),
                "claim_scope": "query_task_requires_independent_verified_source",
            }
            targets.append(target)
        target["semantic_task_id"] = task_id
        target["semantic_kind"] = kind
        target["semantic_anchor_surfaces"] = [v for v in anchors if isinstance(v, str)]
        target["semantic_requested_relations"] = list(task.get("graph_relations") or [])
        target["text_evidence_pending"] = task.get("graph_strategy") == "text_only"
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


def _verified_seed_pair_interaction(item, seed_surface: str, candidate_names) -> bool:
    """Accept a verified interaction edge only if it really involves SEED.

    Shared categories or the interaction of candidate A with unrelated B are
    not evidence that A interacts with the original user-mentioned seed drug.
    Direct drug pairs and independently verified seed ingredient joins are
    allowed as source MENTIONS, not clinically established interactions.
    """
    if not seed_surface or not item.get("graph_verified"):
        return False
    path = item.get("graph_path") or {}
    if not path.get("verified_against_original"):
        return False
    edges = path.get("edges") or []
    candidates = set(candidate_names or ()) - {seed_surface}
    if not candidates:
        return False
    # This record's own product must be a verified category member.
    product = item.get("product")
    member = any(e.get("relation") == "DATASET_CATEGORY_MEMBERSHIP"
                 and (e.get("src") or {}).get("name") == product
                 for e in edges if isinstance(e, dict))
    if not member or product not in candidates:
        return False
    for e in edges:
        if not isinstance(e, dict) or e.get("relation") != "INTERACTION_MENTION":
            continue
        src, dst = e.get("src") or {}, e.get("dst") or {}
        s_name, d_name = src.get("name"), dst.get("name")
        if ({s_name, d_name} == {seed_surface, product}
                and src.get("kind") == "Drug" and dst.get("kind") == "Drug"):
            return True
        # A->ingredient is connected to the SEED only when the exact
        # seed Drug->Ingredient identity has its own verified source edge.
        for comp in edges:
            if not isinstance(comp, dict) or comp.get("relation") != "HAS_INGREDIENT":
                continue
            owner, ingredient = comp.get("src") or {}, comp.get("dst") or {}
            if (owner.get("name") == seed_surface and s_name == product
                    and dst.get("id") and ingredient.get("id") == dst.get("id")):
                return True
            if (owner.get("name") == product and s_name == seed_surface
                    and dst.get("id") and ingredient.get("id") == dst.get("id")):
                return True
    return False


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
                # For interaction queries, member/interaction_example paths
                # support candidate membership only; they never imply a
                # pair interaction with the seed product.
                if target.get("scope") == "dataset_membership_candidates_not_pair_interaction":
                    member_edges = [e for e in path.get("edges", [])
                                    if e.get("relation") == "DATASET_CATEGORY_MEMBERSHIP"
                                    and e.get("src", {}).get("name") == item.get("product")]
                    if not member_edges or item.get("product") not in target.get("candidate_names", []):
                        continue
                    bindings.append({"target_id": target["id"], "source_verified": True,
                                     "scope": "verified_category_candidate_membership_only",
                                     "edge_ids": [e.get("id") for e in member_edges]})
                    continue
                if role not in ("interaction_example", "source_field"):
                    continue
                bindings.append({"target_id": target["id"], "source_verified": True,
                                 "scope": "literal_source_mention_not_clinical_conclusion",
                                 "edge_ids": [e["id"] for e in path.get("edges", [])]})
            elif target.get("id") == "target_seed_interaction":
                if item.get("discovery_role") != "interaction_example":
                    continue
                if _verified_seed_pair_interaction(
                        item, target.get("seed_surface") or "",
                        target.get("candidate_names") or []):
                    edges = [e.get("id") for e in path.get("edges") or []
                             if e.get("relation") in ("INTERACTION_MENTION", "HAS_INGREDIENT")]
                    bindings.append({"target_id": target["id"], "source_verified": True,
                                     "scope": "specific_pair_source_mention_not_clinically_validated",
                                     "edge_ids": edges})
        return bindings
    if item.get("evidence_type") in SECTION_TYPES and any(
            t.get("id") in ("target_discovered_sources", "target_seed_interaction")
            for t in targets):
        owner = str(item.get("drug_name") or "")
        text = original_text(item)
        result = []
        for t in targets:
            if t.get("id") == "target_discovered_sources" and owner in t.get("candidate_names", []):
                if t.get("scope") != "dataset_membership_candidates_not_pair_interaction":
                    result.append({"target_id": t["id"],
                                   "source_verified": bool(item.get("parent_verified")),
                                   "scope": "candidate_source_section_pending_semantic_review",
                                   "edge_ids": []})
            if t.get("id") == "target_seed_interaction" and item.get("section_field") == "drug_interactions":
                seed = t.get("seed_surface") or ""
                candidates = set(t.get("candidate_names") or []) - {seed}
                if item.get("parent_verified") and text and seed and (
                    (owner == seed and any(candidate in text for candidate in candidates))
                    or (owner in candidates and seed in text)
                ):
                    result.append({"target_id": t["id"], "source_verified": True,
                                   "scope": "explicit_named_pair_leaflet_source_pending_review",
                                   "edge_ids": []})
        return result
    if graph and not (item.get("graph_verified") and path.get("verified_against_original")):
        return []
    for target in targets:
        task_id = target.get("semantic_task_id")
        if (task_id and task_id == item.get("task_id")
                and not target.get("text_evidence_pending")
                and item.get("graph_verified") and path.get("verified_against_original")
                and isinstance(path.get("edges"), list)):
            required = set(target.get("semantic_requested_relations") or ())
            relations = {e.get("relation") for e in path["edges"]}
            names = {n.get("name") for n in path.get("nodes") or ()}
            # Verified graph evidence must actually cover its expected
            # entities and at least one required relation; it is only a
            # source-mention binding, not a clinical efficacy verdict.
            if (set(target.get("semantic_anchor_surfaces") or ()).issubset(names)
                    and (not required or bool(required & relations))):
                bindings.append({
                    "target_id": target["id"], "source_verified": True,
                    "scope": "verified_task_graph_source_not_clinical_conclusion",
                    "edge_ids": [edge.get("id") for edge in path["edges"]],
                })
        medicines, field = target["medicines"], target["field"]
        if field == "symptom_drug_candidates":
            if (item.get("evidence_type") in SECTION_TYPES
                    and item.get("section_field") in ("indication", "contraindications", "warnings", "adverse_reactions")
                    and "symptom_drug_source_candidates" in item.get("retrieval_task_ids", [])):
                text = original_text(item)
                if text:
                    bindings.append({"target_id": target["id"],
                                     "source_verified": bool(item.get("parent_verified")),
                                     "scope": "symptom_related_candidate_leaflet_source_not_patient_recommendation",
                                     "edge_ids": []})
            continue
        if field == "patient_context_safety":
            if (item.get("evidence_type") in SECTION_TYPES
                    and item.get("section_field") in ("contraindications", "warnings")
                    and "symptom_drug_source_candidates" in item.get("retrieval_task_ids", [])):
                text = original_text(item)
                if text:
                    bindings.append({"target_id": target["id"],
                                     "source_verified": bool(item.get("parent_verified")),
                                     "scope": "population_context_leaflet_source_not_patient_recommendation",
                                     "edge_ids": []})
            continue
        if field == "symptoms" and graph:
            relevant = [e for e in path.get("edges", [])
                        if e.get("relation") == "HAS_SYMPTOM"
                        and e.get("section_field") == "symptom"
                        and e.get("src", {}).get("name") == target.get("disease")
                        and e.get("quote") and e["quote"] in str(item.get("content") or "")]
            if relevant:
                bindings.append({"target_id": target["id"], "source_verified": True,
                                 "scope": "disease_source_symptom_mention_not_diagnosis",
                                 "edge_ids": [e.get("id") for e in relevant]})
            continue
        if field == "disease_classification" and graph:
            relevant = [e for e in path.get("edges", [])
                        if e.get("relation") == "DISEASE_CATEGORY_MENTION"
                        and e.get("section_field") == "category"
                        and e.get("src", {}).get("name") == target.get("disease")
                        and e.get("quote")
                        and e["quote"] in str(item.get("content") or "")]
            if relevant:
                bindings.append({"target_id": target["id"], "source_verified": True,
                                 "scope": "disease_source_classification_mention_not_authoritative_taxonomy",
                                 "edge_ids": [e.get("id") for e in relevant]})
            continue
        if field in ("disease_drug_candidates", "disease_interactions"):
            # Candidate disease links and generic product warnings are not
            # enough to answer treatment or named interaction questions.
            continue
        if field == "mechanism":
            if graph:
                edges = path.get("edges", [])
                mechanism_edges = [e for e in edges
                                   if e.get("relation") in MECHANISM_RELATIONS
                                   and e.get("quote")
                                   and e["quote"] in str(item.get("content") or "")]
                path_drugs = {n.get("name") for n in path.get("nodes", []) if n.get("kind") == "Drug"}
                covers_target = all(any(_matches(name, medicine) for name in path_drugs)
                                    for medicine in medicines)
                if mechanism_edges and covers_target:
                    bindings.append({"target_id": target["id"], "source_verified": True,
                                     "scope": "verified_mechanism_source_mention_not_causal_validation",
                                     "edge_ids": [e.get("id") for e in mechanism_edges]})
            elif item.get("evidence_type") in SECTION_TYPES and item.get("section_field") in ("drug_interactions", "warnings"):
                owner = str(item.get("drug_name") or "")
                text = original_text(item)
                pair_is_explicit = (not target["pair"] or all(
                    _matches(owner, medicine) or any(name in text for name in [medicine["surface"], *medicine["names"]])
                    for medicine in medicines
                ))
                if text and pair_is_explicit and any(_matches(owner, m) for m in medicines):
                    mechanism_terms = re.compile(r"CYP\w*|细胞色素|代谢|酶|转运|抑制|诱导|活性代谢物|血药浓度|血浆浓度|清除率|半衰期|生物利用度|机制")
                    if mechanism_terms.search(text):
                        bindings.append({"target_id": target["id"],
                                         "source_verified": bool(item.get("parent_verified")),
                                         "scope": "source_mechanism_text_pending_semantic_review", "edge_ids": []})
            continue
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
    # Same original-source edge can satisfy both legacy and semantic target
    # matching rules; do not count twice for source coverage.
    unique = {}
    for binding in bindings:
        unique.setdefault(binding["target_id"], binding)
    return list(unique.values())


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
