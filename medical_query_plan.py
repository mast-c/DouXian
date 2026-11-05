"""One bounded, local retrieval plan for text and source-backed graph retrieval."""
from __future__ import annotations

import hashlib
import json

FIELD_LABELS = {
    "indication": "适应症", "composition": "成分", "dosage": "用法用量",
    "warnings": "注意事项", "contraindications": "禁忌",
    "adverse_reactions": "不良反应", "drug_interactions": "药物相互作用",
}
FIELD_RELATIONS = {
    "indication": "INDICATION_MENTION", "composition": "HAS_INGREDIENT",
    "warnings": "CAUTION_MENTION", "contraindications": "CONTRAINDICATION_MENTION",
    "adverse_reactions": "ADVERSE_REACTION_MENTION", "drug_interactions": "INTERACTION_MENTION",
}
MECHANISM_RELATIONS = [
    "MECHANISM_MENTION", "MECHANISM_STATEMENT_MENTION", "MECHANISM_PARTNER_MENTION",
    "MECHANISM_ENTITY_MENTION", "METABOLISM_MENTION", "ENZYME_MODULATION_MENTION",
    "EXPOSURE_CHANGE_MENTION", "TRANSPORT_MENTION", "MECHANISM_OUTCOME_MENTION",
    "STATEMENT_ENZYME_MENTION", "STATEMENT_PROCESS_MENTION", "STATEMENT_EXPOSURE_MENTION",
    "STATEMENT_OUTCOME_MENTION",
]


def build_retrieval_plan(question: str, understanding: dict) -> dict:
    """Plan evidence lookups; questions and candidate formulations are not facts."""
    medicines = understanding.get("medicines", [])
    from medical_category_discovery import category_discovery_plan
    discovery = category_discovery_plan(question, understanding)
    intent = understanding.get("intent", {})
    fields = list(dict.fromkeys(intent.get("fields", []) + [
        field for field, label in FIELD_LABELS.items()
        if label in question or field == "composition" and "组成" in question
    ]))
    blocked = bool(understanding.get("vague_medicines") or understanding.get("unresolved_medication_mentions")
                   or understanding.get("ambiguous_aliases") or understanding.get("route_conflicts")
                   or understanding.get("alternative_medication_mentions") and intent.get("interaction_question"))
    context = list(dict.fromkeys(t for c in understanding.get("patient_context", [])
                                if c.get("assertion") not in ("negated", "other_person")
                                for t in c.get("retrieval_terms", [])))
    names = [g["surface"] + (
        "（检索候选：" + "、".join(g.get("candidate_drug_names", [])[:2]) + "；实际制剂待核对）"
        if not g.get("actual_patient_product_known") and g.get("candidate_drug_names") else ""
    ) for g in medicines]
    subject = "；".join(names)
    context_text = "；用户描述（不作诊断）：" + "、".join(context[:8]) if context else ""
    time_text = "、".join(t["surface"] for t in understanding.get("timing", []) if t.get("medicine_surface"))
    tasks = []

    def add(key, task_fields, goal, *, clinical_context=True):
        task_fields = [f for f in task_fields if f in FIELD_LABELS]
        tasks.append({"id": key, "fields": task_fields,
                      "query": subject + (context_text if clinical_context else "") + "。原文检索目标：" + goal + "。",
                      "graph_relations": list(dict.fromkeys(FIELD_RELATIONS[f] for f in task_fields if f in FIELD_RELATIONS)),
                      "requires_connected_medicines": key == "interaction"})

    if medicines and not blocked:
        if discovery:
            add("category_candidates", [], "按指定层级查起始药分类，展开同类具体药物并应用排除条件")
            tasks[-1]["graph_relations"] = ["DATASET_CATEGORY_MEMBERSHIP", "DATASET_CATEGORY_PARENT"]
            add("candidate_sources", discovery["fields"], "将上一步候选传入具体制剂原文检索，独立核对关系，不将起始药结论推广到同类药")
            tasks[-1]["depends_on"] = "category_candidates"
            if discovery["other_category"]:
                add("partner_category", [], "核对相互作用对象的分类，按指定的对方分类层级与起始药分类比较")
                tasks[-1]["depends_on"] = "candidate_sources"
                tasks[-1]["graph_relations"] = ["DATASET_CATEGORY_MEMBERSHIP", "DATASET_CATEGORY_PARENT"]
        if not discovery and intent.get("classification_question"):
            add("classification", [], "对应制剂在本地分类文件中的分类路径及来源，不等同权威药理分类")
            tasks[-1]["graph_relations"] = ["DATASET_CATEGORY_MEMBERSHIP", "DATASET_CATEGORY_PARENT", "CLASS_MEMBERSHIP_ASSERTION"]
        if not discovery and intent.get("interaction_question"):
            add("interaction", ["drug_interactions", "warnings"], "对应制剂的药物相互作用及合用条件")
        if not discovery and intent.get("mechanism_question"):
            add("mechanism", ["drug_interactions", "warnings"],
                "检索原文明确记载的药理机制、代谢/转运、酶调节、药物暴露变化及相关结果", clinical_context=False)
            tasks[-1]["graph_relations"] = list(MECHANISM_RELATIONS)
        if not discovery and intent.get("adverse_event_question"):
            add("adverse_event", ["adverse_reactions", "warnings"], "用药后新不适相关的不良反应、注意事项和就医条件，不预设药物因果")
        if not discovery and intent.get("efficacy_question"):
            add("efficacy", ["indication", "warnings"], "症状未改善时的注意事项、适应症及资料明确的时间条件"
                + ("；用户用药时间为“" + time_text + "”，不可擅自换算" if time_text else ""))
        if not discovery and intent.get("risk_question") and not intent.get("adverse_event_question"):
            add("patient_safety", ["contraindications", "warnings"], "患者所述条件相关的禁忌及注意事项")
        remaining = [f for f in fields if not any(f in task["fields"] for task in tasks)]
        # Dosage belongs to a source lookup, not a graph relation or a prescription.
        if intent.get("efficacy_question") or "dosage" in remaining:
            add("administration", ["dosage", "warnings"] if intent.get("efficacy_question") else ["dosage"],
                "用法用量、给药途径及使用方法注意事项，不生成个体化调药建议", clinical_context=False)
            remaining = [f for f in remaining if f not in ("dosage", "warnings")]
        if remaining and not discovery:
            add("source_fields", remaining, "、".join(FIELD_LABELS[f] for f in remaining if f in FIELD_LABELS))
        if not tasks:
            add("original", fields, "用户原问题对应资料：" + question)

    # Merge overflow goals instead of dropping a requested column or drug.
    while len(tasks) > 3:
        extra = tasks.pop()
        target = tasks[-1]
        target["fields"] = list(dict.fromkeys(target["fields"] + extra["fields"]))
        target["graph_relations"] = list(dict.fromkeys(target["graph_relations"] + extra["graph_relations"]))
        target["query"] += " 同时检索" + "、".join(FIELD_LABELS[f] for f in extra["fields"]) + "原文。"
    plan = {
        "version": 1, "original_question": question, "strategy": "local_patient_intents",
        "status": "needs_clarification" if blocked else "planned" if tasks else "original_query",
        "tasks": tasks, "fields": list(dict.fromkeys(fields + [f for t in tasks for f in t["fields"]])),
        "graph_relations": list(dict.fromkeys(r for t in tasks for r in t["graph_relations"])),
        "medicine_surfaces": [m["surface"] for m in medicines],
        "candidate_drug_names": list(dict.fromkeys(name for m in medicines for name in m.get("candidate_drug_names", []))),
        "context": context, "planning_model_calls": 0,
        "limits": {"max_queries": 3, "recall_workers": 2, "recall_per_query": 12,
                   "max_candidates": 36, "rerank_calls": 1, "recall_seconds": 12,
                   "rerank_seconds": 10, "total_seconds": 24},
        "policy": "retrieval_goals_only_not_clinical_claims",
    }
    if discovery:
        plan["category_discovery"] = discovery
        plan["strategy"] = "bounded_dependent_category_discovery"
    plan["plan_id"] = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
    return plan


def section_routing_question(question: str, plan: dict | None) -> str:
    if not plan or plan.get("status") != "planned":
        return question
    return question + "\n待检索原文章节：" + "、".join(FIELD_LABELS[f] for f in plan.get("fields", []) if f in FIELD_LABELS)
