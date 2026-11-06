"""Bounded LLM assistance for query intent and patient-context extraction."""
from __future__ import annotations

import json
import os
import re

from medical_patient_context import context_assertion


SYSTEM_PROMPT = """你是医疗知识库的查询理解器，只负责解析用户问题、规划检索，不回答医学问题。
从原句提取多个意图，以及用户明确提到的药品、药物成分、症状、疾病和特殊人群。
严格输出 JSON：
{"intents":[],"entities":[{"type":"drug|ingredient|symptom|population|disease","surface":"连续的原文字串","canonical":"标准检索名称"}]}
intents 仅可为 symptom_treatment、medication_safety、disease_symptoms、drug_discovery、interaction、
adverse_event、efficacy、mechanism、classification、source_lookup。
规则：
1. 药品 surface 应为最短完整药名，绝不能把“华法林抗凝的房颤患者”当成一个药名：
   “长期服用华法林抗凝的房颤患者，需加用甲硝唑”应提取华法林、甲硝唑和房颤。
2. 不要推断用户未说出的药品、成分、剂型、疾病或关系；不可自行补充盐型/缓释剂型。
3. surface 必须是原句中连续出现的原文；canonical 不确定时使用 surface 原词。
4. “发烧”可标准化为“发热”；“孕妇”“妊娠期人群”可标准化为“妊娠”。
5. 判断“加用甲硝唑可能有什么风险”时，应识别 interaction 和 medication_safety 意图。
6. 否定、假设和他人情况不能改写为用户正在服药或已确诊；抽取实体不等于确认服用事实。
7. 每种信息不超过 8 个实体，总实体数最多 12 个；除 JSON 外不得输出其他内容。"""

INTENT_MAP = {
    "symptom_treatment": ("drug_discovery_question", True),
    "medication_safety": ("risk_question", True),
    "disease_symptoms": ("symptom_question", True),
    "drug_discovery": ("drug_discovery_question", True),
    "interaction": ("interaction_question", True),
    "adverse_event": ("adverse_event_question", True),
    "efficacy": ("efficacy_question", True),
    "mechanism": ("mechanism_question", True),
    "classification": ("classification_question", True),
    "source_lookup": ("source_lookup_question", True),
}
CANONICAL_TERMS = {
    "发烧": "发热", "发热": "发热", "低热": "发热", "高热": "发热",
    "有点烧": "发热", "烧起来": "发热",
    "孕妇": "妊娠", "孕期": "妊娠", "妊娠期": "妊娠",
    "妊娠期人群": "妊娠", "怀孕": "妊娠", "怀孕的人": "妊娠",
    "怀宝宝": "妊娠", "有宝宝": "妊娠",
    "备孕": "备孕", "哺乳期": "哺乳期", "母乳": "哺乳期",
}
# 同义标准化仅用于症状/人群；药品必须通过图谱中的真实名称链接。
_CANONICAL_EQUIVALENTS = {
    "妊娠期人群": "妊娠", "妊娠期": "妊娠", "孕期": "妊娠",
    "孕妇": "妊娠", "怀孕": "妊娠", "发烧": "发热",
}
_ENTITY_CATALOG_KIND = {
    "drug": "Drug", "ingredient": "Ingredient",
    "symptom": "Symptom", "population": "Population", "disease": "Disease",
}
_PATIENT_KIND = {
    "symptom": "symptom_word",
    "population": "special_population",
    "disease": "disease_mention",
}


def _catalog_names(allowed_entity_names) -> dict:
    """Accept a typed graph catalog; keep compatibility with the former flat set."""
    if isinstance(allowed_entity_names, dict):
        result = {}
        for kind, names in allowed_entity_names.items():
            if isinstance(names, (set, frozenset, list, tuple)):
                result[kind] = {name for name in names if isinstance(name, str) and name}
        return result
    if isinstance(allowed_entity_names, (set, frozenset, list, tuple)):
        return {"_flat": {name for name in allowed_entity_names if isinstance(name, str) and name}}
    return {}


def _normalized_spelling(name: str) -> str:
    return re.sub(r"[\s·・\-－_]", "", name).casefold()


def _linked_medication(surface: str, proposed: str, kind: str, catalog: dict):
    """Never use a free-form LLM drug name as proof of a catalog entity.

    Generic ingredient names may be misclassified as drugs by the LLM. In that
    case resolve against Ingredient if the catalog has that *exact* name.
    Do not silently promote a generic ingredient to a specific formulation.
    """
    if not surface or len(surface) > 32:
        return None
    if not proposed:
        proposed = surface
    # Do not permit the LLM to claim a long patient phrase is a brand-name alias.
    if _normalized_spelling(surface) != _normalized_spelling(proposed):
        return None
    wanted = _ENTITY_CATALOG_KIND[kind]
    kinds = (wanted, "Ingredient") if wanted == "Drug" else (wanted,)
    for catalog_kind in kinds:
        available = catalog.get(catalog_kind, set())
        if proposed in available:
            return {"type": "drug" if catalog_kind == "Drug" else "ingredient",
                    "surface": surface, "canonical": proposed,
                    "catalog_kind": catalog_kind}
    return None


def should_enhance(question: str, understanding: dict) -> bool:
    """Run the model for unresolved parsing even when other clues were found."""
    if os.getenv("MEDICAL_LLM_QUERY_UNDERSTANDING", "1").strip().lower() in (
        "0", "false", "off", "no",
    ):
        return False
    if any(understanding.get(key) for key in (
        "unresolved_medication_mentions", "ambiguous_aliases", "route_conflicts",
    )):
        return True
    intent = understanding.get("intent") or {}
    medicines = understanding.get("medicines") or []
    context = understanding.get("patient_context") or []
    if intent.get("interaction_question") and len(medicines) < 2:
        return True
    if not medicines and context and (
        intent.get("drug_discovery_question") or intent.get("risk_question")
    ):
        return True
    recognized_intent = any(intent.get(key) for key in (
        "interaction_question", "mechanism_question", "risk_question",
        "drug_discovery_question", "symptom_question", "adverse_event_question",
        "efficacy_question", "classification_question",
    ))
    return not recognized_intent or not bool(medicines or context)


def _plain_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") if isinstance(part, dict) else str(part)
                       for part in content)
    return str(content or "")


def _parse_response(response) -> dict:
    if isinstance(response, dict) and ("intents" in response or "entities" in response):
        return response
    raw = getattr(response, "content", response)
    if isinstance(raw, dict) and ("intents" in raw or "entities" in raw):
        return raw
    content = _plain_content(raw).strip()
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("query understanding returned no JSON object")
    value = json.loads(content[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("query understanding JSON must be an object")
    return value


def enhance_understanding(
    question: str, understanding: dict, model, *, allowed_entity_names=()
) -> tuple[dict, dict]:
    """Merge verified query hints. Never invent or overwrite verified graph evidence.

    LLM drug mentions are stored separately as `llm_medicine_mentions`; they
    do not masquerade as an upstream verified patient medicine/formulation.
    The graph retriever can turn them into candidate anchors only after a
    second, exact graph-catalog match.
    """
    if not should_enhance(question, understanding):
        return understanding, {
            "status": "skipped", "reason": "sufficient_rule_parse_or_disabled",
            "model_calls": 0,
        }
    try:
        if getattr(model, "root_client", None) is not None and hasattr(model, "model_copy"):
            client = model.root_client.with_options(timeout=4, max_retries=0)
            model = model.model_copy(update={
                "request_timeout": 4, "max_retries": 0,
                "root_client": client, "client": client.chat.completions,
            })
        bounded_model = model.bind(temperature=0, max_tokens=900)
        response = bounded_model.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question[:1200]},
        ])
        payload = _parse_response(response)
        catalog = _catalog_names(allowed_entity_names)
        merged = {
            **understanding,
            "intent": dict(understanding.get("intent") or {}),
            "patient_context": list(understanding.get("patient_context") or []),
            "llm_medicine_mentions": list(understanding.get("llm_medicine_mentions") or []),
            "llm_unlinked_medication_mentions": list(understanding.get("llm_unlinked_medication_mentions") or []),
        }
        accepted_intents = []
        intents = payload.get("intents")
        for value in (intents[:10] if isinstance(intents, list) else []):
            if value not in INTENT_MAP:
                continue
            key, enabled = INTENT_MAP[value]
            merged["intent"][key] = enabled
            accepted_intents.append(value)

        fields = list(merged["intent"].get("fields") or [])
        for flag, names in (
            ("risk_question", ("contraindications", "warnings", "adverse_reactions")),
            ("interaction_question", ("drug_interactions", "warnings")),
            ("adverse_event_question", ("adverse_reactions", "warnings")),
            ("drug_discovery_question", ("indication",)),
        ):
            if merged["intent"].get(flag):
                fields.extend(names)
        merged["intent"]["fields"] = list(dict.fromkeys(fields))

        accepted_context, accepted_medicines = [], []
        rejected = 0
        existing_context = {(x.get("surface"), x.get("kind"))
                            for x in merged["patient_context"] if isinstance(x, dict)}
        existing_medicines = {(x.get("surface"), x.get("catalog_kind"))
                              for x in merged["llm_medicine_mentions"] if isinstance(x, dict)}
        entities = payload.get("entities")
        for item in (entities[:12] if isinstance(entities, list) else []):
            if not isinstance(item, dict):
                rejected += 1
                continue
            surface, kind = item.get("surface"), item.get("type")
            if not isinstance(surface, str) or not surface.strip() or surface not in question:
                rejected += 1
                continue
            if kind not in _ENTITY_CATALOG_KIND:
                rejected += 1
                continue
            proposed = item.get("canonical")
            proposed = proposed.strip() if isinstance(proposed, str) else ""
            if kind in ("drug", "ingredient"):
                linked = _linked_medication(surface, proposed, kind, catalog)
                if linked is None:
                    # An unrecognized *short named substance* must not be
                    # silently discarded from a multi-drug interaction query.
                    # Long patient descriptions are not treated as drug names.
                    if (len(surface) <= 20 and
                            not re.search(r"患者|病人|患有|感染|发烧|症状|长期服用", surface) and
                            _normalized_spelling(surface) == _normalized_spelling(proposed or surface)):
                        candidate = {"surface": surface, "type": kind}
                        if candidate not in merged["llm_unlinked_medication_mentions"]:
                            merged["llm_unlinked_medication_mentions"].append(candidate)
                    rejected += 1
                    continue
                identity = (linked["surface"], linked["catalog_kind"])
                if identity not in existing_medicines:
                    merged["llm_medicine_mentions"].append(linked)
                    existing_medicines.add(identity)
                    accepted_medicines.append(linked)
                continue

            if len(surface) > 40:
                rejected += 1
                continue
            preferred = CANONICAL_TERMS.get(surface)
            proposed_normalized = _CANONICAL_EQUIVALENTS.get(proposed, proposed)
            # Do not turn a free-form model rewrite into a confirmed diagnosis:
            # "牙周感染" and "牙周炎" are not automatically identical.
            if preferred:
                if proposed and proposed_normalized not in (preferred, surface):
                    rejected += 1
                    continue
                canonical = preferred
            elif not proposed or _normalized_spelling(proposed) == _normalized_spelling(surface):
                canonical = surface
            else:
                rejected += 1
                continue
            patient_kind = _PATIENT_KIND[kind]
            identity = (surface, patient_kind)
            if identity in existing_context:
                continue
            terms = [surface, canonical]
            if proposed and proposed != canonical and proposed_normalized == canonical:
                terms.append(proposed)
            linked = {
                "surface": surface, "kind": patient_kind,
                "retrieval_terms": list(dict.fromkeys(terms)),
                "assertion": context_assertion(question, question.find(surface)),
                "diagnosis_asserted": False,
                "extraction_source": "llm_query_understanding",
            }
            merged["patient_context"].append(linked)
            existing_context.add(identity)
            accepted_context.append({
                "type": kind, "surface": surface, "canonical": canonical,
            })

        merged["llm_query_understanding"] = {
            "status": "ok", "intents": accepted_intents,
            "entities": accepted_context,
            "medication_mentions": accepted_medicines,
            "unlinked_medication_mentions": merged["llm_unlinked_medication_mentions"],
        }
        return merged, {
            "status": "ok", "model_calls": 1,
            "accepted_intents": accepted_intents,
            "accepted_entities": accepted_context,
            "accepted_medication_mentions": accepted_medicines,
            "unlinked_medication_mentions": merged["llm_unlinked_medication_mentions"],
            "rejected_entity_count": rejected,
        }
    except Exception as exc:
        # API failure must not silently turn an unverified LLM guess into a
        # drug-identity assertion. Keep the deterministic baseline unchanged.
        return understanding, {
            "status": "error", "model_calls": 1,
            "error_type": type(exc).__name__, "message": str(exc)[:240],
        }
