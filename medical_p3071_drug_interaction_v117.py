# -*- coding: utf-8 -*-
"""P3-07.1: bounded, read-only drug–drug interaction EVIDENCE preflight.

Never creates drug-class or drug–drug interaction graph edges.
Never treats a source excerpt as an independently reviewed clinical fact.
First supported benchmark pair: warfarin / ibuprofen (both orders).
The references are OFFICIAL URLs verified when this release was authored,
but are NOT live-fetched/revalidated by this local runtime.
"""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

STAGE="P3-07.1"
ACTIVATE="MEDICAL_P3071_INTERACTION_ENABLED"
STRICT_PAIR=("warfarin","ibuprofen")
PAIR_META={
    "warfarin":{
        "zh":"华法林",
        "aliases":("华法林","warfarin"),
        "class_name":"香豆素类维生素K拮抗剂抗凝药",
    },
    "ibuprofen":{
        "zh":"布洛芬",
        "aliases":("布洛芬","ibuprofen"),
        "class_name":"非甾体抗炎药（NSAID）",
    },
}
PRIORITY_FIELDS=("drug_interactions","warnings","contraindications")
SECTION_MAX_CHARS=20000
EXCERPT_CHARS=450
PER_DRUG_LIMIT=6
MAX_RECORDS=12
MAX_QUERY_SECONDS=6.0

OFFICIAL_LABEL_REFERENCES=(
    {
        "id":"NLM_DAILYMED_WARFARIN",
        "title":"DailyMed · Warfarin Sodium · Drug Interactions (7.3)",
        "url":"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=541c9a70-adaf-4ef3-94ba-ad4e70dfa057",
        "pair_mentioned_directly":True,
        "source_type":"official_US_prescribing_information_reference",
        "support":"Official label classifies ibuprofen under NSAIDs associated with increased bleeding risk during warfarin use.",
        "live_runtime_verified":False,
    },
    {
        "id":"NLM_DAILYMED_IBUPROFEN",
        "title":"DailyMed · Ibuprofen · Warfarin-type Anticoagulants",
        "url":"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=404df13d-4ada-4324-9eed-e21f5d5043a4",
        "pair_mentioned_directly":True,
        "source_type":"official_US_prescribing_information_reference",
        "support":"Official label describes an increased serious GI bleeding risk for warfarin plus NSAIDs.",
        "live_runtime_verified":False,
    },
)
# Each path is EXACTLY Drug -HAS_RECORD-> SourceRecord -HAS_SECTION-> Section.
# We do not claim Section->DrugClass is a Neo4j relationship.
CYPHER_SECTION_QUERY="""
MATCH (d:Drug)-[:HAS_RECORD]->(r:SourceRecord)
      -[:HAS_SECTION]->(s:Section)
WHERE d.name IS NOT NULL
  AND d.name CONTAINS $ingredient
  AND toLower(coalesce(s.field,'')) IN $fields
  AND s.text IS NOT NULL AND trim(s.text) <> ''
  AND size(s.text) <= $max_section_chars
RETURN d.id AS drug_id, d.name AS drug_name,
       r.id AS record_id, r.source_url AS source_url,
       r.authority_status AS authority_status,
       s.id AS section_id, s.field AS section_field,
       s.text AS section_text
ORDER BY CASE toLower(s.field)
         WHEN 'drug_interactions' THEN 0
         WHEN 'warnings' THEN 1
         ELSE 2 END, d.name, r.id, s.id
LIMIT $row_limit
""".strip()


class InteractionBlocked(RuntimeError):
    pass


def _normalize_question(question:str) -> str:
    if not isinstance(question,str):
        return ""
    return " ".join(question.split())[:500]


def _has_alias(question:str,alias:str) -> bool:
    if all(ord(x)<128 for x in alias):
        # Whole English ingredient name, not a substring of another token.
        return re.search(
            r"(?<![a-z])"+re.escape(alias.lower())+r"(?![a-z])",
            question.lower()
        ) is not None
    return alias in question


def recognize_pair(question:str) -> dict:
    """Explicit, allowlisted ingredient pair only. No fuzzy medical synonyms."""
    q=_normalize_question(question)
    detected=[
        key for key,info in PAIR_META.items()
        if any(_has_alias(q,a) for a in info["aliases"])
    ]
    if len(detected)!=2 or set(detected)!=set(STRICT_PAIR):
        return {
            "status":"OUT_OF_SCOPE_PAIR",
            "detected_keys":detected,
            "recognized_pair":False,
            "pair_display":None,
        }
    return {
        "status":"EXACT_TWO_INGREDIENTS",
        "detected_keys":detected,
        "recognized_pair":True,
        "pair_display":"华法林 × 布洛芬",
        "request_is_safety_or_interaction":bool(re.search(
            r"能否|能不能|可不可以|可否|一起|同服|同时|合用|联用|"
            r"服用|吃|相互作用|冲突|禁忌|风险|安全|是否",q
        )),
    }


def _http_url(text:Any) -> str:
    if not isinstance(text,str) or len(text)>2048:
        return ""
    parsed=urlparse(text.strip())
    return text.strip() if (
        parsed.scheme in ("http","https") and parsed.netloc
    ) else ""


def _match_token(text:str,tokens:tuple[str,...]) -> str|None:
    lower=text.lower()
    for item in tokens:
        if item.lower() in lower:
            return item
    return None


def classify_leaflet_text(text:str,ingredient_key:str) -> dict:
    """Ground literal text mentions. No clinical inference from weak keywords."""
    if ingredient_key not in STRICT_PAIR:
        raise ValueError("Ingredient not allowlisted for this research pilot")
    if not isinstance(text,str) or not text:
        return {"status":"EMPTY_SECTION","match":None,"risk_terms":[]}
    other = "华法林" if ingredient_key=="ibuprofen" else "布洛芬"
    english_other = "warfarin" if ingredient_key=="ibuprofen" else "ibuprofen"
    if ingredient_key=="ibuprofen":
        classes=("抗凝药","香豆素","双香豆素","抗凝剂","anticoagulant","coumarin")
    else:
        classes=("非甾体抗炎","非甾体消炎","NSAID","nsaid","抗炎镇痛","nonsteroidal")
    risk_terms=tuple(
        t for t in ("出血","凝血酶原","胃肠道出血","凝血","血小板",
                    "bleeding","hemorrhage","haemorrhage",
                    "prothrombin")
        if t.lower() in text.lower()
    )
    direct=_match_token(text,(other,english_other))
    class_term=_match_token(text,classes)
    if direct and risk_terms:
        status="DIRECT_OTHER_DRUG_PLUS_RISK_IN_LEAFLET"
    elif class_term and risk_terms:
        status="DRUG_CLASS_PLUS_RISK_IN_LEAFLET"
    elif direct:
        status="OTHER_DRUG_MENTION_WITHOUT_EXPLICIT_RISK"
    elif class_term:
        status="DRUG_CLASS_MENTION_WITHOUT_EXPLICIT_RISK"
    else:
        status="NO_PAIR_OR_CLASS_MENTION"
    match=direct or class_term
    return {
        "status":status,
        "match":match,
        "other_drug_directly_mentioned":bool(direct),
        "class_mentioned":bool(class_term),
        "risk_terms":list(risk_terms)[:8],
        "is_medical_truth_verified":False,
    }


def _literal_quote(full_text:str,anchor:str|None) -> tuple[str,int]:
    """Preserve an EXACT substring from source; no model paraphrase."""
    if not full_text:
        return "",-1
    pos=full_text.lower().find(anchor.lower()) if anchor else -1
    if pos<0:
        return full_text[:EXCERPT_CHARS],0
    a=max(0,pos-175)
    b=min(len(full_text),pos+275)
    return full_text[a:b],a


def _section_to_evidence(row:dict,ingredient:str) -> dict|None:
    if not isinstance(row,dict):
        return None
    info=PAIR_META[ingredient]
    drug=str(row.get("drug_name") or "").strip()
    # Database names are formulations; reject other ingredient names.
    if not drug or info["zh"] not in drug:
        return None
    field=str(row.get("section_field") or "").strip().lower()
    if field not in PRIORITY_FIELDS:
        return None
    full_text=row.get("section_text")
    if not isinstance(full_text,str) or not 0<len(full_text)<=SECTION_MAX_CHARS:
        return None
    ids=[str(row.get(k) or "").strip() for k in
         ("drug_id","record_id","section_id")]
    if not all(ids) or len(set(ids))!=3:
        return None
    cls=classify_leaflet_text(full_text,ingredient)
    quote,offset=_literal_quote(full_text,cls["match"])
    assert quote in full_text  # Guaranteed literal source grounding.
    return {
        "ingredient":ingredient,
        "drug_name":drug,
        "source_status":"third_party_unverified",
        "source_type":"local_neo4j_section",
        "clinical_verified":False,
        "node_labels":["Drug","SourceRecord","Section"],
        "node_ids":ids,
        "relationship_types":["HAS_RECORD","HAS_SECTION"],
        "neo4j_hops":2,
        "section_field":field,
        "section_sha256":hashlib.sha256(
            full_text.encode("utf-8")
        ).hexdigest(),
        "section_length":len(full_text),
        "quote":quote,
        "quote_start":offset,
        "classification":cls,
        "source_url":_http_url(row.get("source_url")),
        "authority_status":str(row.get("authority_status") or "unknown"),
        "graph_linked_path_is_drug_drug_interaction":False,
        "created_new_graph_edges":False,
    }


def _safe_result_status(evidence:list[dict]) -> str:
    if any(x["classification"]["status"]=="DIRECT_OTHER_DRUG_PLUS_RISK_IN_LEAFLET"
           for x in evidence):
        return "LOCAL_DIRECT_PAIR_RISK_MENTION_FOUND"
    if any(x["classification"]["status"]=="DRUG_CLASS_PLUS_RISK_IN_LEAFLET"
           for x in evidence):
        return "LOCAL_CLASS_RISK_MENTION_FOUND"
    if any(x["classification"]["other_drug_directly_mentioned"] or
           x["classification"]["class_mentioned"] for x in evidence):
        return "LOCAL_MENTION_WITHOUT_EXPLICIT_RISK"
    if evidence:
        return "LOCAL_SECTIONS_FOUND_NO_PAIR_EVIDENCE"
    return "NO_LOCAL_GRAPH_SECTIONS"


def _default_provider(*,ingredient:str,limit:int,root:Path) -> list[dict]:
    """Neo4j Query objects use Session.run, never tx.run (P3-06.1A fix)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(root/".env",override=False)
    except ImportError:
        pass
    from neo4j import GraphDatabase, Query, READ_ACCESS
    uri=os.getenv("NEO4J_URI","").strip()
    pw=os.getenv("NEO4J_PASSWORD","")
    user=os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
    db=os.getenv("NEO4J_DATABASE","neo4j")
    if not uri or not pw:
        raise InteractionBlocked("缺少 NEO4J_URI 或 NEO4J_PASSWORD")
    with GraphDatabase.driver(
        uri,auth=(user,pw),connection_timeout=5.0
    ) as driver:
        with driver.session(
            database=db,default_access_mode=READ_ACCESS
        ) as session:
            return [
                row.data() for row in session.run(
                    Query(CYPHER_SECTION_QUERY,timeout=MAX_QUERY_SECONDS),
                    ingredient=PAIR_META[ingredient]["zh"],
                    fields=list(PRIORITY_FIELDS),
                    max_section_chars=SECTION_MAX_CHARS,
                    row_limit=limit,
                )
            ]


def inspect_interaction(
    question:str,*,root:Path|str=".",provider=None,
    max_per_drug:int=PER_DRUG_LIMIT,
) -> dict:
    """Explicit, bounded 2-drug study. Does NOT modify any DB or model answer."""
    if not isinstance(max_per_drug,int) or not 1<=max_per_drug<=6:
        raise InteractionBlocked("max_per_drug 必须为 1～6")
    root=Path(root).resolve()
    parsed=recognize_pair(question)
    base={
        "stage":STAGE,
        "created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question":_normalize_question(question),
        "pair":parsed,
        "status":"OUT_OF_SCOPE_PAIR",
        "research_only":True,
        "clinical_review_performed":False,
        "source_rights_verified":False,
        "graph_writes":0,
        "chroma_writes":0,
        "graph_paths_are_drug_drug_edges":False,
        "official_reference_live_verified":False,
        "local_evidence":[],
        "official_references":[],
        "evidence_chain":[],
        "query_count":0,
        "notes":[
            "药物分类及相互作用的跨来源连接不是 Neo4j 实体关系边；"
            "不能把来源 Section 的两跳技术路径称为药物相互作用两跳关系。"
        ],
    }
    if not parsed["recognized_pair"]:
        return base
    rows=[]
    errors=[]
    for ingredient in STRICT_PAIR:
        try:
            got=(
                provider(ingredient=ingredient,limit=max_per_drug,root=root)
                if provider is not None
                else _default_provider(
                    ingredient=ingredient,limit=max_per_drug,root=root
                )
            )
            if not isinstance(got,list) or len(got)>max_per_drug:
                raise InteractionBlocked("Neo4j 返回行数超过受控预算")
            base["query_count"]+=1
            for row in got:
                processed=_section_to_evidence(row,ingredient)
                if processed is not None:
                    rows.append(processed)
        except Exception as exc:
            # No fabricated graph paths when Neo4j is offline; official
            # URLs can still be cited as non-live-validated references.
            errors.append({
                "ingredient":ingredient,
                "error_type":type(exc).__name__,
                "message":str(exc)[:300],
            })
    keys=set()
    unique=[]
    for item in rows:
        key=(item["ingredient"],tuple(item["node_ids"]))
        if key not in keys:
            keys.add(key)
            unique.append(item)
    base["local_evidence"]=unique[:MAX_RECORDS]
    base["official_references"]=[dict(x) for x in OFFICIAL_LABEL_REFERENCES]
    base["status"]=_safe_result_status(unique)
    if errors:
        base["status"]="PARTIAL_GRAPH_FAILURE" if unique else "GRAPH_UNAVAILABLE"
        base["graph_errors"]=errors
    base["evidence_chain"]=[
        {
            "step":1,
            "kind":"graph_provenance_only",
            "description":"药品制剂记录 -> 来源记录 -> 原文章节",
            "graph_edge_types":["HAS_RECORD","HAS_SECTION"],
            "actual_graph_hops":2,
            "grounded_records":len(unique),
        },
        {
            "step":2,
            "kind":"literal_local_text_check_not_graph_edge",
            "description":"筛选原文中的另一药品名、抗凝药/NSAID 类别与出血等字面风险词",
            "class_risk_support_count":sum(
                x["classification"]["status"]=="DRUG_CLASS_PLUS_RISK_IN_LEAFLET"
                for x in unique
            ),
            "direct_pair_risk_support_count":sum(
                x["classification"]["status"]=="DIRECT_OTHER_DRUG_PLUS_RISK_IN_LEAFLET"
                for x in unique
            ),
            "actual_graph_hops":0,
        },
        {
            "step":3,
            "kind":"external_official_label_reference_not_runtime_fetched",
            "description":"官方标签作为跨来源直接药物对照引用指针；本地脚本不联网验证最新版",
            "reference_ids":[r["id"] for r in OFFICIAL_LABEL_REFERENCES],
            "actual_graph_hops":0,
        },
    ]
    base["research_interpretation"]={
        "candidate_pair":"华法林 × 布洛芬",
        "risk_signal":"出血风险（官方标签引用指针）",
        "scope":"research_evidence_reconciliation",
        "independent_clinical_ground_truth_established_by_this_program":False,
        "drug_drug_graph_relation_exists":False,
        "model_answer_changed":False,
        "patient_specific_recommendation":False,
        "requires_up_to_date_label_review":True,
    }
    return base
