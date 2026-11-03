# -*- coding: utf-8 -*-
"""豆馅医疗 GraphRAG V1.17 · P3-06.1 5~8 hop *real graph edge* retrieval.

BOTH Neo4j and Chroma are READ ONLY. Traversals are fixed, typed,
parameterized relationship patterns, never user-generated Cypher.

5 hops: DiseaseDocument <- DiseaseTerm <- DiseaseCandidate <- Drug
        -> SourceRecord -> Section (requires candidate's SAME record/Section)
6 hops: DiseaseDocument <- Term <- Candidate <- Drug -> Candidate -> Term
        -> DiseaseDocument (shared DRUG MENTION, not treatment)
7 hops: DiseaseDocument <- Term <- Candidate <- Drug -> Candidate -> Term
        <- Candidate <- Drug (two drug sources mentioning a shared term)
8 hops: DiseaseDocument <- Term <- Candidate <- Drug -> SourceRecord
        -> Section <- Candidate -> Term -> DiseaseDocument
        (both candidates source from the SAME Section, not treatment)

All returned Neo4j linked document props must equal P3-04.4B's audited
exact-name plan, and all related source OIDs/line hashes must still match
medical.json. Chroma original text is validated by P3-04.3/P3-04.5.
Graph Section text has only candidate-quote consistency, NOT independently
clinically/legally verified.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

from medical_p3041_medical_json_prepare_v117 import SOURCE_STATUS
from medical_p3043_disease_encyclopedia_runtime_v117 import (
    load_source_index,match_disease_names,question_route,requested_fields,
    search_disease_encyclopedia,
)
from medical_p3045_graph_disease_retrieval_v117 import (
    LINK_TYPE,_expected_graph_properties,_same_values,
    validate_graph_inputs,search_graph_enhanced_disease_encyclopedia,
)
from drug_graph_retriever import _http_url

STAGE="P3-06.1"
ACTIVATE="MEDICAL_P3061_MULTIHOP_ENABLED"
DEFAULT_MIN_HOPS=5
DEFAULT_MAX_HOPS=8
MAX_PATHS=24
MAX_PER_TEMPLATE=6
MAX_DISEASE_ANCHORS=2
MAX_NEO4J_SECONDS=6.0
ALLOWED_SECTION_FIELDS=frozenset((
    "indication","contraindications","warnings","adverse_reactions",
    "drug_interactions","dosage","composition"
))
FIELD_NAMES={
    "indication":"适应症",
    "contraindications":"禁忌",
    "warnings":"注意事项",
    "adverse_reactions":"不良反应",
    "drug_interactions":"药物相互作用",
    "dosage":"用法用量",
    "composition":"成分",
}
PATH_TYPES={
    5:(
        "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
        "MENTIONS_TERM","HAS_CANDIDATE","HAS_RECORD","HAS_SECTION"
    ),
    6:(
        "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
        "MENTIONS_TERM","HAS_CANDIDATE","HAS_CANDIDATE",
        "MENTIONS_TERM","EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA"
    ),
    7:(
        "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
        "MENTIONS_TERM","HAS_CANDIDATE","HAS_CANDIDATE",
        "MENTIONS_TERM","MENTIONS_TERM","HAS_CANDIDATE"
    ),
    8:(
        "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
        "MENTIONS_TERM","HAS_CANDIDATE","HAS_RECORD","HAS_SECTION",
        "SUPPORTED_BY","MENTIONS_TERM",
        "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA"
    ),
}
PATH_LABELS={
    5:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
       "SourceRecord","Section"),
    6:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
       "DiseaseCandidate","DiseaseTerm","DiseaseDocument"),
    7:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
       "DiseaseCandidate","DiseaseTerm","DiseaseCandidate","Drug"),
    8:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
       "SourceRecord","Section","DiseaseCandidate","DiseaseTerm",
       "DiseaseDocument"),
}

# These are fixed implementation-owned queries. Never interpolate question,
# entity names, paths, relationship type strings or LLM output into Cypher.
_CYPHER_HEAD=f"""
MATCH p=(a:DiseaseDocument {{id:$doc_id}})
       <-[la:{LINK_TYPE}]-(ta:DiseaseTerm {{id:$term_id}})
       <-[:MENTIONS_TERM]-(ca:DiseaseCandidate)
       <-[:HAS_CANDIDATE]-(da:Drug)
"""
_COMMON_PROJECTION="""
RETURN
  [n IN nodes(p) | coalesce(n.id,'')] AS node_ids,
  [n IN nodes(p) | labels(n)] AS node_labels,
  [rel IN relationships(p) | type(rel)] AS relation_types,
  a.id AS doc_anchor_id, properties(a) AS doc_anchor_props,
  properties(la) AS link_anchor_props,
  ta.id AS term_anchor_id, ta.name AS name_anchor,
  ca.id AS candidate_a_id, ca.evidence_quote AS candidate_a_quote,
  ca.candidate_status AS candidate_a_status,
  ca.clinical_status AS candidate_a_clinical_status,
  ca.source_status AS candidate_a_source_status,
  da.id AS drug_a_id, da.name AS drug_a_name,
  {extra}
ORDER BY drug_a_name, candidate_a_id, coalesce(record_id, ''),
         coalesce(doc_related_id, '')
LIMIT $row_limit
"""
QUERY_5=(
    _CYPHER_HEAD
    + """
       -[:HAS_RECORD]->(rec:SourceRecord)
       -[:HAS_SECTION]->(sec:Section)
MATCH (ca)-[:FROM_RECORD]->(rec)
MATCH (ca)-[:SUPPORTED_BY]->(sec)
WHERE toLower(coalesce(sec.field,'')) IN $fields
  AND sec.text IS NOT NULL AND trim(sec.text) <> ''
  AND size(sec.text) <= 100000
"""
    + _COMMON_PROJECTION.format(extra="""
  null AS doc_related_id,null AS doc_related_props,
  null AS link_related_props,null AS term_related_id,
  null AS name_related,
  null AS candidate_b_id,null AS candidate_b_quote,
  null AS candidate_c_id,
  null AS drug_b_id,null AS drug_b_name,
  rec.id AS record_id, rec.source_url AS record_source_url,
  rec.authority_status AS record_authority_status,
  sec.id AS section_id, sec.field AS section_field,
  sec.text AS section_text
""")
)
QUERY_6=(
    _CYPHER_HEAD
    + f"""
       -[:HAS_CANDIDATE]->(cb:DiseaseCandidate)
       -[:MENTIONS_TERM]->(tb:DiseaseTerm)
       -[lb:{LINK_TYPE}]->(b:DiseaseDocument)
WHERE ca.id <> cb.id AND ta.id <> tb.id AND a.id <> b.id
"""
    + _COMMON_PROJECTION.format(extra="""
  b.id AS doc_related_id, properties(b) AS doc_related_props,
  properties(lb) AS link_related_props,
  tb.id AS term_related_id, tb.name AS name_related,
  cb.id AS candidate_b_id, cb.evidence_quote AS candidate_b_quote,
  null AS candidate_c_id,
  null AS drug_b_id,null AS drug_b_name,
  null AS record_id,null AS record_source_url,
  null AS record_authority_status,
  null AS section_id,null AS section_field,null AS section_text
""")
)
QUERY_7=(
    _CYPHER_HEAD
    + f"""
       -[:HAS_CANDIDATE]->(cb:DiseaseCandidate)
       -[:MENTIONS_TERM]->(tb:DiseaseTerm)
       <-[:MENTIONS_TERM]-(cc:DiseaseCandidate)
       <-[:HAS_CANDIDATE]-(db:Drug)
MATCH (tb)-[lb:{LINK_TYPE}]->(b:DiseaseDocument)
WHERE ca.id <> cb.id AND cb.id <> cc.id AND ca.id <> cc.id
  AND ta.id <> tb.id AND da.id <> db.id AND a.id <> b.id
"""
    + _COMMON_PROJECTION.format(extra="""
  b.id AS doc_related_id, properties(b) AS doc_related_props,
  properties(lb) AS link_related_props,
  tb.id AS term_related_id, tb.name AS name_related,
  cb.id AS candidate_b_id, cb.evidence_quote AS candidate_b_quote,
  cc.id AS candidate_c_id,
  db.id AS drug_b_id, db.name AS drug_b_name,
  null AS record_id,null AS record_source_url,
  null AS record_authority_status,
  null AS section_id,null AS section_field,null AS section_text
""")
)
QUERY_8=(
    _CYPHER_HEAD
    + f"""
       -[:HAS_RECORD]->(rec:SourceRecord)
       -[:HAS_SECTION]->(sec:Section)
       <-[:SUPPORTED_BY]-(cb:DiseaseCandidate)
       -[:MENTIONS_TERM]->(tb:DiseaseTerm)
       -[lb:{LINK_TYPE}]->(b:DiseaseDocument)
MATCH (ca)-[:FROM_RECORD]->(rec)
MATCH (ca)-[:SUPPORTED_BY]->(sec)
MATCH (cb)-[:FROM_RECORD]->(rec)
MATCH (da)-[:HAS_CANDIDATE]->(cb)
WHERE toLower(coalesce(sec.field,'')) IN $fields
  AND sec.text IS NOT NULL AND trim(sec.text) <> ''
  AND size(sec.text) <= 100000
  AND ca.id <> cb.id AND ta.id <> tb.id AND a.id <> b.id
"""
    + _COMMON_PROJECTION.format(extra="""
  b.id AS doc_related_id, properties(b) AS doc_related_props,
  properties(lb) AS link_related_props,
  tb.id AS term_related_id,tb.name AS name_related,
  cb.id AS candidate_b_id,cb.evidence_quote AS candidate_b_quote,
  null AS candidate_c_id,
  null AS drug_b_id,null AS drug_b_name,
  rec.id AS record_id,rec.source_url AS record_source_url,
  rec.authority_status AS record_authority_status,
  sec.id AS section_id,sec.field AS section_field,
  sec.text AS section_text
""")
)
CYPHER={5:QUERY_5,6:QUERY_6,7:QUERY_7,8:QUERY_8}


class MultiHopBlocked(RuntimeError):
    pass


def require(ok:bool,msg:str):
    if not ok:raise MultiHopBlocked(msg)


def enabled() -> bool:
    return os.getenv(ACTIVATE,"0").lower().strip() in (
        "1","true","yes","on"
    )


def _check_budget(min_hops:int,max_hops:int,max_paths:int,
                  per_template:int) -> None:
    require(
        type(min_hops) is int and type(max_hops) is int
        and 5<=min_hops<=max_hops<=8,
        "P3-06.1 仅支持已存在真实 schema 的 5～8 跳固定模板"
    )
    require(type(max_paths) is int and 1<=max_paths<=MAX_PATHS,
            "max_paths 必须位于 1～24")
    require(type(per_template) is int and 1<=per_template<=MAX_PER_TEMPLATE,
            "per_template 必须位于 1～6")


def _lookup_graph(hops:int,name:str,term_id:str,doc_id:str,
                  per_template:int,provider:Callable[...,list[dict]]|None,
                  project_root:Path) -> list[dict]:
    require(hops in CYPHER,"拒绝未知/动态 Cypher 跳数模板")
    if provider is not None:
        values=provider(
            hops=hops,disease_name=name,
            term_id=term_id,doc_id=doc_id,
            row_limit=per_template
        )
        require(isinstance(values,list) and len(values)<=per_template,
                "模拟/检索服务返回的多跳结果条数超限")
        return values
    from dotenv import load_dotenv
    from neo4j import GraphDatabase,Query,READ_ACCESS
    load_dotenv(project_root/".env",override=False)
    uri=os.getenv("NEO4J_URI","").strip()
    password=os.getenv("NEO4J_PASSWORD","")
    user=os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
    database=os.getenv("NEO4J_DATABASE","neo4j")
    require(bool(uri and password),"Neo4j 连接参数缺失")
    driver=GraphDatabase.driver(
        uri,auth=(user,password),connection_timeout=5
    )
    try:
        with driver.session(
            database=database,default_access_mode=READ_ACCESS,
        ) as session:
            # Neo4j driver contract: Query(metadata/timeout) can be passed
            # to Session.run(), but NOT to tx.run() inside execute_read().
            # Keep the session READ_ACCESS and query timeout, and fully
            # consume results before closing the session.
            result=[
                record.data() for record in session.run(
                    Query(CYPHER[hops],timeout=MAX_NEO4J_SECONDS),
                    doc_id=doc_id,term_id=term_id,
                    fields=sorted(ALLOWED_SECTION_FIELDS),
                    row_limit=per_template
                )
            ]
            require(len(result)<=per_template,"Neo4j 多跳返回条数超过单模板上限")
            return result
    finally:
        driver.close()


def _compact(value:Any) -> str:
    return " ".join(value.split()) if isinstance(value,str) else ""


def _validate_doc(document_id:str, term_id:str, name:str,
                  doc_props:dict,link_props:dict,
                  plan_index:dict,source_index:dict,plan_sha:str) -> dict:
    candidate=plan_index.get(name)
    require(candidate is not None,
            "多跳返回的 DiseaseTerm 名称没有严格同名计划依据")
    oid=candidate["source_record_oid"]
    source=source_index["by_oid"].get(oid)
    require(source is not None
            and source["name"]==name
            and source["line"]==candidate["source_line_number"]
            and source["line_sha"]==candidate["source_line_sha256"],
            "多跳疾病文档与 medical.json 原记录 OID/哈希不一致")
    expected_doc,expected_link=_expected_graph_properties(
        candidate,plan_sha
    )
    require(
        document_id==candidate["document_id"]
        and term_id==candidate["disease_term_id"]
        and _same_values(doc_props,expected_doc)
        and _same_values(link_props,expected_link),
        "多跳疾病文档节点或关系不等于 P3-04.4B 未核验来源基线"
    )
    return candidate


def _validate_path(hops:int,row:dict,
                   anchor:dict,plan_index:dict,
                   source_index:dict,plan_sha:str) -> dict:
    require(isinstance(row,dict),"Neo4j 多跳结果格式错误")
    ids=row.get("node_ids")
    labels=row.get("node_labels")
    relations=row.get("relation_types")
    require(
        isinstance(ids,list) and len(ids)==hops+1
        and all(isinstance(x,str) and x for x in ids)
        and len(set(ids))==len(ids),
        "多跳路径存在重复节点/无 ID/非法深度"
    )
    require(
        isinstance(labels,list) and len(labels)==hops+1
        and all(isinstance(ls,list) and PATH_LABELS[hops][i] in ls
                for i,ls in enumerate(labels)),
        "Neo4j 返回节点 Label 顺序不符"
    )
    require(
        isinstance(relations,list)
        and tuple(relations)==PATH_TYPES[hops],
        "Neo4j 多跳关系类型或边数不符合白名单"
    )
    require(ids[0]==anchor["document_id"]
            and ids[1]==anchor["disease_term_id"]
            and row.get("doc_anchor_id")==ids[0]
            and row.get("term_anchor_id")==ids[1]
            and row.get("name_anchor")==anchor["disease_name"]
            and ids[2]==row.get("candidate_a_id")
            and ids[3]==row.get("drug_a_id"),
            "多跳根节点或主药品来源实体身份不符")
    _validate_doc(
        row["doc_anchor_id"],row["term_anchor_id"],
        row["name_anchor"],row.get("doc_anchor_props"),
        row.get("link_anchor_props"),plan_index,source_index,plan_sha
    )
    require(
        isinstance(row.get("candidate_a_quote"),str)
        and bool(row["candidate_a_quote"].strip())
        and isinstance(row.get("drug_a_name"),str)
        and bool(row["drug_a_name"].strip()),
        "DiseaseCandidate/Drug 缺少真实提及原文或药品名称"
    )
    if hops in (6,7,8):
        doc_b=row.get("doc_related_id")
        name_b=row.get("name_related")
        term_b=row.get("term_related_id")
        _validate_doc(
            doc_b,term_b,name_b,row.get("doc_related_props"),
            row.get("link_related_props"),
            plan_index,source_index,plan_sha
        )
        require(
            doc_b!=anchor["document_id"]
            and isinstance(name_b,str)
            and name_b!=anchor["disease_name"]
            and isinstance(row.get("candidate_b_quote"),str)
            and bool(row["candidate_b_quote"].strip()),
            "多跳目标疾病必须与起点不同且具有候选原文"
        )
    if hops==5:
        require(ids[4]==row.get("record_id")
                and ids[5]==row.get("section_id"),
                "5跳说明书 SourceRecord/Section 路径身份不一致")
    if hops==6:
        require(
            ids[4]==row.get("candidate_b_id")
            and ids[5]==row.get("term_related_id")
            and ids[6]==row.get("doc_related_id"),
            "6跳共现疾病端点不一致"
        )
    if hops==7:
        require(
            ids[4]==row.get("candidate_b_id")
            and ids[5]==row.get("term_related_id")
            and ids[6]==row.get("candidate_c_id")
            and ids[7]==row.get("drug_b_id")
            and isinstance(row.get("drug_b_name"),str)
            and bool(row["drug_b_name"].strip())
            and ids[3]!=ids[7],
            "7跳第二药品/桥接疾病/候选身份不符"
        )
    if hops==8:
        require(
            ids[4]==row.get("record_id")
            and ids[5]==row.get("section_id")
            and ids[6]==row.get("candidate_b_id")
            and ids[7]==row.get("term_related_id")
            and ids[8]==row.get("doc_related_id"),
            "8跳疾病和共同 SourceRecord/Section 端点不符"
        )
    if hops in (5,8):
        field=row.get("section_field")
        text=row.get("section_text")
        require(
            isinstance(field,str) and field.lower() in ALLOWED_SECTION_FIELDS
            and isinstance(text,str) and 0<len(text)<=100000
            and isinstance(row.get("record_id"),str)
            and isinstance(row.get("section_id"),str),
            "药品原文 Section 缺少有效字段、ID 或文本超长"
        )
        normalized=_compact(text)
        require(_compact(row["candidate_a_quote"]) in normalized,
                "说明书 Section 无法逐字支持主疾病候选 quote，拒绝跨文档串联")
        if hops==8:
            require(_compact(row["candidate_b_quote"]) in normalized,
                    "8跳 Section 无法支持第二疾病候选 quote")
    # The code never treats this graph edge as a medically-approved use.
    disease_related=row.get("name_related")
    evidence={
        "hop_count":hops,
        "path_kind":{
            5:"disease_to_drug_original_section",
            6:"shared_drug_mentions_two_diseases",
            7:"bridge_through_second_disease_to_other_drug",
            8:"same_drug_section_mentions_two_diseases",
        }[hops],
        "node_ids":list(ids),
        "node_labels":[PATH_LABELS[hops][i] for i in range(hops+1)],
        "relationship_types":list(relations),
        "disease_name":anchor["disease_name"],
        "disease_document_id":anchor["document_id"],
        "disease_term_id":anchor["disease_term_id"],
        "related_disease_name":disease_related,
        "related_document_id":row.get("doc_related_id"),
        "drug_name":row["drug_a_name"],
        "related_drug_name":row.get("drug_b_name"),
        "candidate_id":row["candidate_a_id"],
        "candidate_quote":row["candidate_a_quote"][:1000],
        "candidate_status":str(row.get("candidate_a_status") or "unknown"),
        "candidate_clinical_status":str(
            row.get("candidate_a_clinical_status") or "unknown"
        ),
        "candidate_source_status":str(
            row.get("candidate_a_source_status") or "unknown"
        ),
        "related_candidate_id":row.get("candidate_b_id"),
        "record_id":row.get("record_id"),
        "section_id":row.get("section_id"),
        "section_field":row.get("section_field"),
        "section_text":row.get("section_text"),
        "record_source_url":_http_url(row.get("record_source_url")),
        "record_authority_status":str(
            row.get("record_authority_status") or "unknown"
        ),
        "source_status":SOURCE_STATUS,
        "clinical_verified":False,
        "treatment_relation":False,
        "graph_relationships_verified":True,
        "graph_section_quote_consistent":hops in (5,8),
        "clinical_truth_verified":False,
    }
    return evidence


def _path_id(path:dict) -> str:
    serial=json.dumps(
        {
            "hops":path["hop_count"],
            "ids":path["node_ids"],
            "relations":path["relationship_types"],
        },ensure_ascii=False,sort_keys=True,separators=(",",":")
    )
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()[:20]


def _path_ranking(p:dict) -> tuple:
    # In a source-evidence request, prioritize grounded original Section:
    # 8-hop is deeper but 5-hop has fewer unsupported co-mention assumptions.
    evidence_first=0 if p["section_id"] else 1
    return (
        evidence_first,
        -p["hop_count"],
        p["drug_name"],
        p["section_id"] or "",
        p["path_id"]
    )


def retrieve_multihop(
    question:str, *,
    project_root:Path,
    source_path:Path|None=None,
    graph_audit_path:Path|None=None,
    plan_path:Path|None=None,
    plan_report_path:Path|None=None,
    min_hops:int=DEFAULT_MIN_HOPS,
    max_hops:int=DEFAULT_MAX_HOPS,
    max_paths:int=12,
    per_template:int=4,
    path_provider:Callable[...,list[dict]]|None=None,
    require_enabled:bool=True,
) -> tuple[list[dict],dict]:
    """Read-only bounded 5/6/7/8-hop graph paths, not clinical inference."""
    start=time.perf_counter()
    _check_budget(min_hops,max_hops,max_paths,per_template)
    trace={
        "stage":STAGE,
        "status":"DISABLED",
        "requested_hops":[n for n in CYPHER if min_hops<=n<=max_hops],
        "query_count":0,
        "found_per_hops":{},
        "returned_paths":0,
        "path_count_before_cap":0,
        "neo4j_writes":0,
        "chroma_writes":0,
        "new_embedding_calls":0,
        "medical_claim_verified":False,
        "drug_treatment_relation_inferred":False,
        "source_status":SOURCE_STATUS,
    }
    if require_enabled and not enabled():
        return [],trace
    root=Path(project_root).resolve()
    source=(
        Path(source_path).resolve() if source_path else
        (root/"data/raw/drug_db/MedGraphRAG-Drug-DB/medical.json").resolve()
    )
    require(source.is_file(),"找不到原始 medical.json")
    from medical_p3041_medical_json_prepare_v117 import sha256_file
    from medical_p3045_graph_disease_retrieval_v117 import EXPECTED_SOURCE_SHA
    source_sha=sha256_file(source)
    require(source_sha==EXPECTED_SOURCE_SHA,"多跳原始疾病来源 SHA256 不匹配")
    plan_index,gate=validate_graph_inputs(
        root,source_sha,
        audit_path=graph_audit_path,
        plan_path=plan_path,
        plan_report_path=plan_report_path,
    )
    source_index=load_source_index(source)
    diseases,extra=match_disease_names(
        question,source_index["names"],limit=MAX_DISEASE_ANCHORS
    )
    anchors=[plan_index[n] for n in diseases if n in plan_index]
    trace["matched_diseases"]=diseases
    trace["unlinked_disease_names"]=[
        n for n in diseases if n not in plan_index
    ]
    trace["additional_disease_name_matches_omitted"]=extra
    trace["graph_plan_sha256"]=gate["plan_sha256"]
    trace["graph_audit_status"]=gate["graph_audit_status"]
    trace["graph_query_ran"]=False
    if not anchors:
        trace["status"]="NO_GRAPH_LINKED_DISEASE_IN_QUESTION"
        trace["elapsed_ms"]=round((time.perf_counter()-start)*1000,2)
        return [],trace
    all_paths={}
    for anchor in anchors:
        for hops in trace["requested_hops"]:
            remaining=max_paths-len(all_paths)
            if remaining<=0:break
            limit=min(remaining,per_template)
            trace["graph_query_ran"]=True
            rows=_lookup_graph(
                hops,anchor["disease_name"],anchor["disease_term_id"],
                anchor["document_id"],limit,path_provider,root
            )
            trace["query_count"]+=1
            for row in rows:
                item=_validate_path(
                    hops,row,anchor,plan_index,source_index,
                    gate["plan_sha256"]
                )
                item["path_id"]=_path_id(item)
                all_paths[item["path_id"]]=item
    ranked=sorted(all_paths.values(),key=_path_ranking)
    paths=ranked[:max_paths]
    trace.update({
        "status":"MULTIHOP_PATHS_VERIFIED" if paths else "NO_MULTIHOP_PATH_FOUND",
        "found_per_hops":dict(sorted(Counter(
            str(p["hop_count"]) for p in paths
        ).items())),
        "returned_paths":len(paths),
        "path_count_before_cap":len(all_paths),
        "max_hops_observed":max((x["hop_count"] for x in paths),default=0),
        "min_hops_observed":min((x["hop_count"] for x in paths),default=0),
        "elapsed_ms":round((time.perf_counter()-start)*1000,2),
    })
    return paths,trace


def _mentions_section_intent(question:str) -> bool:
    return (
        isinstance(question,str)
        and bool(re.search(r"说明书|药品资料|药物资料",question))
        and bool(re.search(r"提到|记载|涉及|提及|描述|出现|包含",question))
    )


def _path_section_excerpt(path:dict) -> dict:
    """Only a 5/8-hop Section with a source-exact candidate quote qualifies."""
    require(path.get("hop_count") in (5,8)
            and path.get("graph_section_quote_consistent") is True,
            "没有可直接引用的 5/8 跳原始 Section")
    field=path["section_field"].lower()
    quote=path["candidate_quote"].strip()
    clipped=path["section_text"][:1600]
    name=path["drug_name"]
    return {
        "id":0,
        "source":"Neo4j 药品资料 Section（未核验来源）",
        "source_url":path["record_source_url"],
        "section_path":(
            "Neo4j 多跳 / 药品说明书疾病提及 / "
            +FIELD_NAMES[field]+"（非治疗关系）"
        ),
        "content":(
            "【多跳图谱原文引用｜不是药物治疗建议】\n"
            f"药品资料：{name}\n"
            f"被提及疾病：{path['disease_name']}\n"
            f"说明书栏目：{FIELD_NAMES[field]}\n"
            f"候选证据片段：{quote[:500]}\n"
            f"Section 原文：{clipped}"
            +("……（原文超过 1600 字）" if len(path["section_text"])>1600 else "")
            +"\n重要：仅证明这份第三方药品资料提及该疾病名称，"
            "不代表治疗、预防、获批适应症或临床可信度。"
        ),
        "drug_name":name,
        "disease_term":path["disease_name"],
        "candidate_id":path["candidate_id"],
        "evidence_quote":path["candidate_quote"],
        "record_id":path["record_id"],
        "section_id":path["section_id"],
        "section_field":field,
        "evidence_type":"unverified_multihop_drug_section",
        "source_status":"third_party_unverified",
        "clinical_status":"not_verified",
        "clinical_verified":False,
        "rights_reviewed":False,
        "parent_verified":False,
        "disease_link_verified":False,
        "multi_hop_verified":True,
        "multi_hop_count":path["hop_count"],
        "multi_hop_path_id":path["path_id"],
        "multi_hop_claim":"drug_document_mentions_disease_only",
        "_retrieval_tool":"search_multihop_neo4j_disease_sources",
    }


def search_multihop_enhanced_disease_encyclopedia(
    question:str, *,project_root:Path,
    audit_path:Path|None=None,
    collection:Any=None,
    source_path:Path|None=None,
    max_documents:int=4,
    graph_audit_path:Path|None=None,
    plan_path:Path|None=None,
    plan_report_path:Path|None=None,
    path_provider:Callable[...,list[dict]]|None=None,
    min_hops:int=5,max_hops:int=8
) -> tuple[list[dict],dict]:
    """Drop-in P3-04.5 wrapper for rag.py; default P3-06.1 OFF.

    Graph path failures NEVER allow forged data into online KB. Existing
    P3-04.5/P3-04.3 evidence is retained as independent fallback.
    """
    base_kwargs={
        "project_root":project_root,
        "audit_path":audit_path,
        "collection":collection,
        "source_path":source_path,
        "max_documents":max_documents,
        "graph_audit_path":graph_audit_path,
        "plan_path":plan_path,
        "plan_report_path":plan_report_path,
    }
    base,trace=search_graph_enhanced_disease_encyclopedia(
        question,**base_kwargs
    )
    mh={
        "stage":STAGE,"status":"DISABLED",
        "returned_paths":0,"graph_query_ran":False,
        "hop_range":[min_hops,max_hops],
        "neo4j_writes":0,"chroma_writes":0,
        "clinical_truth_verified":False,
    }
    trace["multi_hop_graph"]=mh
    if not enabled():
        return base,trace
    if not _mentions_section_intent(question):
        mh["status"]="SKIPPED_NOT_SOURCE_MENTION_INTENT"
        return base,trace
    if not (1<=max_documents<=6):
        mh["status"]="BLOCKED_DOCUMENT_LIMIT"
        return base,trace
    route=question_route(question)
    if route not in ("DRUG_ONLY","MIXED"):
        mh["status"]="SKIPPED_ROUTE"
        return base,trace
    # The independent P3-04.5 graph bridge must be authenticated;
    # DRUG_ONLY intentionally had no P3-04.5 evidence, so its new
    # read-only multihop search independently requires the same graph audit.
    prior=trace.get("graph_disease_bridge",{})
    if route=="MIXED" and prior.get("status") not in (
        "GRAPH_EXACT_OID_PROVENANCE_VERIFIED",
        "GRAPH_PARTIAL_WITH_ENCYCLOPEDIA_FALLBACK",
    ):
        mh["status"]="SKIPPED_P3045_NOT_VERIFIED"
        return base,trace
    try:
        paths,path_trace=retrieve_multihop(
            question,project_root=project_root,
            source_path=source_path,
            graph_audit_path=graph_audit_path,
            plan_path=plan_path,
            plan_report_path=plan_report_path,
            min_hops=min_hops,max_hops=max_hops,
            max_paths=10,per_template=3,
            path_provider=path_provider,
            require_enabled=True
        )
        mh.update({
            k:v for k,v in path_trace.items()
            if k!="status"
        })
        mh["status"]=path_trace["status"]
        mh["verified_paths"]=[
            {
                "path_id":p["path_id"],
                "hop_count":p["hop_count"],
                "path_kind":p["path_kind"],
                "node_ids":p["node_ids"],
                "node_labels":p["node_labels"],
                "relationship_types":p["relationship_types"],
                "drug_name":p["drug_name"],
                "related_disease_name":p["related_disease_name"],
                "graph_section_quote_consistent":
                    p["graph_section_quote_consistent"],
                "treatment_relation":False,
            } for p in paths
        ]
        if not paths:return base,trace
        eligible=[p for p in paths
                  if p["hop_count"] in (5,8)
                  and p["graph_section_quote_consistent"]]
        if not eligible:
            mh["status"]="MULTIHOP_DISCOVERY_ONLY_NO_SECTION_EVIDENCE"
            return base,trace
        # One Section is enough to demonstrate actual multi-hop evidence.
        # Do not inject graph Section if an unrelated explicit drug is named.
        selected=None
        for path in eligible:
            if path["drug_name"] in question:
                selected=path
                break
        open_drug_list=bool(re.search(
            r"(哪些|什么|哪几种)(药品|药物|药)|药品(说明书)?有哪些",
            question
        ))
        if selected is None and open_drug_list:
            selected=eligible[0]
        if selected is None:
            mh["status"]="MULTIHOP_VERIFIED_NO_EXPLICIT_DRUG_MATCH"
            return base,trace
        excerpt=_path_section_excerpt(selected)
        # Preserve independent disease evidence; reserve one slot for
        # explicit source-mention question. The host still runs normal
        # drug safety & evidence coverage checks after merging.
        kept=base[:max(0,max_documents-1)]
        kept.append(excerpt)
        new=[{**x,"id":i+1} for i,x in enumerate(kept)]
        mh.update({
            "status":"MULTIHOP_SOURCE_SECTION_RETURNED",
            "returned_section_evidence":1,
            "selected_path_id":selected["path_id"],
            "selected_hops":selected["hop_count"],
            "clinical_truth_verified":False,
        })
        trace["returned_chunks"]=len(new)
        if route=="DRUG_ONLY":
            trace["status"]="MULTIHOP_DRUG_SECTION_QUOTE_CONSISTENT_UNVERIFIED"
        return new,trace
    except Exception as exc:
        mh["status"]="MULTIHOP_BLOCKED_P3045_FALLBACK"
        mh["reason_type"]=type(exc).__name__
        mh["returned_paths"]=0
        mh["graph_query_ran"]=True
        return base,trace
