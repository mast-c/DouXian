# -*- coding: utf-8 -*-
"""豆馅医疗 GraphRAG V1.17 / P3-04.5 只读疾病图谱->百科 Chunk 检索。

Only if explicitly enabled AND P3-04.4B fully audited:
  DiseaseTerm --EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA--> DiseaseDocument
  --> DiseaseDocument.source_record_oid --> Chroma disease_knowledge
  --> original medical.json chunk provenance exact verification.

No writes, no query embeddings, no trusted medical claims. If graph unready,
use P3-04.3 original encyclopedia search (not drug inference).
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

from medical_p3041_medical_json_prepare_v117 import sha256_file, SOURCE_STATUS
from medical_p3043_disease_encyclopedia_runtime_v117 import (
    _read_chroma_collection, _rank_chunks, _make_excerpt, DiseaseEvidenceBlocked,
    question_route, requested_fields, is_enabled, latest_audit,
    load_source_index, match_disease_names, validate_audit,
    validate_disease_chunk, search_disease_encyclopedia,
    MAX_DISEASE_MATCHES, MAX_RETRIEVED_PER_DISEASE, MAX_RETURN_DOCS,
)

STAGE = "P3-04.5"
LINK_TYPE = "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA"
GRAPH_AUDIT_GLOB = "p3044b_final_audit_*.json"
EXPECTED_LINKS = 1150
EXPECTED_SOURCE_SHA = (
    "01aaf5de1304ea6f4ed84e9802b600b729b9f797284234f2d86d77e55f8dfc49"
)
EXPECTED_STAGE_SHA = (
    "054b14946410cebff2a52f003a786440f1cde8a81c49b7b5570b41c5b88d16c0"
)
EXPECTED_PLAN_SHA = (
    "c2010f693a0d2d37e7efd930d973cdd87750ad08a72b43c20a6d25b2a26dc731"
)
MAX_CANDIDATE_LINE_BYTES = 1500000
REL_QUERY = f"""
MATCH (t:DiseaseTerm)-[r:{LINK_TYPE}]->(d:DiseaseDocument)
WHERE t.name IN $disease_names
RETURN t.id AS term_id, t.name AS term_name,
       d.id AS document_id,
       properties(d) AS document_properties,
       properties(r) AS relationship_properties
ORDER BY term_name, term_id, document_id
"""


class GraphRouteBlocked(RuntimeError):
    pass


def _require(ok:bool,message:str) -> None:
    if not ok:
        raise GraphRouteBlocked(message)


def _enabled() -> bool:
    return os.getenv("MEDICAL_P3045_GRAPH_ENABLED", "0").strip().lower() in (
        "1","true","yes","on"
    )


def _sha_bytes(raw:bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _same_values(left:dict,right:dict) -> bool:
    return (isinstance(left,dict) and isinstance(right,dict)
            and json.dumps(left,ensure_ascii=False,sort_keys=True,
                           separators=(",",":"))
            ==json.dumps(right,ensure_ascii=False,sort_keys=True,
                           separators=(",",":")))


def _path_from_env(root:Path, env:str, prefix:str, *, override:Path|None=None
                   ) -> Path|None:
    if override is not None:
        return Path(override).resolve()
    setting=os.getenv(env,"").strip()
    if setting:
        p=Path(setting)
        return (p if p.is_absolute() else root/p).resolve()
    folder=root/"stage4_reports"
    if not folder.is_dir():
        return None
    matched=list(folder.glob(prefix))
    if not matched:
        return None
    return max(matched,key=lambda p:(p.stat().st_mtime_ns,p.name)).resolve()


def _load_json(file:Path) -> dict:
    _require(file.is_file(),f"找不到验收文件：{file.name}")
    obj=json.loads(file.read_text(encoding="utf-8-sig"))
    _require(isinstance(obj,dict),f"验收文件不是 JSON 对象：{file.name}")
    return obj


def _manifest_paths(root:Path, *, audit_override:Path|None=None,
                    plan_override:Path|None=None,
                    plan_report_override:Path|None=None
                    ) -> tuple[Path|None,Path|None,Path|None]:
    audit=_path_from_env(
        root,"MEDICAL_P3045_GRAPH_AUDIT_PATH",
        GRAPH_AUDIT_GLOB,override=audit_override
    )
    report=_path_from_env(
        root,"MEDICAL_P3045_PLAN_REPORT_PATH",
        "p3044a_graph_plan_*.json",override=plan_report_override
    )
    plan=_path_from_env(
        root,"MEDICAL_P3045_PLAN_PATH",
        "p3044a_candidates_*.jsonl",override=plan_override
    )
    return audit,plan,report


def _expected_graph_properties(candidate:dict,plan_sha:str) -> tuple[dict,dict]:
    """Exact P3-04.4B metadata schema; no dependence on installer script."""
    doc={
        "id":candidate["document_id"],
        "disease_name":candidate["disease_name"],
        "source_record_oid":candidate["source_record_oid"],
        "source_line_number":candidate["source_line_number"],
        "source_line_sha256":candidate["source_line_sha256"],
        "source_file":"medical.json",
        "source_file_sha256":candidate["source_file_sha256"],
        "chroma_collection":"disease_knowledge",
        "source_status":SOURCE_STATUS,
        "source_url":"",
        "source_publisher":"",
        "clinical_verified":False,
        "medical_authority_verified":False,
        "source_rights_verified":False,
        "research_only":True,
        "review_status":"NOT_CLINICALLY_REVIEWED",
        "link_basis":"exact_name_only",
        "plan_sha256":plan_sha,
        "graph_link_version":"P3-04.4B",
    }
    edge={
        "link_basis":"exact_name_only",
        "source_status":SOURCE_STATUS,
        "clinical_verified":False,
        "research_only":True,
        "is_treatment_relation":False,
        "plan_sha256":plan_sha,
        "graph_link_version":"P3-04.4B",
    }
    return doc,edge


def validate_graph_audit(graph_audit:dict, plan_report:dict,
                         plan_sha:str,source_sha:str) -> None:
    _require(
        graph_audit.get("stage")=="P3-04.4B"
        and graph_audit.get("status")=="FULL_GRAPH_LINK_AUDIT_OK"
        and graph_audit.get("mode")=="research_only_exact_name_unverified_graph_links"
        and graph_audit.get("relationship_type")==LINK_TYPE
        and graph_audit.get("expected_nodes_and_links")==EXPECTED_LINKS
        and graph_audit.get("after",{}).get("EXISTING_EXACT")==EXPECTED_LINKS,
        "尚未提供通过的 P3-04.4B 1150 条完整图谱只读审计"
    )
    _require(
        graph_audit.get("source_sha256")==source_sha
        and graph_audit.get("plan_sha256")==plan_sha
        and graph_audit.get("source_status")==SOURCE_STATUS
        and graph_audit.get("chroma_collection")=="disease_knowledge",
        "图谱全量审计的来源或候选 SHA 不一致"
    )
    _require(
        graph_audit.get("clinical_verified") is False
        and graph_audit.get("source_rights_verified") is False
        and graph_audit.get("research_prototype") is True
        and graph_audit.get("medical_review_performed") is False,
        "图谱审计冒充临床审核/来源许可或缺少科研状态"
    )
    _require(
        plan_report.get("stage")=="P3-04.4A"
        and plan_report.get("mode")=="strict_read_only_graph_link_plan"
        and plan_report.get("readiness")=="READY_FOR_HUMAN_REVIEW_NO_WRITES"
        and plan_report.get("source_file_sha256")==source_sha
        and plan_report.get("plan_sha256")==plan_sha
        and plan_report.get("aligned_records_in_review_plan")==EXPECTED_LINKS
        and plan_report.get("planned_document_links")==EXPECTED_LINKS
        and plan_report.get("graph_recheck",{}).get("live_verified") is True
        and plan_report["graph_recheck"].get("document_id_conflict_count")==0
        and plan_report.get("clinical_verified") is False
        and plan_report.get("human_review_done") is False,
        "P3-04.4A 严格同名关联审计身份不符"
    )


@lru_cache(maxsize=3)
def _load_plan_index_cached(
    absolute_path:str,mtime_ns:int,nbytes:int,plan_sha:str,
    source_sha:str) -> dict:
    file=Path(absolute_path)
    rows={}
    ids=set()
    term_ids=set()
    with file.open("r",encoding="utf-8-sig") as stream:
        for line_num,line in enumerate(stream,1):
            _require(bool(line.strip()) and len(line.encode("utf-8"))<=MAX_CANDIDATE_LINE_BYTES,
                     f"候选 JSONL 第 {line_num} 行为空或过大")
            try:
                row=json.loads(line)
            except json.JSONDecodeError as exc:
                raise GraphRouteBlocked(f"候选 JSONL 第 {line_num} 行格式错误") from exc
            _require(isinstance(row,dict),
                     f"候选 JSONL 第 {line_num} 行不是对象")
            name=row.get("disease_name")
            doc_id=row.get("document_id")
            term_id=row.get("disease_term_id")
            oid=row.get("source_record_oid")
            _require(
                isinstance(name,str) and len(name)>1
                and isinstance(oid,str) and re.fullmatch(r"[0-9a-f]{24}",oid)
                and doc_id==f"medical_json:{oid}"
                and isinstance(term_id,str) and re.fullmatch(
                    r"diseaseterm_[a-f0-9]{24}",term_id
                ),f"第 {line_num} 条候选的名称/原始 OID/图谱 ID 不正确"
            )
            _require(
                row.get("source_file_sha256")==source_sha
                and row.get("source_file")=="medical.json"
                and row.get("chroma_collection")=="disease_knowledge"
                and row.get("source_status")==SOURCE_STATUS
                and row.get("review_status")=="PENDING_HUMAN_REVIEW"
                and row.get("name_alignment")=="exact_name_only"
                and row.get("proposed_relationship")==LINK_TYPE
                and row.get("clinical_verified") is False
                and row.get("medical_authority_verified") is False
                and row.get("treatment_relation") is False
                and row.get("write_to_neo4j") is False,
                f"第 {line_num} 条候选审核状态/来源关系标记不合法"
            )
            _require(name not in rows and doc_id not in ids and term_id not in term_ids,
                     "同一图谱术语/文档或疾病名称有重复候选，拒绝关联")
            rows[name]=row
            ids.add(doc_id)
            term_ids.add(term_id)
    _require(len(rows)==EXPECTED_LINKS,
             f"候选 JSONL 预期 {EXPECTED_LINKS} 条，实际 {len(rows)}")
    return rows


def _read_plan_index(path:Path, expected_sha:str,source_sha:str) -> dict:
    _require(path.is_file(),f"候选 JSONL 文件不存在：{path}")
    actual=sha256_file(path)
    _require(actual==expected_sha,
             "候选 JSONL 的原始字节 SHA256 与计划不一致")
    stat=path.stat()
    return _load_plan_index_cached(
        str(path.resolve()),stat.st_mtime_ns,stat.st_size,actual,source_sha
    )


def validate_graph_inputs(root:Path,source_sha:str, *,
                          audit_path:Path|None=None,
                          plan_path:Path|None=None,
                          plan_report_path:Path|None=None) -> tuple[dict,dict]:
    """Return a bounded read-only plan index after multi-stage audit gates."""
    graph_audit_path,plan_file,plan_report_file=_manifest_paths(
        root,audit_override=audit_path,plan_override=plan_path,
        plan_report_override=plan_report_path
    )
    if any(path is None for path in (graph_audit_path,plan_file,plan_report_file)):
        raise GraphRouteBlocked("P3-04.4B 全量验收或 P3-04.4A 来源文件缺失")
    graph=_load_json(graph_audit_path)
    report=_load_json(plan_report_file)
    plan_sha=report.get("plan_sha256")
    _require(plan_sha==EXPECTED_PLAN_SHA,
             "P3-04.4A 计划 SHA256 不是已核对的科研实验批次")
    validate_graph_audit(graph,report,plan_sha,source_sha)
    index=_read_plan_index(plan_file,plan_sha,source_sha)
    return index,{
        "graph_audit":graph_audit_path.name,
        "plan_sha256":plan_sha,
        "graph_audit_status":graph["status"],
        "planned_graph_links":len(index),
    }


def _read_neo4j_links(names:list[str], *,
                      provider:Callable[[list[str]],list[dict]]|None=None,
                      project_root:Path|None=None) -> list[dict]:
    _require(len(names)<=MAX_DISEASE_MATCHES,
             "Neo4j 单次查询的疾病名称超过上限")
    if provider is not None:
        results=provider(names)
        return list(results)
    from neo4j import GraphDatabase, READ_ACCESS
    from dotenv import load_dotenv
    load_dotenv((project_root or Path.cwd())/".env",override=False)
    uri=os.getenv("NEO4J_URI","").strip()
    user=os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
    password=os.getenv("NEO4J_PASSWORD","").strip()
    database=os.getenv("NEO4J_DATABASE","neo4j")
    _require(bool(uri and password),
             "没有可用 Neo4j 连接参数；只能回退 P3-04.3 原检索")
    driver=GraphDatabase.driver(uri,auth=(user,password))
    try:
        driver.verify_connectivity()
        with driver.session(
            database=database,default_access_mode=READ_ACCESS
        ) as session:
            return session.execute_read(
                lambda tx:[
                    record.data() for record in tx.run(
                        REL_QUERY,disease_names=names
                    )
                ]
            )
    finally:
        driver.close()


def validate_link_rows(graph_rows:list[dict],names:list[str],
                       plan_index:dict,source_index:dict,
                       plan_sha:str,source_sha:str) -> dict[str,dict]:
    """Fail entire graph branch if any returned Neo4j relationship is forged."""
    _require(isinstance(graph_rows,list) and len(graph_rows)<=len(names),
             "Neo4j 精确疾病关系存在重复或返回异常")
    selected={}
    for row in graph_rows:
        _require(isinstance(row,dict),"Neo4j 图谱结果不是对象")
        name=row.get("term_name")
        candidate=plan_index.get(name)
        _require(name in names and candidate is not None and name not in selected,
                 "Neo4j 返回未经请求/未审核候选/重复的疾病来源关系")
        oid=candidate["source_record_oid"]
        source_entry=source_index["by_oid"].get(oid)
        _require(
            source_entry is not None
            and source_entry["name"]==name
            and source_entry["line"]==candidate["source_line_number"]
            and source_entry["line_sha"]==candidate["source_line_sha256"],
            f"Neo4j 疾病来源 OID/原始记录行哈希不一致：{name}"
        )
        expected_doc,expected_edge=_expected_graph_properties(candidate,plan_sha)
        _require(
            row.get("term_id")==candidate["disease_term_id"]
            and row.get("document_id")==candidate["document_id"]
            and _same_values(row.get("document_properties"),expected_doc)
            and _same_values(row.get("relationship_properties"),expected_edge),
            f"Neo4j 疾病关系/文档属性与已审计候选不一致：{name}"
        )
        selected[name]=candidate
    return selected


def graph_chroma_evidence(
    selected:list[str], linked:dict[str,dict], *, collection:Any,
    source_index:dict,source_sha:str,fields:list[str],max_documents:int
) -> tuple[list[dict],dict]:
    """Fetch by graph Document OID (NOT Chroma default embedding search)."""
    skipped=Counter()
    rows=[]
    for name in selected:
        candidate=linked.get(name)
        if candidate is None:
            continue
        oid=candidate["source_record_oid"]
        original=source_index["by_oid"][oid]
        result=collection.get(
            where={"record_oid":oid},
            limit=MAX_RETRIEVED_PER_DISEASE,
            include=["documents","metadatas"]
        )
        ids=result.get("ids") or []
        docs=result.get("documents") or []
        meta=result.get("metadatas") or []
        _require(len(ids)==len(docs)==len(meta)==len(set(ids)),
                 "Chroma Graph-OID 查询返回无效结果")
        _require(len(ids)<MAX_RETRIEVED_PER_DISEASE,
                 "Graph-OID 的 Chroma Chunk 达到截断上限")
        for cid,doc,md in zip(ids,docs,meta):
            ok,why=validate_disease_chunk(
                cid,doc,md,original,source_sha
            )
            if not ok:
                # Fail the graph path instead of silently using altered chunks.
                raise GraphRouteBlocked(
                    f"图谱已关联原文的 Chroma Chunk 无法逐字校验：{why}"
                )
            if md["source_field"] not in fields:
                continue
            rows.append({
                "chunk_id":cid,
                "document":doc,
                "metadata":md,
                "disease_name":name,
                "section_field":md["source_field"],
                "chunk_index":md["chunk_index"],
                "source_entry":original,
            })
        if not ids:
            raise GraphRouteBlocked(
                f"图谱文档 {name} 在 Chroma 未找到原始 OID Chunk"
            )
    ranked=_rank_chunks(rows,fields)
    best=[]
    used=set()
    for name in selected:
        item=next((x for x in ranked if x["disease_name"]==name
                   and x["chunk_id"] not in used),None)
        if item and len(best)<max_documents:
            best.append(item)
            used.add(item["chunk_id"])
    for item in ranked:
        if len(best)>=max_documents:
            break
        if item["chunk_id"] not in used:
            best.append(item)
            used.add(item["chunk_id"])
    excerpts=[]
    for item in best:
        source_entry=item["source_entry"]
        name=item["disease_name"]
        candidate=linked[name]
        excerpt=_make_excerpt(item,source_entry)
        excerpt.update({
            "graph_link_verified":True,
            "graph_link_type":LINK_TYPE,
            "graph_research_only":True,
            "disease_term_id":candidate["disease_term_id"],
            "disease_term":name,
            "_retrieval_tool":"neo4j_disease_document_to_chroma_oid",
        })
        excerpts.append(excerpt)
    return excerpts,{
        "source_original_chunks_verified":len(excerpts),
        "fields":dict(Counter(row["section_type"] for row in excerpts)),
        "chroma_query_mode":"record_oid_exact",
        "query_embedding_called":False,
    }


def _merge_graph_and_fallback(graph:list[dict],base:list[dict],
                              linked_names:set[str],route:str,
                              max_documents:int) -> list[dict]:
    selected=list(graph)
    for row in base:
        if row.get("disease_name") in linked_names:
            continue
        selected.append(row)
    unique=[]
    seen=set()
    for entry in selected:
        cid=entry.get("chunk_id")
        if cid not in seen:
            seen.add(cid)
            unique.append(entry)
        if len(unique)>=max_documents:
            break
    return [{**x,"id":i+1} for i,x in enumerate(unique)]


def search_graph_enhanced_disease_encyclopedia(
    question:str, *,project_root:Path,
    audit_path:Path|None=None,
    collection:Any=None,
    source_path:Path|None=None,
    max_documents:int=4,
    graph_audit_path:Path|None=None,
    plan_path:Path|None=None,
    plan_report_path:Path|None=None,
    link_provider:Callable[[list[str]],list[dict]]|None=None,
) -> tuple[list[dict],dict]:
    """Graph-first only after explicit opt-in. Otherwise exact P3-04.3 fallback."""
    start=time.perf_counter()
    project_root=Path(project_root).resolve()
    try:
        from dotenv import load_dotenv
        load_dotenv(project_root/".env",override=False)
    except ImportError:
        pass
    route=question_route(question)
    fallback_kwargs={
        "project_root":project_root,
        "audit_path":audit_path,
        "collection":collection,
        "source_path":source_path,
        "max_documents":max_documents,
    }
    graph_info={
        "status":"DISABLED",
        "graph_query_ran":False,
        "graph_verified_names":[],
        "graph_linked_evidence":0,
        "graph_research_only":True,
        "graph_medically_verified":False,
        "graph_relationship_type":LINK_TYPE,
        "graph_writes":0,
        "chroma_writes":0,
        "query_embedding_called":False,
    }
    def fallback(status:str,reason_type:str=""):
        docs,trace=search_disease_encyclopedia(question,**fallback_kwargs)
        graph_info["status"]=status
        if reason_type:
            graph_info["reason_type"]=reason_type
        graph_info["elapsed_ms"]=round((time.perf_counter()-start)*1000,2)
        trace["graph_disease_bridge"]=dict(graph_info)
        return docs,trace

    if not is_enabled() or not _enabled():
        return fallback(
            "DISABLED" if not _enabled() else "ENCYCLOPEDIA_DISABLED"
        )
    if route in ("NONE","DRUG_ONLY"):
        return fallback("SKIPPED_NON_DISEASE_QUERY")
    try:
        _require(1<=max_documents<=MAX_RETURN_DOCS,
                 "max_documents 必须在 1～6 范围内")
        source=(
            Path(source_path).resolve() if source_path is not None
            else (project_root/"data/raw/drug_db/MedGraphRAG-Drug-DB/medical.json").resolve()
        )
        audited=(
            Path(audit_path).resolve() if audit_path is not None
            else latest_audit(project_root)
        )
        _require(source.is_file() and audited is not None,
                 "疾病百科原文/完整 Chroma 审计文件不存在")
        if collection is None:
            collection=_read_chroma_collection(project_root)
        source_gate=validate_audit(audited,source,collection)
        source_sha=source_gate["source_sha256"]
        _require(source_sha==EXPECTED_SOURCE_SHA
                 and source_gate["stage_sha256"]==EXPECTED_STAGE_SHA,
                 "疾病百科源文件与固定科研批次不一致")
        fields,unsupported=requested_fields(question)
        if not fields:
            return fallback("SKIPPED_UNSUPPORTED_FIELDS")
        index=load_source_index(source)
        diseases,extra=match_disease_names(question,index["names"])
        if not diseases:
            return fallback("SKIPPED_NO_EXACT_DISEASE_NAME")
        plan_index,gate=validate_graph_inputs(
            project_root,source_sha,
            audit_path=graph_audit_path,
            plan_path=plan_path,
            plan_report_path=plan_report_path,
        )
        graph_info["graph_audit_file"]=gate["graph_audit"]
        graph_info["graph_plan_sha256"]=gate["plan_sha256"]
        selected_names=[name for name in diseases if name in plan_index]
        if not selected_names:
            return fallback("NO_GRAPH_PLAN_MATCH_FALLBACK")
        graph_info["graph_query_ran"]=True
        graph_rows=_read_neo4j_links(
            selected_names,provider=link_provider,project_root=project_root
        )
        linked=validate_link_rows(
            graph_rows,selected_names,plan_index,index,
            gate["plan_sha256"],source_sha
        )
        if not linked:
            return fallback("GRAPH_LINK_MISSING_FALLBACK")
        graph_docs,source_trace=graph_chroma_evidence(
            diseases,linked,collection=collection,
            source_index=index,source_sha=source_sha,
            fields=fields,max_documents=max_documents
        )
        if not graph_docs:
            return fallback("NO_GRAPH_SECTION_FALLBACK")
        graph_info.update({
            "status":"GRAPH_EXACT_OID_PROVENANCE_VERIFIED",
            "graph_verified_names":list(linked.keys()),
            "graph_linked_evidence":len(graph_docs),
            "source_trace":source_trace,
        })
        not_linked=set(diseases)-set(linked)
        if not_linked:
            base,base_trace=search_disease_encyclopedia(
                question,**fallback_kwargs
            )
            results=_merge_graph_and_fallback(
                graph_docs,base,set(linked),route,max_documents
            )
            graph_info["status"]="GRAPH_PARTIAL_WITH_ENCYCLOPEDIA_FALLBACK"
            graph_info["fallback_names"]=sorted(not_linked)
            graph_info["fallback_trace_status"]=base_trace["status"]
        else:
            results=[
                {**x,"id":i+1} for i,x in enumerate(graph_docs)
            ]
        trace={
            "route":route,
            "status":"VERIFIED_ORIGINAL_TEXT",
            "matched_diseases":diseases,
            "additional_name_matches_omitted":extra,
            "returned_chunks":len(results),
            "unsupported_sections":unsupported,
            "requested_fields":fields,
            "skipped":{},
            "sections_returned":dict(
                Counter(x.get("section_type","") for x in results)
            ),
            "audit_file":audited.name,
            "source_status":SOURCE_STATUS,
            "source_file_sha256":source_sha,
            "disease_collection_count":source_gate["collection_count"],
            "medical_accuracy_reviewed":False,
            "semantic_embedding_called":False,
            "neo4j_relationship_created":False,
        }
        graph_info["elapsed_ms"]=round((time.perf_counter()-start)*1000,2)
        trace["graph_disease_bridge"]=dict(graph_info)
        return results,trace
    except Exception as exc:
        # Fail CLOSED with respect to Neo4j evidence, but preserve independently
        # audited original P3-04.3 encyclopedia if it is available.
        return fallback("GRAPH_SOURCE_BLOCKED_FALLBACK",type(exc).__name__)
