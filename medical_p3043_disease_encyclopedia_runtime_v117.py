# -*- coding: utf-8 -*-
"""P3-04.3: Independent disease encyclopedia evidence retrieval for GraphRAG V1.17.

Read only: user's medical.json, audited Chroma disease_knowledge collection.
Does NOT generate query embeddings, mutate Chroma/Neo4j, or assert treatment edges.

Medical source is third_party_unverified; text/source matching is technical only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable

from medical_p3041_medical_json_prepare_v117 import (
    DATA_ORIGIN, SOURCE_STATUS, FIELD_SPECS, canonical_json, digest,
    field_text_and_stats, get_oid, iterate_source, sha256_file
)

TARGET_COLLECTION = "disease_knowledge"
EXPECTED_CHUNKS = 39443
MAX_DISEASE_MATCHES = 3
MAX_RETRIEVED_PER_DISEASE = 100
MAX_RETURN_DOCS = 6

FIELD_TITLE = {name: title for name, _, title in FIELD_SPECS}
SECTION_TYPES = {name: kind for name, kind, _ in FIELD_SPECS}
SOURCE_REL = Path("data/raw/drug_db/MedGraphRAG-Drug-DB/medical.json")

ENCYCLOPEDIA_TOPICS = re.compile(
    r"是什么|什么是|介绍|概述|症状|表现|病因|发病机制|原因|为什么"
    r"|预防|如何避免|会传染|风险因素|疾病百科|疾病知识|怎么回事"
    r"|诊断标准|如何诊断|并发症|预后", re.I
)
DRUG_TOPICS = re.compile(
    r"药品|药物|用药|说明书|适应症|适应证|禁忌|剂量|不良反应"
    r"|药物相互作用|副作用|推荐药|治疗药|哪种药|能治|吃什么药|处方",
    re.I
)
UNSUPPORTED_TOPICS = (
    ("诊断", "diagnostic_criteria"), ("检查标准", "diagnostic_criteria"),
    ("并发症", "complications"), ("治疗方案", "treatment_guideline"),
    ("预后", "prognosis")
)


class DiseaseEvidenceBlocked(RuntimeError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise DiseaseEvidenceBlocked(message)


def is_enabled() -> bool:
    return os.getenv("MEDICAL_DISEASE_ENCYCLOPEDIA_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def question_route(question: str) -> str:
    """Do not steal 'how a drug leaflet mentions disease X' from P3-03B."""
    question = (question or "").strip()
    if not question:
        return "NONE"
    disease = bool(ENCYCLOPEDIA_TOPICS.search(question))
    drug = bool(DRUG_TOPICS.search(question))
    if not disease:
        return "DRUG_ONLY" if drug else "NONE"
    if ("说明书" in question
            and any(word in question for word in ("提到","描述","记载"))
            and not any(word in question for word in (
                "是什么","什么是","症状","病因","预防","概述"
            ))):
        return "DRUG_ONLY"
    return "MIXED" if drug else "DISEASE_ONLY"


def requested_fields(question: str) -> tuple[list[str], list[str]]:
    question = (question or "").strip()
    asked = []
    if re.search(r"症状|临床表现|表现有哪些|有什么表现",question):
        asked.append("symptom")
    if re.search(r"病因|发病机制|成因|什么原因|为什么",question):
        asked.append("cause")
    if re.search(r"预防|避免|防范",question):
        asked.append("prevent")
    if re.search(r"是什么|什么是|介绍|概述|怎么回事",question):
        asked.append("desc")
    unsupported = sorted({
        topic for hint, topic in UNSUPPORTED_TOPICS if hint in question
    })
    # "诊断标准是什么" uses 是什么 about the standard, not the disease.
    if unsupported and asked==["desc"] and any(
        phrase in question for phrase in (
            "诊断标准是什么","并发症是什么",
            "预后是什么","治疗方案是什么"
        )
    ):
        asked=[]
    if not asked and not unsupported:
        asked = ["desc"]
    elif asked and "desc" not in asked:
        asked.append("desc")
    return list(dict.fromkeys(asked)), unsupported


def latest_audit(root: Path, configured: str | None = None) -> Path | None:
    raw = (
        os.getenv("MEDICAL_DISEASE_AUDIT_PATH", "")
        if configured is None else configured
    ).strip()
    if raw:
        p = Path(raw)
        return (p if p.is_absolute() else root / p).resolve()
    folder = root / "stage4_reports"
    if not folder.is_dir():
        return None
    matches = list(folder.glob("p3042_final_audit_*.json"))
    if not matches:
        return None
    return max(matches, key=lambda x: (x.stat().st_mtime_ns, x.name)).resolve()


def validate_audit(audit_file: Path, source: Path, collection: Any) -> dict:
    _require(audit_file.is_file(),f"P3-04.2 全量审计文件不存在：{audit_file}")
    summary=json.loads(audit_file.read_text(encoding="utf-8-sig"))
    _require(isinstance(summary,dict),"疾病库全量审计报告格式不正确")
    _require(
        summary.get("stage")=="P3-04.2"
        and summary.get("mode")=="read_only_full_chroma_source_audit"
        and summary.get("status")=="PASS_FULL_DISEASE_CHROMA_PROVENANCE",
        "没有通过 P3-04.2 全量 Chroma 来源审计，疾病百科不得启用"
    )
    _require(summary.get("disease_collection")==TARGET_COLLECTION
             and summary.get("verified_chunks")==EXPECTED_CHUNKS,
             "疾病审计 Collection 或通过 Chunk 数不一致")
    _require(summary.get("source_status")==SOURCE_STATUS
             and summary.get("medical_accuracy_reviewed") is False
             and summary.get("neo4j_disease_term_linked") is False,
             "来源审核与疾病术语状态不符合已审计基线")
    raw_sha=sha256_file(source)
    _require(raw_sha==summary.get("source_sha256"),
             "medical.json SHA256 与 P3-04.2 审计不一致")
    _require(collection.name==TARGET_COLLECTION
             and collection.count()==EXPECTED_CHUNKS,
             f"Chroma 疾病 Collection 当前数量不等于 {EXPECTED_CHUNKS}")
    metadata=collection.metadata or {}
    _require(
        metadata.get("pipeline")=="medical-p3042-disease-v117"
        and metadata.get("source_sha256")==raw_sha
        and metadata.get("stage_sha256")==summary.get("staging_sha256")
        and metadata.get("source_status")==SOURCE_STATUS
        and metadata.get("rights_reviewed") is False,
        "Chroma 疾病 Collection 元信息与审计不一致"
    )
    return {
        "source_sha256":raw_sha,
        "stage_sha256":summary["staging_sha256"],
        "collection_count":EXPECTED_CHUNKS,
        "audit_file":audit_file.name
    }


@lru_cache(maxsize=2)
def _read_source_index(source_name: str, source_mtime_ns: int,
                       source_bytes: int) -> dict:
    source=Path(source_name)
    names=defaultdict(list)
    by_oid={}
    for lineno,obj,line_sha in iterate_source(source):
        oid=get_oid(obj)
        name=obj.get("name")
        _require(oid and isinstance(name,str) and name.strip(),
                 f"原疾病文件 {lineno} 行 ID/疾病名不正确")
        _require(oid not in by_oid,f"原疾病文件存在重复记录 OID：{oid}")
        values = {}
        raw_hashes = {}
        canonical_hashes = {}
        for field in SECTION_TYPES:
            original=obj.get(field)
            _require(
                (field=="symptom" and isinstance(original,list)
                 and all(isinstance(x,str) for x in original))
                or (field!="symptom" and isinstance(original,str)),
                f"原疾病文件 {lineno} 行 {field} 格式错误"
            )
            text,_=field_text_and_stats(obj,field)
            values[field]=text
            raw_hashes[field]=digest(canonical_json(original))
            canonical_hashes[field]=digest(text)
        entry={
            "oid":oid,"name":name.strip(),"line":lineno,
            "line_sha":line_sha,"fields":values,
            "source_field_sha":raw_hashes,
            "canonical_field_sha":canonical_hashes,
        }
        by_oid[oid]=entry
        names[entry["name"]].append(oid)
    _require(bool(by_oid),"疾病源文件为空")
    return {"by_oid":by_oid,"names":dict(names)}


def load_source_index(source: Path) -> dict:
    source=source.resolve()
    stat=source.stat()
    return _read_source_index(str(source),stat.st_mtime_ns,stat.st_size)


def match_disease_names(question: str, all_names: dict,
                        limit: int=MAX_DISEASE_MATCHES) -> tuple[list[str],int]:
    """Exact text spans, longest non-overlapping names first, no fuzzy merging."""
    candidates=[]
    for name in all_names:
        if len(name)<2 or len(name)>120:
            continue
        pos=question.find(name)
        if pos!=-1:
            candidates.append((name,pos,pos+len(name)))
    candidates.sort(key=lambda item:(-len(item[0]),item[1],item[0]))
    chosen=[]
    spans=[]
    for name,start,end in candidates:
        if any(start<b and end>a for a,b in spans):
            continue
        spans.append((start,end))
        chosen.append(name)
    return chosen[:limit], max(0,len(chosen)-limit)


def validate_disease_chunk(row_id: str, document: Any, meta: Any,
                           source_entry: dict, source_sha: str
                           ) -> tuple[bool,str]:
    if not isinstance(document,str) or not isinstance(meta,dict):
        return False,"invalid_document"
    oid=source_entry["oid"]
    field=meta.get("source_field")
    if field not in SECTION_TYPES:
        return False,"unexpected_section"
    if meta.get("record_oid")!=oid or meta.get("disease_name")!=source_entry["name"]:
        return False,"wrong_disease"
    if (meta.get("source_file")!="medical.json"
            or meta.get("source_file_sha256")!=source_sha
            or meta.get("source_line_number")!=source_entry["line"]
            or meta.get("source_line_sha256")!=source_entry["line_sha"]):
        return False,"wrong_source_file_or_line"
    if (meta.get("data_origin")!=DATA_ORIGIN
            or meta.get("source_status")!=SOURCE_STATUS
            or meta.get("source_url")!=""
            or meta.get("source_publisher")!=""
            or meta.get("disease_term_id")!=""):
        return False,"source_status_modified"
    if (meta.get("clinical_verified") is not False
            or meta.get("rights_reviewed") is not False
            or meta.get("recommended_drug_relation") is not False):
        return False,"unsafe_status_elevation"
    if meta.get("disease_document_id")!=f"medical_json:{oid}":
        return False,"document_identity_conflict"
    if meta.get("section_type")!=SECTION_TYPES[field]:
        return False,"wrong_section_type"
    if (meta.get("source_field_sha256")!=source_entry["source_field_sha"][field]
            or meta.get("canonical_field_sha256")!=
                source_entry["canonical_field_sha"][field]):
        return False,"source_field_hash_mismatch"
    index,start,end = (
        meta.get("chunk_index"),meta.get("field_start"),meta.get("field_end")
    )
    if not all(type(v) is int for v in (index,start,end)):
        return False,"offset_type_error"
    full=source_entry["fields"][field]
    if not (index>=0 and 0<=start<end<=len(full)):
        return False,"offset_out_of_bounds"
    title=FIELD_TITLE[field]
    header=f"疾病：{source_entry['name']}\n栏目：{title}\n"
    expected_body=full[start:end]
    if (document != header+expected_body
            or meta.get("document_body_start")!=len(header)):
        return False,"content_not_original"
    if (meta.get("body_sha256")!=digest(expected_body)
            or meta.get("document_sha256")!=digest(document)):
        return False,"document_hash_mismatch"
    expected_id = (
        f"disease_v117_{oid}_{field}_{index:04d}_"
        f"{digest(expected_body)[:12]}"
    )
    if row_id!=expected_id:
        return False,"chunk_id_mismatch"
    return True,"verified_original_text"


def _read_chroma_collection(root: Path) -> Any:
    db=root/"chroma_db"
    _require((db/"chroma.sqlite3").is_file(),
             "找不到现有 Chroma 数据库：chroma_db/chroma.sqlite3")
    import chromadb
    client=chromadb.PersistentClient(path=str(db.resolve()))
    _require(TARGET_COLLECTION in {
        row if isinstance(row,str) else row.name
        for row in client.list_collections()
    }, "当前 Chroma 中没有独立的 disease_knowledge Collection")
    return client.get_collection(name=TARGET_COLLECTION)


def _rank_chunks(chunks: list[dict],fields: list[str]) -> list[dict]:
    preferences={name:i for i,name in enumerate(fields)}
    unique={}
    for entry in chunks:
        cid=entry["chunk_id"]
        if cid not in unique:
            unique[cid]=entry
    return sorted(unique.values(),key=lambda item:(
        preferences.get(item["section_field"],100),
        item["chunk_index"],item["disease_name"],item["chunk_id"]
    ))


def _make_excerpt(row: dict, source_entry: dict) -> dict:
    meta=row["metadata"]
    field=meta["source_field"]
    return {
        "id":0,
        "source":"medical.json（第三方疾病百科，来源待核实）",
        "source_url":"",
        "section_path":f"{source_entry['name']} / {FIELD_TITLE[field]}",
        "content":row["document"],
        "chunk_id":row["chunk_id"],
        "evidence_type":"unverified_disease_encyclopedia",
        "disease_name":source_entry["name"],
        "disease_document_id":f"medical_json:{source_entry['oid']}",
        "disease_term_id":"",
        "disease_term":"",  # no Neo4j name inference
        "section_field":"", # never pretend to be drug Section
        "section_type":SECTION_TYPES[field],
        "source_field":field,
        "source_status":SOURCE_STATUS,
        "clinical_status":"not_verified",
        "clinical_verified":False,
        "rights_reviewed":False,
        "parent_verified":False,
        "source_file_sha256":meta["source_file_sha256"],
        "source_line_number":meta["source_line_number"],
        "record_oid":source_entry["oid"],
        "field_start":meta["field_start"],
        "field_end":meta["field_end"],
        "chunk_index":meta["chunk_index"],
        "_retrieval_tool":"search_disease_encyclopedia"
    }


def search_disease_encyclopedia(question: str, *, project_root: Path,
                                audit_path: Path | None=None,
                                collection: Any=None,
                                source_path: Path | None=None,
                                max_documents: int=4
                                ) -> tuple[list[dict],dict]:
    """No embedding, no Neo4j writes, no inferred disease-drug relationships."""
    route=question_route(question)
    trace={
        "route":route,"status":"SKIPPED",
        "matched_diseases":[],"returned_chunks":0,
        "skipped":{},"unsupported_sections":[],
        "audit_file":"","source_status":SOURCE_STATUS,
        "semantic_embedding_called":False,
        "neo4j_relationship_created":False
    }
    if not is_enabled():
        trace["status"]="DISABLED"
        return [],trace
    if route in ("NONE","DRUG_ONLY"):
        return [],trace
    _require(1<=max_documents<=MAX_RETURN_DOCS,
             "max_documents 必须为 1～6")
    project_root=Path(project_root).resolve()
    source = (
        Path(source_path).resolve() if source_path is not None
        else (project_root/SOURCE_REL).resolve()
    )
    audit = (
        Path(audit_path).resolve() if audit_path is not None
        else latest_audit(project_root)
    )
    if audit is None:
        trace["status"]="BLOCKED_NO_FULL_IMPORT_AUDIT"
        return [],trace
    if not source.is_file():
        trace["status"]="BLOCKED_SOURCE_FILE_MISSING"
        return [],trace
    if collection is None:
        collection=_read_chroma_collection(project_root)
    gate=validate_audit(audit,source,collection)
    trace["audit_file"]=gate["audit_file"]
    index=load_source_index(source)
    selected,extra=match_disease_names(question,index["names"])
    trace["matched_diseases"]=selected
    trace["additional_name_matches_omitted"]=extra
    fields,unsupported=requested_fields(question)
    trace["requested_fields"]=fields
    trace["unsupported_sections"]=unsupported
    if not selected:
        trace["status"]="NO_EXACT_DISEASE_NAME"
        return [],trace
    if not fields:
        trace["status"]="UNSUPPORTED_ONLY_QUESTION"
        return [],trace
    skipped=Counter()
    result_rows=[]
    for disease_name in selected:
        raw=collection.get(
            where={"disease_name":disease_name},limit=MAX_RETRIEVED_PER_DISEASE,
            include=["documents","metadatas"]
        )
        ids=raw.get("ids") or []
        docs=raw.get("documents") or []
        metas=raw.get("metadatas") or []
        _require(
            len(ids)==len(docs)==len(metas)
            and len(ids)==len(set(ids)),
            "Chroma 疾病精确名称查询返回字段数量异常"
        )
        # Needn't speculate that 100 covers whole disease; fail closed if full.
        _require(len(ids)<MAX_RETRIEVED_PER_DISEASE,
                 "单个疾病 Chunk 达到查询安全上限，无法证明结果完整")
        valid_oids=set(index["names"][disease_name])
        for cid,doc,meta in zip(ids,docs,metas):
            oid=meta.get("record_oid") if isinstance(meta,dict) else None
            if oid not in valid_oids:
                skipped["unexpected_source_oid"]+=1
                continue
            orig=index["by_oid"][oid]
            ok,reason=validate_disease_chunk(
                cid,doc,meta,orig,gate["source_sha256"]
            )
            if not ok:
                skipped[reason]+=1
                continue
            if meta["source_field"] not in fields:
                continue
            record={
                "chunk_id":cid,"document":doc,"metadata":meta,
                "disease_name":disease_name,
                "section_field":meta["source_field"],
                "chunk_index":meta["chunk_index"],
                "source_entry":orig,
            }
            result_rows.append(record)
    ranked=_rank_chunks(result_rows,fields)
    # Allocate at least one record per disease when possible, then fill by rank.
    results=[]
    used=set()
    for name in selected:
        first=next((x for x in ranked if x["disease_name"]==name
                    and x["chunk_id"] not in used),None)
        if first is not None and len(results)<max_documents:
            results.append(first)
            used.add(first["chunk_id"])
    for item in ranked:
        if len(results)>=max_documents:
            break
        if item["chunk_id"] not in used:
            results.append(item)
            used.add(item["chunk_id"])
    evidence=[_make_excerpt(x,x["source_entry"]) for x in results]
    trace.update({
        "status":"VERIFIED_ORIGINAL_TEXT" if evidence else
                 "NO_VERIFIABLE_EVIDENCE",
        "returned_chunks":len(evidence),
        "sections_returned":dict(sorted(Counter(
            item["section_type"] for item in evidence
        ).items())),
        "skipped":dict(sorted(skipped.items())),
        "source_file_sha256":gate["source_sha256"],
        "disease_collection_count":gate["collection_count"],
        "medical_accuracy_reviewed":False,
    })
    return evidence,trace


def merge_into_kb(existing: list[dict], new: list[dict], *,
                  route: str, cap: int=8) -> list[dict]:
    """Disease-only: discard drug leaflet matches. Mixed: retain both sources."""
    if not 1<=cap<=32:
        raise ValueError("evidence cap out of range")
    if not new:
        return list(existing[:cap])
    if route=="DISEASE_ONLY":
        selected=list(new)
    else:
        # Do not displace all pre-existing drug safety/leaflet evidence.
        drug_safety=[
            x for x in existing
            if x.get("section_field") in {
                "contraindications","warnings","adverse_reactions",
                "drug_interactions"
            }
        ][:2]
        selected=list(new[:4])+drug_safety+list(existing)
    unique=[]
    keys=set()
    for item in selected:
        cid=item.get("chunk_id")
        key=("id",cid) if isinstance(cid,str) and cid else (
            "text",item.get("source"),item.get("content")
        )
        if key not in keys:
            keys.add(key)
            unique.append(item)
        if len(unique)>=cap:
            break
    # Ensure numbering matches existing evidence guard and citation references.
    return [{**row,"id":i+1} for i,row in enumerate(unique)]


def read_only_preflight(project_root: Path, *, audit_path: Path | None=None,
                        collection: Any=None,
                        source_path: Path | None=None) -> dict:
    """A concrete question from existing disease names; no index/vector write."""
    root=Path(project_root).resolve()
    source=(
        Path(source_path).resolve() if source_path is not None
        else (root/SOURCE_REL).resolve()
    )
    _require(source.is_file(),f"原疾病 medical.json 不存在：{source}")
    index=load_source_index(source)
    # Try common terms but select ONLY those present in real source.
    preferred=("糖尿病","高血压","肺炎","百日咳","肺泡蛋白质沉积症")
    name=next((x for x in preferred if x in index["names"]),None)
    if name is None:
        name=next((x for x in index["names"] if len(x)>=3),None)
    _require(name is not None,"没有有效疾病名称")
    evidence,trace=search_disease_encyclopedia(
        f"{name}有哪些症状？",project_root=root,
        audit_path=audit_path,collection=collection,source_path=source
    )
    return {
        "status":"PRECHECK_OK" if evidence
                 and trace["status"]=="VERIFIED_ORIGINAL_TEXT"
                 else "PRECHECK_NOT_READY",
        "test_question":f"{name}有哪些症状？",
        "trace":trace,
        "verified_excerpts":len(evidence),
        "sample_excerpt":(
            evidence[0]["content"][:220] if evidence else ""
        ),
        "note":"原文追溯通过不等于临床内容被权威验证；不建立治疗关系"
    }
