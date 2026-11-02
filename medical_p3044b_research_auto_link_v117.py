# -*- coding: utf-8 -*-
"""P3-04.4B (research-only): safe auto-link Neo4j DiseaseTerm -> DiseaseDocument.

No manual per-disease clinical review is asserted or required for this
research-only exact-name metadata linkage. The program NEVER creates TREATS,
INDICATED_FOR or medical-confirmation relationships, and it NEVER edits
Chroma, SQLite, drug records or existing Neo4j properties.

Commands:
  default: live Neo4j read-only dry run
  --offline-check: validate only local files, no database connection
  --apply --confirm-research-only RESEARCH_UNVERIFIED_ONLY: create limited links
  --audit: read-only verification of all expected document links
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from medical_p3041_medical_json_prepare_v117 import (
    get_oid, iterate_source, sha256_file,
)

STAGE = "P3-04.4B"
RELATIONSHIP = "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA"
DOC_LABEL = "DiseaseDocument"
CHROMA_COLLECTION = "disease_knowledge"
SOURCE_STATUS = "third_party_unverified"
RESEARCH_CONFIRM = "RESEARCH_UNVERIFIED_ONLY"
EXPECTED_CANDIDATES = 1150
EXPECTED_SOURCE_RECORDS = 8807
EXPECTED_TERM_NODES = 1150
EXPECTED_CHROMA_CHUNKS = 39443
EXPECTED_SOURCE_SHA = (
    "01aaf5de1304ea6f4ed84e9802b600b729b9f797284234f2d86d77e55f8dfc49"
)
OID_PATTERN = re.compile(r"^medical_json:([a-f0-9]{24})$")
TERM_PATTERN = re.compile(r"^diseaseterm_[a-f0-9]{24}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_LINE = 1000000
MAX_BATCH = 100
ALLOWED_WRITE_FIELDS = {
    "id", "disease_name", "source_record_oid", "source_line_number",
    "source_line_sha256", "source_file", "source_file_sha256",
    "chroma_collection", "source_status", "source_url",
    "source_publisher", "clinical_verified", "medical_authority_verified",
    "source_rights_verified", "research_only", "review_status",
    "link_basis", "plan_sha256", "graph_link_version",
}


class LinkBlocked(RuntimeError):
    pass


class ExclusiveResearchLock:
    """Cross-platform process lock; never remove old data or stale lock files."""

    def __init__(self, path: Path):
        self.path=Path(str(Path(path).resolve())+".p3044b.lock")
        self.fd=None

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.fd=os.open(self.path,os.O_RDWR|os.O_CREAT,0o600)
        try:
            if os.fstat(self.fd).st_size==0:
                os.write(self.fd,b"0")
            os.lseek(self.fd,0,os.SEEK_SET)
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.fd,msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self.fd)
            self.fd=None
            raise LinkBlocked(
                "另一个 P3-04.4B 导入进程正在运行；拒绝并发写入 Neo4j"
            ) from exc
        return self

    def __exit__(self,exc_type,exc_value,tb):
        if self.fd is None:
            return
        try:
            os.lseek(self.fd,0,os.SEEK_SET)
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.fd,msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.fd,fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd=None


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise LinkBlocked(reason)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _same_props(a: dict, b: dict) -> bool:
    return isinstance(a,dict) and isinstance(b,dict) and _json_bytes(a)==_json_bytes(b)


def _load_json(path: Path) -> dict:
    try:
        data=json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError,UnicodeError,ValueError) as exc:
        raise LinkBlocked(f"JSON 文件无效：{path}") from exc
    require(isinstance(data,dict),f"文件不是 JSON 对象：{path}")
    return data


def expected_props(row: dict, plan_sha: str) -> tuple[dict,dict]:
    """Exactly the metadata we CREATE; never claims clinical verification."""
    doc={
        "id":row["document_id"],
        "disease_name":row["disease_name"],
        "source_record_oid":row["source_record_oid"],
        "source_line_number":row["source_line_number"],
        "source_line_sha256":row["source_line_sha256"],
        "source_file":"medical.json",
        "source_file_sha256":row["source_file_sha256"],
        "chroma_collection":CHROMA_COLLECTION,
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
        "graph_link_version":STAGE,
    }
    link={
        "link_basis":"exact_name_only",
        "source_status":SOURCE_STATUS,
        "clinical_verified":False,
        "research_only":True,
        "is_treatment_relation":False,
        "plan_sha256":plan_sha,
        "graph_link_version":STAGE,
    }
    return doc,link


def validate_plan(*, source:Path, plan:Path, report:Path,
                  chroma_audit:Path, expected_candidates:int=EXPECTED_CANDIDATES,
                  expected_source_records:int=EXPECTED_SOURCE_RECORDS,
                  expected_graph_terms:int=EXPECTED_TERM_NODES) -> dict:
    """Require original file SHA, candidate hash and source-line fingerprints."""
    paths=[Path(item).resolve() for item in (source,plan,report,chroma_audit)]
    for p in paths:
        require(p.is_file(),f"找不到文件：{p}")
    source,plan,report,chroma_audit=paths
    baseline=_load_json(report)
    audit=_load_json(chroma_audit)
    require(
        baseline.get("stage")=="P3-04.4A"
        and baseline.get("mode")=="strict_read_only_graph_link_plan"
        and baseline.get("readiness")=="READY_FOR_HUMAN_REVIEW_NO_WRITES",
        "P3-04.4A 严格同名审计没有达到技术验收状态"
    )
    require(baseline.get("graph_recheck",{}).get("live_verified") is True
            and baseline["graph_recheck"].get("document_id_conflict_count")==0,
            "P3-04.4A 没有通过 Neo4j 实时复核或已有文档冲突")
    require(
        baseline.get("aligned_records_in_review_plan")==expected_candidates
        and baseline.get("planned_document_links")==expected_candidates
        and baseline.get("aligned_unique_names")==expected_candidates,
        "候选数与之前审计的严格同名匹配数不符"
    )
    require(baseline["graph_recheck"].get("neo4j_term_count")==expected_graph_terms,
            "P3-04.4A DiseaseTerm 节点数量与实验基线不同")
    require(
        baseline.get("source_status")==SOURCE_STATUS
        and baseline.get("clinical_verified") is False
        and baseline.get("source_rights_verified") is False
        and baseline.get("human_review_done") is False,
        "输入审计试图提升医学审核、版权或人工审核状态"
    )
    require(
        baseline.get("new_neo4j_nodes")==0
        and baseline.get("new_neo4j_relationships")==0
        and baseline.get("drug_treatment_relationships_created")==0,
        "P3-04.4A 不应已进行图谱写入"
    )
    require(
        audit.get("stage")=="P3-04.2"
        and audit.get("mode")=="read_only_full_chroma_source_audit"
        and audit.get("status")=="PASS_FULL_DISEASE_CHROMA_PROVENANCE"
        and audit.get("verified_chunks")==EXPECTED_CHROMA_CHUNKS
        and audit.get("disease_collection")==CHROMA_COLLECTION
        and audit.get("source_status")==SOURCE_STATUS
        and audit.get("medical_accuracy_reviewed") is False,
        "P3-04.2 疾病百科 Chroma 全量审计未通过"
    )
    sha=sha256_file(source)
    plan_sha=sha256_file(plan)
    require(sha==EXPECTED_SOURCE_SHA and sha==baseline.get("source_file_sha256")
            and sha==audit.get("source_sha256"),
            "原始 medical.json SHA256 与多阶段审计不一致")
    require(source.name=="medical.json","疾病来源文件名不是 medical.json")
    require(plan_sha==baseline.get("plan_sha256"),
            "候选 JSONL 字节 SHA256 与 P3-04.4A 报告不一致")
    require(
        baseline.get("chroma_audited_chunk_count")==EXPECTED_CHROMA_CHUNKS
        and baseline.get("chroma_audit_staging_sha256")==audit.get("staging_sha256"),
        "P3-04.4A 与 P3-04.2 Chroma 来源身份不一致"
    )
    source_rows={}
    for line_number,obj,line_hash in iterate_source(source):
        oid=get_oid(obj)
        name=obj.get("name")
        require(isinstance(name,str) and name.strip() and oid,
                f"medical.json 第 {line_number} 行缺少有效名称/ID")
        require(oid not in source_rows,f"重复源 OID：{oid}")
        source_rows[oid]=(name.strip(),line_number,line_hash)
    require(len(source_rows)==expected_source_records,
            f"原始疾病记录数量 {len(source_rows)} 不等于审计数量 {expected_source_records}")
    candidates=[]
    doc_ids=set()
    term_ids=set()
    names=set()
    with plan.open("r",encoding="utf-8-sig",newline="") as stream:
        for lineno,line in enumerate(stream,1):
            require(line.strip() and len(line)<=MAX_LINE,
                    f"候选文件第 {lineno} 行为空或超长")
            try:
                obj=json.loads(line)
            except json.JSONDecodeError as exc:
                raise LinkBlocked(f"候选文件第 {lineno} 行 JSON 格式错误") from exc
            require(isinstance(obj,dict),f"候选第 {lineno} 行非 JSON 对象")
            doc_id=obj.get("document_id")
            term_id=obj.get("disease_term_id")
            disease=obj.get("disease_name")
            match=OID_PATTERN.fullmatch(doc_id) if isinstance(doc_id,str) else None
            require(match is not None,f"第 {lineno} 行 Document ID 不是稳定 Mongo OID")
            oid=match.group(1)
            require(TERM_PATTERN.fullmatch(term_id or "") is not None,
                    f"第 {lineno} 行 Term ID 不符合预期")
            require(oid in source_rows,f"第 {lineno} 行无对应原始疾病记录")
            original,line_number,line_hash=source_rows[oid]
            require(
                disease==original
                and obj.get("source_record_oid")==oid
                and obj.get("source_line_number")==line_number
                and obj.get("source_line_sha256")==line_hash
                and obj.get("source_file_sha256")==sha
                and obj.get("source_file")=="medical.json",
                f"第 {lineno} 行源疾病名称/行号/哈希不符"
            )
            require(
                obj.get("name_alignment")=="exact_name_only"
                and obj.get("review_status")=="PENDING_HUMAN_REVIEW"
                and obj.get("proposed_relationship")==RELATIONSHIP
                and obj.get("graph_document_label")==DOC_LABEL
                and obj.get("chroma_collection")==CHROMA_COLLECTION,
                f"第 {lineno} 行不是已审计的严格同名候选"
            )
            require(
                obj.get("source_status")==SOURCE_STATUS
                and obj.get("clinical_verified") is False
                and obj.get("medical_authority_verified") is False
                and obj.get("treatment_relation") is False
                and obj.get("write_to_neo4j") is False
                and obj.get("source_url")=="" and obj.get("source_publisher")=="",
                f"第 {lineno} 行包含未经授权的治疗/审核/写入标志"
            )
            require(doc_id not in doc_ids and term_id not in term_ids and disease not in names,
                    f"第 {lineno} 行候选 ID 或同名疾病重复，停止自动关联")
            doc_ids.add(doc_id)
            term_ids.add(term_id)
            names.add(disease)
            candidates.append(obj)
    require(len(candidates)==expected_candidates,
            f"候选文件仅 {len(candidates)} 条，预期 {expected_candidates}")
    # Avoid mixing a source from another campaign even with the same name.
    require(sha256_file(source)==sha and sha256_file(plan)==plan_sha,
            "扫描期间数据文件被修改")
    return {
        "source_path":str(source),"plan_path":str(plan),
        "source_sha256":sha,"plan_sha256":plan_sha,
        "chroma_audit_sha256":sha256_file(chroma_audit),
        "expected_term_count":expected_graph_terms,
        "rows":candidates,
    }


def _classify_batch(rows:list[dict], *, term_rows:list[dict],
                    doc_rows:list[dict], link_rows:list[dict],
                    plan_sha:str) -> dict[str,str]:
    """Pure diff: MISSING, EXISTING_EXACT, or fail closed on any conflict."""
    grouped_terms={}
    grouped_docs={}
    grouped_links={}
    for record in term_rows:
        term=record.get("term_id")
        require(isinstance(term,str),"Neo4j 返回非法 Term ID")
        grouped_terms.setdefault(term,[]).append(record.get("disease_name"))
    for record in doc_rows:
        key=record.get("document_id")
        require(isinstance(key,str),"Neo4j 返回非法 Document ID")
        grouped_docs.setdefault(key,[]).append(record.get("properties"))
    for record in link_rows:
        key=record.get("document_id")
        require(isinstance(key,str),"Neo4j 返回非法关系 Document ID")
        grouped_links.setdefault(key,[]).append(record)
    statuses={}
    for row in rows:
        cid=row["document_id"]
        term=row["disease_term_id"]
        expected_doc,expected_link=expected_props(row,plan_sha)
        terms=grouped_terms.get(term,[])
        require(len(terms)==1 and terms[0]==row["disease_name"],
                f"DiseaseTerm 不存在/重复/名称不符：{row['disease_name']} ({term})")
        docs=grouped_docs.get(cid,[])
        edges=grouped_links.get(cid,[])
        require(len(docs)<=1 and len(edges)<=1,
                f"Neo4j DiseaseDocument 或其来源关系重复：{cid}")
        if not docs and not edges:
            statuses[cid]="MISSING"
            continue
        require(len(docs)==1 and len(edges)==1,
                f"Neo4j DiseaseDocument 孤立/已有不同来源关系：{cid}")
        edge=edges[0]
        require(
            _same_props(docs[0],expected_doc)
            and edge.get("term_id")==term
            and _same_props(edge.get("properties"),expected_link),
            f"Neo4j 已有同 ID 文档或关系与本次计划冲突，拒绝覆盖：{cid}"
        )
        statuses[cid]="EXISTING_EXACT"
    return statuses


READ_TERMS = """
MATCH (t:DiseaseTerm)
WHERE t.id IN $ids
RETURN t.id AS term_id, t.name AS disease_name
"""
READ_DOCS = """
MATCH (d:DiseaseDocument)
WHERE d.id IN $ids
RETURN d.id AS document_id, properties(d) AS properties
"""
READ_LINKS = f"""
MATCH (t:DiseaseTerm)-[r:{RELATIONSHIP}]->(d:DiseaseDocument)
WHERE d.id IN $ids
RETURN d.id AS document_id, t.id AS term_id, properties(r) AS properties
"""
COUNT_TERMS = "MATCH (t:DiseaseTerm) RETURN count(t) AS n"
# Only CREATE new nodes/relationships after all preflight checks pass.
# No SET applied to pre-existing objects; no MERGE, DELETE, DROP or reset.
CREATE_LINK = f"""
MATCH (t:DiseaseTerm {{id: $term_id, name: $name}})
WITH collect(t) AS terms
WHERE size(terms)=1
WITH terms[0] AS t
OPTIONAL MATCH (old:DiseaseDocument {{id:$doc_id}})
WITH t,collect(old) AS previous
WHERE size(previous)=0
CREATE (d:DiseaseDocument)
SET d = $document_properties
CREATE (t)-[rel:{RELATIONSHIP}]->(d)
SET rel = $relationship_properties
RETURN d.id AS created_document_id
"""


def _query_rows(tx:Any, query:str, **kwargs:Any) -> list[dict]:
    return [dict(row) for row in tx.run(query,**kwargs)]


def _snapshot(tx:Any, rows:list[dict], plan_sha:str) -> dict[str,str]:
    tids=[x["disease_term_id"] for x in rows]
    dids=[x["document_id"] for x in rows]
    return _classify_batch(
        rows,term_rows=_query_rows(tx,READ_TERMS,ids=tids),
        doc_rows=_query_rows(tx,READ_DOCS,ids=dids),
        link_rows=_query_rows(tx,READ_LINKS,ids=dids),
        plan_sha=plan_sha,
    )


def _write_transaction(tx:Any,rows:list[dict],plan_sha:str) -> dict[str,int]:
    before=_snapshot(tx,rows,plan_sha)
    new=0
    reused=0
    for row in rows:
        if before[row["document_id"]]=="EXISTING_EXACT":
            reused+=1
            continue
        document,link=expected_props(row,plan_sha)
        result=_query_rows(
            tx,CREATE_LINK,
            term_id=row["disease_term_id"],
            name=row["disease_name"],
            doc_id=row["document_id"],
            document_properties=document,
            relationship_properties=link,
        )
        require(len(result)==1 and
                result[0].get("created_document_id")==row["document_id"],
                f"原子创建图谱文档失败：{row['document_id']}")
        new+=1
    after=_snapshot(tx,rows,plan_sha)
    require(all(status=="EXISTING_EXACT" for status in after.values()),
            "事务内创建后复核失败，回滚本批")
    return {"new":new,"reused":reused}


def _get_driver():
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(Path.cwd()/".env",override=False)
    uri=os.getenv("NEO4J_URI","").strip()
    username=(os.getenv("NEO4J_USER") or
              os.getenv("NEO4J_USERNAME") or "neo4j")
    password=os.getenv("NEO4J_PASSWORD","")
    require(bool(uri and password),
            "缺少 NEO4J_URI 或 NEO4J_PASSWORD；不允许猜测实例")
    driver=GraphDatabase.driver(uri,auth=(username,password))
    driver.verify_connectivity()
    return driver,os.getenv("NEO4J_DATABASE","neo4j")


def scan_graph(driver:Any, db_name:str, rows:list[dict],
               plan_sha:str,batch_size:int) -> dict:
    count=Counter()
    with driver.session(database=db_name) as session:
        for offset in range(0,len(rows),batch_size):
            piece=rows[offset:offset+batch_size]
            statuses=session.execute_read(
                lambda tx:_snapshot(tx,piece,plan_sha)
            )
            count.update(statuses.values())
    return dict(count)


def run_graph(driver:Any, db_name:str, data:dict, *,
              batch_size:int=40, apply:bool=False,
              max_new:int=20, progress=None) -> dict:
    rows=data["rows"]
    plan_sha=data["plan_sha256"]
    with driver.session(database=db_name) as session:
        actual=session.execute_read(
            lambda tx:tx.run(COUNT_TERMS).single()["n"]
        )
    require(type(actual) is int and actual==data["expected_term_count"],
            f"Neo4j DiseaseTerm 数量为 {actual}，与对齐审计不符")
    before=scan_graph(driver,db_name,rows,plan_sha,batch_size)
    require(before.get("MISSING",0)+before.get("EXISTING_EXACT",0)==len(rows),
            "未完成全量图谱状态扫描")
    added=0
    reused=0
    if apply and before.get("MISSING",0):
        with driver.session(database=db_name) as session:
            for offset in range(0,len(rows),batch_size):
                group=rows[offset:offset+batch_size]
                # Read-only full scan already ensured there were no collisions;
                # per-batch write transaction verifies again before CREATE.
                if added>=max_new:
                    break
                capacity=max_new-added
                group=group[:]
                status=session.execute_read(
                    lambda tx:_snapshot(tx,group,plan_sha)
                )
                selected=[]
                remaining=capacity
                for item in group:
                    if status[item["document_id"]]=="MISSING":
                        if remaining<=0:
                            continue
                        remaining-=1
                    selected.append(item)
                if not selected:
                    continue
                output=session.execute_write(
                    lambda tx:_write_transaction(tx,selected,plan_sha)
                )
                added+=output["new"]
                reused+=output["reused"]
                if progress is not None:
                    progress({
                        "processed_batch_start":offset,
                        "new_nodes_and_links":output["new"],
                        "exact_reused_links":output["reused"],
                        "total_new_this_run":added
                    })
                if added>=max_new:
                    break
    after=scan_graph(driver,db_name,rows,plan_sha,batch_size)
    require(after.get("MISSING",0)+after.get("EXISTING_EXACT",0)==len(rows),
            "写入后图谱全量状态复核失败")
    return {
        "status":(
            "FULL_GRAPH_LINK_AUDIT_OK"
            if after.get("EXISTING_EXACT",0)==len(rows)
            else "PARTIAL_RESEARCH_GRAPH_LINKS"
            if apply else "DRY_RUN_GRAPH_CHECK_OK"
        ),
        "before":before,"after":after,
        "new_nodes_this_run":added,
        "new_relationships_this_run":added,
        "existing_exact_reused_this_run":reused,
        "expected_nodes_and_links":len(rows),
        "no_drug_treats_created":True,
        "medical_review_performed":False,
        "research_only":True,
    }


def _save_report(path:Path|None, output:dict) -> None:
    if path is None:return
    path=Path(path).resolve()
    require(not path.exists(),f"报告已存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(output,f,ensure_ascii=False,indent=2)
        f.write("\n")


def cli(argv:list[str]|None=None) -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--plan-report",type=Path,required=True)
    parser.add_argument("--chroma-audit",type=Path,required=True)
    parser.add_argument("--report",type=Path)
    parser.add_argument("--offline-check",action="store_true")
    parser.add_argument("--apply",action="store_true")
    parser.add_argument("--audit",action="store_true")
    parser.add_argument("--confirm-research-only",default="")
    parser.add_argument("--max-new",type=int,default=20)
    parser.add_argument("--batch-size",type=int,default=40)
    args=parser.parse_args(argv)
    require(not (args.apply and (args.audit or args.offline_check)),
            "--apply 不得与 --audit/--offline-check 同时使用")
    require(not (args.audit and args.offline_check),
            "--audit 与 --offline-check 不得同时使用")
    require(1<=args.batch_size<=MAX_BATCH,"batch_size 必须位于 1～100")
    require(1<=args.max_new<=EXPECTED_CANDIDATES,
            f"max-new 必须位于 1～{EXPECTED_CANDIDATES}")
    require(not args.apply or args.confirm_research_only==RESEARCH_CONFIRM,
            "科研实验写图须提供 --confirm-research-only RESEARCH_UNVERIFIED_ONLY")
    return args


def main(argv:list[str]|None=None, *, injected_driver=None,
         injected_database:str="neo4j") -> int:
    args=cli(argv)
    if args.report:
        require(not args.report.exists(),"目标报告已存在，拒绝覆盖")
    data=validate_plan(
        source=args.source,plan=args.plan,
        report=args.plan_report,chroma_audit=args.chroma_audit
    )
    if args.offline_check:
        result={
            "status":"OFFLINE_INPUTS_VERIFIED_NO_DB",
            "expected_nodes_and_links":len(data["rows"]),
            "source_sha256":data["source_sha256"],
            "plan_sha256":data["plan_sha256"],
            "graph_queried":False,"graph_modified":False,
        }
    else:
        driver=injected_driver
        if driver is None:
            driver,db_name=_get_driver()
        else:
            db_name=injected_database
        try:
            if args.apply:
                with ExclusiveResearchLock(args.plan):
                    result=run_graph(
                        driver,db_name,data,
                        batch_size=args.batch_size,
                        apply=True,
                        max_new=args.max_new,
                        progress=lambda event:print(
                            "[P3-04.4B] batch:",
                            json.dumps(event,ensure_ascii=False),flush=True
                        )
                    )
            else:
                result=run_graph(
                    driver,db_name,data,
                    batch_size=args.batch_size,
                    apply=False,
                    max_new=args.max_new
                )
            if args.audit and result["status"]!="FULL_GRAPH_LINK_AUDIT_OK":
                raise LinkBlocked("全量图谱关联尚未完成，不能通过 --audit")
        finally:
            if injected_driver is None:driver.close()
    result.update({
        "stage":STAGE,
        "mode":"research_only_exact_name_unverified_graph_links",
        "created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "relationship_type":RELATIONSHIP,
        "source_status":SOURCE_STATUS,
        "chroma_collection":CHROMA_COLLECTION,
        "source_sha256":data["source_sha256"],
        "plan_sha256":data["plan_sha256"],
        "clinical_verified":False,
        "source_rights_verified":False,
        "research_prototype":True,
    })
    _save_report(args.report,result)
    print("[P3-04.4B]",result["status"])
    print("Graph candidate count:",len(data["rows"]))
    if "after" in result:
        print("Graph before:",result["before"])
        print("Graph after:",result["after"])
        print("New nodes:",result["new_nodes_this_run"],
              "new relationships:",result["new_relationships_this_run"])
    else:
        print("No Neo4j connection or writes were performed")
    if args.report: print("New report:",args.report.resolve())
    return 0


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
    if hasattr(sys.stderr,"reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8",errors="backslashreplace")
    try:
        sys.exit(main())
    except (LinkBlocked,ValueError,OSError,RuntimeError,ImportError) as exc:
        print("[P3-04.4B] BLOCKED:",str(exc),file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("[P3-04.4B] 操作已中断；已提交批次保持原样，下次安全重跑",file=sys.stderr)
        sys.exit(130)
