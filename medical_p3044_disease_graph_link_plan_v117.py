# -*- coding: utf-8 -*-
"""豆馅医疗 GraphRAG V1.17 — P3-04.4A 疾病百科 -> DiseaseTerm 只读图谱关联审计。

本程序仅生成新 JSONL/CSV/JSON 审计资料。所有 Neo4j 查询只读，
不 CREATE/MERGE/SET/DELETE，不建 Drug-[:TREATS]->Disease 关系。
源文 medical.json 是 JSON Lines，不是普通 JSON 数组。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

from medical_p3041_medical_json_prepare_v117 import (
    get_oid, iterate_source, sha256_file
)
from medical_p3043_disease_neo4j_align_v117 import (
    align_names, neo4j_fetchers, load_env
)

PIPELINE = "P3-04.4A"
LINK_TYPE = "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA"
DOC_LABEL = "DiseaseDocument"
DOC_ID_PREFIX = "medical_json:"
CHROMA_COLLECTION = "disease_knowledge"
SOURCE_STATUS = "third_party_unverified"
MAX_FILE_LINE = 2 * 1024 * 1024
EXPECTED_SOURCE_RECORDS = 8807
EXPECTED_SOURCE_NAMES = 8806
EXPECTED_CHROMA_CHUNKS = 39443

# 图中其他来源可能已经存在 DiseaseDocument，全部视为人工复核事项。
READ_DOCUMENT_CONFLICTS = """
MATCH (doc:DiseaseDocument)
WHERE doc.id IN $document_ids
RETURN doc.id AS document_id
ORDER BY document_id
"""

class PlanBlocked(RuntimeError):
    pass


def _ensure(value: bool, message: str) -> None:
    if not value:
        raise PlanBlocked(message)


def _get_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise PlanBlocked(f"无法读取 JSON：{path}") from exc
    _ensure(isinstance(value,dict),f"JSON 报告必须为对象：{path}")
    return value


def collect_source(source: Path) -> tuple[dict[str,dict],dict[str,list[str]]]:
    """原始记录 OID 不可覆盖，同名源记录不可被合并。"""
    rows={}
    names=defaultdict(list)
    for line_number,entry,line_sha in iterate_source(source):
        oid=get_oid(entry)
        name=entry.get("name")
        _ensure(oid and isinstance(name,str) and name.strip(),
                f"第 {line_number} 行缺少有效 OID/疾病名称")
        _ensure(oid not in rows,f"原始记录 OID 重复：{oid}")
        name=name.strip()
        rows[oid]={
            "source_record_oid":oid,
            "disease_name":name,
            "source_line_number":line_number,
            "source_line_sha256":line_sha,
        }
        names[name].append(oid)
    _ensure(bool(rows),"没有读到疾病记录")
    return rows,dict(names)


def _check_historical_alignment_hash(links_file: Path, expected_sha: str) -> str:
    """Check P3-04.3 mapping integrity on Linux and Windows.

    The P3-04.3 exporter hashes UTF-8 JSONL with LF newlines before writing
    through a text stream. On Windows, Python translates those LF newlines to
    CRLF, changing the file's byte SHA256 but *not* the mapped JSON records.

    Accept either the actual byte-exact hash or one strictly CRLF-delimited
    version of the originally hashed LF JSONL. All other bytes, including
    tampering, mixed line endings, or missing line breaks, remain blocking.
    Does not change or rewrite the source mapping file.
    """
    _ensure(
        isinstance(expected_sha,str)
        and len(expected_sha)==64
        and all(ch in "0123456789abcdef" for ch in expected_sha),
        "P3-04.3 对齐报告中缺少有效 links_sha256"
    )
    actual_sha=sha256_file(links_file)
    if actual_sha==expected_sha:
        return "BYTE_EXACT"
    normalized=hashlib.sha256()
    saw_windows=False
    saw_unix=False
    try:
        with links_file.open("rb") as stream:
            for line_no,line in enumerate(stream,1):
                if line.endswith(b"\r\n"):
                    _ensure(line.count(b"\r")==1,
                            f"映射第 {line_no} 行包含非法回车符")
                    normalized.update(line[:-2]+b"\n")
                    saw_windows=True
                elif line.endswith(b"\n"):
                    _ensure(b"\r" not in line,
                            f"映射第 {line_no} 行包含非法回车符")
                    normalized.update(line)
                    saw_unix=True
                else:
                    raise PlanBlocked(
                        f"P3-04.3 映射文件 SHA256 与对齐审计不一致"
                        f"（第 {line_no} 行没有合法行尾）"
                    )
    except OSError as exc:
        raise PlanBlocked("无法校验 P3-04.3 对齐映射文件") from exc
    _ensure(
        saw_windows and not saw_unix
        and normalized.hexdigest()==expected_sha,
        "P3-04.3 映射文件 SHA256 与对齐审计不一致"
    )
    return "WINDOWS_CRLF_EQUIVALENT"


def _get_alignment(
    report_file:Path,links_file:Path, *,
    source_sha:str,source:dict[str,dict],names:dict[str,list[str]]
) -> tuple[dict,list[dict]]:
    rep=_get_json(report_file)
    _ensure(
        rep.get("stage")=="P3-04.3"
        and rep.get("mode")=="read_only_exact_name_neo4j_alignment"
        and rep.get("status") in (
            "EXACT_NAME_ALIGNMENT_COMPLETE_WITH_GAPS",
            "EXACT_NAMES_ALL_MATCHED"
        )
        and rep.get("new_neo4j_relationships")==0
        and rep.get("medical_conclusion")=="NOT_EVALUATED",
        "P3-04.3 不是合格的只读疾病术语对齐审计报告"
    )
    _ensure(rep.get("source_file_sha256")==source_sha,
            "本地 medical.json 与 P3-04.3 报告 SHA256 不一致")
    rep["link_hash_verified_as"]=_check_historical_alignment_hash(
        links_file,rep.get("links_sha256")
    )
    counts=rep.get("counts")
    _ensure(isinstance(counts,dict),"P3-04.3 报告缺少 counts")
    for field,expected in (
        ("source_records",len(source)),
        ("unique_source_names",len(names)),
        ("matched_source_records",rep.get("links_count"))
    ):
        _ensure(counts.get(field)==expected,
                f"历史 P3-04.3 统计 {field} 不一致")
    _ensure(
        sum(int(counts.get(field,0)) for field in (
            "matched_exact_unique_names","unmatched_names",
            "ambiguous_or_missing_term_id"
        ))==len(names),
        "历史对齐状态数量与疾病名称总数不一致"
    )
    links=[]
    mapped_oids=set()
    name_to_term={}
    term_to_name={}
    with links_file.open("r",encoding="utf-8-sig") as stream:
        for number,line in enumerate(stream,1):
            _ensure(bool(line.strip()),f"第 {number} 行为空映射")
            _ensure(len(line)<=MAX_FILE_LINE,
                    f"第 {number} 行超过映射尺寸上限")
            try:
                item=json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlanBlocked(f"对齐映射第 {number} 行无法解析") from exc
            _ensure(isinstance(item,dict),f"映射第 {number} 行不是对象")
            oid=item.get("medical_json_record_oid")
            name=item.get("disease_name")
            term=item.get("disease_term_id")
            _ensure(
                isinstance(oid,str) and oid in source and oid not in mapped_oids,
                f"映射第 {number} 行无效/重复的原始 OID"
            )
            _ensure(isinstance(name,str) and name==source[oid]["disease_name"],
                    f"映射第 {number} 行疾病名称与原文不符")
            _ensure(isinstance(term,str) and term.strip(),
                    f"映射第 {number} 行 term_id 无效")
            _ensure(
                item.get("match_type")=="exact_name_only"
                and item.get("medical_validation")=="not_performed"
                and item.get("is_treatment_relation") is False,
                f"映射第 {number} 行冒充医学确认或治疗关系"
            )
            _ensure(
                name not in name_to_term or name_to_term[name]==term,
                f"同名疾病 {name} 映射到不一致的 Neo4j Term ID"
            )
            _ensure(
                term not in term_to_name or term_to_name[term]==name,
                f"不同疾病名称错误地共用 Term ID：{term}"
            )
            name_to_term[name]=term
            term_to_name[term]=name
            mapped_oids.add(oid)
            links.append({
                "disease_name":name,
                "medical_json_record_oid":oid,
                "disease_term_id":term,
                "match_type":"exact_name_only",
                "medical_validation":"not_performed",
                "is_treatment_relation":False
            })
    _ensure(len(links)==rep.get("links_count"),
            "映射 JSONL 条数与 P3-04.3 审计报告不一致")
    _ensure(
        len(name_to_term)==counts.get("matched_exact_unique_names"),
        "成功匹配疾病名数量不一致"
    )
    for name,oids in names.items():
        if name in name_to_term:
            _ensure(all(oid in mapped_oids for oid in oids),
                    f"同名疾病 {name} 的部分原始 OID 被遗漏")
        else:
            _ensure(all(oid not in mapped_oids for oid in oids),
                    f"未完成的同名疾病 {name} 不得部分关联")
    return rep,links


def _validate_disease_chroma_audit(source_sha: str, audit: Path) -> dict:
    result=_get_json(audit)
    _ensure(
        result.get("stage")=="P3-04.2"
        and result.get("mode")=="read_only_full_chroma_source_audit"
        and result.get("status")=="PASS_FULL_DISEASE_CHROMA_PROVENANCE",
        "缺少 P3-04.2 真正通过的完整 Chroma 来源审计"
    )
    _ensure(
        result.get("source_sha256")==source_sha,
        "本地 medical.json SHA256 与 P3-04.2 全量审计源文件不一致"
    )
    _ensure(
        result.get("verified_chunks")==EXPECTED_CHROMA_CHUNKS
        and result.get("disease_collection")==CHROMA_COLLECTION
        and result.get("source_status")==SOURCE_STATUS
        and result.get("medical_accuracy_reviewed") is False,
        "P3-04.2 的疾病 Collection 数量、来源或医学状态不一致"
    )
    return result


def _canonical_links(rows: list[dict]) -> list[tuple[str,str,str]]:
    return sorted(
        (x["medical_json_record_oid"],x["disease_name"],
         x["disease_term_id"]) for x in rows
    )


def live_recheck(
    names: dict[str,list[str]], prior: dict, links: list[dict],
    fetch_terms: Callable[[list[str]],list[dict]],
    count_terms: Callable[[],int],
    *,
    fetch_existing_docs: Callable[[list[str]],list[dict]] | None=None,
    batch_size: int=300,
) -> dict:
    """用实际当前 Neo4j 名称+Term ID 重新读一次，拒绝旧映射与新图谱不一致。"""
    if not 1<=batch_size<=1000:
        raise ValueError("batch_size must be between 1 and 1000")
    current_count=count_terms()
    expected_count=prior["counts"].get("neo4j_total_disease_term_nodes")
    _ensure(
        type(current_count) is int and current_count>=0
        and current_count==expected_count,
        f"Neo4j DiseaseTerm 总数已变化（原 {expected_count}，现 {current_count}）；"
        "请重新运行 P3-04.3 同名审计"
    )
    fresh_report,fresh_links=align_names(
        names,fetch_terms,graph_total_terms=current_count,
        batch_size=batch_size
    )
    _ensure(
        _canonical_links(fresh_links)==_canonical_links(links),
        "当前 Neo4j 精确名称/ID 映射与 P3-04.3 审计不一致，拒绝使用旧映射"
    )
    for key in ("matched_exact_unique_names","matched_source_records",
                "unmatched_names","ambiguous_or_missing_term_id"):
        _ensure(
            fresh_report["counts"].get(key,0)==prior["counts"].get(key,0),
            f"当前 Neo4j 对齐计数 {key} 与 P3-04.3 不同"
        )
    doc_collisions=[]
    if fetch_existing_docs is not None:
        doc_ids=sorted({
            DOC_ID_PREFIX+item["medical_json_record_oid"] for item in links
        })
        seen=set()
        for start in range(0,len(doc_ids),batch_size):
            batch=doc_ids[start:start+batch_size]
            returned=fetch_existing_docs(batch)
            for doc in returned:
                raw_id=doc.get("document_id") if isinstance(doc,dict) else None
                _ensure(isinstance(raw_id,str) and raw_id in batch,
                        "现有 DiseaseDocument 查询返回了非请求的文档 ID")
                _ensure(raw_id not in seen,
                        f"已有多个 DiseaseDocument 使用相同 ID：{raw_id}")
                seen.add(raw_id)
                doc_collisions.append(raw_id)
    return {
        "live_verified":True,
        "neo4j_term_count":current_count,
        "rechecked_names":len(names),
        "exact_matched_names":fresh_report["counts"].get(
            "matched_exact_unique_names",0),
        "document_id_conflict_count": (
            len(doc_collisions) if fetch_existing_docs is not None else None
        ),
        "document_id_conflict_examples":doc_collisions[:20],
    }


def build_plan(
    *, source_file:Path, align_report_file:Path, align_links_file:Path,
    chroma_audit_file:Path,
    fetch_terms: Callable[[list[str]],list[dict]] | None=None,
    count_terms:Callable[[],int] | None=None,
    fetch_existing_docs:Callable[[list[str]],list[dict]] | None=None,
    batch_size:int=300,
    expected_records:int=EXPECTED_SOURCE_RECORDS,
    expected_names:int=EXPECTED_SOURCE_NAMES,
) -> tuple[dict,list[dict],list[dict]]:
    """返回摘要/待审核计划/逐记录 CSV，不向任何数据库写入。"""
    source_file=Path(source_file).resolve()
    for path in (source_file,align_report_file,align_links_file,chroma_audit_file):
        _ensure(Path(path).is_file(),f"输入文件不存在：{path}")
    originals,names=collect_source(source_file)
    _ensure(
        len(originals)==expected_records and len(names)==expected_names,
        f"源文件记录/疾病名数量变化：{len(originals)}/{len(names)}，"
        f"预期 {expected_records}/{expected_names}；拒绝使用旧基线"
    )
    original_sha=sha256_file(source_file)
    chroma=_validate_disease_chroma_audit(
        original_sha,Path(chroma_audit_file)
    )
    prior,links=_get_alignment(
        Path(align_report_file),Path(align_links_file),
        source_sha=original_sha,source=originals,names=names
    )
    if (fetch_terms is None)!=(count_terms is None):
        raise ValueError("Neo4j fetch_terms 和 count_terms 必须同时提供")
    recheck={"live_verified":False,
             "neo4j_term_count":None,
             "document_id_conflict_count":None,
             "document_id_conflict_examples":[]}
    if fetch_terms:
        recheck=live_recheck(
            names,prior,links,fetch_terms,count_terms,
            fetch_existing_docs=fetch_existing_docs,batch_size=batch_size
        )
    match_by_oid={x["medical_json_record_oid"]:x for x in links}
    ambiguous={
        x.get("disease_name") for x in prior.get("ambiguous_examples",[])
        if isinstance(x,dict) and isinstance(x.get("disease_name"),str)
    }
    plan=[]
    review=[]
    summary_count=Counter()
    for oid,original in originals.items():
        match=match_by_oid.get(oid)
        name=original["disease_name"]
        if match:
            state="EXACT_NAME_CANDIDATE_UNREVIEWED"
            planned={
                "document_id":DOC_ID_PREFIX+oid,
                "disease_name":name,
                "disease_term_id":match["disease_term_id"],
                "name_alignment":"exact_name_only",
                "proposed_relationship":LINK_TYPE,
                "graph_document_label":DOC_LABEL,
                "chroma_collection":CHROMA_COLLECTION,
                "source_file":"medical.json",
                "source_file_sha256":original_sha,
                "source_record_oid":oid,
                "source_line_number":original["source_line_number"],
                "source_line_sha256":original["source_line_sha256"],
                "source_url":"",
                "source_publisher":"",
                "source_status":SOURCE_STATUS,
                "clinical_verified":False,
                "medical_authority_verified":False,
                "review_status":"PENDING_HUMAN_REVIEW",
                "treatment_relation":False,
                "write_to_neo4j":False,
            }
            plan.append(planned)
        elif name in ambiguous:
            state="AMBIGUOUS_TERM_REQUIRES_REVIEW"
        else:
            # Report lists only the first 30 ambiguous names. Other missing
            # mappings cannot be safely called 'no exact match' without
            # the full live recheck; classify them as unresolved.
            state="UNRESOLVED_NO_UNIQUE_EXACT_MAPPING"
        summary_count[state]+=1
        review.append({
            "disease_name":name,
            "medical_json_record_oid":oid,
            "source_line_number":original["source_line_number"],
            "disease_term_id":match["disease_term_id"] if match else "",
            "technical_match_status":state,
            "source_status":SOURCE_STATUS,
            "medical_review_status":"NOT_PERFORMED",
        })
    _ensure(len(plan)==len(links),
            "待审核计划记录与 P3-04.3 映射数不一致")
    _ensure(len(review)==len(originals),
            "未覆盖源文件全部记录")
    if not plan:
        readiness="NO_EXACT_CANDIDATES_REQUIRES_SOURCE_REVIEW"
    elif not recheck["live_verified"]:
        readiness="OFFLINE_SNAPSHOT_ONLY_NOT_GRAPH_VERIFIED"
    elif recheck["document_id_conflict_count"] is None:
        readiness="GRAPH_VERIFIED_BUT_DOCUMENT_COLLISIONS_NOT_CHECKED"
    elif recheck["document_id_conflict_count"]>0:
        readiness="EXISTING_DOCUMENT_IDS_REQUIRE_REVIEW"
    else:
        readiness="READY_FOR_HUMAN_REVIEW_NO_WRITES"
    report={
        "stage":PIPELINE,
        "mode":"strict_read_only_graph_link_plan",
        "readiness":readiness,
        "created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file_sha256":original_sha,
        "source_file_bytes":source_file.stat().st_size,
        "chroma_collection":CHROMA_COLLECTION,
        "chroma_audit_file":Path(chroma_audit_file).name,
        "chroma_audit_staging_sha256":chroma.get("staging_sha256"),
        "chroma_audited_chunk_count":chroma.get("verified_chunks"),
        "original_disease_records":len(originals),
        "unique_disease_names":len(names),
        "aligned_records_in_review_plan":len(plan),
        "aligned_unique_names":prior["counts"]["matched_exact_unique_names"],
        "alignment_link_hash_mode":prior["link_hash_verified_as"],
        "prior_alignment_counts":prior["counts"],
        "review_record_status_counts":dict(summary_count),
        "graph_recheck":recheck,
        "relation_type_proposed":LINK_TYPE,
        "source_status":SOURCE_STATUS,
        "clinical_verified":False,
        "source_rights_verified":False,
        "human_review_done":False,
        "new_neo4j_nodes":0,
        "new_neo4j_relationships":0,
        "drug_treatment_relationships_created":0,
        "note":(
            "本文件仅提出严格同名的未核验疾病百科来源候选关系，"
            "不意味着确认疾病同义、获批治疗用途或资料拥有合法使用许可。"
            "所有图谱写入必须另行实施经审核的 P3-04.4B。"
        ),
    }
    return report,plan,review


def _save_exclusive(report_path:Path, plan_path:Path,
                    review_path:Path, report:dict,
                    plan:list[dict], review:list[dict]) -> None:
    paths=[p.resolve() for p in (report_path,plan_path,review_path)]
    _ensure(len(set(paths))==3,"三份输出文件不能使用相同路径")
    for target in paths:
        _ensure(not target.exists(),f"目标文件已存在，拒绝覆盖：{target}")
    for target in paths:
        target.parent.mkdir(parents=True,exist_ok=True)
    # 计划和 CSV 先落盘；summary 写最后。失败时不宣称审计完成。
    with paths[1].open("x",encoding="utf-8",newline="\n") as writer:
        for item in plan:
            writer.write(json.dumps(
                item,ensure_ascii=False,sort_keys=True,separators=(",",":")
            )+"\n")
    with paths[2].open("x",encoding="utf-8-sig",newline="") as writer:
        fields=[
            "disease_name","medical_json_record_oid","source_line_number",
            "disease_term_id","technical_match_status","source_status",
            "medical_review_status",
        ]
        csv_writer=csv.DictWriter(writer,fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(review)
    report["plan_sha256"]=sha256_file(paths[1])
    report["review_csv_sha256"]=sha256_file(paths[2])
    report["planned_document_links"]=len(plan)
    with paths[0].open("x",encoding="utf-8") as writer:
        json.dump(report,writer,ensure_ascii=False,indent=2)
        writer.write("\n")


def _live_neo4j_document_reader(driver:Any) -> Callable[[list[str]],list[dict]]:
    database=__import__("os").getenv("NEO4J_DATABASE","neo4j")
    def fetch(document_ids:list[str]) -> list[dict]:
        with driver.session(database=database) as session:
            return session.execute_read(
                lambda tx:[
                    row.data()
                    for row in tx.run(READ_DOCUMENT_CONFLICTS,
                                      document_ids=document_ids)
                ]
            )
    return fetch


def _cli(argv=None) -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--align-report",type=Path,required=True)
    parser.add_argument("--align-links",type=Path,required=True)
    parser.add_argument("--chroma-audit",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--review-csv",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=300)
    parser.add_argument("--offline-preview",action="store_true",
                        help="不读 Neo4j，只校验原始文件与历史对齐快照；不会 READY")
    parser.add_argument("--expect-records",type=int,
                        default=EXPECTED_SOURCE_RECORDS)
    parser.add_argument("--expect-names",type=int,
                        default=EXPECTED_SOURCE_NAMES)
    args=parser.parse_args(argv)
    if args.batch_size<1 or args.batch_size>1000:
        parser.error("--batch-size 必须为 1~1000")
    if args.expect_records<1 or args.expect_names<1:
        parser.error("源记录基线必须大于零")
    return args


def main(argv=None) -> int:
    args=_cli(argv)
    driver=None
    try:
        if args.offline_preview:
            read_terms=count_terms=read_docs=None
        else:
            load_env()
            read_terms,count_terms,driver=neo4j_fetchers()
            read_docs=_live_neo4j_document_reader(driver)
        result,plan,review=build_plan(
            source_file=args.source,
            align_report_file=args.align_report,
            align_links_file=args.align_links,
            chroma_audit_file=args.chroma_audit,
            fetch_terms=read_terms,count_terms=count_terms,
            fetch_existing_docs=read_docs,
            batch_size=args.batch_size,
            expected_records=args.expect_records,
            expected_names=args.expect_names,
        )
        _save_exclusive(
            args.report,args.plan,args.review_csv,result,plan,review
        )
    except (OSError,ValueError,RuntimeError,PlanBlocked) as exc:
        print("[P3-04.4A] BLOCKED:",str(exc),file=sys.stderr)
        return 2
    finally:
        if driver is not None:
            driver.close()
    print("[P3-04.4A]",result["readiness"])
    print("疾病源记录:",result["original_disease_records"])
    print("不同疾病名称:",result["unique_disease_names"])
    print("严格同名匹配名称:",result["aligned_unique_names"])
    print("待人工审核疾病文档关联:",result["aligned_records_in_review_plan"])
    print("Neo4j 查询核验:",result["graph_recheck"]["live_verified"])
    print("实际写入 Neo4j 节点/关系: 0 / 0")
    print("计划 JSONL:",args.plan.resolve())
    print("待审核 CSV:",args.review_csv.resolve())
    print("审计报告:",args.report.resolve())
    return 0


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
    if hasattr(sys.stderr,"reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8",errors="backslashreplace")
    sys.exit(main())
