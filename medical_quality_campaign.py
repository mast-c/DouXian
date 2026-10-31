# -*- coding: utf-8 -*-
"""豆馅 Medical GraphRAG V1.17：本地批量质量基线与人工医学语义审核。

只读取 V1.16 的单轮导出 JSON 与人工 CSV；所有操作仅写报告，不联网，
不对 Neo4j/Chroma 读写，不改 RAG 主程序。
通过 V1.16 工程证据检查 ≠ 医学/临床审核通过。

  python medical_quality_campaign.py collect --input-dir medical_traces --output-dir medical_eval_v117
  python medical_quality_campaign.py score --campaign medical_eval_v117/engineering_manifest.json --reviews medical_eval_v117/manual_review.csv
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from medical_quality_acceptance import analyze_export, MAX_FILE_BYTES

MANIFEST_SCHEMA=1
REVIEW_COLUMNS=(
    "case_id", "source_filename", "source_sha256",
    "question", "answer_excerpt",
    "engineering_status", "engineering_issue_codes",
    "unverified_disease_candidate_count", "review_focus",
    "review_status", "evidence_support", "relation_safety",
    "medical_safety", "source_authority", "reviewer", "review_note",
)
REVIEW_STATUS={"unreviewed","reviewed","excluded"}
SUPPORT_VALUES={"supported","partially_supported","unsupported","uncertain"}
RELATION_VALUES={"adequately_caveated","overstated","not_applicable","uncertain"}
MEDICAL_SAFETY_VALUES={"acceptable","needs_correction","uncertain"}
AUTHORITY_VALUES={"verified_official","third_party_unverified","uncertain"}

MAX_FILES=500
MAX_ANSWER_PREVIEW=200


def digest(raw:bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_question(question:Any) -> str:
    """比较重跑问题时忽略可见空白差异，但保留药名与标点。"""
    return " ".join(str(question or "").split()).strip()


def issue_code_counts(case_reports:Iterable[dict[str,Any]]) -> dict[str,int]:
    counter:Counter[str]=Counter()
    for case in case_reports:
        for item in case["issues"]:
            counter[str(item["code"])]+=1
    return dict(sorted(counter.items(),key=lambda kv:(-kv[1],kv[0])))


def _safe_csv(value:Any) -> str:
    text=str(value if value is not None else "")
    # 用户问题/检索原文属于不可信字符串，不应在 Excel 中执行公式。
    return "'"+text if text.startswith(("=","+","-","@","\t","\r")) else text


def _read_csv(value:Any) -> str:
    text=str(value or "")
    return text[1:] if text.startswith("'") and text[1:2] in ("=","+","-","@","\t","\r") else text


def _assert_new_paths(paths:Iterable[Path]) -> None:
    existing=[str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError(
            "为避免覆盖已人工标注的 CSV 或旧评测成果，停止执行。"
            "请使用新的 --output-dir / --output：\n"+"\n".join(existing)
        )


def _source_title(payload:dict[str,Any]) -> str:
    return str(payload.get("question") or "")[:400]


def _focus(question:Any, report:dict[str,Any], trace:dict[str,Any]) -> str:
    text=canonical_question(question)
    focus=[]
    if re.search(r"有哪些药|药物有哪些|什么药|哪些药|治疗.{0,20}药",text):
        focus.append("disease_drug_scope")
    if re.search(r"不良反应|副作用|注意事项|禁忌|相互作用",text):
        focus.append("risk_fields")
    if re.search(r"适应症|适应证|治疗|预防|风险|病因",text):
        focus.append("medical_relation")
    if any(
        x.get("evidence_type") == "unverified_drug_disease_candidate"
        for x in trace.get("knowledge_evidence",[])
        if isinstance(x,dict)
    ):
        focus.append("unreviewed_candidate")
    if report["status"] == "FAIL":
        focus.append("engineering_error")
    return "|".join(dict.fromkeys(focus)) or "citation_grounding"


def _load_export(path:Path) -> tuple[dict[str,Any],bytes]:
    if path.is_symlink():
        raise ValueError(f"不接受符号链接作为验收输入：{path}")
    if not path.is_file():
        raise ValueError(f"文件不是普通文件：{path}")
    size=path.stat().st_size
    if size<=0 or size>MAX_FILE_BYTES:
        raise ValueError(f"导出文件为空或超过单轮 15MB 上限：{path}")
    raw=path.read_bytes()
    payload=json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload,dict) or not isinstance(payload.get("trace"),dict):
        raise ValueError(f"缺少真实单轮 trace 对象：{path}")
    if payload.get("demo_warning") or payload.get("is_synthetic"):
        raise ValueError(
            f"检测到合成/演示数据，不能混进真实质量基线：{path}"
        )
    if payload.get("schema_version") not in (None,1):
        raise ValueError(f"不支持的导出版本：{path}")
    if not isinstance(payload.get("question"),str) or not payload["question"].strip():
        raise ValueError(f"导出缺少本轮真实问题：{path}")
    if not isinstance(payload.get("answer"),str) or not payload["answer"].strip():
        raise ValueError(f"导出缺少本轮真实回答：{path}")
    if not isinstance(payload["trace"].get("knowledge_evidence"),list):
        raise ValueError(f"导出不含可检查的 evidence 列表：{path}")
    return payload,raw


def evaluate_files(
    input_dir:Path, *, pattern:str="medical_trace_*.json",
) -> tuple[list[dict[str,Any]],list[dict[str,str]]]:
    """读取已有单轮 exports，防止重复导出/损坏文件夸大覆盖率。"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"找不到导出目录：{input_dir}")
    if "/" in pattern or "\\" in pattern or ".." in pattern:
        raise ValueError("只允许目录内简单文件名通配符，例如 medical_trace_*.json")
    paths=sorted(input_dir.glob(pattern),key=lambda p:p.name)
    if not paths:
        raise FileNotFoundError(
            f"目录中没有匹配 {pattern!r} 的单轮导出 JSON：{input_dir}"
        )
    if len(paths)>MAX_FILES:
        raise ValueError(f"每次最多处理 {MAX_FILES} 个导出文件")

    results=[]
    invalid=[]
    already_seen:set[str]=set()
    for path in paths:
        try:
            payload,raw=_load_export(path)
            fingerprint=digest(raw)
            if fingerprint in already_seen:
                invalid.append({
                    "source_filename":path.name,
                    "reason":"导出内容与前一个文件完全相同，已去重，避免重复计入样本",
                    "kind":"duplicated_export",
                })
                continue
            already_seen.add(fingerprint)
            check=analyze_export(payload)
            trace=payload["trace"]
            evidence=trace.get("knowledge_evidence",[])
            diseases=sum(
                isinstance(item,dict)
                and item.get("evidence_type")=="unverified_drug_disease_candidate"
                for item in evidence
            )
            results.append({
                "case_id":"QA"+fingerprint[:16].upper(),
                "source_filename":path.name,
                "source_sha256":fingerprint,
                "question":payload["question"],
                "answer_excerpt":payload["answer"][:MAX_ANSWER_PREVIEW],
                "review_focus":_focus(payload["question"],check,trace),
                "unverified_disease_candidate_count":diseases,
                "engineering_status":check["status"],
                "engineering_issue_codes":[x["code"] for x in check["issues"]],
                "engineering_summary":check["summary"],
                "metrics":check["metrics"],
                "issues":check["issues"],
            })
        except (ValueError,UnicodeError,TypeError,KeyError,IndexError,OSError) as exc:
            invalid.append({
                "source_filename":path.name,
                "reason":str(exc)[:280],
                "kind":"invalid_export",
            })
    return results,invalid


def engineering_summary(cases:list[dict[str,Any]],invalid:list[dict[str,str]]) -> dict[str,Any]:
    statuses=Counter(c["engineering_status"] for c in cases)
    counts=Counter()
    multi_expected=multi_cited=multi_missing=0
    invalid_citations=0
    for case in cases:
        m=case["metrics"]
        counts["evidence_cards"]+=int(m.get("evidence_cards") or 0)
        invalid_citations+=len(m.get("invalid_cited_ids") or [])
        multi=m.get("multi_drug")
        if isinstance(multi,dict):
            counts["multi_drug_cases"]+=1
            multi_expected+=int(multi.get("expected_cells") or 0)
            multi_cited+=int(multi.get("correctly_cited_cells") or 0)
            multi_missing+=int(multi.get("false_missing_cells") or 0)
        counts["candidate_cards"]+=case["unverified_disease_candidate_count"]
        counts["definite_duplicates"]+=int(m.get("definite_section_duplicates") or 0)

    n=len(cases)
    distinct_questions=len({
        canonical_question(case["question"]) for case in cases
    })
    return {
        "valid_unique_runs":n,
        "distinct_questions":distinct_questions,
        "repeated_question_runs":max(0,n-distinct_questions),
        "invalid_or_duplicate_files":len(invalid),
        "engineering_status_counts":{
            key:statuses[key] for key in ("PASS","WARN","FAIL")
        },
        "engineering_pass_fraction":round(statuses["PASS"]/n,4) if n else None,
        "evidence_cards":counts["evidence_cards"],
        "invalid_body_citations":invalid_citations,
        "definite_duplicate_sections":counts["definite_duplicates"],
        "unverified_candidate_cards":counts["candidate_cards"],
        "structured_multi_drug_runs":counts["multi_drug_cases"],
        "multi_drug_expected_cells":multi_expected,
        "multi_drug_correctly_cited_cells":multi_cited,
        "multi_drug_citation_coverage":round(multi_cited/multi_expected,4) if multi_expected else None,
        "multi_drug_false_missing_cells":multi_missing,
        "issue_code_frequency":issue_code_counts(cases),
        "notes":[
            "自动验收仅检查引用编号、同源 Section 身份、药品栏目覆盖等工程约束。",
            "不能由工程 PASS 推定答案医学正确、官方批准或临床安全。",
            "如果本次没有人工医学审核，不能声称医疗问答的真实正确率。",
        ],
    }


def _write_json(path:Path,content:Any) -> None:
    path.write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding="utf-8")


def _write_review_csv(path:Path,cases:list[dict[str,Any]]) -> None:
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for case in cases:
            cells={
                "case_id":case["case_id"],
                "source_filename":case["source_filename"],
                "source_sha256":case["source_sha256"],
                "question":case["question"],
                "answer_excerpt":case["answer_excerpt"],
                "engineering_status":case["engineering_status"],
                "engineering_issue_codes":"|".join(case["engineering_issue_codes"]),
                "unverified_disease_candidate_count":case["unverified_disease_candidate_count"],
                "review_focus":case["review_focus"],
                "review_status":"unreviewed",
                "evidence_support":"",
                "relation_safety":"",
                "medical_safety":"",
                "source_authority":"",
                "reviewer":"",
                "review_note":"",
            }
            writer.writerow({k:_safe_csv(cells[k]) for k in REVIEW_COLUMNS})


def collect(
    input_dir:Path,output_dir:Path,*,pattern:str="medical_trace_*.json",
    snapshot_label:str="",
) -> dict[str,Any]:
    cases,invalid=evaluate_files(input_dir,pattern=pattern)
    if not cases:
        raise ValueError("没有有效的单轮导出文件，拒绝生成空评测基线")
    paths=[
        output_dir/"engineering_manifest_v117.json",
        output_dir/"manual_review_v117.csv",
        output_dir/"engineering_overview_v117.md",
    ]
    _assert_new_paths(paths)
    output_dir.mkdir(parents=True,exist_ok=True)
    stats=engineering_summary(cases,invalid)
    manifest={
        "schema_version":MANIFEST_SCHEMA,
        "campaign_version":"V1.17",
        "snapshot_label":str(snapshot_label or ""),
        "source_directory":str(input_dir.resolve()),
        "glob_pattern":pattern,
        "summary":stats,
        "cases":cases,
        "invalid_files":invalid,
    }
    _write_json(paths[0],manifest)
    _write_review_csv(paths[1],cases)
    overview=[
        "# 医疗 GraphRAG V1.17 工程评测汇总",
        "",
        f"- 有效且去重后的运行样本：**{stats['valid_unique_runs']}**",
        f"- 不同的实际问题：**{stats['distinct_questions']}**",
        f"- 重复问题的复测次数：**{stats['repeated_question_runs']}**",
        f"- 无效或重复导出：**{stats['invalid_or_duplicate_files']}**",
        "- 工程状态 PASS / WARN / FAIL："
        + " / ".join(str(stats["engineering_status_counts"][x])
                     for x in ("PASS","WARN","FAIL")),
        f"- 证据卡：**{stats['evidence_cards']}**",
        f"- 不存在的引用编号：**{stats['invalid_body_citations']}**",
        f"- 确定同源 Section 重复：**{stats['definite_duplicate_sections']}**",
        f"- 多药栏目正确引用：**{stats['multi_drug_correctly_cited_cells']}/"
        f"{stats['multi_drug_expected_cells']}**",
        f"- 错误声明栏目缺失：**{stats['multi_drug_false_missing_cells']}**",
        "",
        "## 说明",
        "",
        "这份报告只衡量工程证据/引用约束，不计算医学正确率。",
        "请先在 `manual_review_v117.csv` 中逐条做人工医学语义审阅。",
        "人工审核后再用 `score` 生成独立的人工质量指标。",
    ]
    paths[2].write_text("\n".join(overview)+"\n",encoding="utf-8")
    return manifest


def load_manifest(path:Path) -> dict[str,Any]:
    data=json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data,dict) or data.get("schema_version")!=MANIFEST_SCHEMA:
        raise ValueError("V1.17 工程 Manifest 版本不匹配")
    cases=data.get("cases")
    if not isinstance(cases,list) or not cases:
        raise ValueError("评测 Manifest 没有有效案例")
    ids=[r.get("case_id") for r in cases if isinstance(r,dict)]
    if len(ids)!=len(set(ids)) or len(ids)!=len(cases) or not all(ids):
        raise ValueError("Manifest case_id 缺失或重复")
    if not all(
        isinstance(c.get("source_sha256"),str)
        and len(c["source_sha256"])==64
        and c.get("case_id")=="QA"+c["source_sha256"][:16].upper()
        for c in cases
    ):
        raise ValueError("Manifest 案例 ID 与导出 SHA256 不一致")
    reported=(data.get("summary") or {}).get("engineering_status_counts")
    actual=Counter(c.get("engineering_status") for c in cases)
    if not isinstance(reported,dict) or any(
        reported.get(s,0)!=actual.get(s,0) for s in ("PASS","WARN","FAIL")
    ):
        raise ValueError("Manifest 工程统计与案例明细不一致")
    return data


def parse_reviews(path:Path,manifest:dict[str,Any]) -> tuple[list[dict[str,Any]],dict[str,int]]:
    expected={case["case_id"]:case for case in manifest["cases"]}
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or not set(REVIEW_COLUMNS).issubset(reader.fieldnames):
            raise ValueError("CSV 缺少 V1.17 人工审核模板要求的列")
        rows=[]
        seen=set()
        for line_no,src in enumerate(reader,2):
            row={name:_read_csv(src.get(name,"")).strip()
                 for name in REVIEW_COLUMNS}
            case_id=row["case_id"]
            if case_id not in expected or case_id in seen:
                raise ValueError(f"第 {line_no} 行 case_id 不属于本次 Manifest 或重复")
            seen.add(case_id)
            if row["source_sha256"]!=expected[case_id]["source_sha256"]:
                raise ValueError(
                    f"{case_id} 的导出文件 SHA256 被改动，不能跨不同版本混合标注"
                )
            for key in ("question","source_filename","engineering_status"):
                if row[key]!=str(expected[case_id][key]):
                    raise ValueError(f"{case_id} 的 {key} 与 Manifest 不一致")
            status=row["review_status"]
            if status not in REVIEW_STATUS:
                raise ValueError(f"{case_id} 的 review_status 无效：{status}")
            if status=="reviewed":
                for column,choices in (
                    ("evidence_support",SUPPORT_VALUES),
                    ("relation_safety",RELATION_VALUES),
                    ("medical_safety",MEDICAL_SAFETY_VALUES),
                    ("source_authority",AUTHORITY_VALUES),
                ):
                    if row[column] not in choices:
                        raise ValueError(f"{case_id} 的 {column} 未审核或不在允许的取值中")
                if not row["reviewer"]:
                    raise ValueError(f"{case_id} 标为 reviewed，但缺少审核人")
                if (row["source_authority"]=="verified_official"
                        and not row["review_note"]):
                    raise ValueError(
                        f"{case_id} 标记官方核验，review_note 必须记录具体产品、"
                        "文件版本与核查依据"
                    )
            elif status=="excluded" and not row["review_note"]:
                raise ValueError(f"{case_id} 排除样本必须写 review_note")
            rows.append(row)
    if seen!=set(expected):
        raise ValueError("CSV 缺少 Manifest 中的案例，不能计算有代表性的分母")
    return rows,dict(Counter(row["review_status"] for row in rows))


def manual_review_summary(
    manifest:dict[str,Any],rows:list[dict[str,Any]],
    *,min_reviewed:int=20,
) -> dict[str,Any]:
    if not 1<=min_reviewed<=500:
        raise ValueError("min_reviewed 必须介于 1～500")
    reviewed=[x for x in rows if x["review_status"]=="reviewed"]
    excluded=[x for x in rows if x["review_status"]=="excluded"]
    unreviewed=[x for x in rows if x["review_status"]=="unreviewed"]
    denominator=len(reviewed)
    source_questions={
        c["case_id"]:canonical_question(c["question"])
        for c in manifest["cases"]
    }
    reviewed_unique_questions=len({
        source_questions[x["case_id"]] for x in reviewed
    })
    counter={
        "evidence_support":Counter(row["evidence_support"] for row in reviewed),
        "relation_safety":Counter(row["relation_safety"] for row in reviewed),
        "medical_safety":Counter(row["medical_safety"] for row in reviewed),
        "source_authority":Counter(row["source_authority"] for row in reviewed),
    }

    def fraction(key:str,value:str) -> float|None:
        return (round(counter[key][value]/denominator,4) if denominator else None)

    critical=[
        {
            "case_id":case["case_id"],
            "evidence_support":case["evidence_support"],
            "relation_safety":case["relation_safety"],
            "medical_safety":case["medical_safety"],
        }
        for case in reviewed
        if case["evidence_support"] in ("unsupported","partially_supported")
        or case["relation_safety"]=="overstated"
        or case["medical_safety"]=="needs_correction"
    ]
    engineering_failures=int(
        manifest.get("summary",{})
        .get("engineering_status_counts",{}).get("FAIL") or 0
    )
    unresolved=[
        row["case_id"] for row in reviewed
        if row["evidence_support"]=="uncertain"
        or row["relation_safety"]=="uncertain"
        or row["medical_safety"]=="uncertain"
    ]
    if engineering_failures:
        quality_status="ENGINEERING_NEEDS_FIX"
    elif critical:
        quality_status="REVIEW_AND_FIX"
    elif not reviewed:
        quality_status="NO_HUMAN_REVIEW"
    elif reviewed_unique_questions<min_reviewed:
        quality_status="INSUFFICIENT_SAMPLE"
    elif unresolved:
        quality_status="REVIEW_UNCERTAIN"
    else:
        quality_status="BASELINE_RECORDED"

    # These are separate denominators, intentionally not combined into a
    # clinical "confidence score".
    return {
        "schema_version":MANIFEST_SCHEMA,
        "campaign_version":"V1.17",
        "quality_status":quality_status,
        "engineering_status_counts":manifest["summary"]["engineering_status_counts"],
        "review_counts":{
            "all_valid_cases":len(rows),"reviewed":denominator,
            "reviewed_unique_questions":reviewed_unique_questions,
            "repeated_reviewed_question_runs":max(
                0,denominator-reviewed_unique_questions
            ),
            "unreviewed":len(unreviewed),"excluded":len(excluded),
            "minimum_distinct_questions_for_baseline":min_reviewed,
        },
        "human_review_rates":{
            "fully_supported_fraction":fraction("evidence_support","supported"),
            "unsupported_fraction":fraction("evidence_support","unsupported"),
            "partially_supported_fraction":fraction("evidence_support","partially_supported"),
            "overstated_relation_fraction":fraction("relation_safety","overstated"),
            "medical_needs_correction_fraction":fraction("medical_safety","needs_correction"),
            "third_party_only_fraction":fraction("source_authority","third_party_unverified"),
        },
        "annotation_counts":{
            key:dict(value) for key,value in counter.items()
        },
        "critical_cases":critical,
        "unresolved_case_ids":unresolved,
        "case_ids":[row["case_id"] for row in reviewed],
        "notes":[
            "评分分母仅为 reviewed 的人工评测运行，不包括待审核与排除；达到正式最小样本量还要求足够数量的不同问题，同一题重复运行不能充当多道不同题。",
            "工程状态 PASS 并非医学审核通过；临床或药监核准状态需独立确认。",
            "BASELINE_RECORDED 只表示达到了本轮最低人工样本量且未标出关键错误，"
            "不代表系统安全性通过临床验证。",
        ],
    }


def attach_retrieval_reports(paths:list[Path]) -> dict[str,Any]:
    """V1.13 BM25/Hybrid 指标单独附录，拒绝跨语料冒充同条件对比。

    这两份指标评的是已标注的原文 Section 检索；V1.17 单轮问答
    评的是引用/回答质量。二者样本和分母不同，绝不能相加或算混合准确率。
    """
    if len(paths)>2:
        raise ValueError("最多导入 2 份 V1.13 检索评分报告（BM25 + Hybrid）")
    summaries=[]
    full=[]
    for path in paths:
        data=json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data,dict):
            raise ValueError(f"{path} 不是合法 V1.13 检索报告")
        summary=data.get("summary")
        if (not isinstance(summary,dict)
            or not isinstance(summary.get("raw"),dict)
            or not isinstance(summary.get("scoped"),dict)
            or not isinstance(summary.get("top_k"),int)
            or not data.get("mode")
            or not data.get("collection")):
            raise ValueError(f"{path} 缺少 V1.13 分开检索结果的必需字段")
        if not isinstance(data.get("results"),list) or not data["results"]:
            raise ValueError(f"{path} 没有 Gold Section 测试结果")
        full.append(data)
        summaries.append({
            "filename":path.name,
            "mode":str(data["mode"]),
            "collection":str(data["collection"]),
            "collection_fingerprint":str(data.get("collection_fingerprint") or ""),
            "top_k":summary["top_k"],
            "gold_positive_cases":summary.get("positive_reviewed_cases"),
            "raw":summary["raw"],
            "scoped":summary["scoped"],
        })
    outcome={
        "reports":summaries,
        "sample_sets_independent_of_generation_quality":True,
        "comparison_status":"ONLY_ONE_REPORT" if len(full)==1 else "NOT_REQUESTED",
        "comparison":None,
        "warning":"V1.13 检索 Gold 与 V1.17 回答医学审核不共享分母，不能合并成一个准确率。",
    }
    if len(full)==2:
        a,b=full
        # 用 V1.13 原有对照规则确保两份报告的 Gold 与语料完全相同。
        from medical_eval_workbench import compare_evaluation_reports
        try:
            comparison=compare_evaluation_reports(a,b)
            outcome["comparison_status"]="COMPARABLE_SAME_SNAPSHOT_AND_GOLD"
            outcome["comparison"]=comparison
        except (ValueError,KeyError,TypeError) as exc:
            outcome["comparison_status"]="NOT_COMPARABLE"
            outcome["warning"]+=f" 这两份检索报告不可公平比较：{exc}"
    return outcome


def run_score(manifest_file:Path,reviews_file:Path,output_file:Path,
              *,min_reviewed:int=20,
              retrieval_reports:list[Path]|None=None) -> dict[str,Any]:
    _assert_new_paths([output_file])
    manifest=load_manifest(manifest_file)
    reviews,_=parse_reviews(reviews_file,manifest)
    report=manual_review_summary(manifest,reviews,min_reviewed=min_reviewed)
    if retrieval_reports:
        report["v113_retrieval_benchmark"] = attach_retrieval_reports(
            retrieval_reports
        )
    output_file.parent.mkdir(parents=True,exist_ok=True)
    _write_json(output_file,report)
    return report


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(
        description="V1.17 医疗 GraphRAG 多轮工程验收与人工语义质量基线"
    )
    sub=parser.add_subparsers(dest="command",required=True)
    collect_parser=sub.add_parser("collect",help="只读批量检验 V1.16 的单轮 JSON 导出")
    collect_parser.add_argument("--input-dir",type=Path,required=True)
    collect_parser.add_argument("--output-dir",type=Path,required=True)
    collect_parser.add_argument("--pattern",default="medical_trace_*.json")
    collect_parser.add_argument(
        "--snapshot-label",default="",
        help="手工备注所用数据库快照标签；脚本不据此冒充已自动校验数据库版本"
    )
    score_parser=sub.add_parser("score",help="对人工审核完成的 CSV 生成真实质量基线")
    score_parser.add_argument("--campaign",required=True,type=Path)
    score_parser.add_argument("--reviews",required=True,type=Path)
    score_parser.add_argument("--output",type=Path,required=True)
    score_parser.add_argument("--min-reviewed",type=int,default=20)
    score_parser.add_argument(
        "--retrieval-report",type=Path,action="append",default=[],
        help="可选附加 V1.13 Gold Section 检索报告（最多两份，不混算分母）"
    )

    args=parser.parse_args(argv)
    try:
        if args.command=="collect":
            result=collect(
                args.input_dir,args.output_dir,pattern=args.pattern,
                snapshot_label=args.snapshot_label,
            )
            m=result["summary"]
            print(f"[COLLECT] 有效样本 {m['valid_unique_runs']}，"
                  f"无效/重复 {m['invalid_or_duplicate_files']}")
            print("[ENGINEERING]",m["engineering_status_counts"])
            print(
                f"[CITATION] 错误引用 {m['invalid_body_citations']}，"
                f"重复 Section {m['definite_duplicate_sections']}，"
                f"多药栏目正确引用 {m['multi_drug_correctly_cited_cells']}/"
                f"{m['multi_drug_expected_cells']}"
            )
            print("输出目录：",args.output_dir.resolve())
            return 2 if any(x["kind"]=="invalid_export"
                            for x in result["invalid_files"]) else 0
        report=run_score(
            args.campaign,args.reviews,args.output,
            min_reviewed=args.min_reviewed,
            retrieval_reports=args.retrieval_report,
        )
        print("[HUMAN REVIEW]",report["quality_status"])
        print("[SAMPLES]",report["review_counts"])
        print("[RATES]",report["human_review_rates"])
        print("人工审核质量报告：",args.output.resolve())
        return 0
    except (OSError,ValueError,KeyError,FileExistsError) as exc:
        parser.exit(2, f"执行失败：{exc}\n")


if __name__=="__main__":
    raise SystemExit(main())
