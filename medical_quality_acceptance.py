# -*- coding: utf-8 -*-
"""豆馅 Medical GraphRAG V1.16：离线检索证据质量验收。

输入是聊天页面导出的“单轮问题 + 回答 + trace”JSON。
只使用 Python 标准库：不连接 Neo4j/Chroma、不开云端、不修改原数据。
工程级质量检查不能替代官方说明书与医学专家审核。

运行：
    python medical_quality_acceptance.py --input medical_trace_001.json
    python medical_quality_acceptance.py --input medical_trace_001.json --output report.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable

EXPORT_SCHEMA = 1
MAX_FILE_BYTES = 15 * 1024 * 1024

EVIDENCE_TYPES = {
    "unverified_drug_disease_candidate": "疾病提及候选",
    "unverified_drug_source_section": "Neo4j 原文章节",
    "json_neo4j_text_verified": "Chroma → Neo4j 原文核验",
}
SECTION_TYPES = {"unverified_drug_source_section", "json_neo4j_text_verified"}
FIELDS_CN = {
    "indication": "适应症",
    "adverse_reactions": "不良反应",
    "warnings": "注意事项",
    "contraindications": "禁忌",
    "drug_interactions": "药物相互作用",
}
EXPORTED_TRACE_KEYS = (
    "knowledge_evidence", "retrieval_pipeline", "retrieval_summary",
    "multi_drug_scope", "multi_drug_coverage", "medical_answer_mode",
    "graph_drug_scope", "evidence_entity_gate",
    "evidence_coverage", "self_rag_retry", "medical_safety_guidance", "total_seconds",
)
_REF_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*{1,2})?"
    r"(?:参考资料|参考文献|引用来源)"
    r"(?:（[^）\n]{0,100}）|\([^)]{0,100}\))?"
    r"(?:\*{1,2})?\s*[:：]?(?:\s|$)"
)
_CITES = re.compile(r"(?<!\[)\[(\d{1,4})\](?!\()")
_HEADING = re.compile(r"^\s*###\s+(.+?)\s*$")
_SECTION_ORIGINAL = ("证据原文：", "原文：")
_SOURCE_END_MARKERS = ("\n重要：", "\n资料状态：", "\n来源核验状态：")


def make_export_payload(question: str, answer: str, trace: Any) -> dict[str, Any]:
    """只打包当前回合需要的字段，不导出其他会话、账号或密钥。"""
    if not isinstance(trace, dict):
        trace = {}
    return {
        "schema_version": EXPORT_SCHEMA,
        "question": str(question or ""),
        "answer": str(answer or ""),
        "trace": {
            key: trace[key] for key in EXPORTED_TRACE_KEYS if key in trace
        },
        "scope_notice": (
            "此文件包含本轮用户问题、模型回答及检索的第三方药品原文。"
            "它用于本地工程验收，并非权威临床审核证明。"
        ),
    }


def split_answer_and_reference_footer(text: str) -> tuple[str, int]:
    """去掉程序自动添加的参考资料块再统计正文真正使用的 [n]。"""
    lines = str(text or "").splitlines()
    indices = [i for i, line in enumerate(lines) if _REF_HEADING.match(line)]
    return "\n".join(lines[:indices[0]]) if indices else "\n".join(lines), len(indices)


def cited_numbers(text: str) -> set[int]:
    return {int(num) for num in _CITES.findall(text or "")}


def extracted_source_text(item: dict[str, Any]) -> str:
    """仅用于判断栏目证据是否存在；不作为医学正确性的推断。"""
    if not isinstance(item, dict):
        return ""
    source = str(item.get("content") or "")
    for marker in _SECTION_ORIGINAL:
        if marker in source:
            text = source.split(marker, 1)[1]
            for stop in _SOURCE_END_MARKERS:
                text = text.split(stop, 1)[0]
            return text.strip()
    return ""


def evidence_index(trace: dict[str, Any]) -> list[dict[str, Any]]:
    items = trace.get("knowledge_evidence")
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def section_duplicate_groups(
    evidence: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """仅相同药品+同一 record_id/section_id 的原文章节可判断确定重复。

    单药不同来源、不同制剂、不同栏目，以及 DiseaseCandidate 不参与此去重；
    多个不同 Chroma Chunk 属于同一父 Section 时只标“待人工核对”，
    不能因两个 Chunk 都在一个 Section 就误报为重复医学来源。
    """
    group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    missing: list[dict[str, Any]] = []
    for item in evidence:
        if item.get("evidence_type") not in SECTION_TYPES:
            continue
        drug = str(item.get("drug_name") or "").strip()
        record = str(item.get("record_id") or "").strip()
        parent = str(item.get("section_id") or "").strip()
        if not (drug and record and parent):
            missing.append({
                "evidence_id": item.get("id"),
                "drug_name": drug,
                "record_id": record,
                "section_id": parent,
            })
            continue
        group[(drug, record, parent)].append(item)

    repeats = []
    for (drug, record, parent), items in group.items():
        if len(items) <= 1:
            continue
        methods = {str(x.get("evidence_type")) for x in items}
        full_neo = sum(
            x.get("evidence_type") == "unverified_drug_source_section"
            for x in items
        )
        chroma = [x for x in items
                  if x.get("evidence_type") == "json_neo4j_text_verified"]
        exact_chroma = (
            len(chroma) > 1 and
            len({str(x.get("content") or "") for x in chroma}) < len(chroma)
        )
        definite = (
            (full_neo > 0 and bool(chroma))
            or full_neo > 1
            or exact_chroma
        )
        repeats.append({
            "drug_name": drug,
            "record_id": record, "section_id": parent,
            "evidence_ids": [x.get("id") for x in items],
            "methods": sorted(methods),
            "duplicate_type": ("definite" if definite else "same_parent_different_chunks"),
        })
    return repeats, missing


def candidate_same_quote_groups(
    evidence: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """识别同一句话拆成多个 UI 候选卡的情况，保留疾病术语差异。

    属于“可以合并展示的原文句子”，不是数据库候选关系本身多余；
    不擅自删除原始 DiseaseCandidate。
    """
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        if item.get("evidence_type") != "unverified_drug_disease_candidate":
            continue
        quote = " ".join(str(item.get("evidence_quote") or "").split())
        record = str(item.get("record_id") or "").strip()
        parent = str(item.get("section_id") or "").strip()
        drug = str(item.get("drug_name") or "").strip()
        if not (quote and record and parent and drug):
            continue
        key = (
            drug,record,parent,quote,
            str(item.get("source_url") or ""),
            str(item.get("candidate_status") or ""),
            str(item.get("clinical_status") or ""),
            str(item.get("source_status") or ""),
        )
        grouped[key].append(item)
    return [{
        "drug_name": key[0],
        "record_id":key[1],
        "section_id":key[2],
        "evidence_ids":[x.get("id") for x in items],
        "disease_terms": sorted({
            str(x.get("disease_term") or "")
            for x in items if x.get("disease_term")
        }),
    } for key,items in grouped.items() if len(items)>1]


def parse_drug_blocks(answer_body: str) -> tuple[dict[str, str], list[str]]:
    """解析 V1.15 的 “### 药品名” 分段，不把参考资料误读成回答。"""
    by_drug: dict[str, str] = {}
    order = []
    active = None
    buffer = []
    for line in answer_body.splitlines():
        match = _HEADING.match(line)
        if match:
            if active is not None:
                by_drug[active] = "\n".join(buffer)
            active = html.unescape(match.group(1)).strip()
            order.append(active)
            buffer = []
        elif active is not None:
            buffer.append(line)
    if active is not None:
        by_drug[active] = "\n".join(buffer)
    return by_drug, order


def _field_line(block: str, field: str) -> str | None:
    label = FIELDS_CN[field]
    pattern = re.compile(
        r"^\s*-\s*\*\*" + re.escape(label) +
        r"(?:（原文节选）)?\*\*\s*[:：]\s*(.*)$"
    )
    for row in block.splitlines():
        found = pattern.match(row)
        if found:
            return found.group(1).strip()
    return None


def _positive_field_evidence(
    evidence: list[dict[str, Any]], drug: str, field: str
) -> list[dict[str, Any]]:
    return [
        item for item in evidence
        if item.get("evidence_type") in SECTION_TYPES
        and item.get("drug_name") == drug
        and item.get("section_field") == field
        and extracted_source_text(item)
        and isinstance(item.get("id"), int)
    ]


def analyze_export(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("trace"), dict):
        raise ValueError("验收输入必须包含 trace 对象；请用豆馅的单轮导出按钮")
    trace: dict[str, Any] = data["trace"]
    question = str(data.get("question") or "")
    full_answer = str(data.get("answer") or "")
    body, footer_count = split_answer_and_reference_footer(full_answer)
    evidence = evidence_index(trace)
    issues: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    def issue(severity: str, code: str, description: str, **context: Any) -> None:
        issues.append({
            "severity": severity, "code": code, "description": description,
            **context,
        })

    if not evidence:
        issue("WARN","no_evidence_cards",
              "未找到保存的证据卡，无法证明本轮 RAG 检索结果的完整性")
    if not question:
        issue("WARN","missing_question",
              "导出文件缺少本轮问题，不能独立复现所问的范围")

    ids = [x.get("id") for x in evidence]
    valid_ids = {num for num in ids if type(num) is int and num > 0}
    if len(ids) != len(valid_ids) or sorted(valid_ids) != list(range(1,len(evidence)+1)):
        issue("FAIL", "evidence_ids_not_contiguous",
              "已保存证据编号必须是唯一、连续的 1..N")

    citations = cited_numbers(body)
    unsupported_citations = sorted(citations - valid_ids)
    for num in unsupported_citations:
        issue("FAIL", "invalid_citation",
              "回答中的引文编号没有对应可展开的真实证据", evidence_id=num)
    if evidence and not citations:
        issue("WARN", "no_body_citations", "答案正文没有使用本轮任何证据编号")

    if footer_count > 1:
        issue("FAIL", "duplicated_reference_footer", "回答末尾出现了多份参考资料列表")
    if footer_count == 0 and citations:
        issue("WARN", "missing_reference_footer", "回答引用了证据，但未看到程序参考资料区")

    groups, missing_parent = section_duplicate_groups(evidence)
    for group in groups:
        if group["duplicate_type"] == "definite":
            issue("FAIL","duplicate_cross_source_section",
                  "相同药品+record_id+section_id 的原文被重复保留",
                  drug_name=group["drug_name"], record_id=group["record_id"],
                  section_id=group["section_id"],
                  evidence_ids=group["evidence_ids"])
        else:
            issue("WARN","same_parent_multiple_chunks",
                  "同一 Section 有多条不同 Chroma Chunk，可能是合法分块，不自动删除",
                  drug_name=group["drug_name"],
                  evidence_ids=group["evidence_ids"])
    if missing_parent:
        issue("WARN","missing_parent_identifiers",
              "部分证据缺少 record_id/section_id，不能精确判定跨库重复",
              evidence_ids=[x["evidence_id"] for x in missing_parent])

    repeated_quotes = candidate_same_quote_groups(evidence)
    for group in repeated_quotes:
        issue("WARN", "repeated_candidate_quote_cards",
              "相同药品+同 Section 的一句话拆成多张疾病候选卡；"
              "可合并展示但原始候选关系应保留",
              drug_name=group["drug_name"],
              evidence_ids=group["evidence_ids"],
              disease_terms=group["disease_terms"])

    pipeline = trace.get("retrieval_pipeline")
    pipeline = pipeline if isinstance(pipeline,dict) else {}
    merge = pipeline.get("merge")
    merge = merge if isinstance(merge,dict) else {}
    retained = merge.get("retained")
    if isinstance(retained,int) and retained != len(evidence):
        issue("WARN","pipeline_count_diff",
              "检索统计与历史证据卡数量不同，可能包含后续补检或旧数据格式",
              retained_reported=retained, evidence_cards=len(evidence))

    if "根据审核通过的证据" in body:
        issue("FAIL", "false_medical_approval_wording",
              "程序级原文核验被错误表述为医学审核通过")

    metrics.update({
        "question": question[:250],
        "answer_mode": str(trace.get("medical_answer_mode") or "unspecified"),
        "evidence_cards": len(evidence),
        "evidence_types": dict(Counter(str(x.get("evidence_type") or "unknown")
                                       for x in evidence)),
        "cited_ids": sorted(citations),
        "invalid_cited_ids": unsupported_citations,
        "unused_evidence_ids": sorted(valid_ids - citations),
        "reference_footer_blocks": footer_count,
        "repeat_section_groups": len(groups),
        "same_quote_candidate_card_groups": len(repeated_quotes),
        "definite_section_duplicates": sum(
            r["duplicate_type"] == "definite" for r in groups),
        "reported_merged_evidence": retained if isinstance(retained,int) else None,
        "reported_neo4j_parent_verified": (
            ((pipeline.get("chroma_hybrid") or {}).get("parent_verified"))
            if isinstance(pipeline.get("chroma_hybrid"),dict) else None
        ),
    })

    mode = str(trace.get("medical_answer_mode") or "")
    scope = trace.get("multi_drug_scope")
    scope = scope if isinstance(scope,dict) else {}
    selected = [str(name) for name in scope.get("listed_drugs",[])
                if isinstance(name,str) and name.strip()] if isinstance(scope.get("listed_drugs"),list) else []
    fields = [field for field in scope.get("supplement_fields",[])
              if field in FIELDS_CN and field != "indication"] if isinstance(scope.get("supplement_fields"),list) else []
    if mode == "structured_disease_drug_source_overview":
        if not selected or not fields:
            issue("FAIL","missing_structured_scope",
                  "结构化回答缺少本轮列出的药名或补查栏目")
        blocks, order = parse_drug_blocks(body)
        expected_set = set(selected)
        actual_set = set(order)
        missing_drugs = sorted(expected_set - actual_set)
        extra_drugs = sorted(actual_set - expected_set)
        for drug in missing_drugs:
            issue("FAIL","drug_omitted_from_answer",
                  "已发现的药品没有独立回答段落",drug_name=drug)
        for drug in extra_drugs:
            issue("WARN","drug_outside_scope_answered",
                  "回答出现不在本轮疾病候选清单中的药品段落",drug_name=drug)
        if len(order) != len(actual_set):
            issue("FAIL","duplicate_drug_heading",
                  "同一药品在回答中出现重复段落")
        for row in evidence:
            if selected and row.get("drug_name") and row["drug_name"] not in expected_set:
                issue("FAIL","off_scope_drug_in_evidence",
                      "非所列药品进入最终证据卡",
                      drug_name=str(row["drug_name"]),evidence_id=row.get("id"))
        supported_cells = correctly_answered = false_missing = 0
        cell_details = []
        for drug in selected:
            block = blocks.get(drug,"")
            candidates = [i for i in evidence
                          if i.get("drug_name") == drug and
                          i.get("evidence_type") == "unverified_drug_disease_candidate"]
            if candidates and block and not any(
                i.get("id") in cited_numbers(block) for i in candidates
            ):
                issue("FAIL","missing_candidate_citation",
                      "该药疾病提及候选未在对应药品段落引用",drug_name=drug)
            for field in fields:
                positive = _positive_field_evidence(evidence, drug, field)
                line = _field_line(block,field) if block else None
                field_ids = {int(i["id"]) for i in positive}
                cited_in_line = cited_numbers(line or "")
                has_source = bool(positive)
                supported_cells += int(has_source)
                valid_for_field = bool(field_ids & cited_in_line)
                cell_details.append({
                    "drug_name":drug, "field":field, "source_present":has_source,
                    "source_evidence_ids":sorted(field_ids),
                    "answer_citation_valid":valid_for_field,
                })
                if line is None and block:
                    issue("FAIL","field_not_answered",
                          "该药已被列出，但没有对应的说明书栏目回答",
                          drug_name=drug,field=field)
                if has_source and "本轮没有获得" in (line or ""):
                    false_missing += 1
                    issue("FAIL","false_missing_source_claim",
                          "实际有栏目原文，却在回答中声明本轮没有资料",
                          drug_name=drug,field=field,
                          evidence_ids=sorted(field_ids))
                if has_source and not valid_for_field:
                    issue("FAIL","field_evidence_not_cited",
                          "药品字段已有原文但未引用对应药品+字段的证据编号",
                          drug_name=drug,field=field,
                          evidence_ids=sorted(field_ids))
                if has_source and valid_for_field:
                    correctly_answered += 1
                if line is not None:
                    wrong_ids = sorted(
                        x for x in cited_in_line if x in valid_ids and x not in field_ids
                    )
                    if wrong_ids:
                        issue("FAIL","cross_drug_or_field_citation",
                              "栏目引用了其他药品或其他栏目证据编号",
                              drug_name=drug,field=field,evidence_ids=wrong_ids)
                if not has_source and line is not None and "本轮没有获得" not in line:
                    issue("FAIL","unsupported_field_answer",
                          "该药该栏目没有检索证据，回答却没有标注资料缺失",
                          drug_name=drug,field=field)
        expected_cells = len(selected)*len(fields)
        metrics["multi_drug"] = {
            "selected_drugs": selected, "requested_fields": fields,
            "expected_cells":expected_cells,
            "cells_with_source": supported_cells,
            "correctly_cited_cells":correctly_answered,
            "false_missing_cells":false_missing,
            "cell_citation_coverage":(
                round(correctly_answered/expected_cells,4) if expected_cells else None
            ),
            "cells":cell_details,
        }
        reported = trace.get("multi_drug_coverage")
        if isinstance(reported,dict) and isinstance(reported.get("supported_cells"),int):
            if reported["supported_cells"] != supported_cells:
                issue("WARN","reported_field_coverage_mismatch",
                      "回答生成过程保存的栏目覆盖数与已保存真实证据不符",
                      reported=reported["supported_cells"],
                      observed=supported_cells)
    elif selected:
        issue("WARN","non_structured_multi_drug",
              "本轮不是 V1.15 结构化原文回答，需人工检查多药栏目遗漏")

    severity_counts=Counter(i["severity"] for i in issues)
    status = ("FAIL" if severity_counts["FAIL"] else
              "WARN" if severity_counts["WARN"] else "PASS")
    return {
        "schema_version":EXPORT_SCHEMA,
        "status": status,
        "summary": {
            "failures":severity_counts["FAIL"],"warnings":severity_counts["WARN"],
            "checks": "工程引文/栏目覆盖/跨库 Section 身份，不包含临床疗效验证",
        },
        "metrics":metrics,
        "issues":issues,
    }


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(
        description="V1.16 医疗 GraphRAG 单轮证据和回答离线验收"
    )
    parser.add_argument("--input",required=True,type=Path,
                        help="从豆馅 V1.16 聊天界面导出的单轮 JSON")
    parser.add_argument("--output",type=Path,default=None,
                        help="可选：生成本地 JSON 验收报告")
    parser.add_argument("--overwrite",action="store_true",
                        help="明确覆盖既有报告文件")
    args=parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"文件不存在：{args.input}")
    if args.input.stat().st_size > MAX_FILE_BYTES:
        parser.error("输入文件超过 15MB；请只导出单轮，而不是整个会话")
    data=json.loads(args.input.read_text(encoding="utf-8-sig"))
    report=analyze_export(data)
    print(f"验收状态：{report['status']} | "
          f"失败项={report['summary']['failures']} | "
          f"警告项={report['summary']['warnings']}")
    print(f"证据卡：{report['metrics']['evidence_cards']} | "
          f"正文引用：{report['metrics']['cited_ids']} | "
          f"确定重复 Section：{report['metrics']['definite_section_duplicates']}")
    coverage=report['metrics'].get('multi_drug')
    if coverage:
        print(f"多药字段：证据覆盖 {coverage['cells_with_source']}/"
              f"{coverage['expected_cells']}；正确引用 "
              f"{coverage['correctly_cited_cells']}/"
              f"{coverage['expected_cells']}；错误缺失 "
              f"{coverage['false_missing_cells']}")
    for row in report['issues']:
        print(f"  [{row['severity']}] {row['code']}：{row['description']}")
    if args.output:
        if args.output.exists() and not args.overwrite:
            parser.error(f"报告已存在：{args.output}；请更换文件名或加 --overwrite")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(
            json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
        )
        print("报告：",args.output.resolve())
    return 2 if report['status']=="FAIL" else 0


if __name__=="__main__":
    raise SystemExit(main())
