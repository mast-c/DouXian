# -*- coding: utf-8 -*-
"""Add P3-07.2 generic drug-pair evidence to the EXISTING chat evidence HTML.

No Streamlit `pages/` entry. No clinical claim or new answer citation.
"""
from __future__ import annotations
import html
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from medical_p3072_corpus_index_v117 import CorpusBlocked
from medical_p3072_pair_retrieval_v117 import (
    ENABLE_FLAG, TRACE_KEY, inspect_pair, index_path
)

CSS=r"""
.kb-p3072{margin:16px 0 12px 20px;border-top:1px solid #e1e7f1;padding-top:12px}
.kb-p3072-head{font-size:13.5px;font-weight:700;color:#37567d}
.kb-p3072-pill{background:#f0f5ff;border-radius:100px;padding:2px 9px;
  font-size:10.5px;color:#526d97;margin-left:8px}
.kb-p3072-note{font-size:11.5px;color:#8b94a4;line-height:1.85;margin:8px 0}
.kb-p3072-warning{color:#875c32;background:#fff8ee;border:1px solid #f1e0c7;
  border-radius:7px;padding:9px 11px;font-size:11.8px;line-height:1.75;margin:10px 0}
.kb-p3072-card{border:1px solid #e1e7f2;border-radius:9px;background:#fcfdff;
  margin:9px 0;overflow:hidden}
.kb-p3072-card summary{cursor:pointer;padding:10px 12px;font-size:12.3px;
  font-weight:650;color:#405875;overflow-wrap:anywhere}
.kb-p3072-detail{padding:11px 12px;border-top:1px solid #e5edf7;
  font-size:11.7px;color:#536881;line-height:1.8}
.kb-p3072-nodes{display:flex;gap:6px;align-items:center;flex-wrap:wrap;
  background:#f1f6fd;padding:8px;border-radius:7px}
.kb-p3072-nodes span{background:white;padding:4px 7px;border-radius:5px}
.kb-p3072-quote{border-left:2px solid #a5b8dc;margin:10px 0 6px;
  padding:8px 11px;background:#f7faff;color:#536a89;white-space:pre-wrap;
  overflow-wrap:anywhere}
.kb-p3072-meta{font-size:10.5px;color:#8392a4;overflow-wrap:anywhere}
.kb-p3072 a{color:#496da0;text-decoration:none}
.kb-p3072-archive>summary{cursor:pointer;color:#6e829a;font-size:12px;padding:9px 0}
@media(max-width:780px){.kb-p3072{margin-left:0}}
"""

def _safe(x,limit=300)->str:
    return html.escape(str(x or "")[:limit],quote=True)


def _href(x)->str:
    if not isinstance(x,str):return ""
    try:
        parsed=urlsplit(x)
    except ValueError:
        return ""
    if parsed.scheme not in ("https","http") or not parsed.netloc:
        return ""
    return _safe(x,2048)


def prepare_full_corpus_interaction_trace(trace,question,*,root=None):
    """Post-generation. Read-only local SQLite. Never changes old citations."""
    if not isinstance(trace,dict) or not isinstance(question,str):
        return {"status":"INVALID_TRACE"}
    if TRACE_KEY in trace and isinstance(trace[TRACE_KEY],dict):
        return trace[TRACE_KEY]
    location=Path(root or Path(__file__).resolve().parent)
    try:
        from dotenv import load_dotenv
        load_dotenv(location/".env",override=False)
    except ImportError:
        pass
    if os.getenv(ENABLE_FLAG,"0").lower().strip() not in ("1","true","on","yes"):
        return {"status":"DISABLED"}
    if not re.search(
        r"一起|同服|合用|联用|同用|同时|能不能|可以|能否|服用|吃|"
        r"相互作用|配伍|冲突|禁忌|风险|安全|联合",question,re.I
    ):
        return {"status":"NO_PAIR_SAFETY_INTENT"}
    try:
        mapping=os.getenv("MEDICAL_P3072_CURATED_CLASS_MAP","").strip()
        curated=None
        if mapping:
            q=Path(mapping)
            curated=q if q.is_absolute() else location/q
        outcome=inspect_pair(
            question,root=location,curated_map=curated,
            max_sections_per_group=400,max_evidence=12,
        )
    except Exception as exc:
        outcome={
            "stage":"P3-07.2",
            "status":"INDEX_UNAVAILABLE_OR_QUERY_BLOCKED",
            "error_type":type(exc).__name__,
            "error_message":str(exc)[:150],
            "question":question[:300],
            "source_evidence":[],
            "medical_decision":None,
            "pair_safe_to_combine":None,
            "clinical_verified":False,
        }
    trace[TRACE_KEY]=outcome
    return outcome


def interaction_full_corpus_html(trace)->str:
    """Same HTML <details> under existing original trace evidence list."""
    if not isinstance(trace,dict):
        return ""
    report=trace.get(TRACE_KEY)
    if not isinstance(report,dict) or report.get("stage")!="P3-07.2":
        return ""
    matches=report.get("name_resolution",{}).get("matches",[])
    if not isinstance(matches,list):matches=[]
    status=str(report.get("status") or "")
    if status in (
        "NO_CATALOG_DRUG_DETECTED","ONLY_ONE_DRUG_DETECTED",
        "EMPTY_QUESTION","NO_PAIR_SAFETY_INTENT"
    ):
        return ""
    snapshot=report.get("index_snapshot") or {}
    sources=report.get("source_evidence") or []
    cards=[]
    for idx,entry in enumerate(sources[:12],1):
        if not isinstance(entry,dict) or entry.get("clinical_verified") is True:
            continue
        if entry.get("edge_types")!=["HAS_RECORD","HAS_SECTION"]:
            continue
        if entry.get("actual_graph_hops")!=2:
            continue
        ids=entry.get("node_ids")
        if not isinstance(ids,list) or len(ids)!=3:
            continue
        analysis=entry.get("analysis")
        if not isinstance(analysis,dict):continue
        literal=analysis.get("matched_literal")
        state=analysis.get("status")
        quote=entry.get("analysis",{}).get("quote","")
        href=_href(entry.get("source_url"))
        link=(
            '<a target="_blank" rel="noopener noreferrer" href="'
            +href+'">查看原始资料 ↗</a>'
        ) if href else "无安全来源链接"
        cards.append(
            '<details class="kb-p3072-card">'
            '<summary>来源 '+str(idx)+' · '
            +_safe(entry.get("owner_drug_name"),65)+' / '
            +_safe(entry.get("section_field"),32)
            +' · '+_safe(state,70)+'</summary>'
            '<div class="kb-p3072-detail">'
            '<div class="kb-p3072-nodes">'
            '<span>Drug</span> → HAS_RECORD → '
            '<span>SourceRecord</span> → HAS_SECTION → '
            '<span>Section</span></div>'
            '<div>原文匹配：'+_safe(literal,100)+'；风险词：'
            +_safe("、".join(analysis.get("risk_terms_near_match") or []),140)
            +'；提示词：'
            +_safe("、".join(analysis.get("caution_terms_near_match") or []),140)
            +'</div><div class="kb-p3072-quote">'
            +_safe(quote,500)+'</div>'
            '<div class="kb-p3072-meta">SourceRecord ID: '
            +_safe(ids[1],130)+' · Section ID: '+_safe(ids[2],130)
            +' · SHA256 '+_safe(entry.get("section_sha256"),65)
            +'</div><div class="kb-p3072-meta">技术证据：静态快照来源链；'
            '源文本字面命中不代表批准的相互作用关系。'
            +' · '+link+'</div>'
            '</div></details>'
        )
    names=" × ".join(
        _safe(x.get("canonical_key"),65)
        for x in matches[:2] if isinstance(x,dict)
    )
    coverage=(
        str(snapshot.get("drugs_with_safety_sections","?"))
        +"/"+str(snapshot.get("drugs_indexed","?"))
    )
    limits=(
        ' <strong>本次扫描触及预算上限，未穷举该药所有安全章节。</strong>'
        if report.get("search_budget_truncated") else ""
    )
    if status=="INDEX_UNAVAILABLE_OR_QUERY_BLOCKED":
        message="全库派生索引暂不可读取，请先检查 P3-07.2 索引构建情况。"
    elif status=="EXACTLY_TWO_CATALOG_GROUPS_DETECTED":
        message="已识别两种药品，正在核对本地来源文本。"
    elif status in (
        "NO_PAIR_SPECIFIC_EVIDENCE_FOUND",
        "ONLY_ONE_DRUG_DETECTED","AMBIGUOUS_CATALOG_ALIAS",
        "THREE_OR_MORE_DRUGS_NOT_A_PAIR"
    ):
        message="没有获得足以确认该药物组合的本地来源关联证据；不能推断可以安全联用。"
    elif status=="LOCAL_UNRESOLVED_CLASS_WARNING_ONLY":
        message="检索到药品类别警告，但尚未验证另一药属于该类别，不能作为药物对相互作用结论。"
    else:
        message="下方展示的是可能相关的原文字面提示，仍需核对适用产品的官方最新版说明书。"
    body = (
        '<style>'+CSS+'</style>'
        '<section class="kb-p3072" aria-label="全库药物相互作用来源核验">'
        '<div class="kb-p3072-head">🔎 全库药物相互作用来源检索'
        '<span class="kb-p3072-pill">原证据栏 · 独立来源链</span></div>'
        '<div class="kb-p3072-note">当前药物对：'
        + (names or "未解析成明确双药")
        +'；药品安全章节覆盖：'+_safe(coverage,50)
        +'；索引版本：'+_safe(snapshot.get("created_utc"),70)
        +'；来源记录为第三方资料、研究用途。'
        +limits+'</div>'
        '<div class="kb-p3072-warning">'+_safe(message,300)
        +' 本模块不自动确认任何药物组合的安全性、风险等级或疗效；'
        '未命中≠安全。已有处方用药请先咨询医生或药师。</div>'
        '<div class="kb-p3072-note">检索状态：'+_safe(status,110)
        +' · 来源章节：'+str(len(cards))
        +' · 已证实的 Neo4j 药物—药物相互作用边：0（本阶段未创建或确认）'
        '</div>'
        +''.join(cards)
        +'</section>'
    )
    # Keep supplemental discovery inspectable without competing with cited paths.
    return (
        '<details class="kb-p3072-archive"><summary>补充来源线索 · '
        + str(len(cards)) + ' 条（包含其他剂型，不参与本轮回答）</summary>'
        + body + '</details>'
    )
