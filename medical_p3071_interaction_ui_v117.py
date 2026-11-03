# -*- coding: utf-8 -*-
"""HTML evidence panel for P3-07.1. No Streamlit import or database writes."""
from __future__ import annotations

import html
import os
from pathlib import Path
from urllib.parse import urlparse

from medical_p3071_drug_interaction_v117 import (
    ACTIVATE, inspect_interaction, recognize_pair,
)

TRACE_KEY="p3071_interaction"
SECTION_LIMIT=8


def _esc(text,lim=400)->str:
    return html.escape(str(text or "")[:lim],quote=True)


def _url(value)->str:
    if not isinstance(value,str):
        return ""
    u=urlparse(value)
    return value if u.scheme=="https" and u.hostname in {
        "dailymed.nlm.nih.gov","www.dailymed.nlm.nih.gov",
    } else ""


def prepare_interaction_trace_for_display(trace,question,*,project_root=None):
    """After generation: cache separate structured references for the UI.

    Never modifies trace.knowledge_evidence, merged_excerpts, answer, citations.
    If no Neo4j, does not synthesize graph evidence from label URLs.
    """
    if not isinstance(trace,dict) or not isinstance(question,str):
        return {"status":"INVALID_TRACE"}
    root=Path(project_root or Path(__file__).resolve().parent).resolve()
    try:
        from dotenv import load_dotenv
        load_dotenv(root/".env",override=False)
    except ImportError:
        pass
    if os.getenv(ACTIVATE,"0").lower().strip() not in ("1","on","true","yes"):
        return {"status":"DISABLED"}
    if TRACE_KEY in trace and isinstance(trace[TRACE_KEY],dict):
        return trace[TRACE_KEY]
    parsed=recognize_pair(question)
    if not(parsed["recognized_pair"] and parsed["request_is_safety_or_interaction"]):
        return {"status":"OUT_OF_SCOPE"}
    try:
        outcome=inspect_interaction(question,root=root)
    except Exception as ex:
        outcome={
            "stage":"P3-07.1",
            "status":"BLOCKED",
            "error_type":type(ex).__name__,
            "question":question[:500],
            "local_evidence":[],
            "official_references":[],
        }
    trace[TRACE_KEY]=outcome
    return outcome


def interaction_evidence_html(trace:dict)->str:
    if not isinstance(trace,dict):
        return ""
    report=trace.get(TRACE_KEY)
    if not isinstance(report,dict) or report.get("stage")!="P3-07.1":
        return ""
    rows=report.get("local_evidence")
    refs=report.get("official_references")
    if not isinstance(rows,list):rows=[]
    if not isinstance(refs,list):refs=[]
    # No risk signal when nothing resolved; no fabricated evidence.
    if not rows and not refs:
        return ""
    chain=[]
    for idx,row in enumerate(rows[:SECTION_LIMIT],1):
        if not isinstance(row,dict):
            continue
        if (row.get("relationship_types")!=["HAS_RECORD","HAS_SECTION"]
                or row.get("neo4j_hops")!=2):
            continue
        if row.get("clinical_verified") is True:
            continue
        names=row.get("node_ids")
        if not isinstance(names,list) or len(names)!=3:continue
        cls=row.get("classification") or {}
        quote=_esc(row.get("quote"),450)
        label=_esc(row.get("drug_name"),90)
        field=_esc(row.get("section_field"),45)
        source_status=_esc(cls.get("status"),80)
        short_id=_esc(row.get("section_sha256"),18)
        chain.append(
            '<details class="p3071-card">'
            '<summary>原文 P'+str(idx)+' · '+label+' · '
            +field+' · '+source_status+'</summary>'
            '<div class="p3071-body">'
            '<div class="p3071-steps">'
            '<span class="p3071-node">Drug</span>'
            '<span class="p3071-edge">HAS_RECORD →</span>'
            '<span class="p3071-node">SourceRecord</span>'
            '<span class="p3071-edge">HAS_SECTION →</span>'
            '<span class="p3071-node">Section</span>'
            '</div><div class="p3071-source">'
            '来源 ID：'+_esc(names[1],100)+' / '+_esc(names[2],100)
            +' · SHA256：'+short_id+'</div>'
            '<blockquote class="p3071-quote">'+quote+'</blockquote>'
            '<div class="p3071-warning">'
            '这只是本地第三方说明书来源 2 跳路径；其中提及的'
            '“抗凝药”等类别是原文关键词，不是已存在的 Neo4j 药物分类边。'
            '</div></div></details>'
        )
    external=[]
    for idx,ref in enumerate(refs[:4],1):
        if not isinstance(ref,dict):continue
        url=_url(ref.get("url"))
        if not url:continue
        external.append(
            '<li><a target="_blank" rel="noopener noreferrer" href="'
            +_esc(url,2048)+'">'+_esc(ref.get("title"),130)+'</a>'
            '<span class="p3071-small"> · 美国 DailyMed 官方标签引用指针，'
            '运行时未实时核对最新版</span></li>'
        )
    if not chain and not external:return ""
    section=(
        '<section class="p3071-panel" aria-label="药物相互作用证据链">'
        '<div class="p3071-title">⚠ 药物相互作用核验链 '
        '<span class="p3071-pill">科研展示 · 非自动临床判定</span></div>'
        '<div class="p3071-warning">'
        '华法林与布洛芬联用有增加严重出血风险的官方警告。'
        '本地 Neo4j 查询只能证明 Drug → SourceRecord → Section 的'
        '来源路径，不能单凭一条类别关键词证明存在经过审核的药物相互作用关系。'
        '涉及真实用药请咨询医生或药师，不要自行联用或停用处方华法林。'
        '</div>'
        '<div class="p3071-small">状态：'+_esc(report.get("status"),100)
        +' · 本地原文章节：'+str(len(chain))
        +' 条 · 真正药物—药物 Neo4j 关系：0 条（尚未构建）</div>'
        +''.join(chain)
        +('<div class="p3071-refs">权威标签引用（外部核实）：<ul>'
          +''.join(external)+'</ul></div>' if external else "")
        +'</section>'
    )
    return '<style>'+STYLE+'</style>'+section


STYLE=r"""
.p3071-panel{
    margin:14px 0 14px 20px;padding:13px 15px;border-radius:11px;
    border:1px solid #e9d5b3;background:#fffdfa;color:#41516a;
}
.p3071-title{font-size:14px;font-weight:700;color:#865e27;margin-bottom:9px}
.p3071-pill{font-size:10px;margin-left:8px;background:#fff0da;
    padding:3px 8px;border-radius:99px;color:#8b5f2e}
.p3071-small{color:#8290a3;font-size:11px;line-height:1.8}
.p3071-warning{font-size:11.8px;line-height:1.85;color:#825f36;margin:8px 0}
.p3071-card{margin:9px 0;border:1px solid #ebe2d7;border-radius:8px;
    background:white;overflow:hidden}
.p3071-card>summary{cursor:pointer;padding:10px 12px;font-size:12.2px;
    font-weight:600;overflow-wrap:anywhere}
.p3071-body{padding:9px 12px;border-top:1px solid #f0eae3}
.p3071-steps{display:flex;align-items:center;overflow-x:auto;gap:8px;
    padding:7px 0}
.p3071-node{flex:none;padding:7px 9px;border-radius:7px;
    background:#f1f5fc;font-size:11px}
.p3071-edge{flex:none;font-size:10px;color:#6384b4}
.p3071-source{font-size:11px;color:#8a94a1;overflow-wrap:anywhere;margin:7px 0}
.p3071-quote{border-left:2px solid #b7c5dd;padding:9px 12px;margin:8px 0;
    background:#f9fbff;font-size:12px;overflow-wrap:anywhere;white-space:pre-wrap}
.p3071-refs{font-size:12px;margin:11px 0 0}
.p3071-refs a{color:#476c9f;text-decoration:none;overflow-wrap:anywhere}
.p3071-refs li{margin:4px 0}
@media(max-width:780px){.p3071-panel{margin-left:0;padding:11px}}
"""
