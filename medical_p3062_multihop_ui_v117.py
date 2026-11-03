# -*- coding: utf-8 -*-
"""P3-06.2 safe, read-only Streamlit visualization of 5–8-hop Neo4j paths.

Can render:
  (1) P3-06.1 read-only preflight JSON (saved or uploaded);
  (2) verified_paths in a prior rag.py disease_encyclopedia.multi_hop_graph trace;
  (3) a live read-only result from medical_p3061_multihop_preflight_v117.inspect.

This module does NOT import Streamlit at import time (pure helpers test offline).
Never labels graph co-mentioning as approved therapy or clinical evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from medical_p3061_multihop_graph_v117 import PATH_LABELS, PATH_TYPES

STAGE="P3-06.2"
MAX_REPORT_BYTES=8*1024*1024
MAX_VISIBLE_PATHS=24
MAX_VISIBLE_ID_LEN=180
MAX_STRING_LEN=4000
MAX_QUESTION_LEN=500
PATH_DIRECTIONS={
    5:("back","back","back","forward","forward"),
    6:("back","back","back","forward","forward","forward"),
    7:("back","back","back","forward","forward","back","back"),
    8:("back","back","back","forward","forward","back","forward","forward"),
}
PATH_INTENT={
    5:"疾病来源 → 药品说明书 Section",
    6:"同一药品资料中的两个疾病提及候选",
    7:"通过第二疾病术语连接另一药品资料",
    8:"同一 Section 提及两个疾病（8 跳）",
}
TYPE_ENGLISH={
    "DiseaseDocument":"DiseaseDocument",
    "DiseaseTerm":"DiseaseTerm",
    "DiseaseCandidate":"Candidate",
    "Drug":"Drug",
    "SourceRecord":"SourceRecord",
    "Section":"Section",
}
TYPE_COLORS={
    "DiseaseDocument":"#DFF3ED",
    "DiseaseTerm":"#E2ECF9",
    "DiseaseCandidate":"#FFF1DD",
    "Drug":"#FFE5D7",
    "SourceRecord":"#EDF0F4",
    "Section":"#F2E8FF",
}
# Labels for graph layout (not medical claims).
EDGE_SHORT={
    "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA":"EXACT_NAME_MATCH\n(unverified)",
    "MENTIONS_TERM":"MENTIONS_TERM",
    "HAS_CANDIDATE":"HAS_CANDIDATE",
    "HAS_RECORD":"HAS_RECORD",
    "HAS_SECTION":"HAS_SECTION",
    "SUPPORTED_BY":"SUPPORTED_BY",
}


class VisualizeBlocked(ValueError):
    pass


def require(condition:bool, message:str) -> None:
    if not condition:
        raise VisualizeBlocked(message)


def _short(value:Any,limit:int=60) -> str:
    if not isinstance(value,str):
        return ""
    s=" ".join(value.split())
    return s if len(s)<=limit else s[:limit-1]+"…"


def dot_string(value:str) -> str:
    """A quoted DOT-safe string. No user data is used as DOT syntax."""
    value=str(value or "")[:MAX_STRING_LEN]
    value=value.replace("\\","\\\\").replace('"','\\"')
    value=value.replace("\r"," ").replace("\n","\\n").replace("\t"," ")
    value="".join(ch for ch in value if ord(ch)>=32)
    return '"'+value+'"'


def _fallback_path_id(hops:int,node_ids:list[str],rels:list[str]) -> str:
    raw=json.dumps(
        {"h":hops,"ids":node_ids,"rels":rels},
        ensure_ascii=False,sort_keys=True,separators=(",",":")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _normalize_path(obj:dict,position:int) -> dict:
    require(isinstance(obj,dict),f"第 {position} 条不是路径对象")
    hops=obj.get("hop_count")
    require(type(hops) is int and hops in PATH_LABELS,
            f"第 {position} 条 hop_count 不是 5～8")
    ids=obj.get("node_ids")
    rels=obj.get("relationship_types")
    labels=obj.get("node_labels")
    require(
        isinstance(ids,list) and len(ids)==hops+1
        and all(isinstance(x,str) and 0<len(x)<=MAX_VISIBLE_ID_LEN
                for x in ids)
        and len(set(ids))==len(ids),
        f"第 {position} 条路径节点 ID 缺失/重复/异常"
    )
    require(
        isinstance(rels,list) and tuple(rels)==PATH_TYPES[hops],
        f"第 {position} 条路径关系类型/方向模板不符"
    )
    require(
        isinstance(labels,list) and len(labels)==hops+1
        and all(
            (isinstance(value,str) and value==PATH_LABELS[hops][i])
            or (isinstance(value,list) and PATH_LABELS[hops][i] in value)
            for i,value in enumerate(labels)
        ),
        f"第 {position} 条路径节点类型顺序不符"
    )
    require(
        obj.get("clinical_verified") is not True
        and obj.get("clinical_truth_verified") is not True
        and obj.get("treatment_relation") is not True
        and obj.get("source_status","third_party_unverified")
        =="third_party_unverified",
        "报告将未经验证的来源提升为医疗权威/治疗关系，拒绝展示为有效路径"
    )
    pid=obj.get("path_id") or _fallback_path_id(hops,ids,rels)
    require(isinstance(pid,str)
            and re.fullmatch(r"[A-Za-z0-9_-]{1,120}",pid) is not None,
            f"第 {position} 条路径 ID 非法")
    row={
        **{key:value for key,value in obj.items()
           if key not in ("section_text","candidate_quote")},
        "path_id":pid,
        "hop_count":hops,
        "node_ids":list(ids),
        "node_labels":list(PATH_LABELS[hops]),
        "relationship_types":list(PATH_TYPES[hops]),
        "treatment_relation":False,
        "source_status":"third_party_unverified",
        "clinical_verified":False,
    }
    # Normalize optional previews, including direct _path fields.
    quote=obj.get("candidate_quote_preview") or obj.get("candidate_quote")
    body=obj.get("section_text_preview") or obj.get("section_text")
    row["candidate_quote_preview"]=_short(quote,240)
    row["section_text_preview"]=_short(body,280)
    row["drug_name"]=_short(obj.get("drug_name"),130)
    row["disease_name"]=_short(obj.get("disease_name"),130)
    row["related_disease_name"]=_short(obj.get("related_disease_name"),130)
    row["related_drug_name"]=_short(obj.get("related_drug_name"),130)
    row["section_field"]=_short(obj.get("section_field"),45)
    return row


def normalize_report(payload:dict,*,mode:str="saved_report") -> dict:
    """Normalize a P3-06.1 report or an entire exported online trace.

    mode is advisory UI text only. It does not affect proof validity.
    """
    require(isinstance(payload,dict),"报告必须是 JSON 对象")
    question=payload.get("question","")
    if payload.get("stage")=="P3-06.1" and isinstance(payload.get("paths"),list):
        raw_paths=payload["paths"]
        trace=payload.get("graph_trace") or {}
        status=str(payload.get("graph_path_status") or trace.get("status") or "")
        created_utc=payload.get("created_utc")
        source_format="preflight_report"
    else:
        outer=payload.get("trace") if isinstance(payload.get("trace"),dict) else payload
        disease=outer.get("disease_encyclopedia") if isinstance(outer,dict) else {}
        mh=disease.get("multi_hop_graph") if isinstance(disease,dict) else None
        if not isinstance(mh,dict) and isinstance(outer,dict):
            mh=outer.get("multi_hop_graph")
        require(isinstance(mh,dict),"找不到 P3-06.1 多跳 trace 或 paths")
        raw_paths=mh.get("verified_paths",[])
        trace=mh
        status=str(mh.get("status") or "")
        created_utc=payload.get("created_utc")
        source_format="online_trace"
    require(isinstance(raw_paths,list) and len(raw_paths)<=MAX_VISIBLE_PATHS,
            "路径记录不是 list 或超过 24 条上限")
    require(isinstance(trace,dict),"graph_trace 必须是 JSON 对象")
    paths=[_normalize_path(p,i+1) for i,p in enumerate(raw_paths)]
    ids=[p["path_id"] for p in paths]
    require(len(set(ids))==len(ids),
            "不同路径 ID 重复；为防止混淆，拒绝展示")
    # A saved (potentially user-supplied) JSON can only claim previous
    # validation. The UI does not cryptographically prove report origin.
    return {
        "stage":STAGE,
        "source_format":source_format,
        "mode":mode,
        "question":_short(question,MAX_QUESTION_LEN),
        "created_utc":created_utc,
        "status":status or ("TRACE_WITH_PATHS" if paths else "NO_PATHS"),
        "paths":paths,
        "path_count":len(paths),
        "actual_depths":sorted({p["hop_count"] for p in paths}),
        "per_depth":dict(sorted(Counter(p["hop_count"] for p in paths).items())),
        "query_count":trace.get("query_count"),
        "unverified_source_warning":True,
    }


def read_report_bytes(raw:bytes,*,mode:str="uploaded_report") -> dict:
    require(isinstance(raw,bytes) and 0<len(raw)<=MAX_REPORT_BYTES,
            "报告大小必须在 1 Byte～8 MiB 之间")
    try:
        report=json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError,ValueError) as exc:
        raise VisualizeBlocked("报告不是可解析的 UTF-8 JSON") from exc
    return normalize_report(report,mode=mode)


def read_report_file(file:Path) -> dict:
    file=Path(file)
    require(file.is_file(),"未找到历史多跳审计报告")
    require(0<file.stat().st_size<=MAX_REPORT_BYTES,
            "历史报告文件大小超限")
    return read_report_bytes(file.read_bytes(),mode="saved_report")


def list_saved_reports(root:Path) -> list[Path]:
    """List ONLY local P3-06.1 report files; never search user file library."""
    folder=Path(root)/"stage4_reports"
    if not folder.is_dir():
        return []
    seen=set()
    paths=[]
    for pattern in ("p3061_8hop_*.json",
                    "p3061_multihop_*.json",
                    "p3061_online_preview_*.json"):
        for p in folder.glob(pattern):
            if p.is_file() and p not in seen:
                seen.add(p)
                paths.append(p)
    return sorted(paths,key=lambda p:(p.stat().st_mtime_ns,p.name),reverse=True)[:80]


def _node_display(path:dict,index:int) -> str:
    kind=PATH_LABELS[path["hop_count"]][index]
    name=""
    if kind in ("DiseaseDocument","DiseaseTerm"):
        if index in (0,1):
            name=path.get("disease_name","")
        else:
            name=path.get("related_disease_name","")
    elif kind=="Drug":
        name=(path.get("drug_name","")
              if index==3 else path.get("related_drug_name",""))
    elif kind=="Section":
        name=path.get("section_field","")
    elif kind=="SourceRecord":
        name=_short(path.get("record_id",""),20)
    elif kind=="DiseaseCandidate":
        name=_short(path.get("node_ids",[])[index],14)
    return (f"{TYPE_ENGLISH[kind]}\n{_short(name,24)}"
            if name else TYPE_ENGLISH[kind])


def _edge_label(relation:str) -> str:
    return EDGE_SHORT.get(relation,relation)


def build_dot(path:dict,*,orientation:str="LR") -> str:
    """Safe Graphviz visualization of ONE exact-length graph path."""
    row=_normalize_path(path,1)
    require(orientation in ("LR","TB"),"仅支持横向或纵向的路径布局")
    hops=row["hop_count"]
    lines=[
        "digraph MultiHop {",
        f"graph [rankdir={orientation}, bgcolor=transparent, pad=0.18, "
        "nodesep=0.32, ranksep=0.8, splines=polyline];",
        "node [fontname=\"Microsoft YaHei\", fontsize=11, color=\"#78899b\","
        " penwidth=1.0, shape=box, style=\"rounded,filled\", margin=\"0.12,0.10\"];",
        "edge [fontname=\"Arial\", fontsize=9, color=\"#6a8498\", "
        "fontcolor=\"#496a7e\", penwidth=1.35, arrowsize=0.65];",
    ]
    for i,kind in enumerate(PATH_LABELS[hops]):
        fill=TYPE_COLORS[kind]
        label=_node_display(row,i)
        lines.append(
            f"n{i} [label={dot_string(label)}, fillcolor={dot_string(fill)}];"
        )
    for i,(rtype,direction) in enumerate(
        zip(row["relationship_types"],PATH_DIRECTIONS[hops])
    ):
        lines.append(
            f"n{i} -> n{i+1} [label={dot_string(_edge_label(rtype))},"
            f" dir={direction}];"
        )
    lines.append("}")
    return "\n".join(lines)


def edge_details(path:dict) -> list[dict]:
    row=_normalize_path(path,1)
    depth=row["hop_count"]
    details=[]
    for i,(rel,way) in enumerate(
        zip(row["relationship_types"],PATH_DIRECTIONS[depth])
    ):
        details.append({
            "Step":i+1,
            "From":TYPE_ENGLISH[PATH_LABELS[depth][i]],
            "Direction":"←" if way=="back" else "→",
            "Relation":rel,
            "To":TYPE_ENGLISH[PATH_LABELS[depth][i+1]],
            "From ID":_short(row["node_ids"][i],50),
            "To ID":_short(row["node_ids"][i+1],50),
        })
    return details


def _path_summary(row:dict) -> str:
    h=row["hop_count"]
    disease=row.get("disease_name") or "Disease"
    drug=row.get("drug_name") or "Drug"
    related=row.get("related_disease_name")
    connection=f"{drug} → {related}" if related else f"{drug} → Section"
    return (
        f"{h} hops · {_short(disease,16)} · "
        f"{_short(connection,40)} · ID {row['path_id'][:12]}"
    )


def render_path_report(
    report:dict,*,key_prefix:str="p3062",
    title:str="多跳证据路径",view_mode:str|None=None
) -> None:
    """Render already normalized report; no automatic live DB activity."""
    import streamlit as st
    # Always validate again when called from a user-supplied dictionary.
    if report.get("stage")!=STAGE or not isinstance(report.get("paths"),list):
        report=normalize_report(report)
    paths=[_normalize_path(x,i+1) for i,x in enumerate(report["paths"])]
    st.subheader(title)
    st.caption(
        "Neo4j 图谱路径展示 · 技术来源一致性 ≠ 临床疗效或监管批准"
    )
    status=report.get("status","UNKNOWN")
    c1,c2,c3=st.columns(3)
    c1.metric("路径数量",len(paths))
    c2.metric("最大跳数",max((x["hop_count"] for x in paths),default=0))
    c3.metric("可见深度",", ".join(
        str(x) for x in sorted({p["hop_count"] for p in paths})
    ) or "无")
    if view_mode is None:
        view_mode=report.get("mode","saved_report")
    if view_mode=="live_query":
        st.info("本次结果来自刚执行的 Neo4j 只读查询，已运行来源核验。")
    else:
        st.info(
            "正在展示保存的报告或在线 trace：这里不重新查询数据库，"
            "原路径校验结果以生成该报告时的状态为准。"
        )
    st.caption(
        f"检索状态：{status}　"
        f"原问题：{report.get('question','（未记录）')}"
    )
    if not paths:
        st.warning(
            "未找到可展示的 5～8 跳路径。请检查题目是否包含具体疾病词，"
            "或回到实时查询模式选择其他疾病。"
        )
        return
    selectable=["全部"]+[str(depth) for depth in sorted({
        p["hop_count"] for p in paths
    })]
    choose=st.radio(
        "按路径深度筛选",selectable,
        horizontal=True,key=key_prefix+"_depth"
    )
    filtered=[
        p for p in paths
        if choose=="全部" or str(p["hop_count"])==choose
    ]
    if not filtered:
        st.info("当前深度没有返回路径。")
        return
    available_drugs=sorted({
        p["drug_name"] for p in filtered if p.get("drug_name")
    })
    if available_drugs:
        drug_filter=st.selectbox(
            "按药品资料名称过滤已返回的路径",
            ["全部药品"]+available_drugs,
            key=key_prefix+"_drug_filter"
        )
        if drug_filter!="全部药品":
            filtered=[
                row for row in filtered if row["drug_name"]==drug_filter
            ]
    st.dataframe(
        [{
            "Depth":p["hop_count"],
            "Path ID":p["path_id"][:16],
            "Drug":p.get("drug_name") or "—",
            "Start disease":p.get("disease_name") or "—",
            "Related disease":p.get("related_disease_name") or "—",
            "Section":p.get("section_field") or "—",
        } for p in filtered],
        hide_index=True,use_container_width=True
    )
    st.caption("本表仅展示当前检索上限内返回的路径，不代表全图谱穷举结果。")
    ids=[p["path_id"] for p in filtered]
    pick=st.selectbox(
        "选择要检查的路径",ids,
        format_func=lambda pid:_path_summary(
            next(p for p in filtered if p["path_id"]==pid)
        ),
        key=key_prefix+"_path"
    )
    target=next(p for p in filtered if p["path_id"]==pick)
    horizontal=st.checkbox(
        "横向展示路径（取消勾选后纵向）",
        value=True,key=key_prefix+"_horizontal"
    )
    dot=build_dot(target,orientation="LR" if horizontal else "TB")
    try:
        st.graphviz_chart(dot,use_container_width=True)
    except (ImportError,RuntimeError,ValueError) as exc:
        st.warning(
            "当前 Streamlit 环境无法渲染 Graphviz；已提供文本路径和逐跳表。"
        )
        st.code(dot,language="dot")
    st.markdown(
        f"**{target['hop_count']} 条真实关系边** · "
        f"路径标识 `{target['path_id']}`"
    )
    with st.expander("逐跳节点和关系详情",expanded=True):
        st.dataframe(
            edge_details(target),
            hide_index=True,use_container_width=True
        )
    with st.expander("原始来源与验证范围",expanded=False):
        st.write(
            "起点疾病：",target.get("disease_name") or "未记录",
            "；药品资料：",target.get("drug_name") or "未记录"
        )
        if target.get("related_disease_name"):
            st.write(
                "另一疾病术语：",target["related_disease_name"]
            )
        if target.get("section_id"):
            st.write(
                "Section ID：",target["section_id"],
                "；栏目：",target.get("section_field") or "未知"
            )
            if target.get("section_text_preview"):
                st.text_area(
                    "药品资料 Section 原文预览（非全文）",
                    value=target["section_text_preview"],
                    height=110,disabled=True
                )
            if target.get("section_text_sha256"):
                st.code(target["section_text_sha256"])
        if target.get("candidate_quote_preview"):
            st.write(
                "提及候选原文片段：",target["candidate_quote_preview"]
            )
        st.warning(
            "DiseaseCandidate 表示药品资料中存在疾病术语提及。"
            "路径连通、同一 Section 的字符串核验，"
            "不等于临床真实有效性、已核准适应症或疾病因果关系。"
        )
    csv_rows=edge_details(target)
    report_export={
        "stage":STAGE,
        "source_status":"third_party_unverified",
        "path_id":target["path_id"],
        "hop_count":target["hop_count"],
        "steps":csv_rows,
        "medically_verified":False,
    }
    st.download_button(
        "导出当前路径 JSON",
        data=json.dumps(
            report_export,ensure_ascii=False,indent=2
        ).encode("utf-8"),
        file_name=f"p3062_path_{target['path_id'][:16]}.json",
        mime="application/json",
        key=key_prefix+"_download"
    )


def render_multihop_trace_panel(
    trace:dict,*,key_prefix:str="p3062_inline"
) -> None:
    """Optional two-line integration into an existing Streamlit answer page.

    from medical_p3062_multihop_ui_v117 import render_multihop_trace_panel
    render_multihop_trace_panel(trace)
    """
    import streamlit as st
    try:
        report=normalize_report(trace,mode="online_trace")
    except VisualizeBlocked:
        return  # Existing pipeline has no multi-hop trace; no UI disruption.
    if report["paths"]:
        with st.expander("🔎 本轮问答的多跳图谱路径",expanded=False):
            render_path_report(
                report,key_prefix=key_prefix,
                title="本轮问答的来源关联路径",
                view_mode="online_trace"
            )


def inspect_live(
    question:str,*,root:Path,
    graph_audit_path:Path|None=None,
    min_hops:int=8,max_hops:int=8,
    max_paths:int=8,per_template:int=4,
    path_provider=None
) -> dict:
    """Explicit invocation only. Never triggered during page rendering."""
    from medical_p3061_multihop_preflight_v117 import inspect
    result=inspect(
        question,root=Path(root).resolve(),
        graph_audit=graph_audit_path,
        min_hops=min_hops,max_hops=max_hops,
        max_paths=max_paths,per_template=per_template,
        path_provider=path_provider,
        online_preview=False,
    )
    return normalize_report(result,mode="live_query")
