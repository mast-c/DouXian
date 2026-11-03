# -*- coding: utf-8 -*-
"""P3-06.3: append *display-only* multi-hop path cards to existing RAG trace.

The existing Streamlit evidence panel likely renders trace["knowledge_evidence"].
This adapter writes additional distinct path display cards to that same array.
IMPORTANT: these cards are NEVER passed into merged_excerpts, Evidence Guard,
RAG generation, citations, retrieval hit statistics, or clinical conclusions.

One query per answer, cached inside trace to survive P3-04.3 Self-RAG's later
replacement of trace["knowledge_evidence"]. Defaults to OFF.
All new graph queries use the existing P3-06.1 audited 5-8 hop read-only runner.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
from typing import Any

from medical_p3061_multihop_graph_v117 import (
    PATH_LABELS,PATH_TYPES,retrieve_multihop,
)

VERSION="P3-06.3"
FLAG="MEDICAL_P3063_INLINE_PATH_CARDS"
TRACE_KEY="p3063_multihop_path_cards"
CARD_TYPE="unverified_multihop_path_display_only"
# DEFAULT: user has personally verified the 8-hop path, so show 4 such paths.
DEFAULT_MIN_HOPS=8
DEFAULT_MAX_HOPS=8
DEFAULT_MAX_PATHS=4
DEFAULT_PER_TEMPLATE=4
PATH_CAP=16
TEMPLATES={5,6,7,8}
DIRECTIONS={
    5:("←","←","←","→","→"),
    6:("←","←","←","→","→","→"),
    7:("←","←","←","→","→","←","←"),
    8:("←","←","←","→","→","←","→","→"),
}
CN_REL={
    "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA":"严格同名来源关联（未经临床验证）",
    "MENTIONS_TERM":"提及疾病术语",
    "HAS_CANDIDATE":"药品疾病提及候选",
    "HAS_RECORD":"药品来源记录",
    "HAS_SECTION":"来源章节",
    "SUPPORTED_BY":"候选来源章节",
}


def _env_int(key:str,default:int,lower:int,upper:int) -> int:
    try:
        raw=int(os.getenv(key,str(default)).strip())
    except (ValueError,AttributeError):
        return default
    return max(lower,min(upper,raw))


def _config() -> dict:
    minimum=_env_int("MEDICAL_P3063_PATH_MIN_HOPS",DEFAULT_MIN_HOPS,5,8)
    maximum=_env_int("MEDICAL_P3063_PATH_MAX_HOPS",DEFAULT_MAX_HOPS,5,8)
    # If env accidentally reverses depth, constrain instead of inventing queries.
    maximum=max(maximum,minimum)
    return {
        "min_hops":minimum,
        "max_hops":maximum,
        "max_paths":_env_int(
            "MEDICAL_P3063_PATH_MAX_PATHS",DEFAULT_MAX_PATHS,1,PATH_CAP
        ),
        "per_template":_env_int(
            "MEDICAL_P3063_PATH_PER_TEMPLATE",
            DEFAULT_PER_TEMPLATE,1,6
        ),
    }


def _enabled() -> bool:
    return os.getenv(FLAG,"0").strip().lower() in (
        "1","true","yes","on"
    )


def _valid_path(path:dict) -> bool:
    if not isinstance(path,dict):
        return False
    hops=path.get("hop_count")
    if type(hops) is not int or hops not in TEMPLATES:
        return False
    ids=path.get("node_ids")
    rels=path.get("relationship_types")
    labels=path.get("node_labels")
    return (
        isinstance(ids,list) and len(ids)==hops+1
        and all(isinstance(v,str) and bool(v) for v in ids)
        and len(set(ids))==len(ids)
        and isinstance(rels,list) and tuple(rels)==PATH_TYPES[hops]
        and isinstance(labels,list) and len(labels)==hops+1
        and all(
            (isinstance(ls,str) and ls==PATH_LABELS[hops][i])
            or (isinstance(ls,list) and PATH_LABELS[hops][i] in ls)
            for i,ls in enumerate(labels)
        )
        and path.get("clinical_verified") is not True
        and path.get("treatment_relation") is not True
        and path.get("clinical_truth_verified") is not True
        and path.get("graph_relationships_verified") is True
        and path.get("source_status")
            =="third_party_unverified"
    )


def _path_description(row:dict) -> str:
    hops=row["hop_count"]
    nodes=row["node_ids"]
    kinds=PATH_LABELS[hops]
    arrows=DIRECTIONS[hops]
    fields={
        0:row.get("disease_name"),
        1:row.get("disease_name"),
        3:row.get("drug_name"),
    }
    if hops in (6,7,8):
        fields[hops]=row.get("related_disease_name") if hops!=7 else row.get("related_drug_name")
        if hops==7:
            fields[5]=row.get("related_disease_name")
        elif hops==6:
            fields[5]=row.get("related_disease_name")
        elif hops==8:
            fields[7]=row.get("related_disease_name")
    if hops in (5,8):
        fields[5]=row.get("section_field")
    lines=[]
    for i,kind in enumerate(kinds):
        shown=str(fields.get(i) or "").strip()
        label=kind+(f"：{shown[:60]}" if shown else "")
        lines.append(f"{i+1:02d}. {label}  [{nodes[i][:110]}]")
        if i<hops:
            rel=PATH_TYPES[hops][i]
            lines.append(
                f"      {arrows[i]} {rel}"
                f"（{CN_REL.get(rel,'来源关联')}）"
            )
    return "\n".join(lines)


def _card_for_path(row:dict,index:int,source_count:int) -> dict:
    if not _valid_path(row):
        raise ValueError("拒绝展示来源或多跳结构不符合白名单的路径")
    hops=row["hop_count"]
    drug=str(row.get("drug_name") or "未命名药品资料")[:90]
    disease=str(row.get("disease_name") or "疾病")[:80]
    related=str(row.get("related_disease_name") or "")
    path_id=str(row.get("path_id") or "").strip()
    if not path_id:
        canonical="|".join(row["node_ids"])+"|"+"|".join(
            row["relationship_types"]
        )
        path_id=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    caption=(
        f"{disease} → {drug} → {related}"
        if related else f"{disease} → {drug} → 药品原文 Section"
    )
    title=f"GraphRAG 多跳路径 P{index:02d} · {hops} 跳（仅展示，不是回答引用）"
    snippet=(
        f"【图谱路径 P{index:02d}｜{hops} 跳｜未核验医疗来源】\n"
        f"{caption}\n"
        f"Path ID：{path_id}\n"
        f"图谱关系类型：严格同名来源关联 / 提及候选 / 原始章节。\n"
        f"逐跳节点与边：\n{_path_description(row)}\n"
    )
    if row.get("section_id") and hops in (5,8):
        snippet+=(
            "\nSection ID："+str(row["section_id"])[:120]
            +"\nSection 原文候选片段："
            +str(row.get("candidate_quote") or "")[:280]+"\n"
        )
    snippet+=(
        "\n【重要】这是图谱关系的来源关联路径，不是新的医学原文答案；"
        "药品资料提及某疾病不代表该药治疗该疾病，也不代表其获批适应症。"
        "此路径不参加模型生成、引用编号或证据充分性评分。"
    )
    return {
        # Keep numeric IDs so existing generic card renderers can iterate,
        # while the title explicitly disclaims answer citation status.
        "id":source_count+index,
        "source":title,
        "section_path":f"Neo4j / {hops} 跳 / {caption}"[:240],
        "preview":(
            f"路径 ID {path_id}；{caption}。"
            "这仅是疾病名称共同提及的来源路径，不代表治疗关联。"
        )[:240],
        "content":snippet,
        "source_url":"",
        "evidence_type":CARD_TYPE,
        "candidate_status":"graph_path_display_only",
        "clinical_status":"not_verified",
        "source_status":"third_party_unverified",
        "clinical_verified":False,
        "rights_reviewed":False,
        "drug_name":drug,
        "disease_term":disease,
        "evidence_quote":str(row.get("candidate_quote") or "")[:500],
        "candidate_id":str(row.get("candidate_id") or ""),
        "record_id":str(row.get("record_id") or ""),
        "section_id":str(row.get("section_id") or ""),
        "section_field":str(row.get("section_field") or ""),
        "parent_verified":False,
        "path_id":path_id,
        "hop_count":hops,
        "node_ids":list(row["node_ids"]),
        "node_labels":list(PATH_LABELS[hops]),
        "relationship_types":list(PATH_TYPES[hops]),
        "graph_verified":True,
        "display_only":True,
        "counts_as_answer_evidence":False,
        "counts_as_retrieval_hit":False,
        "is_answer_citation":False,
        "treatment_relation":False,
    }


def attach_multihop_path_cards(
    trace:dict,question:str,*,project_root:Path
) -> dict:
    """Append multiple path cards to trace["knowledge_evidence"], no answer change.

    Safe to call twice: 2nd call reattaches cached paths after Self-RAG
    reassigns trace["knowledge_evidence"], WITHOUT repeating Neo4j queries.
    """
    if not isinstance(trace,dict):
        return {"status":"SKIPPED_NO_TRACE"}
    from dotenv import load_dotenv
    root=Path(project_root).resolve()
    load_dotenv(root/".env",override=False)
    if not _enabled():
        return {"status":"DISABLED"}
    rows=trace.get("knowledge_evidence")
    if not isinstance(rows,list):
        return {"status":"SKIPPED_NO_EVIDENCE_LIST"}
    if not isinstance(question,str) or not question.strip():
        return {"status":"SKIPPED_EMPTY_QUESTION"}
    # Never alter the original retrieval evidence; only append display cards.
    originals=[
        row for row in rows
        if not(isinstance(row,dict) and row.get("evidence_type")==CARD_TYPE)
    ]
    metadata=trace.get(TRACE_KEY)
    if not isinstance(metadata,dict):
        config=_config()
        metadata={
            "stage":VERSION,
            "status":"NOT_STARTED",
            "source_evidence_count":len(originals),
            "path_cards_count":0,
            "graph_writes":0,
            "chroma_writes":0,
            "display_only":True,
            "excluded_from_generation":True,
            "excluded_from_citation_scoring":True,
            "max_paths":config["max_paths"],
            "requested_hops":[
                x for x in range(config["min_hops"],config["max_hops"]+1)
            ],
            "created_utc":datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
        }
        trace[TRACE_KEY]=metadata
        # Explicit exclusion of queries with no disease evidence route.
        disease_trace=trace.get("disease_encyclopedia")
        if isinstance(disease_trace,dict):
            if disease_trace.get("route") in ("NONE","DRUG_ONLY") and (
                not disease_trace.get("matched_disease_names")
            ):
                metadata["status"]="SKIPPED_NO_DISEASE_ROUTE"
        if metadata["status"]=="NOT_STARTED":
            try:
                paths,graph_trace=retrieve_multihop(
                    question,
                    project_root=root,
                    min_hops=config["min_hops"],
                    max_hops=config["max_hops"],
                    max_paths=config["max_paths"],
                    per_template=config["per_template"],
                    require_enabled=False,
                )
                # Only real P3-06.1-validated nodes+relationships are
                # accepted; no path is invented from the model answer.
                validated=[
                    {
                        **{
                            k:v for k,v in x.items()
                            if k not in ("section_text",)
                        },
                        "candidate_quote":str(
                            x.get("candidate_quote") or ""
                        )[:500],
                    }
                    for x in paths if _valid_path(x)
                ]
                metadata["status"]=(
                    "MULTIHOP_PATH_CARDS_READY"
                    if validated else "NO_VERIFIED_PATHS"
                )
                metadata["path_cards_count"]=len(validated)
                metadata["paths"]=validated
                metadata["graph_trace"]={
                    key:graph_trace.get(key)
                    for key in (
                        "status","query_count","found_per_hops",
                        "graph_audit_status","max_hops_observed",
                        "requested_hops","elapsed_ms"
                    )
                }
            except Exception as exc:
                metadata["status"]="PATH_QUERY_BLOCKED_KEEP_ORIGINAL_EVIDENCE"
                metadata["reason_type"]=type(exc).__name__
                metadata["paths"]=[]
                metadata["path_cards_count"]=0
    cached=metadata.get("paths",[])
    if not isinstance(cached,list):
        cached=[]
    # no duplication when called again; Self-RAG can replace evidence midway
    rows[:]=originals
    source_count=max(
        (int(x.get("id",0)) for x in originals
         if isinstance(x,dict) and type(x.get("id")) is int),
        default=0
    )
    append_generic=os.getenv(
        "MEDICAL_P3063_APPEND_GENERIC_CARDS","1"
    ).strip().lower() not in ("0","false","off","no")
    cards=[]
    if append_generic:
        for i,path in enumerate(cached[:PATH_CAP],1):
            try:
                cards.append(_card_for_path(path,i,source_count))
            except (KeyError,TypeError,ValueError):
                continue
    rows.extend(cards)
    metadata["source_evidence_count"]=len(originals)
    metadata["path_cards_count"]=len(cached)
    metadata["appended_card_count"]=len(cards)
    metadata["displayed_card_count"]=len(rows)
    metadata["status"]=(
        "MULTIHOP_PATH_CARDS_APPENDED" if cards
        else "MULTIHOP_PATHS_READY_FOR_RICH_UI" if cached
        else metadata.get("status","NO_VERIFIED_PATHS")
    )
    # Deliberately DO NOT change retrieval_pipeline or knowledge_sources.
    return {
        "status":metadata["status"],
        "original_evidence_count":len(originals),
        "display_path_card_count":len(cards),
        "paths_available_for_rich_ui":len(cached),
    }


def _dot_quote(value:str) -> str:
    """Escapes all report-controlled text before adding it to Graphviz DOT."""
    value=str(value or "")[:200]
    value=value.replace("\\","\\\\").replace('"','\\"')
    value=value.replace("\r"," ").replace("\n","\\n")
    value="".join(ch for ch in value if ord(ch)>=32)
    return '"'+value+'"'


def _inline_graph_dot(path:dict) -> str:
    if not _valid_path(path):
        raise ValueError("Graphviz 节点/关系不在受约束的多跳白名单中")
    h=path["hop_count"]
    colors={
        "DiseaseDocument":"#DFF3ED",
        "DiseaseTerm":"#E2ECF9",
        "DiseaseCandidate":"#FFF1DD",
        "Drug":"#FFE5D7",
        "SourceRecord":"#EDF0F4",
        "Section":"#F2E8FF",
    }
    labels=PATH_LABELS[h]
    nodes=path["node_ids"]
    rels=PATH_TYPES[h]
    directions=DIRECTIONS[h]
    lines=[
        "digraph InlineEvidence {",
        "graph [rankdir=LR,bgcolor=transparent,pad=0.2,"
        "nodesep=0.34,ranksep=0.8];",
        'node [shape=box,style="rounded,filled",fontname="Microsoft YaHei",'
        'fontsize=11,color="#78899b",margin="0.12,0.10"];',
        'edge [fontname="Arial",fontsize=9,color="#6a8498",'
        'fontcolor="#496a7e",arrowsize=0.6];'
    ]
    for i,kind in enumerate(labels):
        detail=""
        if i in (0,1):
            detail=str(path.get("disease_name") or "")
        elif i==3:
            detail=str(path.get("drug_name") or "")
        elif kind=="Drug":
            detail=str(path.get("related_drug_name") or "")
        elif kind in ("DiseaseDocument","DiseaseTerm"):
            detail=str(path.get("related_disease_name") or "")
        elif kind=="Section":
            detail=str(path.get("section_field") or "")
        elif kind=="SourceRecord":
            detail=str(path.get("record_id") or "")[:16]
        title=kind+("\n"+detail[:24] if detail else "")
        lines.append(
            f"n{i} [label={_dot_quote(title)},"
            f"fillcolor={_dot_quote(colors[kind])}];"
        )
    for i,(rel,direction) in enumerate(zip(rels,directions)):
        arrow="back" if direction=="←" else "forward"
        lines.append(
            f"n{i} -> n{i+1} [label={_dot_quote(rel)},dir={arrow}];"
        )
    lines.append("}")
    return "\n".join(lines)


def _inline_edge_details(path:dict) -> list[dict]:
    if not _valid_path(path):
        raise ValueError("Path metadata invalid")
    hops=path["hop_count"]
    return [{
        "Step":i+1,
        "From":PATH_LABELS[hops][i],
        "Direction":DIRECTIONS[hops][i],
        "Relationship":PATH_TYPES[hops][i],
        "To":PATH_LABELS[hops][i+1],
        "From ID":path["node_ids"][i][:100],
        "To ID":path["node_ids"][i+1][:100],
    } for i in range(hops)]


def render_rich_multihop_paths_inside_evidence(
    trace:dict,*,key_prefix:str="p3063"
) -> int:
    """Optional placement directly AFTER the existing evidence cards loop.

    No database queries; uses verified paths cached by attach() hook.
    Set MEDICAL_P3063_APPEND_GENERIC_CARDS=0 to prevent duplicate cards if
    this richer Streamlit rendering is placed in the main UI.
    """
    import streamlit as st
    # Standalone DOT builder: no dependence on P3-06.2 installation.
    info=trace.get(TRACE_KEY) if isinstance(trace,dict) else None
    if not isinstance(info,dict):
        return 0
    paths=info.get("paths")
    if not isinstance(paths,list):
        return 0
    valid=[row for row in paths if _valid_path(row)][:PATH_CAP]
    if not valid:
        return 0
    st.markdown(
        f"**🔗 关联路径（{len(valid)} 条）· 独立于原文引用**"
    )
    st.caption(
        "下列是 Neo4j 图谱来源路径，不计入已通过证据充分性审核的原文数量。"
        "路径关联或共同提及，不等于药物疗效、疾病因果或临床审核。"
    )
    for i,path in enumerate(valid,1):
        hops=path["hop_count"]
        name=str(path.get("drug_name") or "药品资料")
        related=str(path.get("related_disease_name") or "Section")
        pid=str(path.get("path_id") or "")[:20]
        label=(
            f"路径 P{i:02d} · {hops} 跳 · "
            f"{name[:42]} → {related[:42]}"
        )
        # Streamlit forbids nesting expanders; the surrounding evidence
        # section may itself already be an st.expander. Use st.container.
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption("Path ID: "+pid)
            graph_source=_inline_graph_dot(path)
            try:
                st.graphviz_chart(
                    graph_source,use_container_width=True
                )
            except (ImportError,RuntimeError,ValueError):
                st.code(graph_source,language="dot")
            if st.checkbox(
                f"查看路径 P{i:02d} 的逐跳关系与来源",
                value=False,
                key=f"{key_prefix}_details_{i}_{pid}"
            ):
                st.dataframe(
                    _inline_edge_details(path),
                    hide_index=True,
                    use_container_width=True
                )
                if path.get("section_id"):
                    st.write("Section ID:",path["section_id"])
                if path.get("candidate_quote"):
                    st.write(
                        "资料提及候选句：",
                        str(path["candidate_quote"])[:300]
                    )
            st.caption(
                "路径已通过技术来源核验；医学事实和治疗关系未经核验。"
            )
    return len(valid)
