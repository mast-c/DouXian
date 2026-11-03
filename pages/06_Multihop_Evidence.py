# -*- coding: utf-8 -*-
"""P3-06.2 Streamlit multipage dashboard: safe 5–8-hop evidence paths.

Run normally as a page inside existing Streamlit app `pages/`, or launch
standalone: streamlit run pages/06_Multihop_Evidence.py
No Neo4j connection happens until the user clicks "执行只读检索".
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from medical_p3062_multihop_ui_v117 import (
    read_report_bytes,
    read_report_file,
    list_saved_reports,
    render_path_report,
    inspect_live,
    VisualizeBlocked,
)

st.set_page_config(
    page_title="多跳证据链 | 豆馅医疗 GraphRAG",
    page_icon="🔗",
    layout="wide",
)
st.title("🔗 豆馅医疗 · 多跳证据链")
st.caption(
    "GraphRAG V1.17 / P3-06.2 · 5–8 hop Neo4j paths · "
    "基于真实节点与关系的来源关联可视化"
)
st.warning(
    "本页仅可视化第三方未核验数据中的疾病提及及来源关系。"
    "疾病术语与药品同处图谱路径不代表药物治疗疾病、已核准适应症或临床有效性。"
)

tab_history,tab_live,tab_upload=st.tabs([
    "📂 历史多跳报告","🔎 实时 Neo4j（只读）","📤 上传 JSON／在线 Trace"
])

with tab_history:
    st.markdown("#### 查看刚才运行生成的 8 跳报告")
    saved=list_saved_reports(PROJECT_ROOT)
    if not saved:
        st.info(
            "stage4_reports 中尚无 p3061_8hop_*.json 或 "
            "p3061_multihop_*.json。可直接切换到实时查询，"
            "或上传之前生成的报告。"
        )
    else:
        file=st.selectbox(
            "选择历史审计 JSON",
            options=saved,
            format_func=lambda value:value.name,
            key="p3062_history_choice"
        )
        try:
            report=read_report_file(file)
        except (OSError,ValueError,TypeError) as exc:
            st.error(f"无法读取历史多跳报告：{exc}")
        else:
            render_path_report(
                report,key_prefix="p3062_history",
                title="历史报告的多跳图谱路径",
                view_mode="saved_report"
            )
            st.caption("源文件："+file.name)

with tab_live:
    st.markdown("#### 直接查询真实 Neo4j 的关系路径")
    st.caption(
        "每次点击才执行数据库查询。"
        "查询采用固定 Cypher 白名单、只读会话和结果数量限制。"
    )
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT/".env",override=False)

    audit_files=sorted(
        (PROJECT_ROOT/"stage4_reports").glob("p3044b_final_audit_*.json"),
        key=lambda value:(value.stat().st_mtime_ns,value.name),
        reverse=True,
    ) if (PROJECT_ROOT/"stage4_reports").is_dir() else []

    with st.form("p3062_query_form",clear_on_submit=False):
        question=st.text_area(
            "请输入要分析的疾病与药品来源问题",
            value="哪些药品说明书提到了过敏性鼻炎？",
            max_chars=500,
            height=90,
            help="只提取明确疾病词名称；不会把药品共现自动当作疗效关系。",
        )
        mode=st.selectbox(
            "路径搜索范围",
            options=[
                "仅查询 8 跳（展示深度关联）",
                "查询 5～8 跳（完整探索）",
                "仅查询 5 跳（原文 Section）",
                "仅查询 6 跳",
                "仅查询 7 跳",
            ],
            index=0,
        )
        ca,cb=st.columns(2)
        with ca:
            max_paths=st.slider(
                "路径总上限",min_value=1,max_value=24,value=16
            )
        with cb:
            per_template=st.slider(
                "单模板最多返回",min_value=1,max_value=6,value=4
            )
        st.caption(
            "建议 5～8 跳一起查时，将路径总上限设置为"
            "至少「每模板最多返回 × 4」；"
            "否则前面的短路径可能先占满名额，导致 8 跳未被执行。"
            "想确认 8 跳时建议选「仅查询 8 跳」。"
        )
        chosen_audit=st.selectbox(
            "P3-04.4B 完整图谱审计",
            options=audit_files if audit_files else [None],
            format_func=lambda f:f.name if f is not None else "未找到",
        )
        clicked=st.form_submit_button(
            "🔎 执行 Neo4j 只读多跳检索",
            type="primary",
            use_container_width=True,
        )
    if clicked:
        if chosen_audit is None:
            st.error(
                "缺少 p3044b_final_audit_*.json。"
                "先完成 P3-04.4B 完整来源关联审计。"
            )
        elif not question.strip():
            st.error("请输入具体疾病名称。")
        else:
            if mode.startswith("仅查询 8"):
                depth=(8,8)
            elif mode.startswith("查询 5"):
                depth=(5,8)
            elif mode.startswith("仅查询 5"):
                depth=(5,5)
            elif mode.startswith("仅查询 6"):
                depth=(6,6)
            else:
                depth=(7,7)
            try:
                with st.spinner(
                    f"正在只读查询真实 Neo4j（{depth[0]}～{depth[1]} 跳）……"
                ):
                    outcome=inspect_live(
                        question.strip(),root=PROJECT_ROOT,
                        graph_audit_path=chosen_audit,
                        min_hops=depth[0],max_hops=depth[1],
                        max_paths=max_paths,
                        per_template=per_template,
                    )
                    st.session_state["p3062_live_report"]=outcome
            except (RuntimeError,OSError,ValueError,ImportError,TypeError) as exc:
                st.session_state.pop("p3062_live_report",None)
                st.error(
                    f"多跳来源校验未通过：{type(exc).__name__}: {exc}"
                )
    if "p3062_live_report" in st.session_state:
        render_path_report(
            st.session_state["p3062_live_report"],
            key_prefix="p3062_live",
            title="本次实时查询的 5–8 跳路径",
            view_mode="live_query",
        )

with tab_upload:
    st.markdown("#### 上传独立预检报告或在线问答 Trace")
    st.caption(
        "支持 p3061_8hop_*.json、p3061_multihop_*.json，"
        "也支持包含 disease_encyclopedia.multi_hop_graph "
        "字段的完整问答 trace。上传不会修改本地文件或数据库。"
    )
    uploaded=st.file_uploader(
        "选择 JSON（最大 8 MiB）",
        type=["json"],key="p3062_uploaded_json",
    )
    if uploaded is not None:
        try:
            result=read_report_bytes(
                uploaded.getvalue(),mode="uploaded_report"
            )
            render_path_report(
                result,key_prefix="p3062_upload",
                title="上传报告的多跳图谱路径",
                view_mode="uploaded_report",
            )
        except (ValueError,OSError,TypeError) as exc:
            st.error(f"上传报告格式或来源标记不合法：{exc}")

with st.sidebar:
    st.markdown("### 多跳检索说明")
    st.caption(
        "5 hops：Disease→Drug→Section\n\n"
        "6 hops：same Drug → two Diseases\n\n"
        "7 hops：bridge Disease → other Drug\n\n"
        "8 hops：same Section → two Diseases"
    )
    st.divider()
    st.caption("科研研究用途 · 未完成临床验证")
    with st.expander("接入现有聊天结果页（可选）"):
        st.code(
            "from medical_p3062_multihop_ui_v117 import render_multihop_trace_panel\n"
            "render_multihop_trace_panel(trace)",
            language="python"
        )
        st.caption(
            "将第二行放在已获得在线 trace 的问答结果渲染位置。"
            "无需修改 Neo4j 或 rag.py；若原前端变量不叫 trace，"
            "请传入它实际保存的完整 trace 字典。"
        )
