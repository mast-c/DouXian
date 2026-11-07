# -*- coding: utf-8 -*-
"""Optional Streamlit renderer for VERIFIED direct and multi-hop graph paths.

The existing frontend implementation was not supplied in the current upload;
this component must be invoked explicitly from the user's Streamlit page.
It does not decide clinical sufficiency and never converts an unverified
structural connection to an answer claim.
"""
from __future__ import annotations


def render_verified_graph_paths(trace: dict, st=None, *, limit: int = 12):
    """Show independent source-verified graph paths, including one-hop edges.

    Example after ``trace`` is populated:
        from medical_graph_path_ui import render_verified_graph_paths
        render_verified_graph_paths(trace)

    Returns number of rendered paths, for tests/metrics.
    """
    if st is None:
        import streamlit as st
    graph = (trace or {}).get("semantic_graph") or {}
    paths = [path for path in graph.get("paths") or []
             if isinstance(path, dict) and path.get("verified_against_original")
             and path.get("edges") and path.get("id")]
    paths = list({path["id"]: path for path in paths}.values())
    paths.sort(key=lambda p: (int(p.get("hops") or 0), p["id"]))
    direct = sum(p.get("hops") == 1 for p in paths)
    multi = sum(int(p.get("hops") or 0) >= 2 for p in paths)
    diverse = sum(len(set(p.get("relations") or ())) >= 2 for p in paths)
    st.markdown(f"**来源可核验的图谱关系：{len(paths)} 条**（单跳 {direct} / 多跳 {multi} / 多类关系 {diverse}）")
    st.caption("仅表示原始资料中的关系提及，不等于临床因果或安全性结论；答案仍需证据审核。")
    if graph.get("status") == "source_verification_blocked":
        st.warning("图谱来源索引或快照校验失败；本轮未展示未经核验的候选路径。")
        return 0
    if not paths:
        st.info("当前没有通过原始来源校验的图谱路径。结构连接和可信医学证据并不等价。")
        return 0
    for num, path in enumerate(paths[:max(0, min(limit, 25))], 1):
        hops = int(path.get("hops") or len(path["edges"]))
        rels = len(set(path.get("relations") or ()))
        with st.expander(f"路径 {num} · {hops} 跳 · {rels} 类关系 · 来源已回查"):
            for pos, edge in enumerate(path.get("edges") or [], 1):
                origin = (edge.get("src") or {}).get("name", "未知起点")
                target = (edge.get("dst") or {}).get("name", "未知终点")
                relation = edge.get("relation_label") or edge.get("relation") or "未知关系"
                st.write(f"{pos}. {origin} —{relation}→ {target}")
                st.caption("来源：" + str(edge.get("source_file") or "未标注")
                           + " · 章节：" + str(edge.get("section_field") or "未标注"))
                quote = edge.get("quote")
                if isinstance(quote, str) and quote:
                    st.text(quote[:1400])
            st.caption("已核验原文位置，不代表医学临床验证已完成。")
    if len(paths) > limit:
        st.caption(f"仅显示前 {limit} 条，共有 {len(paths)} 条核验路径。")
    return min(len(paths), max(0, min(limit, 25)))
