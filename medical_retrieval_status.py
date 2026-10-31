# -*- coding: utf-8 -*-
"""V1.11 医疗 Hybrid GraphRAG 检索状态：区分取回、原文核验、过滤和去重。

只根据实际工具报告及实际进入各处理阶段的证据计数。
没有网络访问、不操作数据库、不推断医学有效性。
这个模块同时被 rag.py 和 Streamlit 页面使用，确保统计口径相同。
"""
from __future__ import annotations

import html
from typing import Any

SCHEMA_VERSION = 1

STATUS_TEXT = {
    "ok": "已完成",
    "no_match": "未命中",
    "empty_collection": "向量库为空",
    "error": "执行失败",
    "invalid_response": "结果格式异常",
    "provenance_rejected": "原文校验未通过",
    "no_relevant_sections": "栏目不相关",
    "not_run": "未执行",
}


def _count(value: Any) -> int:
    """容错转换，不把 bool、负数或无效字符串当作命中数量。"""
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _part(reports: dict, tool: str) -> dict:
    value = reports.get(tool)
    return value if isinstance(value, dict) else {}


def _origin_count(items: list[dict], tool_name: str) -> int:
    return sum(
        isinstance(item, dict) and item.get("_retrieval_tool") == tool_name
        for item in items
    )


def build_retrieval_pipeline(
    tool_reports: dict[str, Any],
    before_entity_filter: list[dict[str, Any]],
    after_entity_filter: list[dict[str, Any]],
    after_dedup: list[dict[str, Any]],
    entity_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """用真实四阶段数据创建可持久化的轻量诊断快照。

    returned 是工具返回的可用证据数量，不是底层 BM25/向量库的
    原始候选总数；parent_verified 是 Chroma 证据执行 Neo4j 原文校验
    通过的数量；retained 是跨工具去重之后仍作为独立证据保留的数量。
    """
    reports = tool_reports if isinstance(tool_reports, dict) else {}
    graph = _part(reports, "search_drug_graph")
    chroma = _part(reports, "search_knowledge_base")
    mv = chroma.get("medical_validation")
    mv = mv if isinstance(mv, dict) else {}
    er = entity_report if isinstance(entity_report, dict) else {}

    graph_stage = {
        "status": str(graph.get("status") or "not_run"),
        "returned": _count(graph.get("hits_before_entity_gate", graph.get("hits"))),
        "direct_sections": _count(graph.get("section_count")),
        "candidate_cards": _count(graph.get("candidate_count")),
        "raw_candidate_nodes": _count(graph.get("raw_candidate_nodes")),
        "grouped_same_quote_nodes": _count(graph.get("grouped_same_quote_nodes")),
        "after_entity_filter": _origin_count(after_entity_filter, "search_drug_graph"),
        "retained": _origin_count(after_dedup, "search_drug_graph"),
        "had_partial_error": bool(graph.get("partial_error")),
    }
    chroma_stage = {
        "status": str(chroma.get("status") or "not_run"),
        "returned": _count(chroma.get("hits_before_entity_gate", chroma.get("hits"))),
        "indexed_chunks": _count(chroma.get("chroma_chunks")),
        "parent_verified": _count(mv.get("verified")),
        "parent_rejected": _count(mv.get("failed_provenance")),
        "section_not_relevant": _count(mv.get("skipped_section")),
        "legacy_not_verified": _count(mv.get("legacy_unverified")),
        "after_entity_filter": _origin_count(after_entity_filter, "search_knowledge_base"),
        "retained": _origin_count(after_dedup, "search_knowledge_base"),
    }
    before = len(before_entity_filter)
    scoped = len(after_entity_filter)
    final = len(after_dedup)
    merge = {
        "unique_before_entity_filter": before,
        "after_entity_filter": scoped,
        "removed_by_entity_filter": max(0, before - scoped),
        "removed_by_dedup": max(0, scoped - final),
        "retained": final,
        "retained_neo4j_verified": sum(
            bool(item.get("parent_verified"))
            for item in after_dedup if isinstance(item, dict)
        ),
        "explicit_drugs": [
            str(name)[:120]
            for name in er.get("explicit_drugs", [])
            if isinstance(name, str)
        ][:6] if isinstance(er.get("explicit_drugs"), list) else [],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "neo4j_direct": graph_stage,
        "chroma_hybrid": chroma_stage,
        "merge": merge,
    }


def retrieval_heading(trace: Any) -> str | None:
    """仅供新版历史记录/当前状态条使用；旧 trace 返回 None。"""
    if not isinstance(trace, dict):
        return None
    pipeline = trace.get("retrieval_pipeline")
    if not isinstance(pipeline, dict) or pipeline.get("schema_version") != SCHEMA_VERSION:
        return None
    graph = _part(pipeline, "neo4j_direct")
    chroma = _part(pipeline, "chroma_hybrid")
    merge = _part(pipeline, "merge")
    n = _count(merge.get("retained"))
    verified = _count(chroma.get("parent_verified"))
    errors = any(
        item.get("status") in ("error", "invalid_response", "provenance_rejected")
        for item in (graph, chroma)
    )
    label = f"医疗检索完成 · 合并去重后 {n} 条证据"
    if verified:
        label += f" · Neo4j 原文核验 {verified} 条"
    elif _count(graph.get("returned")):
        label += f" · Neo4j 直查 {_count(graph['returned'])} 条"
    if errors:
        label += "（部分通道异常）"
    return label


def _number(value: Any) -> str:
    return str(_count(value)) + " 条"


def retrieval_status_html(trace: Any) -> str:
    """渲染克制、可读的 HTML 统计，不用 Neo4j 0 暗示数据库未参与。"""
    if not isinstance(trace, dict):
        return ""
    pipeline = trace.get("retrieval_pipeline")
    if not isinstance(pipeline, dict) or pipeline.get("schema_version") != SCHEMA_VERSION:
        return ""

    graph = _part(pipeline, "neo4j_direct")
    chroma = _part(pipeline, "chroma_hybrid")
    merge = _part(pipeline, "merge")
    g_status = str(graph.get("status") or "not_run")
    c_status = str(chroma.get("status") or "not_run")

    def metric(label: str, count: int, detail: str) -> str:
        # label/detail are authored text or enumerated statuses only.
        return (
            '<div class="kb-stat">'
            f'<dt>{html.escape(label)}</dt>'
            f'<dd>{html.escape(_number(count))}</dd>'
            f'<span class="kb-stat-detail">{html.escape(detail)}</span>'
            '</div>'
        )

    cards = [
        metric(
            "Neo4j 直接检索",
            graph.get("returned"),
            f"去重后独立保留 {_number(graph.get('retained'))} · "
            + STATUS_TEXT.get(g_status, "状态未知")
        ),
        metric(
            "Chroma 文本证据",
            chroma.get("returned"),
            f"去重后独立保留 {_number(chroma.get('retained'))} · "
            + STATUS_TEXT.get(c_status, "状态未知")
        ),
        metric(
            "Neo4j 原文核验",
            chroma.get("parent_verified"),
            f"来源检查未通过 {_number(chroma.get('parent_rejected'))}"
        ),
        metric(
            "合并后证据",
            merge.get("retained"),
            f"同原文合并 {_number(merge.get('removed_by_dedup'))} · "
            f"药品实体过滤 {_number(merge.get('removed_by_entity_filter'))}"
        ),
    ]
    notes: list[str] = []
    g_returned = _count(graph.get("returned"))
    g_kept = _count(graph.get("retained"))
    parent_verified = _count(chroma.get("parent_verified"))
    dedup_removed = _count(merge.get("removed_by_dedup"))
    if parent_verified:
        notes.append(
            f"Chroma 的 {parent_verified} 条文本证据通过了 Neo4j 父级 Section 原文核对；"
            "该核对是技术溯源，不代表医学专家审核。"
        )
    if g_returned and not g_kept:
        if _count(graph.get("after_entity_filter")) and dedup_removed:
            notes.append(
                f"Neo4j 直接返回过 {g_returned} 条证据；"
                "与其他通道的相同原文合并后，没有单独保留的图谱证据卡。"
            )
        else:
            notes.append(
                f"Neo4j 直接返回过 {g_returned} 条证据；"
                "经过药品匹配或合并后未单独保留。"
            )
    if g_status in ("error", "invalid_response"):
        notes.append("Neo4j 直接检索失败，不能将该通道的 0 条解释为查询无匹配。")
    elif c_status in ("error", "invalid_response"):
        notes.append("Chroma 文本检索失败，未能统计正常检索覆盖。")
    if graph.get("had_partial_error"):
        notes.append("Neo4j 部分子查询失败；剩余检索证据仍按现有逻辑保留。")
    if _count(chroma.get("section_not_relevant")):
        notes.append(
            f"Chroma 还有 {_count(chroma['section_not_relevant'])} 条候选因章节不相关被排除。"
        )

    note_html = (
        '<p class="kb-stat-note">' + html.escape(" ".join(notes)) + '</p>'
        if notes else ""
    )
    return (
        '<section class="kb-retrieval-status" aria-label="检索通道统计">'
        '<div class="trace-step-title">'
        '<span class="trace-step-icon" aria-hidden="true">⌕</span>'
        '<span>检索与原文核验统计</span></div>'
        '<dl class="kb-retrieval-metrics">' + "".join(cards) + '</dl>'
        + note_html + '</section>'
    )
