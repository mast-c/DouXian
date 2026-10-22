# -*- coding: utf-8 -*-
"""豆馅 V2.8 图谱辅助检索：只做查询扩展，不生成事实或引用。

必须使用可信的本地 V2.7 图谱与完全一致的原 Markdown，SHA256 不匹配时
原样返回子查询；不会执行向量搜索、大模型调用、Reranker 或文件改写。
"""
from functools import lru_cache
import json
import os
from pathlib import Path
import re

if __package__:
    from .graph_chunk_linker_v211 import check_graph_source
else:
    from graph_chunk_linker_v211 import check_graph_source

SCHEMA = "douxian-kg-2.7-offline-1"


# 只允许用已核对原文的语义关系提供检索关键词。
SEMANTIC_RELATIONS = frozenset({
    "IS_A", "SPECIAL_CASE_OF", "HAS_PART", "USES", "BASED_ON",
    "SUBFIELD_OF", "COMPUTES",
})
RELATION_INTENT = re.compile(
    r"关系|从属|属于|特例|分类|类别|子方法|哪些算法|有哪些方法|"
    r"包含什么|包括什么|由什么组成|由哪些.*组成|有哪些层|"
    r"有哪些组成|结构包含|有什么组成"
)
MAX_ADDED_TERMS = 3
MAX_QUERY_CHARACTERS = 220


def _contains(text, candidate):
    if not candidate:
        return False
    if any(ord(ch) >= 128 for ch in candidate):
        return candidate in text
    return re.search(
        r"(?<![A-Za-z])" + re.escape(candidate) + r"(?![A-Za-z])",
        text, re.IGNORECASE
    ) is not None


@lru_cache(maxsize=4)
def _read_verified_graph(
    graph_file: str, graph_mtime_ns: int, graph_size: int,
    source_file: str, source_mtime_ns: int, source_size: int,
):
    """以路径+文件状态为键缓存；仅第一次加载或文件变化时验证全部来源证据。"""
    del graph_mtime_ns, graph_size, source_mtime_ns, source_size
    path = Path(graph_file)
    graph = json.loads(path.read_text(encoding="utf-8"))
    metadata = graph.get("metadata") or {}
    if metadata.get("schema") != SCHEMA:
        raise ValueError("知识图谱 schema 与 V2.7 不匹配")
    if metadata.get("source_file") != Path(source_file).name:
        raise ValueError("图谱来源文件名与当前教材不匹配")
    check_graph_source(graph, Path(source_file))
    return graph


def _normalized_nodes_edges(graph):
    nodes = {}
    for n in graph["nodes"]:
        if n.get("type") == "Concept" and isinstance(n.get("label"), str):
            nodes[n["id"]] = n

    # 只允许有 evidence.extraction=exact_source_phrase 的语义边参与查询扩展。
    relations = []
    for e in graph["edges"]:
        if (
            e.get("relation") in SEMANTIC_RELATIONS
            and e.get("source") in nodes and e.get("target") in nodes
            and e.get("evidence", {}).get("extraction") == "exact_source_phrase"
        ):
            relations.append(e)
    return nodes, relations


def suggest_retrieval_queries(question, subqueries, graph_file, source_file):
    """安全返回 (queries, diagnostics)，结果长度与原 subqueries 永远一致。

    graph_file: 事先离线构建的 JSON 图谱。
    source_file: 原始 Markdown（按 SHA256 逐字验证）。
    任何错误/缺失/过期都退回 V2.6 原检索词，不改变模型调用次数。
    """
    originals = list(subqueries) if isinstance(subqueries, (list, tuple)) else []
    details = {
        "status": "no_match",
        "added_terms": [],
        "queries_before": len(originals),
        "queries_after": len(originals),
        "graph_semantic_edges_used": 0,
    }
    if not originals or not isinstance(question, str) or not question.strip():
        return originals, details

    graph_path = Path(graph_file).expanduser()
    source_path = Path(source_file).expanduser()
    if not graph_path.is_file() or not source_path.is_file():
        details["status"] = "missing_file"
        return originals, details

    try:
        graph_stat, source_stat = graph_path.stat(), source_path.stat()
        graph = _read_verified_graph(
            str(graph_path.resolve()), graph_stat.st_mtime_ns, graph_stat.st_size,
            str(source_path.resolve()), source_stat.st_mtime_ns, source_stat.st_size,
        )
        nodes, relationships = _normalized_nodes_edges(graph)
    except (ValueError, KeyError, TypeError, OSError, AssertionError, json.JSONDecodeError):
        details["status"] = "invalid_or_stale"
        return originals, details

    # 必须在用户原问题中提及，且在该条子查询中也能定位术语；不污染其他子问题。
    matched_nodes = []
    for node in nodes.values():
        terms = (node["label"], *node.get("aliases_for_search", []))
        if any(_contains(question, term) for term in terms):
            matched_nodes.append(node)
    if not matched_nodes:
        return originals, details

    normalized_queries = []
    added_records = []
    relation_count = 0
    relationship_query = bool(RELATION_INTENT.search(question))
    for index, query in enumerate(originals):
        if not isinstance(query, str):
            normalized_queries.append(query)
            continue
        extras = []
        extra_ids = set()
        semantic_candidates = set()
        matched_here = []
        for n in matched_nodes:
            terms = (n["label"], *n.get("aliases_for_search", []))
            if any(_contains(query, term) for term in terms):
                matched_here.append(n)
                # 原问题/分词中出现英语或者不同中文译名时，补齐本地图谱的规范名。
                if not _contains(query, n["label"]) and n["id"] not in extra_ids:
                    extras.append(n["label"])
                    extra_ids.add(n["id"])
        if relationship_query:
            for n in matched_here:
                for e in relationships:
                    if e["source"] == n["id"]:
                        candidate_id = e["target"]
                    elif e["target"] == n["id"] and e["relation"] in {"IS_A", "HAS_PART"}:
                        candidate_id = e["source"]
                    else:
                        continue
                    if candidate_id not in nodes or candidate_id in extra_ids:
                        continue
                    new_name = nodes[candidate_id]["label"]
                    if new_name == n["label"] or _contains(query, new_name):
                        continue
                    extras.append(new_name)
                    extra_ids.add(candidate_id)
                    semantic_candidates.add(new_name)
                    if len(extras) >= MAX_ADDED_TERMS:
                        break
                if len(extras) >= MAX_ADDED_TERMS:
                    break
        # 只加入少量真实术语，不附带图谱文字作为生成证据。
        extras = extras[:MAX_ADDED_TERMS]
        revised = query
        if extras:
            suffix = " " + " ".join(extras)
            if len(query) + len(suffix) <= MAX_QUERY_CHARACTERS:
                revised += suffix
                relation_count += sum(term in semantic_candidates for term in extras)
                added_records.append({"subquery_index": index + 1, "terms": extras})
        normalized_queries.append(revised)
    details["added_terms"] = added_records
    details["status"] = "applied" if added_records else "no_match"
    details["graph_semantic_edges_used"] = relation_count
    return normalized_queries, details


def default_paths(project_folder):
    root = Path(project_folder)
    default_source = root / "graph_data" / "sources" / "神经网络与深度学习.md"
    return (
        os.getenv("DOUXIAN_GRAPH_FILE") or str(root / "graph_data" / "knowledge_graph_v27.json"),
        os.getenv("DOUXIAN_GRAPH_SOURCE") or str(default_source),
    )
