"""Self-RAG V1.5：证据不足时，最多一次定向补检 + 再次校验。

只服务于知识库模式；不访问互联网，不更新 Chroma，不重新 Embedding。
原有证据 ID 不改变；仅在得到新的 Chunk 时再进行一次充分性审核。
"""

import json
import re
from typing import Any, Callable


REWRITE_PROMPT = """你是本地 RAG 系统的查询改写器，只负责定位缺失的知识库资料。

给定用户原始问题、审核后证据不足的方面，以及已经尝试过的查询，
生成最多两条互补的、新的检索问题，专门寻找尚未覆盖的知识。

规则：
1. 保留原始用户询问的对象、限制条件、时间和来源限制。
2. 不重复之前尝试过的查询；优先改换检索角度或同义表达。
3. 尽量询问实际功能、前提、输入与输出、相关实例或相邻概念间的关系，
   但不要凭空补充具体的技术名称、年代、性能数据或结论。
4. 你生成的是【查询】，不是答案；不假定某事实必定存在。
5. 不得扩展用户原始问题范围；不得把无关背景查询变成必答知识。
6. 只能输出 JSON：{"queries":["检索问题1","检索问题2"]}。
"""


def _plain_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def _norm_query(query: str) -> str:
    return re.sub(r"\s+", "", query).rstrip("？！?!.。")


def _rewrite_queries(
    model: Any,
    question: str,
    missing_aspects: list[str],
    original_queries: list[str],
    max_queries: int = 2,
) -> list[str]:
    """调用一次 Qwen 生成定向补检查询；解析失败时返回 []。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    payload = json.dumps({
        "original_question": question,
        "missing_aspects": missing_aspects,
        "previous_queries": original_queries,
    }, ensure_ascii=False)

    response = model.bind(temperature=0).invoke([
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=payload),
    ])
    raw = _plain_content(response.content)
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b < a:
        return []

    try:
        parsed = json.loads(raw[a:b + 1])
    except (TypeError, ValueError):
        return []
    items = parsed.get("queries") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return []

    seen = {_norm_query(q) for q in original_queries if isinstance(q, str)}
    seen.add(_norm_query(question))
    result = []
    for item in items:
        if not isinstance(item, str):
            continue
        value = " ".join(item.split()).strip()
        key = _norm_query(value)
        if 5 <= len(value) <= 180 and key not in seen:
            result.append(value)
            seen.add(key)
        if len(result) >= max_queries:
            break
    return result


def reassess_fixed_aspects(
    *,
    model: Any,
    question: str,
    excerpts: list[dict],
    retrieval_queries: list[str] | None,
    original_coverage: dict,
) -> dict:
    """二次审核时固定首轮各子问题的文字和顺序，防止漏评。"""
    from evidence_guard import (
        GUARD_SYSTEM_PROMPT,
        validate_judge_output,
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    required = [
        item["question"]
        for item in original_coverage.get("aspects", [])
        if isinstance(item, dict) and isinstance(item.get("question"), str)
    ]
    if not required:
        return {"decision": "error", "message": "首次审核没有有效的待回答方面"}

    payload = json.dumps({
        "original_question": question,
        "fixed_aspects": required,
        "retrieval_queries_for_context_only": (retrieval_queries or [])[:8],
        "knowledge_evidence": excerpts,
    }, ensure_ascii=False)

    fixed_prompt = (
        GUARD_SYSTEM_PROMPT
        + "\n额外强制要求：本轮 fixed_aspects 已由第一次审核确认，"
        "必须逐字、按顺序、逐项输出它们，"
        "不得更改 question 字段，不得增删合并拆分。"
        "仅对这些固定方面重新判断是否有充分证据。"
    )

    response = model.bind(temperature=0).invoke([
        SystemMessage(content=fixed_prompt),
        HumanMessage(content=payload),
    ])
    raw = _plain_content(response.content)
    return validate_judge_output(raw, excerpts)


def retry_insufficient_evidence(
    *,
    model: Any,
    question: str,
    excerpts: list[dict],
    coverage: dict,
    original_queries: list[str] | None,
    search_fn: Callable[[str], list[Any]],
    judge_fn: Callable[..., dict],
    rewrite_fn: Callable[[str, list[str], list[str]], list[str]] | None = None,
    max_queries: int = 2,
    max_new_chunks: int = 3,
    max_total_chunks: int = 8,
) -> dict:
    """一次补检，不存在递归或无限循环。

    返回：
      coverage: 最终审核结果
      excerpts: 对应于最终审核的证据列表（原 ID 稳定）
      retry_queries: 本轮实际尝试的补检查询
      new_chunks: 新增的 Chunk 数量
      status: skipped / no_queries / no_new_evidence /
              retry_error / judge_error / unchanged / improved

    注意：二次审核只有覆盖范围真正改善时才替换原结论；
    出错或变差均保留第一次审核结果。
    """
    current = list(excerpts)
    default = {
        "coverage": coverage,
        "excerpts": current,
        "retry_queries": [],
        "new_chunks": 0,
        "status": "skipped",
    }

    if not isinstance(coverage, dict) or coverage.get("decision") not in ("partial", "refuse"):
        return default

    if not callable(search_fn) or not callable(judge_fn):
        return {**default, "status": "retry_error"}

    missing = coverage.get("refused_questions") or [
        aspect.get("question") for aspect in coverage.get("aspects", [])
        if isinstance(aspect, dict) and aspect.get("status") == "insufficient"
    ]
    missing = [x for x in missing if isinstance(x, str) and x.strip()]
    if not missing:
        return {**default, "status": "no_queries"}

    previous = [
        q for q in (original_queries or [])
        if isinstance(q, str) and q.strip()
    ]
    try:
        proposed = (
            rewrite_fn(question, missing, previous)
            if rewrite_fn is not None
            else _rewrite_queries(model, question, missing, previous, max_queries)
        )
    except Exception as exc:
        print(f"[Self-RAG] 查询改写失败：{type(exc).__name__}: {exc}")
        return {**default, "status": "retry_error"}

    if not isinstance(proposed, list):
        return {**default, "status": "no_queries"}

    tried = {_norm_query(question), *(_norm_query(q) for q in previous)}
    queries = []
    for raw in proposed:
        if not isinstance(raw, str):
            continue
        query = " ".join(raw.split()).strip()
        key = _norm_query(query)
        if 5 <= len(query) <= 180 and key not in tried:
            queries.append(query)
            tried.add(key)
        if len(queries) >= max_queries:
            break

    if not queries:
        return {**default, "status": "no_queries"}

    existing = {
        (str(item.get("source", "")), str(item.get("content", "")))
        for item in current if isinstance(item, dict)
    }
    capacity = max(0, min(max_new_chunks, max_total_chunks - len(current)))
    if capacity == 0:
        return {**default, "status": "no_new_evidence"}

    # 各查询分别召回两个候选；交替选取，避免第一条查询占满补检名额。
    groups = []
    attempted = []
    for query in queries:
        attempted.append(query)
        try:
            docs = search_fn(query)
            groups.append(docs if isinstance(docs, list) else [])
        except Exception as exc:
            print(f"[Self-RAG] 补检失败：{query[:80]} ({type(exc).__name__}: {exc})")
            groups.append([])

    added = []
    for rank in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if rank >= len(group):
                continue
            doc = group[rank]
            content = getattr(doc, "page_content", None)
            metadata = getattr(doc, "metadata", None)
            if not isinstance(content, str) or not content.strip():
                continue
            if not isinstance(metadata, dict):
                metadata = {}
            source = str(metadata.get("source") or "未知文件")
            dedup_key = (source, content)
            if dedup_key in existing:
                continue
            existing.add(dedup_key)
            added.append({
                "id": len(current) + len(added) + 1,
                "source": source,
                "section_path": str(metadata.get("section_path") or "未标注章节"),
                "chunk_id": str(metadata.get("chunk_id") or ""),
                "content": content,
            })
            if len(added) >= capacity:
                break
        if len(added) >= capacity:
            break

    if not added:
        return {
            **default, "retry_queries": attempted,
            "status": "no_new_evidence",
        }

    augmented = current + added
    try:
        revised = judge_fn(
            model=model,
            question=question,
            excerpts=augmented,
            retrieval_queries=previous + attempted,
        )
    except Exception as exc:
        print(f"[Self-RAG] 二次审核失败：{type(exc).__name__}: {exc}")
        return {
            **default, "retry_queries": attempted,
            "new_chunks": len(added), "status": "judge_error",
        }

    if not isinstance(revised, dict) or revised.get("decision") == "error":
        return {
            **default, "retry_queries": attempted,
            "new_chunks": len(added), "status": "judge_error",
        }

    old_accepted = len(coverage.get("answer_questions") or [])
    new_accepted = len(revised.get("answer_questions") or [])
    old_aspects = len(coverage.get("aspects") or [])
    new_aspects = len(revised.get("aspects") or [])

    # 二次审核不能改写、遗漏待回答方面，也不能把原本通过的方面降级。
    original_items = coverage.get("aspects") or []
    revised_items = revised.get("aspects") or []
    same_questions = (
        new_aspects == old_aspects
        and all(
            _norm_query(old.get("question", ""))
            == _norm_query(new.get("question", ""))
            for old, new in zip(original_items, revised_items)
            if isinstance(old, dict) and isinstance(new, dict)
        )
        and all(isinstance(x, dict) for x in original_items + revised_items)
    )
    preserves_old = same_questions and all(
        previous.get("status") == new.get("status")
        for previous, new in zip(original_items, revised_items)
        if previous.get("status") in ("supported", "refuted")
    )

    improved = (
        same_questions
        and preserves_old
        and new_accepted > old_accepted
        and revised.get("decision") in ("answer", "partial")
    )
    if improved:
        return {
            "coverage": revised,
            "excerpts": augmented,
            "retry_queries": attempted,
            "new_chunks": len(added),
            "status": "improved",
        }

    return {
        **default,
        "retry_queries": attempted,
        "new_chunks": len(added),
        "status": "unchanged",
    }
