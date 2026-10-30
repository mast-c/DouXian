# -*- coding: utf-8 -*-
"""药品实体约束：仅对用户明确提到的药名执行精确匹配。

放在 Neo4j + Chroma 多路证据合并后；不替代 BM25/RRF/Reranker。
- 药品+疾病：不让其他药品因疾病词共现混入引用
- 仅询问疾病：不推断唯一药品、不屏蔽多药品结果
- 多药品对比：只保留用户明确点名的所有药品
- 资料没有药品 Metadata 时：须在来源标题或正文包含精确药名
不会用“包含同一个疾病”证明药品间的治疗关系。
"""
from __future__ import annotations

from typing import Any, Iterable


def _names_in_question(question: str, names: Iterable[str]) -> tuple[str, ...]:
    """以长名称优先处理嵌套药名，不将简称误当作另一种药。"""
    raw = str(question or "").strip()
    if not raw:
        return ()
    normalized = {str(x).strip() for x in names if isinstance(x, str) and len(x.strip()) >= 2}
    spans: list[tuple[int, int, str]] = []
    for name in sorted(normalized, key=lambda n: (-len(n), n)):
        index = raw.find(name)
        while index >= 0:
            spans.append((index, index + len(name), name))
            index = raw.find(name, index + 1)
    if not spans:
        return ()
    chosen: list[tuple[int, int, str]] = []
    for first, last, name in sorted(spans, key=lambda x: (-(x[1] - x[0]), x[0])):
        if all(last <= other_start or first >= other_end
               for other_start, other_end, _ in chosen):
            chosen.append((first, last, name))
    return tuple(sorted({name for _, _, name in chosen}, key=lambda n: (-len(n), n)))


def query_drug_names(question: str, evidence: Iterable[dict[str, Any]],
                     additional_names: Iterable[str] = ()) -> tuple[str, ...]:
    """仅以显式药名（在问题中完整出现）建立作用域。"""
    names = list(additional_names)
    for item in evidence:
        if isinstance(item, dict):
            name = item.get("drug_name")
            if isinstance(name, str) and name.strip():
                names.append(name)
    return _names_in_question(question, names)


def _text_matches_named_drug(item: dict[str, Any], names: set[str]) -> bool:
    drug_name = str(item.get("drug_name") or "").strip()
    if drug_name:
        return drug_name in names
    # 对旧 Chroma 文件名、正文原样保守匹配。无相关药品名则不能作为
    # “某个具体药品”的证据；没有元数据的文档也不会伪装为原文核验。
    searchable = (
        str(item.get("source") or "") + "\n"
        + str(item.get("section_path") or "") + "\n"
        + str(item.get("content") or "")
    )
    return any(name in searchable for name in names)


def filter_evidence_for_query(
    question: str,
    evidence: list[dict[str, Any]],
    *,
    known_drugs: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """返回保留的证据及报告。编号由该函数重新从 1 顺序分配。"""
    scope = query_drug_names(question, evidence, known_drugs)
    if not scope:
        items = [{**item, "id": i + 1} for i, item in enumerate(evidence)]
        return items, {
            "mode": "unrestricted", "explicit_drugs": [],
            "removed_other_drugs": 0, "removed_unattributed": 0,
            "kept": len(items),
        }

    allowed = set(scope)
    selected = []
    other = missing = 0
    for item in evidence:
        if _text_matches_named_drug(item, allowed):
            selected.append({**item, "id": len(selected) + 1})
        elif str(item.get("drug_name") or "").strip():
            other += 1
        else:
            missing += 1
    return selected, {
        "mode": "explicit_drugs", "explicit_drugs": list(scope),
        "removed_other_drugs": other,
        "removed_unattributed": missing,
        "kept": len(selected),
    }


def filter_documents_for_drugs(
    documents: Iterable[Any], named_drugs: Iterable[str]
) -> list[Any]:
    """用于 Self-RAG 补检，同样拒绝跨药品的 Document。"""
    names = set(named_drugs)
    if not names:
        return list(documents)
    result = []
    for doc in documents:
        meta = getattr(doc, "metadata", None)
        meta = meta if isinstance(meta, dict) else {}
        evidence = {
            "drug_name": meta.get("drug_name"),
            "source": meta.get("source"),
            "section_path": meta.get("section_path"),
            "content": getattr(doc, "page_content", ""),
        }
        if _text_matches_named_drug(evidence, names):
            result.append(doc)
    return result
