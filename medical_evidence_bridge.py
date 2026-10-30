# -*- coding: utf-8 -*-
"""医疗 Hybrid 检索结果的章节路由与 Neo4j 原文校验。

不更改 BM25、RRF 和 Reranker 的算法；负责在模型看到证据前核对
Chroma 中 data_origin=json_matched_to_neo4j_section 的每条记录。

精确回溯只能证明技术上的原文一致，绝不代表临床或来源权威审核。
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Iterable

_NEW_JSON_ORIGIN = "json_matched_to_neo4j_section"

_MEDICAL_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"适应症|适应证|适用于|治疗什么|治什么|能治|预防|"
     r"用于什么病|哪些疾病|什么疾病|什么病|与.{0,35}(?:疾病|病症|症状|"
     r"动脉粥样硬化).{0,20}(?:关系|关联)|"
     r"关联|有何关系|有什么关系|有何关联|有什么关联|疗效",
     ("indication",)),
    (r"禁忌|禁用|不宜|不能用|过敏", ("contraindications",)),
    (r"慎用|注意事项|警告|监测|风险提示", ("warnings",)),
    (r"不良反应|副作用|药物反应|不适反应", ("adverse_reactions",)),
    (r"相互作用|联用|合用|配伍|与其他药|一起吃", ("drug_interactions", "warnings")),
    (r"用法|用量|剂量|怎么服|服药次数|给药", ("dosage",)),
    (r"成分|活性物质|组成", ("composition",)),
)


def expected_section_fields(question: str) -> frozenset[str] | None:
    """确定用户明显指向的药品说明书栏目。

    返回 None 表示不做栏目筛选；不会把预防、风险条件
    自动转换成已确认治疗关系。
    """
    text = str(question or "").strip()
    if not text:
        return None
    result: set[str] = set()
    for pattern, fields in _MEDICAL_KEYWORDS:
        if re.search(pattern, text, flags=re.I):
            result.update(fields)
    return frozenset(result) if result else None


def _source_matches(meta: dict[str, Any], row: dict[str, Any]) -> bool:
    return (
        row.get("drug_name") == meta.get("drug_name")
        and str(row.get("field") or "").lower()
        == str(meta.get("section_field") or "").lower()
        and row.get("source_url") == meta.get("source_url")
    )


def verify_json_chunk(
    metadata: dict[str, Any],
    content: str,
    parent: dict[str, Any] | None,
) -> tuple[bool, str]:
    """验证 id 关联的父级 Section 内容和原文坐标（默认失败封闭）。"""
    if not parent:
        return False, "parent_missing"
    if not isinstance(metadata, dict) or not isinstance(content, str):
        return False, "invalid_input"
    if not _source_matches(metadata, parent):
        return False, "source_mismatch"
    full = parent.get("full_section")
    if not isinstance(full, str) or not full:
        return False, "parent_text_missing"
    try:
        start = int(metadata["quote_start"])
        end = int(metadata["quote_end"])
    except (KeyError, ValueError, TypeError):
        return False, "invalid_offsets"
    if not (0 <= start < end <= len(full)):
        return False, "offset_out_of_range"
    observed_sha = hashlib.sha256(full.encode("utf-8")).hexdigest()
    if observed_sha != metadata.get("section_sha256"):
        return False, "parent_changed"
    parts = content.split("\n证据原文：", 1)
    if len(parts) != 2 or parts[1] != full[start:end]:
        return False, "exact_quote_mismatch"
    if not all(isinstance(metadata.get(field), str) and metadata[field]
               for field in ("drug_id", "record_id", "section_id", "source_url")):
        return False, "missing_binding"
    return True, "text_verified"


def prepare_medical_documents(
    documents: Iterable[Any],
    question: str,
    fetch_parent: Callable[[dict[str, Any]], dict[str, Any] | None],
    *,
    limit: int = 3,
) -> tuple[list[Any], dict[str, Any]]:
    """通过题意字段筛选、原文溯源校验，输出可进入证据链的 Document。

    对本次 JSON 精确绑定的文档采取 fail-closed；普通旧库文档原样放行，
    但绝不将它们标注为“已回溯核验”。每条调用只读父级 Section。
    """
    fields = expected_section_fields(question)
    selected: list[Any] = []
    reports: dict[str, Any] = {
        "considered": 0, "skipped_section": 0, "failed_provenance": 0,
        "verified": 0, "legacy_unverified": 0, "reasons": {},
        "section_gate": sorted(fields) if fields else [],
    }
    for doc in documents:
        if len(selected) >= limit:
            break
        meta = getattr(doc, "metadata", None)
        content = getattr(doc, "page_content", None)
        if not isinstance(meta, dict) or not isinstance(content, str) or not content.strip():
            continue
        reports["considered"] += 1

        field = str(meta.get("section_field") or "").lower()
        if fields and field and field not in fields:
            reports["skipped_section"] += 1
            continue

        if meta.get("data_origin") == _NEW_JSON_ORIGIN:
            try:
                parent = fetch_parent(meta)
                ok, reason = verify_json_chunk(meta, content, parent)
            except Exception as exc:
                ok, reason = False, f"lookup_{type(exc).__name__}"
            if not ok:
                reports["failed_provenance"] += 1
                reports["reasons"][reason] = reports["reasons"].get(reason, 0) + 1
                continue

            # 不将整个旧版 Section 塞给生成模型：其部分数据包含多个来源字段；
            # 精确 Quote 才是当前已核验的单条证据。保留原文及元数据。
            enriched = {
                **meta, "evidence_type": "json_neo4j_text_verified",
                "provenance_verified": True,
                "source_status": str(meta.get("source_status") or "third_party_unverified"),
            }
            selected.append(type(doc)(page_content=content, metadata=enriched))
            reports["verified"] += 1
        else:
            selected.append(doc)
            reports["legacy_unverified"] += 1

    reports["selected"] = len(selected)
    return selected, reports


def collapse_duplicate_parent_sections(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """若 Chroma 片段已回溯到同一 Section，则隐藏重复的 Neo4j 原文章节节点。

    候选 DiseaseCandidate 和精确原文是两种不同证据，必须保留。
    """
    verified_parents = {
        (str(item.get("record_id")), str(item.get("section_id")))
        for item in items
        if item.get("evidence_type") == "json_neo4j_text_verified"
        and item.get("record_id") and item.get("section_id")
    }
    result = []
    for item in items:
        key = (str(item.get("record_id")), str(item.get("section_id")))
        if item.get("evidence_type") == "unverified_drug_source_section" and key in verified_parents:
            continue
        result.append({**item, "id": len(result) + 1})
    return result
