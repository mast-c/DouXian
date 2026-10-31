# -*- coding: utf-8 -*-
"""V1.15 医疗多药品、多字段问答：按真实证据组织，杜绝凭生成模型遗漏栏目。

只读取已返回的 Neo4j/Chroma 证据；不访问数据库、不推断临床疗效。
适用：开放式疾病→多药品清单并询问不良反应/注意事项等栏目。
一般单药问答仍走原来的 Evidence Guard + Qwen。
"""
from __future__ import annotations

import html
import re
from typing import Any, Iterable

CANDIDATE = "unverified_drug_disease_candidate"
SECTION_TYPES = {"unverified_drug_source_section", "json_neo4j_text_verified"}
FIELDS = {
    "indication": "适应症",
    "adverse_reactions": "不良反应",
    "warnings": "注意事项",
    "contraindications": "禁忌",
    "drug_interactions": "药物相互作用",
}
UNKNOWN_STATEMENTS = {"尚不明确", "尚不清楚", "不详", "未明确", "暂无资料"}


def _safe_text(value: Any) -> str:
    """来源文本是数据，不是 Markdown/HTML 指令，也不能伪造 [99] 引用。"""
    text = " ".join(str(value or "").split())
    text = html.escape(text, quote=False)
    text = (text.replace("[", "&#91;").replace("]", "&#93;")
            .replace("|", "&#124;").replace("`", "&#96;"))
    return text.replace("*", "&#42;").replace("_", "&#95;")


def limit_quote(text: str, *, max_chars: int = 155) -> str:
    """显示明确标注的原文节选；长片段尽量在语句边界停止。"""
    text = _safe_text(text)
    if len(text) <= max_chars:
        return text
    breaks = [text.rfind(sep, 0, max_chars) for sep in ("。", "；", ";", "！", "？")]
    boundary = max(breaks)
    if boundary >= max_chars // 2:
        return text[:boundary + 1] + "…（节选）"
    return text[:max_chars].rstrip() + "…（节选）"


def original_section_text(item: dict[str, Any]) -> str:
    """只从本轮结构化证据的原文区提取，避免混入状态、系统提示和来源声明。"""
    if not isinstance(item, dict):
        return ""
    content = str(item.get("content") or "")
    if "证据原文：" in content:
        body = content.split("证据原文：", 1)[1]
    elif "原文：" in content:
        body = content.split("原文：", 1)[1]
    else:
        return ""
    for stop in ("\n重要：", "\n资料状态：", "\n来源核验状态："):
        body = body.split(stop, 1)[0]
    return body.strip()


def scope_graph_discovered_drugs(
    items: Iterable[dict[str, Any]],
    drug_names: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """疾病-only 问题对 Chroma 进行图谱候选药名约束，排除共词但无关药。

    与用户显式药名 Entity Gate 区分：这些名称来自图谱候选，
    而不是声称用户亲自说出了四个药名。
    """
    allowed_list = list(dict.fromkeys(
        name.strip() for name in drug_names
        if isinstance(name, str) and name.strip()
    ))[:4]
    if not allowed_list:
        return list(items), {
            "applied": False, "retrieved_drugs": [],
            "removed_other_drugs": 0, "removed_unattributed": 0,
        }

    allowed = set(allowed_list)
    kept: list[dict[str, Any]] = []
    other = unattributed = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        drug = str(item.get("drug_name") or "").strip()
        if drug in allowed:
            kept.append({**item, "id": len(kept) + 1})
        elif drug:
            other += 1
        else:
            # 没有药品身份的检索内容不能作为某特定药品的不良反应证据。
            unattributed += 1

    return kept, {
        "applied": True,
        "retrieved_drugs": allowed_list,
        "removed_other_drugs": other,
        "removed_unattributed": unattributed,
        "kept": len(kept),
    }


def _field_evidence(items: list[dict[str, Any]], drug: str, field: str
                    ) -> dict[str, Any] | None:
    matched = [
        item for item in items
        if item.get("drug_name") == drug
        and item.get("section_field") == field
        and item.get("evidence_type") in SECTION_TYPES
        and original_section_text(item)
    ]
    if not matched:
        return None
    # 同一栏目有多个来源时，优先引用通过 Neo4j 父级回溯的 Chroma 原文，
    # 不会把多条不同产品来源假装已经交叉医学验证。
    matched.sort(key=lambda item: (
        1 if item.get("parent_verified") else 0,
        1 if item.get("evidence_type") == "json_neo4j_text_verified" else 0,
    ), reverse=True)
    return matched[0]


def _candidate(items: list[dict[str, Any]], drug: str) -> dict[str, Any] | None:
    for item in items:
        if item.get("drug_name") == drug and item.get("evidence_type") == CANDIDATE:
            return item
    return None


def _source_quote(item: dict[str, Any], *, max_chars: int) -> str:
    return limit_quote(original_section_text(item), max_chars=max_chars)


def build_multi_drug_source_answer(
    question: str,
    evidence: list[dict[str, Any]],
    selected_drugs: list[str],
    requested_fields: list[str],
) -> tuple[str, dict[str, Any]]:
    """只从当前证据逐药组织引用；不生成医学推荐或虚构缺失栏目。

    返回 Markdown 答案和逐药覆盖状态。参考资料链接仍交由原有
    render_kb_references() 从实际引用编号生成。
    """
    names = list(dict.fromkeys(
        n.strip() for n in selected_drugs
        if isinstance(n, str) and n.strip()
    ))[:4]
    fields = [f for f in dict.fromkeys(requested_fields) if f in FIELDS and f != "indication"]
    if not names:
        return "", {"listed_drugs": [], "supported_cells": 0, "missing_cells": 0}

    candidates = [_candidate(evidence, name) for name in names]
    disease_terms = {
        str(item.get("disease_term") or "").strip()
        for item in candidates if item and item.get("disease_term")
    }
    disease_label = f"“{_safe_text(next(iter(disease_terms)))}”" if len(disease_terms) == 1 else "所问疾病"
    lines = [
        f"**本轮资料中与{disease_label}有关的药品提及候选：{len(names)} 种。**",
        (
            "以下按药品逐项列出数据库中的疾病提及和说明书栏目原文。"
            if fields else "以下按药品逐项列出本轮检索到的疾病提及候选原文。"
        )
        + "**这不是完整的治疗药物清单，也不代表已经确认的获批适应症或处方推荐。**",
    ]
    covered = missing = 0
    by_drug: list[dict[str, Any]] = []

    for drug in names:
        name = _safe_text(drug)
        candidate = _candidate(evidence, drug)
        lines.append(f"\n### {name}")
        if candidate and candidate.get("id"):
            quote = limit_quote(candidate.get("evidence_quote"), max_chars=115)
            lines.append(f"**疾病关联原文（待人工审核）**：{quote} [{candidate['id']}]")
        else:
            lines.append("**疾病关联**：本轮未找到可引用的疾病提及原文，不能据此判断治疗关系。")

        found_fields: list[str] = []
        missing_fields: list[str] = []
        for field in fields:
            label = FIELDS[field]
            item = _field_evidence(evidence, drug, field)
            if not item or not item.get("id"):
                missing += 1
                missing_fields.append(field)
                lines.append(
                    f"- **{label}**：本轮没有获得该药对应栏目的可引用原文；"
                    "不能据此认定该药没有相关风险或限制。"
                )
                continue
            covered += 1
            found_fields.append(field)
            source = original_section_text(item)
            if source.strip().rstrip("。；;：: ") in UNKNOWN_STATEMENTS:
                lines.append(
                    f"- **{label}**：资料原文为“{_safe_text(source)}”；"
                    f"这不等于已确认没有相关风险。[{item['id']}]"
                )
            else:
                lines.append(
                    f"- **{label}（原文节选）**：{_source_quote(item, max_chars=170)} "
                    f"[{item['id']}]"
                )
        by_drug.append({
            "drug_name": drug, "candidate_present": candidate is not None,
            "found_fields": found_fields, "missing_fields": missing_fields,
        })

    lines.append(
        "\n*来源说明：以上为本地第三方资料的原文提及与栏目摘录，"
        "仍需结合实际制剂、病因及最新版权威说明书核对；"
        "具体用药由医师或药师判断。长原文请展开对应编号的证据卡。*"
    )
    return "\n\n".join(lines), {
        "listed_drugs": names,
        "supported_cells": covered,
        "missing_cells": missing,
        "by_drug": by_drug,
        "mode": "deterministic_source_excerpts_no_llm",
    }
