"""GraphRAG 候选关系的阅读友好型展示（纯函数，无数据库/LLM 依赖）。

保留安全边界：仅作为原文提及；绝不将候选节点升级为治疗/预防关系。
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit
from typing import Any


_STATUS = {
    "pending_human_review": "待人工复核",
    "not_validated": "尚未完成医学验证",
    "third_party_unverified": "第三方来源，未核验",
    "unknown": "状态未知",
}


def human_status(value: Any) -> str:
    """将常见审核值变成简短中文；对其他值采用保守状态文字。"""
    key = str(value or "unknown").strip()
    return _STATUS.get(key, "状态待核对")


def md_inline(value: Any, *, max_chars: int = 240) -> str:
    """普通数据转为 Markdown 文本，不允许插入图片/标题/链接结构。"""
    text = " ".join(str(value or "").split())[:max_chars]
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r'([\\`*_{}\[\]()#+.!|>~\-])', r'\\\1', text)


def _source_link(candidate: dict[str, Any]) -> str:
    """仅在已有安全 http(s) 来源时生成短链接；不渲染裸 URL。"""
    raw = candidate.get("source_url") or candidate.get("source")
    if not isinstance(raw, str) or len(raw) > 2048:
        return "原始来源链接未提供"
    url = raw.strip()
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "原始来源链接未提供"
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        return "原始来源链接未提供"
    # 对 Markdown 目标中的 () 和空格进行 URL 编码；业务数据不进入 HTML。
    from urllib.parse import quote
    safe_url = quote(url, safe=":/?#[]@!$&'*+,;=%-._~")
    return f"[查看原始药品资料]({safe_url})"


def _quote_kind(item: dict[str, Any]) -> str:
    """仅根据原文的明确字面措辞作极有限的解释，不做临床推断。"""
    term = "".join(str(item.get("disease_term") or "").split())
    quote = "".join(str(item.get("evidence_quote") or "").split())
    if term and (term + "危险" in quote or term + "风险" in quote):
        return "原文这里谈到的是该疾病的**风险条件**，而不是直接描述该药治疗这一疾病。"
    return "这是药品资料中的疾病**提及记录**，不能仅凭名称共现就推断治疗、预防或禁忌关系。"


def build_candidate_answer(items: list[dict[str, Any]]) -> str:
    """生成可流式显示和持久化的 Markdown 答案，不重复输出调试状态和长 URL。"""
    if not items:
        return "当前没有找到能够展示的药品—疾病候选证据。"
    clean_items = [item for item in items if isinstance(item, dict)]
    if not clean_items:
        return "当前没有找到能够展示的药品—疾病候选证据。"

    names = {(str(c.get("drug_name") or ""), str(c.get("disease_term") or "")) for c in clean_items}
    single_pair = len(names) == 1
    if single_pair:
        drug, disease = next(iter(names))
        lead = (
            f"目前找到 **{md_inline(drug)}** 与 **{md_inline(disease)}** 的原文提及记录。"
            "但**这不是已经确认的治疗关系**，也不能据此判断该药具有对应的获批适应症。"
        )
    else:
        lead = (
            f"本次找到 **{len(clean_items)} 条药品—疾病提及候选**。"
            "这些记录用于定位原文，**不能直接当作治疗或预防关系**。"
        )

    lines = [lead, "", "**原文依据**", ""]
    for i, item in enumerate(clean_items[:5], 1):
        drug = md_inline(item.get("drug_name") or "未知药品")
        disease = md_inline(item.get("disease_term") or "未命名疾病词条")
        quote = md_inline(item.get("evidence_quote") or "没有提供原文", max_chars=260)
        suffix = "（节选）" if len(" ".join(str(item.get("evidence_quote") or "").split())) > 260 else ""
        if len(clean_items) > 1:
            lines.append(f"**证据 {i} · {drug} / {disease}**")
            lines.append("")
        lines.append(f"> {quote}{suffix}")
        lines.append("")
        lines.append(_quote_kind(item))
        lines.append("")
        lines.append(f"*资料来源：{_source_link(item)}*  ·  *{human_status(item.get('candidate_status'))}*  ·  *{human_status(item.get('clinical_status'))}*")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*说明：这些材料来自尚未完成权威核验的药品数据。若需确认正式适应症，请核对具体药品的最新版核准说明书。*")
    return "\n".join(lines).strip()
