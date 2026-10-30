# -*- coding: utf-8 -*-
"""面向 drug_db_v2_cleaned.json 的结构感知 Chunk；不调用模型、不写数据库。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

FIELD_LABELS = {
    "indication": "适应症",
    "contraindications": "禁忌",
    "adverse_reactions": "不良反应",
    "warnings": "注意事项",
    "drug_interactions": "药物相互作用",
    "dosage": "用法用量",
    "composition": "药品成分",
}
DEFAULT_FIELDS = ("indication", "contraindications", "adverse_reactions",
                  "warnings", "drug_interactions")

# 前缀只用于检索召回；原文片段本身保持逐字符不变。
def document_text(drug_name: str, field: str, quote: str) -> str:
    return (
        f"药品：{drug_name}\n栏目：{FIELD_LABELS[field]}\n"
        "资料状态：第三方来源尚未完成医学权威核验\n"
        f"证据原文：{quote}"
    )


@dataclass(frozen=True)
class RawChunk:
    drug_name: str
    field: str
    item_index: int
    item_start: int
    item_end: int
    json_path: str
    quote: str
    forced_cut: bool
    # 来源 Section 中的准确坐标在联结成功后才生成。


def _boundaries(text: str) -> list[int]:
    cuts = {0, len(text)}
    for m in re.finditer(r"[。！？；\n]", text):
        cuts.add(m.end())
    # 仅对有明确空白边界的项目序号补充切分候选位置，
    # 不在 1.0、LDL-C 这种医学计量表达内部强制断开。
    for m in re.finditer(r"(?<=\s)(?:\d{1,2}[.．、]|[（(][一二三四五六七八九十0-9]{1,3}[）)])", text):
        cuts.add(m.start())
    return sorted(cuts)


def split_long_entry(text: str, *, max_chars: int, overlap: int) -> list[tuple[int, int, bool]]:
    """短 JSON 条目完整保留；长条目优先在句末或条目边界切分。

    返回相对原 JSON 字段条目内容的起止下标；绝不改写原文。
    第三个值表示是否被迫在非自然边界切分。
    """
    if not text:
        return []
    if not 100 <= max_chars <= 20000:
        raise ValueError("max_chars 必须在 100～20000 之间")
    if not 0 <= overlap < max_chars // 2:
        raise ValueError("overlap 应大于等于 0，且小于窗口的一半")
    if len(text) <= max_chars:
        return [(0, len(text), False)]

    cuts = _boundaries(text)
    valid_cuts = set(cuts)
    windows = []
    start = 0
    while start < len(text):
        max_end = min(len(text), start + max_chars)
        lower = start + min(100, max(1, (max_end-start)//2))
        possible_ends = [c for c in cuts if lower <= c <= max_end]
        end = max(possible_ends) if possible_ends else max_end
        if end <= start:
            end = max_end
        forced = end not in valid_cuts
        windows.append((start, end, forced))
        if end == len(text):
            break
        near = max(start + 1, end - overlap)
        # 尽量让重叠起点也位于语义边界；没有则保留字符重叠。
        possible_starts = [c for c in cuts if near <= c < end]
        next_start = min(possible_starts) if possible_starts else near
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return windows


def extract_raw_chunks(
    drug_name: str,
    drug: dict[str, Any],
    fields: tuple[str, ...],
    *,
    chunk_size: int,
    overlap: int,
) -> list[RawChunk]:
    """保持原始 JSON 字段及数组条目边界。"""
    if not isinstance(drug, dict):
        raise TypeError("药品数据必须为 JSON 对象")
    chunks: list[RawChunk] = []
    for field in fields:
        if field not in FIELD_LABELS:
            raise ValueError(f"不支持的字段：{field}")
        value = drug.get(field)
        if isinstance(value, str):
            entries = [(0, value)]
        elif isinstance(value, list):
            entries = [(i, item) for i, item in enumerate(value) if isinstance(item, str)]
        elif value in (None, ""):
            continue
        else:
            raise TypeError(f"{field} 的 JSON 类型不在允许范围内: {type(value).__name__}")

        for index, original in entries:
            # 不清洗原文，不删除标点；起止位置仍然对应原 JSON 字符串。
            if not original.strip():
                continue
            full_prefix = document_text(drug_name, field, "")
            max_body = chunk_size - len(full_prefix)
            if max_body < 100:
                raise ValueError("chunk_size 太小，无法保留原文和检索前缀")
            safe_overlap = min(overlap, max(0, max_body//3))
            for start, end, forced in split_long_entry(
                original, max_chars=max_body, overlap=safe_overlap
            ):
                if not original[start:end].strip():
                    continue
                path = field if isinstance(value, str) else f"{field}[{index}]"
                chunks.append(RawChunk(
                    drug_name=drug_name, field=field, item_index=index,
                    item_start=start, item_end=end,
                    json_path=path, quote=original[start:end], forced_cut=forced
                ))
    return chunks


def safe_url(value: Any) -> str:
    from urllib.parse import urlsplit
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if len(value) > 2048:
        return ""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    return value if parsed.scheme in ("https", "http") and bool(parsed.netloc) else ""


def align_to_neo4j(
    raw_chunks: list[RawChunk],
    rows: list[dict[str, Any]],
    original_source_url: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """给每个 JSON 原文片段找到唯一的 Neo4j Section 和精确字符位置。

    多匹配/零匹配/来源 URL 不一致均拒绝自动连接，避免证据误挂靠。
    rows 来自现有 Drug -> SourceRecord -> Section 只读路径。
    """
    linked, errors = [], []
    expected_url = safe_url(original_source_url)
    if not expected_url:
        return [], ["JSON 中没有可验证的 HTTP(S) 来源 URL，禁止声称精确关联 Neo4j"]

    for chunk in raw_chunks:
        matches = []
        for row in rows:
            if row.get("drug_name") != chunk.drug_name:
                continue
            if str(row.get("field") or "").lower() != chunk.field:
                continue
            if safe_url(row.get("source_url")) != expected_url:
                continue
            if not all(str(row.get(name) or "") for name in ("drug_id", "record_id", "section_id")):
                continue
            section_text = row.get("section_text")
            if not isinstance(section_text, str):
                continue
            pos = section_text.find(chunk.quote)
            while pos >= 0:
                matches.append((row, pos))
                pos = section_text.find(chunk.quote, pos + 1)

        if len(matches) != 1:
            errors.append(
                f"{chunk.drug_name} / {chunk.json_path} / "
                f"片段[{chunk.item_start}:{chunk.item_end}] 匹配数={len(matches)}"
            )
            continue
        row, start = matches[0]
        end = start + len(chunk.quote)
        assert row["section_text"][start:end] == chunk.quote
        linked.append({
            "chunk": chunk,
            "drug_id": str(row["drug_id"]),
            "record_id": str(row["record_id"]),
            "section_id": str(row["section_id"]),
            "section_start": start,
            "section_end": end,
            "source_url": expected_url,
            "source_status": str(row.get("source_status") or "not_verified"),
            "section_hash": hashlib.sha256(
                row["section_text"].encode("utf-8")
            ).hexdigest(),
        })
    return linked, errors


def make_chroma_record(
    item: dict[str, Any], *, chunk_index: int, chunk_total: int,
    json_filename: str
) -> dict[str, Any]:
    """生成 Chroma 允许的扁平 Metadata；所有 ID 来自精确匹配的图谱。"""
    chunk: RawChunk = item["chunk"]
    key = "|".join((
        "medical-json-v2", item["drug_id"], item["record_id"], item["section_id"],
        chunk.json_path, str(chunk.item_start), str(chunk.item_end)
    ))
    chunk_id = "medical_v2_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:36]
    content = document_text(chunk.drug_name, chunk.field, chunk.quote)
    metadata = {
        "chunk_id": chunk_id,
        "doc_id": hashlib.sha256(
            f'{item["drug_id"]}|{item["record_id"]}'.encode("utf-8")
        ).hexdigest(),
        "source": f"药品资料：{chunk.drug_name}",
        "section_path": f"{chunk.drug_name} / {FIELD_LABELS[chunk.field]} / {chunk.json_path}",
        "source_url": item["source_url"],
        "source_status": item["source_status"],
        "data_origin": "json_matched_to_neo4j_section",
        "drug_name": chunk.drug_name,
        "drug_id": item["drug_id"],
        "record_id": item["record_id"],
        "section_id": item["section_id"],
        "section_field": chunk.field,
        "json_path": chunk.json_path,
        "json_filename": json_filename,
        "quote_start": item["section_start"],
        "quote_end": item["section_end"],
        "section_sha256": item["section_hash"],
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "forced_cut": chunk.forced_cut,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
    }
    return {"id": chunk_id, "text": content, "metadata": metadata}
