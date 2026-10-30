# -*- coding: utf-8 -*-
"""V1.7 Chroma 医疗 Chunk 完整性与 Neo4j 来源抽样验收（只读）。

用法：
  python medical_chroma_acceptance.py --expect-count 88 --sample-parent 12

默认只读取 Chroma 的 Collection/Metadata/正文，不调用云端、不写 Chroma/Neo4j。
--sample-parent N 时从现有 Neo4j 只读回查 N 条代表性 JSON Chunk。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ORIGIN = "json_matched_to_neo4j_section"
MANDATORY = (
    "chunk_id", "drug_id", "record_id", "section_id", "section_field",
    "drug_name", "source_url", "source_status", "json_path",
    "section_sha256", "content_sha256",
)
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_HTTP = {"http", "https"}


def inspect_chunk(chunk_id: str, content: str, metadata: dict[str, Any]) -> list[str]:
    """无需连接数据库的 Chroma 数据一致性检查。

    只对显式标记 JSON-Section 绑定的 Chunk 执行严格规则；
    历史普通文档不伪装为经过 Neo4j 核验的医学证据。
    """
    problems: list[str] = []
    if not isinstance(content, str) or not content.strip():
        return ["正文为空或类型错误"]
    if not isinstance(metadata, dict):
        return ["Metadata 不是字典"]
    if metadata.get("data_origin") != ORIGIN:
        return []
    for key in MANDATORY:
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            problems.append(f"Metadata 缺失或为空: {key}")
    if metadata.get("chunk_id") != chunk_id:
        problems.append("内部 chunk_id 与 Chroma ID 不一致")

    actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if actual_hash != metadata.get("content_sha256"):
        problems.append("Chunk 正文 SHA256 不一致")
    if not _HEX_64.fullmatch(str(metadata.get("section_sha256") or "")):
        problems.append("父级 Section SHA256 格式错误")

    body = content.split("\n证据原文：", 1)
    if len(body) != 2 or not body[1]:
        problems.append("未找到非空的证据原文")
    else:
        if "药品：" + str(metadata.get("drug_name") or "") + "\n" not in content:
            problems.append("检索前缀药品与 Metadata 不一致")
    try:
        start = int(metadata["quote_start"])
        end = int(metadata["quote_end"])
        if not 0 <= start < end:
            problems.append("原文起止位置不合法")
        if len(body) == 2 and end-start != len(body[1]):
            problems.append("原文位置长度与 Chunk 原文长度不符")
    except (ValueError, TypeError, KeyError):
        problems.append("原文 quote_start / quote_end 不是有效整数")

    url = metadata.get("source_url")
    try:
        parts = urlsplit(str(url or ""))
        if parts.scheme not in _HTTP or not parts.netloc:
            problems.append("来源 URL 不是 HTTP(S)")
    except ValueError:
        problems.append("来源 URL 无法解析")
    return problems


def audit_records(
    rows: list[tuple[str, str, dict[str, Any]]],
    *,
    parent_sample_size: int = 0,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], str]], list[str]]:
    """合并统计 Chroma 和新版 JSON Chunk；父级抽样尽可能覆盖不同药品。"""
    issues: list[str] = []
    counts: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    sources: set[str] = set()
    drug_set: set[str] = set()
    sections: set[tuple[str, str]] = set()
    ids: set[str] = set()
    samples: list[tuple[dict[str, Any], str]] = []
    sample_drugs: set[str] = set()
    sample_keys: set[tuple[str, str]] = set()

    for index, (chunk_id, content, meta) in enumerate(rows, 1):
        if chunk_id in ids:
            issues.append(f"记录 #{index}：Chroma ID 重复")
        ids.add(chunk_id)
        if not isinstance(meta, dict):
            issues.append(f"记录 #{index}：Metadata 格式异常")
            counts["invalid_metadata"] += 1
            continue
        origin = str(meta.get("data_origin") or "legacy_or_other")
        counts["all"] += 1
        if origin != ORIGIN:
            counts["legacy_or_other"] += 1
            # 旧 Chunk 不具备本轮精确来源映射，不应误标为核验通过。
            if not isinstance(content, str) or not content.strip():
                issues.append(f"记录 #{index}：旧文档正文为空")
            continue
        counts["json_matched"] += 1
        if bool(meta.get("forced_cut")):
            counts["forced_cut"] += 1
        drug = str(meta.get("drug_name") or "")
        field = str(meta.get("section_field") or "")
        field_key = (str(meta.get("record_id") or ""), str(meta.get("section_id") or ""))
        fields[field] += 1
        drug_set.add(drug)
        sections.add(field_key)
        sources.add(str(meta.get("source_url") or ""))
        errors = inspect_chunk(chunk_id, content, meta)
        if errors:
            counts["invalid_json"] += 1
            issues.extend([f"记录 #{index}（{drug or chunk_id}）：{e}" for e in errors])
            continue
        counts["valid_json"] += 1
        # 优先一药取一条、一父级取一条，避免 N 条全来自同一种药品。
        if len(samples) < parent_sample_size and (
            drug not in sample_drugs or field_key not in sample_keys
        ):
            samples.append((meta, content))
            sample_drugs.add(drug)
            sample_keys.add(field_key)

    result = {
        "scanned": counts["all"] + counts["invalid_metadata"],
        "json_structured_chunks": counts["json_matched"],
        "json_metadata_valid": counts["valid_json"],
        "json_metadata_invalid": counts["invalid_json"],
        "legacy_or_other": counts["legacy_or_other"],
        "forced_cuts": counts["forced_cut"],
        "drug_count": len(drug_set),
        "parent_section_count": len(sections),
        "independent_source_urls": len(sources),
        "fields": dict(fields.most_common()),
    }
    return result, samples, issues


def fetch_chroma() -> tuple[int, list[tuple[str, str, dict[str, Any]]], str]:
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    import config_data as config
    path = Path(config.persist_directory).expanduser().resolve()
    db = path / "chroma.sqlite3"
    if not db.is_file():
        raise FileNotFoundError(f"找不到 Chroma 数据库：{db}")
    import chromadb
    client = chromadb.PersistentClient(path=str(path))
    # get_collection，不使用 get_or_create_collection；若配置错误就直接报错。
    collection = client.get_collection(name=str(config.collection_name))
    total = int(collection.count())
    records: list[tuple[str, str, dict[str, Any]]] = []
    offset = 0
    while offset < total:
        data = collection.get(
            limit=min(200, total-offset),
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        if not ids:
            raise RuntimeError("Chroma 分页提前结束，请暂停并发写入后重试")
        if not len(ids) == len(documents) == len(metadatas):
            raise RuntimeError("Chroma 返回的 IDs、正文、元数据数量不一致")
        records.extend(zip(ids, documents, metadatas))
        offset += len(ids)
    if int(collection.count()) != total:
        raise RuntimeError("扫描期间 Chroma 记录总数发生变化，请停止并发入库后重试")
    return total, records, f"{path} / {config.collection_name}"


def verify_parents(samples: list[tuple[dict, str]]) -> tuple[int, list[str]]:
    if not samples:
        return 0, []
    from medical_evidence_bridge import verify_json_chunk
    from drug_graph_retriever import DrugGraphRetriever
    retriever = DrugGraphRetriever()
    checked = 0
    errors = []
    try:
        for meta, content in samples:
            checked += 1
            try:
                parent = retriever.fetch_parent_section(meta)
                ok, reason = verify_json_chunk(meta, content, parent)
            except Exception as exc:
                ok, reason = False, f"查询异常：{type(exc).__name__}"
            drug = str(meta.get("drug_name") or "")
            field = str(meta.get("section_field") or "")
            print(
                f"  [PARENT] {drug} / {field} "
                f"=> {'精确匹配' if ok else '未通过'} ({reason})"
            )
            if not ok:
                errors.append(f"{drug} / {field}: {reason}")
    finally:
        retriever.close()
    return checked, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Chroma JSON 医疗 Chunk 完整性与原文关联验收（只读）")
    parser.add_argument("--expect-count", type=int, default=None,
                        help="预期 Chroma 总 Chunk 数量，如本轮 88；变化时可不填写")
    parser.add_argument("--sample-parent", type=int, default=0,
                        help="从不同药品/Section 抽取 N 条，再读取 Neo4j 精确回溯")
    args = parser.parse_args()
    if args.expect_count is not None and args.expect_count < 0:
        parser.error("--expect-count 不能为负")
    if not 0 <= args.sample_parent <= 200:
        parser.error("--sample-parent 需在 0~200 之间")
    total, rows, collection = fetch_chroma()
    print("=== 医疗 Chroma 只读验收 ===")
    print("Collection:", collection)
    print("Chroma 当前 Chunk:", total)
    report, samples, issues = audit_records(rows, parent_sample_size=args.sample_parent)
    if args.expect_count is not None and args.expect_count != total:
        issues.append(f"Chroma 总 Chunk {total} 与预期 {args.expect_count} 不一致")
    if args.sample_parent:
        print("=== Neo4j 父级 Section 抽样 ===")
        count, parent_issues = verify_parents(samples)
        report["sampled_neo4j_parents"] = count
        report["sampled_neo4j_failed"] = len(parent_issues)
        issues.extend(parent_issues)
    print("=== 统计结果 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        print("=== 发现问题 ===")
        for entry in issues[:30]:
            print(" -", entry)
        if len(issues) > 30:
            print(f" ...另有 {len(issues)-30} 项未列出")
        print("结论：发现数据异常；不要据此宣称全部来源匹配。")
        return 2
    print("验收结果：当前扫描与抽检项目通过（技术一致性，不代表临床权威审核）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
