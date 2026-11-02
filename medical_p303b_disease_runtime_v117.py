# -*- coding: utf-8 -*-
"""豆馅医疗 GraphRAG V1.17 / P3-03B 在线疾病候选→药品说明书证据直连。

读取 P3-03A 只读审计的 JSONL 映射并用 Chunk ID 在已有 Chroma 集合中
精确读取，不进行 embedding、不更新图数据库，不建立任何医疗治疗关系。
每次使用前以 Neo4j 父级 Section 再次核验来源、原文偏移和内容 SHA。
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import ntpath
import os
from pathlib import Path
from typing import Any, Callable

from medical_evidence_bridge import verify_json_chunk

LINKED = "LINKED_EXACT_CHUNK"
JSON_ORIGIN = "json_matched_to_neo4j_section"
ASSOCIATION = "unverified_disease_mention"
VALIDATED_EVIDENCE_TYPE = "json_neo4j_text_verified"
CANDIDATE_EVIDENCE_TYPE = "unverified_drug_disease_candidate"

# 每次用户提问的数据库访问上限：不触发全库扫描或 N 条 API 请求。
MAX_CANDIDATES = 10
MAX_DOCS = 6
MAX_CHUNK_IDS = 24
MAX_LINK_ROWS = 100000
MAX_LINK_FILE_BYTES = 40 * 1024 * 1024


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _safe_url(value: Any) -> str:
    from medical_json_chunker import safe_url
    return safe_url(value)


def choose_links_file(project_dir: Path, configured: str | None = None) -> Path | None:
    """优先显式路径；否则只选择项目 stage3_reports 中最新的配对审计文件。"""
    root = Path(project_dir).resolve()
    if os.getenv("MEDICAL_DISEASE_LINKS_ENABLED", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        return None
    text = (configured if configured is not None else
            os.getenv("MEDICAL_DISEASE_LINKS_PATH", "")).strip()
    if text:
        value = Path(text)
        return (value if value.is_absolute() else root / value).resolve()

    reports = root / "stage3_reports"
    if not reports.is_dir():
        return None
    pairs = []
    for item in reports.glob("p3_03a_links_*.jsonl"):
        if item.is_file() and _paired_report_path(item).is_file():
            pairs.append(item)
    if not pairs:
        return None
    return max(pairs, key=lambda item: (item.stat().st_mtime_ns, item.name)).resolve()


def _paired_report_path(path: Path) -> Path:
    name = path.name
    prefix = "p3_03a_links_"
    if not name.startswith(prefix) or not name.endswith(".jsonl"):
        raise ValueError("关联文件名必须是 p3_03a_links_*.jsonl")
    report_name = "p3_03a_report_" + name[len(prefix):-6] + ".json"
    return path.with_name(report_name)


def _integrity_path(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    if stat.st_size < 1 or stat.st_size > MAX_LINK_FILE_BYTES:
        raise ValueError("P3-03A 映射文件大小不在安全范围")
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=3)
def _load_index_cached(
    file_path: str, file_mtime_ns: int, file_bytes: int,
    report_path: str, report_mtime_ns: int, report_bytes: int
) -> tuple[dict[str, dict], dict]:
    """每进程仅首次加载快照；文件改动时以路径+mtime+大小刷新。"""
    path = Path(file_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError("P3-03A 审计报告不是 JSON 对象")
    if (report.get("stage") != "P3-03A"
            or report.get("mode") != "read_only_cross_store_evidence_association"
            or report.get("association_type") != ASSOCIATION
            or report.get("status") not in (
                "PARTIAL_WITH_UNRESOLVED_LINKS", "PASS_ALL_LINKED"
            )):
        raise ValueError("P3-03A 关联报告不满足接入条件")
    full_rows = int(report.get("candidate_drug_term_pairs", -1))
    expected_linked = int(report.get("direct_chroma_links", -1))
    counts = report.get("status_counts")
    if (not isinstance(counts, dict)
            or not 1 <= full_rows <= MAX_LINK_ROWS
            or not 1 <= expected_linked <= full_rows):
        raise ValueError("P3-03A 审计数量或成功映射数异常")
    from_report = ntpath.basename(str(report.get("rows_output") or ""))
    if from_report != path.name:
        raise ValueError("P3-03A 报告关联的 JSONL 文件名不一致")

    index: dict[str, dict] = {}
    actual_counts = Counter()
    seen_candidate_keys = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, start=1):
            if line_no > MAX_LINK_ROWS:
                raise ValueError("P3-03A JSONL 超过读取上限")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"P3-03A JSONL 第 {line_no} 行不是对象")
            status = row.get("status")
            if not isinstance(status, str):
                raise ValueError(f"P3-03A JSONL 第 {line_no} 行没有 status")
            actual_counts[status] += 1
            candidate_id = _valid_text(row.get("candidate_id"))
            term_id = _valid_text(row.get("disease_term_id"))
            key = (candidate_id, term_id)
            if not all(key) or key in seen_candidate_keys:
                raise ValueError("P3-03A JSONL 候选—疾病术语存在空 ID 或重复")
            seen_candidate_keys.add(key)
            if status != LINKED:
                continue
            # 保守策略：候选 ID 如映射到多个 DiseaseTerm，不允许用单值推断。
            if candidate_id in index:
                raise ValueError(
                    "P3-03A 一个 candidate_id 对应多个成功链接；"
                    "在线映射需要二级索引，当前版本拒绝猜测"
                )
            required = (
                "drug_name", "drug_id", "record_id", "section_id",
                "source_url", "section_sha256", "disease_name",
                "evidence_quote_sha256",
            )
            if any(not _valid_text(row.get(k)) for k in required):
                raise ValueError("P3-03A 某个已关联记录缺少不可为空的来源键")
            if row.get("association_type") != ASSOCIATION:
                raise ValueError("P3-03A 关联语义不是未经验证的提及")
            if row.get("section_field") != "indication":
                raise ValueError("P3-03A 关联源栏目不是 indication")
            if not _safe_url(row["source_url"]):
                raise ValueError("P3-03A 有无效来源 URL")
            ids = row.get("chunk_ids")
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 8
                    or not all(isinstance(id_, str) and id_ for id_ in ids)
                    or len(ids) != len(set(ids))):
                raise ValueError("P3-03A 精确关联记录没有合法 Chunk ID")
            start, end = row.get("quote_start"), row.get("quote_end")
            if (not isinstance(start, int) or isinstance(start, bool)
                    or not isinstance(end, int) or isinstance(end, bool)
                    or not 0 <= start < end):
                raise ValueError("P3-03A 候选证据坐标无效")
            index[candidate_id] = row
    if sum(actual_counts.values()) != full_rows:
        raise ValueError("P3-03A JSONL 行数与审计总数不一致")
    expected_counts = {str(k): int(v) for k, v in counts.items()}
    if dict(actual_counts) != expected_counts:
        raise ValueError("P3-03A JSONL 状态数与审计报告不一致")
    if len(index) != expected_linked:
        raise ValueError("P3-03A JSONL 精确关联数与审计报告不一致")
    return index, {
        "candidate_pairs": full_rows, "audited_linked_pairs": expected_linked,
        "source_report": Path(report_path).name,
        "source_links": path.name,
    }


def load_audited_index(path: Path) -> tuple[dict[str, dict], dict]:
    file_path = Path(path).resolve()
    report_path = _paired_report_path(file_path)
    first = _integrity_path(file_path)
    second = _integrity_path(report_path)
    return _load_index_cached(*first, *second)


def _candidate_ids(item: dict) -> list[str]:
    raw = item.get("candidate_ids")
    if isinstance(raw, list):
        ids = [value for value in raw if isinstance(value, str) and value]
    else:
        id_ = item.get("candidate_id")
        ids = [id_] if isinstance(id_, str) and id_ else []
    return list(dict.fromkeys(ids))[:MAX_CANDIDATES]


def _candidate_matches_map(candidate: dict, row: dict) -> bool:
    """检查图谱查询候选与审计行是否同一药品、原文、来源及疾病术语。"""
    terms = candidate.get("disease_terms")
    if isinstance(terms, list) and terms:
        disease_match = row["disease_name"] in terms
    else:
        disease_match = row["disease_name"] == _valid_text(candidate.get("disease_term"))
    return (
        candidate.get("evidence_type") == CANDIDATE_EVIDENCE_TYPE
        and candidate.get("drug_name") == row["drug_name"]
        and candidate.get("record_id") == row["record_id"]
        and candidate.get("section_id") == row["section_id"]
        and candidate.get("source_url") == row["source_url"]
        and disease_match
    )


def _verify_one(
    candidate: dict, link: dict, cid: str,
    metadata: dict, document: str, parent: dict
) -> tuple[bool, str]:
    if not isinstance(metadata, dict) or not isinstance(document, str):
        return False, "missing_chunk_payload"
    if metadata.get("data_origin") != JSON_ORIGIN:
        return False, "wrong_chroma_origin"
    if metadata.get("chunk_id") != cid:
        return False, "chunk_id_conflict"
    if metadata.get("content_sha256") != _digest(document):
        return False, "content_sha_mismatch"
    for key in ("drug_id", "record_id", "section_id"):
        if metadata.get(key) != link[key]:
            return False, f"{key}_mismatch"
    if metadata.get("drug_name") != link["drug_name"]:
        return False, "drug_name_mismatch"
    if metadata.get("section_field") != "indication":
        return False, "wrong_section_field"
    if metadata.get("source_url") != link["source_url"]:
        return False, "source_url_mismatch"

    # 即使 JSONL 通过初始审计，在线使用时也检查 Neo4j 的当前原文。
    verified, reason = verify_json_chunk(metadata, document, parent)
    if not verified:
        return False, f"parent_{reason}"
    full = parent["full_section"]
    if _digest(full) != link["section_sha256"]:
        return False, "audit_section_changed"
    start, end = link["quote_start"], link["quote_end"]
    if not 0 <= start < end <= len(full):
        return False, "candidate_offsets_out_of_range"
    quote = full[start:end]
    if _digest(quote) != link["evidence_quote_sha256"]:
        return False, "audit_candidate_quote_changed"
    if quote.strip() != _valid_text(candidate.get("evidence_quote")).strip():
        return False, "graph_candidate_quote_changed"
    cstart, cend = metadata.get("quote_start"), metadata.get("quote_end")
    if not isinstance(cstart, int) or not isinstance(cend, int):
        return False, "chunk_offsets_type_invalid"
    if not cstart <= start < end <= cend:
        return False, "candidate_not_covered"
    return True, "verified"


def resolve_disease_links(
    candidates: list[dict], *, links_path: Path,
    native_chroma: Any,
    fetch_parent: Callable[[dict], dict | None],
    max_documents: int = 4,
) -> tuple[list[dict], dict]:
    """直接按 Chunk ID 从 Chroma 读取、用 Neo4j 父 Section 复核，不执行向量搜索。"""
    if not 1 <= max_documents <= MAX_DOCS:
        raise ValueError("max_documents 超出允许范围")
    report: dict[str, Any] = {
        "status": "NO_CANDIDATES", "candidate_items": 0, "mapped_ids": 0,
        "verified_links": 0, "returned_documents": 0,
        "skipped": {}, "link_report": "",
    }
    selected = [
        item for item in candidates
        if isinstance(item, dict)
        and item.get("evidence_type") == CANDIDATE_EVIDENCE_TYPE
    ][:MAX_CANDIDATES]
    report["candidate_items"] = len(selected)
    if not selected:
        return [], report

    audit_index, source_info = load_audited_index(links_path)
    report["link_report"] = source_info["source_report"]
    candidates_and_rows: list[tuple[dict, dict]] = []
    reasons = Counter()
    for candidate in selected:
        for candidate_id in _candidate_ids(candidate):
            row = audit_index.get(candidate_id)
            if row is None:
                reasons["unresolved_or_absent_mapping"] += 1
                continue
            if not _candidate_matches_map(candidate, row):
                reasons["graph_candidate_audit_key_mismatch"] += 1
                continue
            candidates_and_rows.append((candidate, row))
    report["mapped_ids"] = len(candidates_and_rows)
    if not candidates_and_rows:
        report.update({
            "status": "NO_ELIGIBLE_LINKS",
            "skipped": dict(reasons)
        })
        return [], report

    # 以审计确认的 Chunk ID 获取文档，避免重新运行 BM25/Embedding。
    all_ids: list[str] = []
    for _, row in candidates_and_rows:
        for cid in row["chunk_ids"]:
            if cid not in all_ids:
                all_ids.append(cid)
            if len(all_ids) >= MAX_CHUNK_IDS:
                break
        if len(all_ids) >= MAX_CHUNK_IDS:
            break
    if len(all_ids) > MAX_CHUNK_IDS:
        raise ValueError("本轮 Chunk ID 数超过安全限制")
    result = native_chroma.get(
        ids=all_ids, include=["documents", "metadatas"]
    )
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metas = result.get("metadatas") or []
    if not (len(ids) == len(documents) == len(metas)
            and len(ids) == len(set(ids))):
        raise ValueError("Chroma 指定 ID 读取结果异常")
    stored = {cid: (doc, meta)
              for cid, doc, meta in zip(ids, documents, metas)}
    # 同一 Section 的多个候选共享一次 Neo4j 查询。
    parents: dict[tuple[str, str, str], dict | None] = {}
    linked: dict[str, dict] = {}
    for candidate, row in candidates_and_rows:
        key = (row["drug_id"], row["record_id"], row["section_id"])
        if key not in parents:
            try:
                parents[key] = fetch_parent({
                    "drug_id": key[0], "record_id": key[1], "section_id": key[2]
                })
            except Exception:
                parents[key] = None
        parent = parents[key]
        if not isinstance(parent, dict) or not parent:
            reasons["parent_lookup_failed"] += 1
            continue
        if (parent.get("drug_name") != row["drug_name"]
                or _valid_text(parent.get("field")).lower() != "indication"
                or parent.get("source_url") != row["source_url"]):
            reasons["parent_key_mismatch"] += 1
            continue
        for cid in row["chunk_ids"]:
            if len(linked) >= max_documents and cid not in linked:
                reasons["document_limit"] += 1
                continue
            if cid not in stored:
                reasons["chunk_missing"] += 1
                continue
            doc, meta = stored[cid]
            valid, why = _verify_one(candidate, row, cid, meta, doc, parent)
            if not valid:
                reasons[why] += 1
                continue
            report["verified_links"] += 1
            if cid in linked:
                existing = linked[cid]
                if row["candidate_id"] not in existing["linked_candidate_ids"]:
                    existing["linked_candidate_ids"].append(row["candidate_id"])
                if row["disease_name"] not in existing["disease_terms"]:
                    existing["disease_terms"].append(row["disease_name"])
                continue
            # 未更改原始 document：引用永远是 Chroma 保存的原文正文。
            linked[cid] = {
                "id": 0,
                "source": _valid_text(meta.get("source")) or (
                    "药品资料：" + row["drug_name"]
                ),
                "section_path": _valid_text(meta.get("section_path")) or (
                    row["drug_name"] + " / 适应症"
                ),
                "chunk_id": cid,
                "content": doc,
                "evidence_type": VALIDATED_EVIDENCE_TYPE,
                "parent_verified": True,
                "disease_link_verified": True,
                "association_type": ASSOCIATION,
                "link_retrieval_route": "neo4j_candidate_to_chroma_id",
                "candidate_id": row["candidate_id"],
                "linked_candidate_ids": [row["candidate_id"]],
                "disease_term": row["disease_name"],
                "disease_terms": [row["disease_name"]],
                "evidence_quote": _valid_text(candidate.get("evidence_quote")),
                "drug_name": row["drug_name"],
                "drug_id": row["drug_id"],
                "record_id": row["record_id"],
                "section_id": row["section_id"],
                "section_field": "indication",
                "source_url": row["source_url"],
                # 技术性原文校验绝不升级第三方来源的监管权威性。
                "source_status": "third_party_unverified",
                "_retrieval_tool": "search_drug_graph",
            }
    report["skipped"] = dict(sorted(reasons.items()))
    report["returned_documents"] = len(linked)
    report["status"] = "LINKS_VERIFIED" if linked else "NO_VERIFIABLE_CHUNKS"
    return list(linked.values()), report


def merge_disease_evidence(
    excerpts: list[dict], verified_chunks: list[dict], *,
    cap: int
) -> tuple[list[dict], dict]:
    """保留候选，优先插入直连原文；重复原文只保留一条；不超过既有 cap。"""
    if not 1 <= cap <= 128:
        raise ValueError("evidence cap 越界")
    if not verified_chunks:
        return list(excerpts[:cap]), {
            "linked_inserted": 0, "existing_duplicates": 0,
            "trimmed_items": max(0, len(excerpts) - cap)
        }
    chosen = []
    for item in excerpts:
        if item.get("evidence_type") == CANDIDATE_EVIDENCE_TYPE:
            chosen.append(item)
    # 保留最多两条原流程已经选中的药品安全栏目证据，避免疾病提及
    # 直连占满引用配额并挤掉禁忌/注意事项/不良反应的原文。
    safety_fields = {
        "contraindications", "warnings", "adverse_reactions",
        "drug_interactions", "dosage",
    }
    safety = [
        item for item in excerpts
        if item.get("evidence_type") != CANDIDATE_EVIDENCE_TYPE
        and item.get("section_field") in safety_fields
    ][:2]
    chosen.extend(safety)
    chosen.extend(verified_chunks)
    chosen.extend(
        item for item in excerpts
        if item.get("evidence_type") != CANDIDATE_EVIDENCE_TYPE
        and item not in safety
    )
    keys = set()
    unique = []
    duplicates = 0
    for item in chosen:
        chunk_id = _valid_text(item.get("chunk_id"))
        key = ("chunk", chunk_id) if chunk_id else (
            "source_content",
            _valid_text(item.get("source")),
            _valid_text(item.get("content")),
        )
        if key in keys:
            duplicates += 1
            continue
        keys.add(key)
        unique.append(item)
    final = unique[:cap]
    return final, {
        "linked_inserted": sum(bool(x.get("disease_link_verified")) for x in final),
        "existing_duplicates": duplicates,
        "trimmed_items": max(0, len(unique) - cap),
    }
