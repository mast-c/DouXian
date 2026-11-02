# -*- coding: utf-8 -*-
"""豆馅医疗 GraphRAG V1.17 P3-04.1

真实 medical.json (JSON Lines, one disease per line) 的疾病百科独立入库准备。
对数据分栏、自然边界切块、原文哈希、稳定 Chunk ID 与来源标记。
不会连接 Chroma/Neo4j/SQLite，也不会调用 Embedding 或产生医疗疗效关系。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterator

HEX24 = re.compile(r"^[0-9a-f]{24}$")
FIELD_SPECS = (
    ("desc", "overview", "疾病概述"),
    ("cause", "causes", "病因与发病机制"),
    ("prevent", "prevention", "预防与健康教育"),
    ("symptom", "symptoms", "症状（原始列表）"),
)
# 独立疾病文档中这些字段均未经医学审查；
# 药品推荐、患病率、治愈率与收费信息不进入疾病 Embedding。
EXCLUDED_HIGH_RISK_FIELDS = (
    "recommand_drug", "common_drug", "drug_detail",
    "cured_prob", "get_prob", "cure_lasttime", "cost_money",
    "cure_way", "do_eat", "not_eat", "recommand_eat",
)
BOUNDARIES = "\n。！？!?；;，,、：:"
MIN_CHUNK = 180
MAX_CHUNK = 1600
DEFAULT_CHUNK = 850
SOURCE_STATUS = "third_party_unverified"
MAX_ISSUE_EXAMPLES = 25
MAX_SOURCE_LINE = 2 * 1024 * 1024  # bytes
DATA_ORIGIN = "local_medical_json_disease_encyclopedia_unverified"


def digest(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True)


def get_oid(record: dict) -> str:
    root = record.get("_id")
    raw = root.get("$oid") if isinstance(root, dict) else None
    return raw if isinstance(raw, str) and HEX24.fullmatch(raw) else ""


def is_truncated_item(text: str) -> bool:
    # 源文档有形如“发热伴咳嗽、咯...” 的截断词条；
    # 不允许当作完整症状知识自动入库。
    return ("..." in text or "…" in text or "⋯" in text
            or text.endswith("．．．"))


def verify_field_shape(record: dict) -> None:
    oid = get_oid(record)
    if not oid:
        raise ValueError("Mongo _id.$oid 缺失或非 24 位十六进制")
    if not isinstance(record.get("name"), str) or not record["name"].strip():
        raise ValueError("疾病名称缺失")
    for field in ("desc", "cause", "prevent"):
        if not isinstance(record.get(field), str):
            raise ValueError(f"{field} 必须为字符串")
    symptom = record.get("symptom")
    if not isinstance(symptom, list) or not all(
        isinstance(value, str) for value in symptom
    ):
        raise ValueError("symptom 必须为字符串数组")


def split_with_offsets(full_text: str, *, chunk_size: int
                       ) -> list[tuple[int, int, str]]:
    """保留原文精确字符偏移，不补造标点，不切断正文字符。"""
    if not MIN_CHUNK <= chunk_size <= MAX_CHUNK:
        raise ValueError(f"chunk_size 必须位于 {MIN_CHUNK}～{MAX_CHUNK}")
    start = 0
    result = []
    while start < len(full_text):
        limit = min(start + chunk_size, len(full_text))
        if limit == len(full_text):
            end = limit
        else:
            possibilities = [
                k + 1 for k in range(start, limit)
                if full_text[k] in BOUNDARIES
            ]
            if not possibilities:
                raise ValueError(
                    f"没有自然切分位置：偏移[{start}:{limit}]"
                )
            # 优先取靠后的自然断点，避免大量非常短的片段。
            end = possibilities[-1]
        if not start < end <= len(full_text):
            raise RuntimeError("chunk 位置非法")
        body = full_text[start:end]
        if full_text[start:end] != body:
            raise RuntimeError("切块未能匹配原始文本")
        if body.strip():
            result.append((start, end, body))
        start = end
    if not result:
        raise ValueError("该栏目的正文为空")
    return result


def field_text_and_stats(record: dict, field: str) -> tuple[str, dict]:
    data = record[field]
    if isinstance(data, str):
        return data, {"excluded_truncated_symptoms": 0,
                      "excluded_empty_symptoms": 0,
                      "retained_symptoms": 0}
    if field != "symptom" or not isinstance(data, list):
        raise ValueError("当前仅支持 symptom 原始列表构建文本")
    keep = []
    truncated = 0
    empty = 0
    for value in data:
        value = value.strip()
        if not value:
            empty += 1
            continue
        if is_truncated_item(value):
            truncated += 1
            continue
        keep.append(value)
    # 列表的规范化表示：保持顺序，不杜撰任何新症状。
    # 偏移基准为此字段重建的全文，不是源 JSON 文件的字节坐标。
    return "\n".join(keep), {
        "excluded_truncated_symptoms": truncated,
        "excluded_empty_symptoms": empty,
        "retained_symptoms": len(keep)
    }


def assemble_chunks(record: dict, *, line_number: int,
                    source_file_sha256: str,
                    source_filename: str,
                    source_line_sha256: str,
                    chunk_size: int) -> tuple[list[dict], list[dict], dict]:
    verify_field_shape(record)
    name = record["name"].strip()
    oid = get_oid(record)
    all_chunks = []
    errors = []
    counters = Counter()
    for field, section_type, title in FIELD_SPECS:
        text, stats = field_text_and_stats(record, field)
        counters.update(stats)
        if not text.strip():
            counters["empty_fields"] += 1
            continue
        source_raw_hash = digest(canonical_json(record[field]))
        field_hash = digest(text)
        try:
            spans = split_with_offsets(text, chunk_size=chunk_size)
        except ValueError as exc:
            errors.append({
                "record_id": oid, "disease_name": name, "field": field,
                "line": line_number, "reason": str(exc)
            })
            continue
        for index, (start, end, body) in enumerate(spans):
            # 标题是明确标记的结构化辅助文本；body 才是来自源栏目的逐字文本。
            header = f"疾病：{name}\n栏目：{title}\n"
            document = header + body
            chunk_id = (
                f"disease_v117_{oid}_{field}_{index:04d}_"
                f"{digest(body)[:12]}"
            )
            metadata = {
                "data_origin": DATA_ORIGIN,
                "source_status": SOURCE_STATUS,
                "clinical_verified": False,
                "recommended_drug_relation": False,
                "disease_document_id": f"medical_json:{oid}",
                "record_oid": oid,
                "disease_name": name,
                "disease_term_id": "",  # 等待 Neo4j 精确匹配与审核
                "source_file": source_filename,
                "source_file_sha256": source_file_sha256,
                "source_line_sha256": source_line_sha256,
                "source_line_number": line_number,
                "source_field": field,
                "source_field_type": "array" if field == "symptom" else "text",
                "section_type": section_type,
                "section_title": title,
                "source_field_sha256": source_raw_hash,
                "canonical_field_sha256": field_hash,
                "field_start": start,
                "field_end": end,
                "body_sha256": digest(body),
                "document_sha256": digest(document),
                "document_body_start": len(header),
                "chunk_index": index,
                "source_url": "",  # 原资料无 URL，绝不杜撰
                "source_publisher": "",
                "source_published_at": "",
                "rights_reviewed": False,
            }
            all_chunks.append({
                "id": chunk_id, "text": document,
                "metadata": metadata
            })
            counters[f"chunks_{field}"] += 1
            counters["embedding_chars"] += len(document)
        counters[f"nonempty_{field}"] += 1
    return all_chunks, errors, dict(counters)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            data = stream.read(1024 * 1024)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def iterate_source(path: Path) -> Iterator[tuple[int, dict, str]]:
    with path.open("rb") as stream:
        for line_no, raw in enumerate(stream, start=1):
            if len(raw) > MAX_SOURCE_LINE:
                raise ValueError(f"源文件第 {line_no} 行过长，超过限制")
            if not raw.strip():
                continue
            try:
                line = raw.decode("utf-8-sig")
                record = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"源文件第 {line_no} 行 JSON 无法解析") from exc
            if not isinstance(record, dict):
                raise ValueError(f"源文件第 {line_no} 行不是 JSON 对象")
            yield line_no, record, digest(raw.rstrip(b"\r\n"))


def _tmpfile(target: Path, *, suffix: str) -> tuple[Path, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=".p3041_", suffix=suffix, dir=str(target.parent)
    )
    return Path(name), os.fdopen(fd, "w", encoding="utf-8", newline="\n")


def prepare(
    source: Path, *, output: Path | None, report_path: Path,
    limit_records: int = 0, chunk_size: int = DEFAULT_CHUNK,
    expected_sha256: str = "",
) -> dict:
    source = source.resolve()
    report_path = report_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"疾病原始资料不存在：{source}")
    if not MIN_CHUNK <= chunk_size <= MAX_CHUNK:
        raise ValueError(f"chunk_size 必须在 {MIN_CHUNK}～{MAX_CHUNK}")
    if limit_records < 0:
        raise ValueError("limit_records 不能为负")
    if output:
        output = output.resolve()
        if output == source or output == report_path:
            raise ValueError("输出路径不得等于原始 JSON 或报告路径")
        if output.exists():
            raise FileExistsError(f"输出 JSONL 已存在，拒绝覆盖：{output}")
    if report_path.exists():
        raise FileExistsError(f"报告已存在，拒绝覆盖：{report_path}")
    checksum = sha256_file(source)
    if expected_sha256 and checksum.lower() != expected_sha256.lower():
        raise ValueError("原始文件 SHA256 与预期不符，拒绝生成")
    stats = Counter()
    duplicate_names = []
    quality_examples = []
    names = {}
    record_ids = set()
    problems = []
    tmp_chunks = None
    stream = None
    try:
        if output:
            tmp_chunks, stream = _tmpfile(output, suffix=".jsonl.tmp")
        for line_no, record, line_sha in iterate_source(source):
            if limit_records and stats["records"] >= limit_records:
                break
            verify_field_shape(record)
            oid = get_oid(record)
            if oid in record_ids:
                raise ValueError(f"重复源 Mongo _id：{oid}；拒绝跨记录覆盖")
            record_ids.add(oid)
            name = record["name"].strip()
            if name in names:
                duplicate_names.append({
                    "name": name, "first_record_oid": names[name],
                    "repeated_record_oid": oid
                })
            else:
                names[name] = oid
            stats["records"] += 1
            if any(record.get(f) for f in EXCLUDED_HIGH_RISK_FIELDS):
                stats["records_with_high_risk_fields"] += 1
            chunks, errors, counts = assemble_chunks(
                record, line_number=line_no,
                source_file_sha256=checksum,
                source_filename=source.name,
                source_line_sha256=line_sha,
                chunk_size=chunk_size
            )
            if errors:
                stats["unsplittable_section_count"] += len(errors)
                problems.extend(errors[:max(0, MAX_ISSUE_EXAMPLES - len(problems))])
            for key,value in counts.items():
                stats[key] += value
            if counts["excluded_truncated_symptoms"] and len(quality_examples)<MAX_ISSUE_EXAMPLES:
                quality_examples.append({
                    "record_oid": oid,
                    "disease_name": name,
                    "excluded_symptom_items": counts["excluded_truncated_symptoms"]
                })
            stats["chunks"] += len(chunks)
            if stream:
                for chunk in chunks:
                    stream.write(json.dumps(
                        chunk, ensure_ascii=False,
                        separators=(",", ":")
                    ) + "\n")
        if not stats["records"]:
            raise ValueError("文件无有效疾病记录")
        if stream:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            stream = None
        # 对分块失败的栏目不能谎称全量成功：
        # 文件仍可用于诊断，但需人工审查再使用其对应记录。
        blockers = stats["unsplittable_section_count"]
        status = (
            "PREVIEW_ONLY" if output is None
            else "STAGED_PARTIAL_REVIEW_REQUIRED" if limit_records
            else "STAGED_WITH_BLOCKERS" if blockers
            else "STAGED_UNVERIFIED_DATA"
        )
        report = {
            "stage": "P3-04.1",
            "mode": "local_disease_encyclopedia_prepare_no_db",
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_path": str(source),
            "source_file_sha256": checksum,
            "source_file_bytes": source.stat().st_size,
            "data_format": "JSON Lines despite .json suffix",
            "source_records_processed": stats["records"],
            "distinct_disease_names": len(names),
            "duplicate_disease_names": duplicate_names[:MAX_ISSUE_EXAMPLES],
            "distinct_source_record_oids": len(record_ids),
            "chunk_size": chunk_size,
            "index_ready_records": stats["chunks"],
            "embedding_characters_estimated": stats["embedding_chars"],
            "field_counts": {
                k: stats.get(k, 0) for k in (
                    "nonempty_desc", "nonempty_cause", "nonempty_prevent",
                    "nonempty_symptom",
                    "chunks_desc", "chunks_cause", "chunks_prevent",
                    "chunks_symptom",
                    "empty_fields",
                )
            },
            "truncated_symptoms_skipped": stats["excluded_truncated_symptoms"],
            "truncated_symptoms_examples": quality_examples,
            "unsplittable_section_count": blockers,
            "unsplittable_section_examples": problems,
            "records_with_high_risk_fields": stats["records_with_high_risk_fields"],
            "high_risk_fields_excluded_from_embeddings": list(EXCLUDED_HIGH_RISK_FIELDS),
            "source_provenance_status": SOURCE_STATUS,
            "source_publication_and_rights": "UNKNOWN_REQUIRE_REVIEW",
            "disease_term_link_status": "NOT_MATCHED_TO_NEO4J",
            "output_chunks_path": str(output) if output else None,
            "output_chunks_sha256": sha256_file(tmp_chunks) if tmp_chunks else None,
            "safety": (
                "仅本地标准化准备，药品推荐/治愈率/费用/患病率不索引；"
                "来源身份和使用授权均未核验；疾病术语尚未和 Neo4j 链接；"
                "没有对 Chroma/Neo4j/SQLite 进行任何写操作"
            )
        }
        report_temp, report_stream = _tmpfile(report_path, suffix=".json.tmp")
        try:
            with report_stream:
                json.dump(report, report_stream, ensure_ascii=False, indent=2)
                report_stream.write("\n")
                report_stream.flush()
                os.fsync(report_stream.fileno())
            # 再次检查不覆盖现有文件。
            if (report_path.exists() or
                    (output is not None and output.exists())):
                raise FileExistsError("准备完成前发现同名输出，拒绝覆盖")
            if tmp_chunks and output:
                os.replace(tmp_chunks, output)
                tmp_chunks = None
            os.replace(report_temp, report_path)
        finally:
            if report_temp.exists():
                report_temp.unlink()
        return report
    finally:
        if stream:
            stream.close()
        if tmp_chunks and tmp_chunks.exists():
            tmp_chunks.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P3-04.1 将 medical.json JSONL 转为带原始哈希的疾病 Chunk（不写数据库）"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="可选；指定则导出 Chroma 待索引 JSONL，否则只生成质量报告")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--expected-sha256", default="")
    args = parser.parse_args(argv)
    try:
        report = prepare(
            args.input, output=args.output,
            report_path=args.report, limit_records=args.limit_records,
            chunk_size=args.chunk_size,
            expected_sha256=args.expected_sha256
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print("[P3-04.1] BLOCKED:", exc, file=sys.stderr)
        return 2
    print("[P3-04.1]",report["status"])
    print("疾病记录:", report["source_records_processed"])
    print("不同疾病名称:", report["distinct_disease_names"])
    print("准备 Chunk:", report["index_ready_records"])
    print("疑似截断症状条目已跳过:", report["truncated_symptoms_skipped"])
    print("强制截断阻断栏目:", report["unsplittable_section_count"])
    print("来源状态:", report["source_provenance_status"])
    print("报告:", args.report.resolve())
    if args.output:
        print("待索引 JSONL:", args.output.resolve())
    return 0 if not report["unsplittable_section_count"] else 2


if __name__ == "__main__":
    sys.exit(main())
