# -*- coding: utf-8 -*-
"""豆馅医疗 GraphRAG V1.17 / P3-04.0 独立疾病资料源盘点（只读扫描）。

仅读取用户本地指定目录中的文件名、有限文件样本及可选来源登记表；
创建新的诊断 JSON/CSV，不访问 Neo4j/Chroma/Embedding，不修改任何数据。
所有识别都是候选分类，绝不将药品说明书的疾病提及视为疾病百科资料。
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

SUPPORTED = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".txt",
             ".md", ".markdown", ".html", ".htm", ".xml", ".pdf", ".docx", ".zip"}
STRUCTURED = {".json", ".jsonl", ".ndjson", ".csv", ".tsv"}
TEXT_FORMATS = {".txt", ".md", ".markdown", ".html", ".htm"}
BINARY_DOCS = {".pdf", ".docx"}
PRUNED_DIRS = {"chroma_db", "stage3_reports", "stage4_reports", ".git",
               "__pycache__", "node_modules", "venv", ".venv", ".idea",
               ".vscode", "dist", "build"}
DRUG_INDICATORS = {"drugs", "drug_name", "药品名称", "药物名称", "药品通用名称",
                   "generic_name", "contraindications", "adverse_reactions",
                   "药品说明书"}
NAME_KEYS = ("disease_name", "disease", "disease_title", "疾病名称",
             "疾病名", "疾病", "疾病名称（中文）", "name", "title", "名称")
CONTENT_FIELDS = ("overview", "definition", "summary", "description",
                  "symptoms", "symptom", "causes", "etiology",
                  "diagnosis", "diagnostic_criteria", "treatment",
                  "treatment_principles", "complications", "prevention",
                  "risk_factors", "epidemiology", "临床表现", "概述", "定义",
                  "简介", "症状", "症状表现", "病因", "诊断", "诊断标准",
                  "治疗原则", "治疗", "并发症", "预防", "危险因素", "流行病学")
CONTENT_SET = {k.casefold() for k in CONTENT_FIELDS}
DISEASE_HINT = re.compile(
    r"disease|diseases|clinical.guide|illness|symptom|"
    r"疾病|病症|疾病百科|临床指南|诊疗指南|诊断标准|病因|症状", re.I
)
# 文件名/路径只是提示；包含 drug 或 药品 的文件不会被自动当疾病百科。
DRUG_HINT = re.compile(r"drug_db|drug.db|drug_db_v2|药品说明书|药品数据库",re.I)
MAX_STRUCTURED_READ = 24 * 1024 * 1024
MAX_TEXT_READ = 24 * 1024
MAX_HASH_BYTES = 64 * 1024 * 1024
MAX_SAMPLE_ROWS = 120
MAX_EXAMPLES = 4
PROVENANCE_KEYS = ("publisher", "source_url", "document_version", "rights",
                   "retrieved_at")


def _text(v: Any) -> str:
    if isinstance(v, str):
        return v.strip()
    return ""


def _record_probe(record: Any) -> tuple[bool, bool, str]:
    """返回 (像疾病知识条目, 像药品条目, 疾病名称)。"""
    if not isinstance(record, dict):
        return False, False, ""
    keys = {str(k).strip().casefold() for k in record}
    drug_score = len(keys & {k.casefold() for k in DRUG_INDICATORS})
    content_score = len(keys & CONTENT_SET)
    # 仅按存在字段名判断，是结构候选，不代表条目内容有真实性。
    name = ""
    for k, value in record.items():
        if str(k).casefold() in {x.casefold() for x in NAME_KEYS}:
            name = _text(value)[:100]
            if name:
                break
    # name + 真实疾病内容栏目的出现是最低门槛；
    # 单纯 disease_name 或药品 indication 中出现疾病不算。
    disease = bool(name and content_score and drug_score < 2)
    drug = (drug_score >= 2 or
            (drug_score >= 1 and not content_score))
    return disease, drug, name if disease else ""


def _entries(container: Any) -> list[Any]:
    """仅取少量样本，避免对巨大的来源文件做全部推断。"""
    if isinstance(container, list):
        return container[:MAX_SAMPLE_ROWS]
    if not isinstance(container, dict):
        return []
    for field in ("diseases", "disease_records", "疾病", "疾病资料",
                  "records", "items", "entries", "data"):
        value = container.get(field)
        if isinstance(value, list):
            return value[:MAX_SAMPLE_ROWS]
        if isinstance(value, dict):
            return _entries(value)
    if _record_probe(container)[0]:
        return [container]
    # 疾病名称作为 JSON key 的字典式结构。
    example = []
    for title, value in list(container.items())[:MAX_SAMPLE_ROWS]:
        if isinstance(value, dict) and any(
            str(k).casefold() in CONTENT_SET for k in value.keys()
        ):
            example.append({"disease_name": str(title), **value})
    return example


def _examine_records(items: list[Any]) -> tuple[str, dict]:
    good = 0
    drugs = 0
    examples = []
    for item in items:
        is_disease, is_drug, name = _record_probe(item)
        good += bool(is_disease)
        drugs += bool(is_drug)
        if is_disease and name and len(examples) < MAX_EXAMPLES:
            examples.append(name)
    if good > 0 and drugs > 0:
        scope = "MIXED_RECORDS_NEEDS_REVIEW"
    elif good > 0:
        scope = "POSSIBLE_INDEPENDENT_DISEASE_DOCUMENTS"
    elif drugs > 0:
        scope = "DRUG_RECORDS_NOT_DISEASE_CORPUS"
    else:
        scope = "SCHEMA_UNKNOWN"
    return scope, {
        "sampled_rows": len(items),
        "disease_shaped_sample_rows": good,
        "drug_shaped_sample_rows": drugs,
        "example_disease_names": examples,
    }


def _read_structured(path: Path, ext: str, limit: int) -> tuple[str, dict]:
    # JSONL/NDJSON 可以流式读取前 120 条，不受整个文件大小限制；
    # JSON、CSV/TSV 在本轮保守地限制整体大小以防解析负担。
    if ext not in {".jsonl", ".ndjson"} and path.stat().st_size > limit:
        return "LARGE_FILE_NOT_PARSED", {
            "note": f"文件超过 {limit} 字节限制，未解析正文",
            "sampled_rows": 0, "disease_shaped_sample_rows": 0,
            "drug_shaped_sample_rows": 0, "example_disease_names": [],
        }
    try:
        if ext == ".json":
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(obj, dict) and isinstance(obj.get("drugs"), (dict, list)):
                if not any(k in obj for k in ("diseases", "disease_records", "疾病资料")):
                    return "DRUG_RECORDS_NOT_DISEASE_CORPUS", {
                        "note": "JSON 顶层为 drugs；不是独立疾病百科资料",
                        "sampled_rows": 0, "disease_shaped_sample_rows": 0,
                        "drug_shaped_sample_rows": 1, "example_disease_names": [],
                    }
            return _examine_records(_entries(obj))
        if ext in {".jsonl", ".ndjson"}:
            samples = []
            with path.open("r", encoding="utf-8-sig") as stream:
                # 每行最多读取 2 MiB+1 字符，避免畸形超长行拉爆内存。
                for _ in range(MAX_SAMPLE_ROWS * 3):
                    line = stream.readline(2 * 1024 * 1024 + 1)
                    if not line:
                        break
                    if len(line) > 2 * 1024 * 1024 and not line.endswith("\n"):
                        raise ValueError("一行超过 2 MiB")
                    if not line.strip():
                        continue
                    samples.append(json.loads(line))
                    if len(samples) >= MAX_SAMPLE_ROWS:
                        break
            return _examine_records(samples)
        delimiter = "\t" if ext == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            samples = []
            for item in csv.DictReader(stream, delimiter=delimiter):
                samples.append(item)
                if len(samples) >= MAX_SAMPLE_ROWS:
                    break
            return _examine_records(samples)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        return "READ_OR_PARSE_ERROR", {
            "note": type(exc).__name__, "sampled_rows": 0,
            "disease_shaped_sample_rows": 0, "drug_shaped_sample_rows": 0,
            "example_disease_names": [],
        }
    return "SCHEMA_UNKNOWN", {}


def _read_unstructured(path: Path, ext: str) -> tuple[str, dict]:
    if ext in BINARY_DOCS:
        is_disease_hint = bool(DISEASE_HINT.search(str(path)))
        return (
            "POSSIBLE_DISEASE_PDF_DOCX_NEEDS_EXTRACTION"
            if is_disease_hint else "GENERIC_DOCUMENT_NEEDS_REVIEW"
        ), {
            "note": "仅登记文件，不读取 PDF/DOCX 内容；需提取后检查授权和原文"
        }
    if ext == ".zip":
        is_disease_hint = bool(DISEASE_HINT.search(str(path)))
        return (
            "POSSIBLE_DISEASE_ARCHIVE_NEEDS_SAFE_INSPECTION"
            if is_disease_hint else "GENERIC_ARCHIVE_NEEDS_REVIEW"
        ), {
            "note": "未打开或解压 ZIP；请先在隔离目录安全检查归档内容"
        }
    if ext == ".xml":
        return "XML_NEEDS_SCHEMA_REVIEW", {
            "note": "XML 仅登记文件，需要结构检查，不自动信任术语与关系"
        }
    try:
        with path.open("rb") as stream:
            buf = stream.read(MAX_TEXT_READ)
        text = buf.decode("utf-8-sig", errors="replace")
        signal = sum(bool(re.search(pattern, text, flags=re.I)) for pattern in
                     ("症状|临床表现|symptoms?", "诊断|diagnosis|diagnostic",
                      "病因|causes?|etiology", "疾病|病症|disease"))
        if signal >= 2:
            return "POSSIBLE_UNSTRUCTURED_DISEASE_SOURCE", {
                "note": "仅由文本标题/栏目初筛，需人工核验内容是否为疾病专文"
            }
    except OSError as exc:
        return "READ_OR_PARSE_ERROR", {"note": type(exc).__name__}
    return "UNSTRUCTURED_NEEDS_REVIEW", {
        "note": "尚无结构化疾病记录；不自动转换为临床事实"
    }


def _sha256_small(path: Path, size: int) -> str:
    if size > MAX_HASH_BYTES:
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            data = stream.read(1024 * 1024)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def _manifest_records(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    values = doc.get("sources") if isinstance(doc, dict) else None
    if not isinstance(values, list):
        raise ValueError("来源登记表必须是包含 sources 数组的 JSON 对象")
    result = {}
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("来源登记表每项必须为对象")
        key = _text(item.get("path")).replace("\\", "/").casefold().strip("/")
        if not key or key in result:
            raise ValueError("来源登记表 path 缺失或重复")
        result[key] = item
    return result


def _rel_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _file_probe(path: Path, *, root: Path, manifest: dict[str, dict]) -> dict:
    ext = path.suffix.lower()
    size = path.stat().st_size
    rel = _rel_path(path, root)
    hint = str(path).casefold()
    info = {
        "path": rel, "extension": ext, "bytes": size,
        "sha256": "", "classification": "", "metadata_status": "NOT_REGISTERED",
        "publisher": "", "source_url": "", "document_version": "",
        "rights": "", "retrieved_at": "",
        "sampled_rows": 0, "disease_shaped_sample_rows": 0,
        "drug_shaped_sample_rows": 0, "example_disease_names": [],
        "note": ""
    }
    if DRUG_HINT.search(hint):
        scope = "DRUG_RECORDS_NOT_DISEASE_CORPUS"
        stats = {"note": "按文件名/目录排除药品资料；避免当作疾病百科"}
    elif ext in STRUCTURED:
        scope, stats = _read_structured(path, ext, MAX_STRUCTURED_READ)
    else:
        scope, stats = _read_unstructured(path, ext)
    info["classification"] = scope
    info.update(stats)
    key = rel.replace("\\", "/").casefold().strip("/")
    registered = manifest.get(key)
    if registered:
        for field in PROVENANCE_KEYS:
            info[field] = _text(registered.get(field))
        declared_complete = all(info[field] for field in PROVENANCE_KEYS)
        info["metadata_status"] = (
            "DECLARED_COMPLETE_NEEDS_RIGHTS_REVIEW"
            if declared_complete else "INCOMPLETE_REGISTRATION"
        )
        claimed_hash = _text(registered.get("sha256")).lower()
        if claimed_hash:
            calculated = _sha256_small(path, size)
            info["sha256"] = calculated
            if not calculated:
                info["metadata_status"] = "HASH_NOT_CHECKED_TOO_LARGE"
            elif claimed_hash != calculated:
                info["metadata_status"] = "SHA256_MISMATCH"
        # 即使登记字段齐全也不自动证明来源可信或有权再分发。
    if (scope in {"POSSIBLE_INDEPENDENT_DISEASE_DOCUMENTS",
                  "POSSIBLE_UNSTRUCTURED_DISEASE_SOURCE"}
            and not info["sha256"]):
        info["sha256"] = _sha256_small(path, size)
    return info


def _default_scan_roots(root: Path) -> list[Path]:
    return [
        root / "data" / "raw", root / "raw",
        root / "disease_db", root / "data" / "disease_db",
        root / "disease_knowledge", root / "data" / "disease_knowledge",
        root / "medical_documents", root / "data" / "medical_documents",
        root / "knowledge", root / "data" / "knowledge",
    ]


def _walk_files(
    folders: list[Path], *, max_files: int = 20000,
    max_depth: int = 12
) -> tuple[list[Path], list[str]]:
    selected: list[Path] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for folder in folders:
        if not folder.exists():
            warnings.append(f"扫描目录不存在：{folder}")
            continue
        if not folder.is_dir():
            warnings.append(f"非目录、未扫描：{folder}")
            continue
        # 避免重复扫描嵌套路径；禁止跟随符号链接走出资料根目录。
        for current, dirs, names in os.walk(folder, followlinks=False):
            current_path = Path(current)
            depth = len(current_path.relative_to(folder).parts)
            dirs[:] = sorted(
                name for name in dirs
                if name.lower() not in PRUNED_DIRS
                and not (current_path / name).is_symlink()
                and depth < max_depth
            )
            for name in sorted(names):
                path = current_path / name
                if path.suffix.lower() not in SUPPORTED or path.is_symlink():
                    continue
                identity = str(path.resolve()).casefold()
                if identity in seen:
                    continue
                seen.add(identity)
                selected.append(path)
                if len(selected) >= max_files:
                    warnings.append(
                        f"达到 max_files={max_files}，结果为部分扫描，不可宣称全量覆盖"
                    )
                    return selected, warnings
    return selected, warnings


def analyze(
    *, root: Path, scan_dirs: list[Path] | None = None,
    manifest_path: Path | None = None, max_files: int = 20000
) -> tuple[dict, list[dict]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"项目目录不存在：{root}")
    paths = scan_dirs if scan_dirs is not None else _default_scan_roots(root)
    folders = [(p if p.is_absolute() else root / p).resolve() for p in paths]
    manifest = _manifest_records(manifest_path)
    files, warnings = _walk_files(folders, max_files=max_files)
    records = []
    for path in files:
        try:
            records.append(_file_probe(path, root=root, manifest=manifest))
        except (OSError, ValueError) as exc:
            records.append({
                "path": _rel_path(path, root),
                "extension": path.suffix.lower(),
                "classification": "READ_OR_PARSE_ERROR",
                "note": type(exc).__name__, "bytes": -1,
                "metadata_status": "UNAVAILABLE",
            })
    classes = Counter(r["classification"] for r in records)
    types = Counter(r["extension"] for r in records)
    possible = [
        r for r in records
        if r["classification"] in (
            "POSSIBLE_INDEPENDENT_DISEASE_DOCUMENTS",
            "POSSIBLE_UNSTRUCTURED_DISEASE_SOURCE",
            "POSSIBLE_DISEASE_PDF_DOCX_NEEDS_EXTRACTION",
            "POSSIBLE_DISEASE_ARCHIVE_NEEDS_SAFE_INSPECTION",
        )
    ]
    structured = [
        r for r in records
        if r["classification"] == "POSSIBLE_INDEPENDENT_DISEASE_DOCUMENTS"
    ]
    truncated = any("max_files" in w for w in warnings)
    inconclusive = any(r["classification"] in (
        "LARGE_FILE_NOT_PARSED", "READ_OR_PARSE_ERROR",
        "MIXED_RECORDS_NEEDS_REVIEW",
    ) for r in records)
    readiness = (
        "PARTIAL_SCAN_REQUIRES_RERUN" if truncated
        else "CANDIDATE_SOURCES_REQUIRE_PROVENANCE_REVIEW"
        if possible else "INCONCLUSIVE_FILES_REQUIRE_REVIEW"
        if inconclusive else "NO_INDEPENDENT_DISEASE_SOURCES_IDENTIFIED"
    )
    report = {
        "schema_version": 1,
        "stage": "P3-04.0",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "local_source_inventory_only",
        "readiness": readiness,
        "project_root": str(root),
        "scanned_roots": [str(f) for f in folders],
        "manifest_file": str(manifest_path) if manifest_path else None,
        "files_scanned": len(files),
        "candidate_source_files": len(possible),
        "candidate_structured_source_files": len(structured),
        "disease_shaped_sample_rows": sum(
            int(r.get("disease_shaped_sample_rows", 0)) for r in structured
        ),
        "classifications": dict(sorted(classes.items())),
        "extensions": dict(sorted(types.items())),
        "candidates": [
            {"path": r["path"], "classification": r["classification"],
             "sample_names": r.get("example_disease_names", []),
             "metadata_status": r.get("metadata_status", "NOT_REGISTERED"),
             "sha256": r.get("sha256", "")}
            for r in possible[:60]
        ],
        "warnings": warnings,
        "next_step": (
            "核对来源、版本、版权/授权与独立疾病正文；"
            "准备疾病标准化 JSONL 后单独建立 Chroma disease_knowledge collection。"
            "不可直接将 1150 DiseaseTerm 或药品说明书提及批量冒充疾病百科。"
        ),
        "safety_note": (
            "本报告只评估来源文件形状和元数据，绝不证明临床正确性或合法授权；"
            "没有读写 Neo4j/Chroma/SQLite，也没有调用云端模型。"
        ),
    }
    return report, records


def save_report(report: dict, files: list[dict], *, json_out: Path,
                csv_out: Path) -> None:
    if json_out.resolve() == csv_out.resolve():
        raise ValueError("JSON 和 CSV 输出必须是两个不同路径")
    if json_out.exists() or csv_out.exists():
        raise FileExistsError("目标报告已存在，拒绝覆盖历史审计数据")
    for output in (json_out, csv_out):
        output.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    columns = (
        "path", "classification", "extension", "bytes", "sha256",
        "metadata_status", "publisher", "source_url", "document_version",
        "rights", "retrieved_at", "sampled_rows", "disease_shaped_sample_rows",
        "drug_shaped_sample_rows", "example_disease_names", "note"
    )
    with csv_out.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in files:
            selected = {key: row.get(key, "") for key in columns}
            selected["example_disease_names"] = "、".join(
                row.get("example_disease_names") or []
            )
            writer.writerow(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P3-04.0 只读盘点本地疾病资料来源，拒绝将药品 Chunk 当疾病百科"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--scan", type=Path, action="append",
                        help="要扫描的相对或绝对目录；可重复；省略使用默认目录")
    parser.add_argument("--manifest", type=Path,
                        help="可选 JSON 来源登记表，字段 publisher/source_url/version/rights 等")
    parser.add_argument("--max-files", type=int, default=20000)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--files-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.max_files <= 100000:
        parser.error("--max-files 必须在 1~100000")
    try:
        report, files = analyze(
            root=args.root, scan_dirs=args.scan,
            manifest_path=args.manifest, max_files=args.max_files
        )
        save_report(report, files, json_out=args.report,
                    csv_out=args.files_csv)
    except (OSError, ValueError) as exc:
        print("[P3-04.0] BLOCKED:", exc, file=sys.stderr)
        return 2
    print(f"[P3-04.0] {report['readiness']}")
    print("文件扫描:", report["files_scanned"])
    print("候选疾病资料文件:", report["candidate_source_files"])
    print("结构化疾病资料候选:", report["candidate_structured_source_files"])
    print("仅采样中的疾病形状记录:", report["disease_shaped_sample_rows"])
    print("分类:", json.dumps(report["classifications"], ensure_ascii=False))
    print("报告:", args.report.resolve())
    print("文件清单:", args.files_csv.resolve())
    if report["warnings"]:
        print("注意：", "；".join(report["warnings"][:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
