# -*- coding: utf-8 -*-
"""豆馅医疗 GraphRAG V1.17 / P3-04.2 疾病百科 Chroma 独立 Collection 安全导入。

默认 DRY-RUN，不打开 Chroma、不建立 SQLite、不调用 embedding。
--apply --allow-cloud-embedding --confirm-data-rights ... 才使用现有
KnowledgeBaseService 的 embedding 函数，将经过 P3-04.1 审计的 Chunk
写入同一 Chroma 持久目录下的 disease_knowledge Collection。

全程不会调用/修改 Neo4j，不对已有 rag Collection 执行任何写操作。
每批前后对 rag Collection 的 ID、正文、metadata 做全量指纹核验。
单独 SQLite 检查点 + 精确 ID 比较；崩溃后可从 Chroma 逐条恢复。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Callable, Iterable
from urllib.parse import quote as uri_quote

from medical_p3041_medical_json_prepare_v117 import (
    DATA_ORIGIN, SOURCE_STATUS, FIELD_SPECS, digest, sha256_file
)

PIPELINE = "medical-p3042-disease-v117"
DISEASE_COLLECTION = "disease_knowledge"
DRUG_COLLECTION = "rag"
EXPECT_STAGE_CHUNKS = 39443
EXPECT_RAG_CHUNKS = 52104
RIGHTS_TOKEN = "I_CONFIRM_DATA_RIGHTS_AND_EMBEDDING_COSTS"
MAX_SELECT = 500
MAX_BATCH = 32
MAX_PASSES = 300
MAX_EMBED_CHARS = 500000
FIELD_TYPES = {name: section for name, section, _ in FIELD_SPECS}
CRITICAL_METADATA = (
    "data_origin", "source_status", "clinical_verified",
    "recommended_drug_relation", "disease_document_id", "record_oid",
    "disease_name", "disease_term_id", "source_file",
    "source_file_sha256", "source_line_sha256", "source_line_number",
    "source_field", "source_field_type", "section_type", "section_title",
    "source_field_sha256", "canonical_field_sha256",
    "field_start", "field_end", "body_sha256", "document_sha256",
    "document_body_start", "chunk_index", "source_url",
    "source_publisher", "source_published_at", "rights_reviewed",
)


class ImportBlocked(RuntimeError):
    pass


class ImportFileLock:
    """OS-managed inter-process lock, auto-released on abnormal exit/Ctrl+C."""

    def __init__(self, checkpoint_path: Path):
        self.lock_path=Path(str(Path(checkpoint_path).resolve())+".lock")
        self.fd=None
        self.is_windows=(os.name=="nt")

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True,exist_ok=True)
        self.fd=os.open(str(self.lock_path),os.O_CREAT|os.O_RDWR,0o600)
        try:
            if os.fstat(self.fd).st_size==0:
                os.write(self.fd,b"0")
            os.lseek(self.fd,0,os.SEEK_SET)
            if self.is_windows:
                import msvcrt
                msvcrt.locking(self.fd,msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,IOError) as exc:
            os.close(self.fd)
            self.fd=None
            raise ImportBlocked(
                "已有另一个 P3-04.2 进程持有同一检查点导入锁，"
                "请关闭旧进程后再执行"
            ) from exc
        return self

    def __exit__(self,exc_type,exc,tb):
        if self.fd is None:
            return
        try:
            os.lseek(self.fd,0,os.SEEK_SET)
            if self.is_windows:
                import msvcrt
                msvcrt.locking(self.fd,msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.fd,fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd=None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ImportBlocked(message)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def validate_inputs(
    source: Path, chunks: Path, preparation: Path, audit: Path, *,
    expect_count: int = EXPECT_STAGE_CHUNKS
) -> dict:
    """验源 SHA、完整 staging SHA、准备报告和全量原文审计链。绝不联网。"""
    paths = [Path(x).resolve() for x in (source, chunks, preparation, audit)]
    for value in paths:
        _require(value.is_file(), f"找不到输入文件：{value}")
    source, chunks, preparation, audit = paths
    try:
        plan = json.loads(preparation.read_text(encoding="utf-8-sig"))
        verified = json.loads(audit.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ImportBlocked("准备报告或审计报告不是有效 JSON") from exc
    _require(isinstance(plan, dict) and isinstance(verified, dict),
             "准备报告和审计报告必须是 JSON 对象")
    _require(plan.get("stage") == "P3-04.1"
             and plan.get("status") == "STAGED_UNVERIFIED_DATA",
             "P3-04.1 必须是完整的 STAGED_UNVERIFIED_DATA 状态")
    _require(plan.get("mode") == "local_disease_encyclopedia_prepare_no_db",
             "准备报告来源模式不正确")
    _require(verified.get("stage") == "P3-04.1"
             and verified.get("status") == "PASS_ALL_SOURCE_ALIGNED",
             "P3-04.1 原文全量验证未通过")
    _require(
        verified.get("mode") == "read_only_all_chunk_source_verification",
        "审计不是全量逐条原文回溯")
    _require(
        plan.get("source_provenance_status") == SOURCE_STATUS
        and plan.get("disease_term_link_status") == "NOT_MATCHED_TO_NEO4J"
        and plan.get("source_publication_and_rights") ==
            "UNKNOWN_REQUIRE_REVIEW",
        "第三方来源状态或 DiseaseTerm 审核边界出现异常")
    _require(plan.get("unsplittable_section_count") == 0,
             "存在强制切分的栏目，拒绝导入")
    count = plan.get("index_ready_records")
    _require(isinstance(count, int) and count == expect_count,
             f"待入库 Chunk 数 {count!r} 与 --expect-chunks={expect_count} 不一致")
    _require(verified.get("verified_chunk_ids") == count,
             "审计通过的 Chunk 数与计划不一致")
    _require(verified.get("verified_source_records") ==
             plan.get("source_records_processed"),
             "原始疾病记录数不一致")
    _require(verified.get("source_file_sha256") ==
             plan.get("source_file_sha256"),
             "审计源 SHA256 与计划不一致")
    _require(verified.get("staging_sha256") ==
             plan.get("output_chunks_sha256"),
             "审计 staging SHA256 与计划不一致")
    _require(verified.get("medical_validation") == "NOT_PERFORMED"
             and verified.get("chroma_import") == "NOT_PERFORMED",
             "审计报告不是导入前的原始准备状态")
    source_sha = sha256_file(source)
    stage_sha = sha256_file(chunks)
    _require(source_sha == plan["source_file_sha256"],
             f"本地 medical.json SHA256 不一致，当前：{source_sha}")
    _require(stage_sha == plan["output_chunks_sha256"],
             f"本地 staging JSONL SHA256 不一致，当前：{stage_sha}")
    _require(source.name == Path(str(plan.get("source_path", ""))).name,
             "疾病原始文件名与准备报告不一致")
    _require(source.name == "medical.json", "预计原始数据应为 medical.json")
    _require(str(plan.get("chunk_size")) == "850",
             "本工具仅对已审计的 chunk_size=850 基线开放")
    # 不能仅信任审计报告中自称 PASS 的状态：
    # 每次独立重跑 P3-04.1 的全量原文/字段/ID/偏移核验。
    from medical_p3041_verify_staging_v117 import verify
    try:
        independently_verified = verify(source,chunks,preparation)
    except (ValueError, RuntimeError) as exc:
        raise ImportBlocked(
            "独立原文回溯失败，拒绝任何 Chroma 写入"
        ) from exc
    _require(
        independently_verified.get("status") == "PASS_ALL_SOURCE_ALIGNED"
        and independently_verified.get("verified_chunk_ids") == count
        and independently_verified.get("source_file_sha256") == source_sha
        and independently_verified.get("staging_sha256") == stage_sha,
        "本地重新执行的原文核验未通过"
    )
    return {
        "source_sha256": source_sha,
        "stage_sha256": stage_sha,
        "source_path": str(source),
        "chunks_path": str(chunks),
        "count": count,
        "prepare_sha256": sha256_file(preparation),
        "audit_sha256": sha256_file(audit),
        "source_records": plan.get("source_records_processed"),
    }


def validate_staging_row(row: dict, *, stage_source_sha: str,
                         source_basename: str) -> dict:
    """在从待索引 JSONL 选择写入前，独立验证每条 Chunk 的 ID/正文标记。"""
    _require(isinstance(row, dict), "staging JSONL 某行不是对象")
    cid, body, meta = row.get("id"), row.get("text"), row.get("metadata")
    _require(isinstance(cid, str) and cid, "Chunk 缺少字符串 ID")
    _require(isinstance(body, str) and body, f"Chunk {cid} 正文缺失")
    _require(isinstance(meta, dict), f"Chunk {cid} metadata 非对象")
    oid = meta.get("record_oid")
    field = meta.get("source_field")
    _require(isinstance(oid, str) and len(oid) == 24
             and all(x in "0123456789abcdef" for x in oid),
             f"Chunk {cid} 原始记录 ID 不合法")
    _require(field in FIELD_TYPES, f"Chunk {cid} 意外字段：{field!r}")
    index = meta.get("chunk_index")
    _require(type(index) is int and index >= 0,
             f"Chunk {cid} chunk_index 不合法")
    start = meta.get("document_body_start")
    _require(type(start) is int and 0 < start < len(body),
             f"Chunk {cid} document_body_start 不合法")
    chunk_body = body[start:]
    _require(meta.get("document_sha256") == digest(body)
             and meta.get("body_sha256") == digest(chunk_body),
             f"Chunk {cid} 内容校验哈希失败")
    expected_id = (
        f"disease_v117_{oid}_{field}_{index:04d}_"
        f"{digest(chunk_body)[:12]}"
    )
    _require(cid == expected_id, f"Chunk {cid} 与稳定 Chunk ID 规则不符")
    _require(meta.get("data_origin") == DATA_ORIGIN
             and meta.get("source_status") == SOURCE_STATUS
             and meta.get("clinical_verified") is False
             and meta.get("recommended_drug_relation") is False
             and meta.get("rights_reviewed") is False,
             f"Chunk {cid} 来源/医学审核状态不正确")
    _require(meta.get("disease_term_id") == ""
             and meta.get("source_url") == ""
             and meta.get("source_publisher") == "",
             f"Chunk {cid} 被填入未经验证的 Neo4j ID/来源")
    _require(meta.get("source_file") == source_basename
             and meta.get("source_file_sha256") == stage_source_sha,
             f"Chunk {cid} 源文件哈希不一致")
    _require(meta.get("section_type") == FIELD_TYPES[field],
             f"Chunk {cid} section_type 不正确")
    _require(meta.get("disease_document_id") == f"medical_json:{oid}",
             f"Chunk {cid} disease_document_id 不正确")
    _require(all(k in meta for k in CRITICAL_METADATA),
             f"Chunk {cid} 缺少必要 metadata")
    field_start, field_end = meta["field_start"], meta["field_end"]
    _require(type(field_start) is int and type(field_end) is int
             and 0 <= field_start < field_end and
             field_end - field_start == len(chunk_body),
             f"Chunk {cid} 规范化原文偏移不合法")
    return row


def read_next_rows(chunks: Path, completed: set[str], *, max_items: int,
                   max_chars: int, source_sha: str,
                   expect_stage_count: int) -> tuple[list[dict], int, bool]:
    """扫描完整快照，跳过已完成 ID；按两个独立限额取下一批。"""
    chosen: list[dict] = []
    selected_chars = 0
    total = 0
    remaining = False
    with chunks.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                raise ImportBlocked(f"staging 第 {line_no} 行为空")
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ImportBlocked(
                    f"staging 第 {line_no} 行 JSON 损坏"
                ) from exc
            # 全量审计 + staging SHA 验证了源数据的完整覆盖；
            # 仍在实际写入前逐条检查所有被选中的记录。
            cid = record.get("id")
            _require(isinstance(cid, str) and cid,
                     f"staging 第 {line_no} 行缺 ID")
            if cid in completed:
                continue
            if len(chosen) >= max_items:
                remaining = True
                continue
            record = validate_staging_row(
                record, stage_source_sha=source_sha,
                source_basename="medical.json"
            )
            length = len(record["text"])
            if selected_chars + length > max_chars:
                if not chosen:
                    raise ImportBlocked(
                        f"Chunk {cid} 单条超过 max_embed_chars={max_chars}")
                remaining = True
                continue
            chosen.append(record)
            selected_chars += length
    _require(total == expect_stage_count,
             f"staging 文件共 {total} 行，与审计 {expect_stage_count} 行不一致")
    return chosen, selected_chars, remaining


def _get_rows(collection: Any, ids: list[str]) -> dict[str, tuple[str, dict]]:
    if not ids:
        return {}
    result = collection.get(
        ids=ids, include=["documents", "metadatas"]
    )
    got_ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    _require(len(got_ids) == len(docs) == len(metas)
             and len(set(got_ids)) == len(got_ids),
             "Chroma 指定 ID 读取结果不一致或重复")
    _require(all(cid in ids for cid in got_ids),
             "Chroma 返回了本批未请求的 ID")
    return dict(zip(got_ids, zip(docs, metas)))


def check_stored_chunk(
    requested: dict, stored: tuple[str, dict], *,
    require_all_meta: bool = True
) -> None:
    cid = requested["id"]
    text, meta = stored
    _require(text == requested["text"], f"Chroma Chunk {cid} 正文冲突")
    _require(isinstance(meta, dict), f"Chroma Chunk {cid} metadata 类型冲突")
    required = (tuple(requested["metadata"].keys()) if require_all_meta
                else CRITICAL_METADATA)
    differs = [
        key for key in required
        if meta.get(key) != requested["metadata"].get(key)
    ]
    _require(not differs,
             f"Chroma Chunk {cid} 来源 Metadata 冲突：{differs[:8]}")


def partition_existing(collection: Any, items: list[dict]
                       ) -> tuple[list[dict], list[dict]]:
    found = _get_rows(collection, [r["id"] for r in items])
    missing, reused = [], []
    for row in items:
        if row["id"] in found:
            check_stored_chunk(row, found[row["id"]])
            reused.append(row)
        else:
            missing.append(row)
    return missing, reused


def _as_vectors(vectors: Any, expected: int) -> list[list[float]]:
    _require(vectors is not None and len(vectors) == expected,
             "Embedding 返回的向量条数不匹配")
    out = []
    dims = None
    for vector in vectors:
        try:
            vals = [float(number) for number in vector]
        except (TypeError, ValueError) as exc:
            raise ImportBlocked("Embedding 返回了非数值向量") from exc
        _require(vals and all(math.isfinite(x) for x in vals),
                 "Embedding 返回空向量或 NaN/Inf")
        if dims is None:
            dims = len(vals)
        _require(len(vals) == dims, "同一批 Embedding 维度不一致")
        out.append(vals)
    return out


def embed_and_add(collection: Any, embedder: Any, rows: list[dict]) -> int:
    """向量必须由现有项目 embedding 提供；禁用 Chroma 默认 Embedding。"""
    if not rows:
        return 0
    vectors = _as_vectors(
        embedder.embed_documents([item["text"] for item in rows]),len(rows)
    )
    collection.add(
        ids=[item["id"] for item in rows],
        documents=[item["text"] for item in rows],
        metadatas=[item["metadata"] for item in rows],
        embeddings=vectors,
    )
    # Chroma.add 可能在异常或重复时部分完成；提交检查点前必须读回校验。
    stored = _get_rows(collection, [r["id"] for r in rows])
    _require(len(stored) == len(rows),
             "写入后 Chroma 缺少部分 Chunk，不能标记 checkpoint=done")
    for row in rows:
        check_stored_chunk(row, stored[row["id"]])
    return len(rows)


def _list_collection_names(client: Any) -> set[str]:
    names = set()
    for row in client.list_collections():
        if isinstance(row, str):
            names.add(row)
        else:
            names.add(row.name)
    return names


def get_drug_collection(client: Any, *, expect_count: int) -> Any:
    _require(DRUG_COLLECTION in _list_collection_names(client),
             "Chroma 缺少原有 rag Collection，拒绝运行")
    collection = client.get_collection(name=DRUG_COLLECTION)
    observed = collection.count()
    if expect_count >= 0:
        _require(observed == expect_count,
                 f"rag Chunk 数 {observed} 与预期 {expect_count} 不一致，"
                 "请先审查现有库，不得修改历史集合")
    return collection


def rag_fingerprint(collection: Any, *, page: int = 400) -> dict:
    """只读：完整 ID/正文/metadata 指纹，保持与记录分页顺序无关。"""
    count = collection.count()
    identities = []
    for offset in range(0, count, page):
        rows = collection.get(
            limit=min(page,count-offset),offset=offset,
            include=["documents", "metadatas"]
        )
        ids = rows.get("ids") or []
        docs = rows.get("documents") or []
        metas = rows.get("metadatas") or []
        _require(len(ids) == len(docs) == len(metas),
                 "rag 集合分页返回的数据长度异常")
        _require(len(ids) == min(page,count-offset),
                 "rag 集合分页返回数量异常，可能并发写入")
        for cid, doc, meta in zip(ids,docs,metas):
            _require(isinstance(cid,str) and isinstance(doc,str)
                     and isinstance(meta,dict),
                     "rag 原有 Chunk 格式异常")
            identities.append((cid,digest(doc),digest(json_dumps(meta))))
    _require(collection.count() == count and len(identities) == count
             and len({cid for cid,_,_ in identities}) == count,
             "rag 集合扫描期间发生变化或包含重复 ID")
    h = hashlib.sha256()
    for identity in sorted(identities):
        h.update(json_dumps(identity).encode("utf-8"))
        h.update(b"\n")
    return {"count":count,"document_metadata_sha256":h.hexdigest()}


def _identity(data: dict, *, chroma_path: Path, embedding_model: str,
              rag_fp: dict) -> dict:
    return {
        "pipeline": PIPELINE,
        "collection": DISEASE_COLLECTION,
        "drug_collection": DRUG_COLLECTION,
        "chroma_path": str(chroma_path.resolve()),
        "source_sha256": data["source_sha256"],
        "stage_sha256": data["stage_sha256"],
        "stage_count": data["count"],
        "chunk_size": 850,
        "embedding_model": embedding_model,
        "rag_fingerprint": rag_fp,
        "source_provenance_status": SOURCE_STATUS,
    }


class DiseaseCheckpoint:
    def __init__(self, path: Path, identity: dict):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=30)
        self.db.execute("PRAGMA busy_timeout=30000")
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS identity(
                    key TEXT PRIMARY KEY, content TEXT NOT NULL)
            """)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS chunk_progress(
                    chunk_id TEXT PRIMARY KEY,
                    document_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                       status IN ('in_progress', 'done')),
                    origin TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL)
            """)
            key = json_dumps(identity)
            stored = self.db.execute(
                "SELECT content FROM identity WHERE key='manifest'"
            ).fetchone()
            if stored is None:
                _require(self.db.execute(
                    "SELECT COUNT(*) FROM chunk_progress"
                ).fetchone()[0] == 0,
                "现有检查点有记录但无身份，拒绝接管")
                self.db.execute(
                    "INSERT INTO identity(key,content) VALUES('manifest',?)",
                    (key,)
                )
            else:
                _require(stored[0] == key,
                         "疾病导入检查点身份已绑定其他文件、模型、"
                         "Chroma 路径或 rag 基线，拒绝重置")
            self.db.commit()
        except Exception:
            self.db.rollback()
            self.db.close()
            raise

    def completed(self) -> set[str]:
        return {
            row[0] for row in self.db.execute(
                "SELECT chunk_id FROM chunk_progress WHERE status='done'"
            )
        }

    def stats(self) -> dict[str, int]:
        return {
            status: int(n) for status, n in self.db.execute(
                "SELECT status, COUNT(*) FROM chunk_progress GROUP BY status"
            )
        }

    def mark(self, rows: list[dict], status: str, *, origin: str = "") -> None:
        _require(status in {"in_progress","done"}, "未知检查点状态")
        at=now()
        with self.db:
            for row in rows:
                cid, sha = row["id"],row["metadata"]["document_sha256"]
                old=self.db.execute(
                    "SELECT document_sha256 FROM chunk_progress WHERE chunk_id=?",
                    (cid,)
                ).fetchone()
                _require(not old or old[0]==sha,
                         f"检查点中 {cid} 存在不同正文指纹，拒绝覆盖")
                self.db.execute("""
                    INSERT INTO chunk_progress
                      (chunk_id,document_sha256,status,origin,updated_at)
                    VALUES (?,?,?,?,?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                      status=excluded.status,
                      origin=excluded.origin,
                      updated_at=excluded.updated_at
                """,(cid,sha,status,origin,at))

    def close(self) -> None:
        self.db.close()


def read_checkpoint_status(path: Path) -> dict:
    path=Path(path).resolve()
    if not path.is_file():
        return {"status":"NO_CHECKPOINT","counts":{},"identity":None}
    uri="file:"+uri_quote(path.as_posix(),safe="/:")+"?mode=ro"
    db=sqlite3.connect(uri,uri=True,timeout=10)
    try:
        db.execute("PRAGMA query_only=ON")
        try:
            counts=dict(db.execute(
                "SELECT status, COUNT(*) FROM chunk_progress GROUP BY status"
            ).fetchall())
            ident=db.execute(
                "SELECT content FROM identity WHERE key='manifest'"
            ).fetchone()
        except sqlite3.Error as exc:
            raise ImportBlocked("现有 SQLite 不属于 P3-04.2 独立疾病检查点") from exc
    finally:
        db.close()
    return {
        "status":"EXISTS", "counts":counts,
        "identity":json.loads(ident[0]) if ident else None
    }


def disease_metadata(identity: dict) -> dict:
    return {
        "pipeline": PIPELINE,
        "data_origin": DATA_ORIGIN,
        "source_sha256": identity["source_sha256"],
        "stage_sha256": identity["stage_sha256"],
        "embedding_model": identity["embedding_model"],
        "source_status": SOURCE_STATUS,
        "rights_reviewed": False,
    }


def get_or_create_disease_collection(
    client: Any, *, identity: dict
) -> Any:
    names=_list_collection_names(client)
    expected=disease_metadata(identity)
    if DISEASE_COLLECTION in names:
        collection=client.get_collection(name=DISEASE_COLLECTION)
        metadata=collection.metadata or {}
        _require(all(metadata.get(k)==v for k,v in expected.items()),
                 "已有 disease_knowledge Collection 的模型/数据来源/版本"
                 "不是本次任务，拒绝覆盖或接管")
    else:
        collection=client.create_collection(
            name=DISEASE_COLLECTION,metadata=expected
        )
    return collection


def _load_runtime(*, path: Path) -> tuple[Any, Any, str]:
    """只有 --apply 时调用，确保模型与项目现有 QA 检索一致。"""
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    import config_data as config
    configured_path=Path(config.persist_directory).resolve()
    _require(configured_path==path.resolve(),
             f"配置的 Chroma 路径 {configured_path} 与目标 {path} 不同，拒绝跨库")
    _require(str(config.collection_name)==DRUG_COLLECTION,
             "项目当前原药品 Collection 配置不是 rag，拒绝写入")
    model=os.getenv("DASHSCOPE_EMBEDDING_MODEL","").strip()
    _require(bool(os.getenv("DASHSCOPE_API_KEY","")) and bool(model),
             "缺少 DASHSCOPE_API_KEY 或 DASHSCOPE_EMBEDDING_MODEL")
    from knowledge_base import KnowledgeBaseService
    service=KnowledgeBaseService()
    provider=getattr(service.chroma,"_embedding_function",None)
    _require(provider is not None and callable(
        getattr(provider,"embed_documents",None)),
        "知识库当前 LangChain Chroma 对象缺少 _embedding_function.embed_documents；"
        "未使用默认 Chroma Embedding，也不猜测其他模型")
    # 部分版本的 Embedding 对象会暴露模型名；发现明确不一致时拒绝混用。
    for name in ("model","model_name"):
        actual=getattr(provider,name,None)
        if isinstance(actual,str) and actual.strip():
            _require(actual.strip()==model,
                     f"项目 Embedding 对象的 {name}={actual!r}，"
                     f"与 DASHSCOPE_EMBEDDING_MODEL={model!r} 不一致")
            break
    import chromadb
    client=chromadb.PersistentClient(path=str(path.resolve()))
    return client,provider,model


def _apply_job_impl(
    *, source: Path, chunks: Path, prepare_report: Path, audit_report: Path,
    chroma_path: Path, state_path: Path,
    embedding_model: str, client: Any, embedder: Any,
    expect_chunks: int, expect_rag_count: int,
    max_new_chunks: int, embedding_batch: int, max_embed_chars: int,
    max_passes: int, continuous: bool, pause_seconds: float,
    pass_callback: Callable[[dict],None] | None = None,
) -> dict:
    data=validate_inputs(
        source,chunks,prepare_report,audit_report,expect_count=expect_chunks
    )
    _require(Path(chroma_path).is_dir(),
             f"Chroma 路径不存在：{chroma_path}")
    _require(Path(state_path).resolve() != Path(
        chroma_path).resolve() / "chroma.sqlite3",
        "独立 checkpoint 不能指向 Chroma 数据库")
    rag=get_drug_collection(client,expect_count=expect_rag_count)
    rag_fp=rag_fingerprint(rag)
    identity=_identity(
        data,chroma_path=Path(chroma_path),
        embedding_model=embedding_model,rag_fp=rag_fp
    )
    checkpoint=DiseaseCheckpoint(state_path,identity)
    # 在检查点身份和原 rag 基线已确认后，才允许创建新 Collection。
    try:
        disease=get_or_create_disease_collection(client,identity=identity)
        total_new=0
        total_reused=0
        total_selected=0
        passes=[]
        last_status="NOT_STARTED"
        for pass_number in range(1,max_passes+1):
            before=rag_fingerprint(rag)
            _require(before==rag_fp,
                     "原 rag Collection 在本轮开始前发生变化，禁止继续")
            completed=checkpoint.completed()
            selected,chars,remaining=read_next_rows(
                chunks,completed,max_items=max_new_chunks,
                max_chars=max_embed_chars,source_sha=data["source_sha256"],
                expect_stage_count=expect_chunks
            )
            if not selected:
                last_status="ALL_IMPORTED" if len(completed)==expect_chunks \
                    else "NO_MORE_ELIGIBLE"
                break
            # 先完整比较所有已存在 ID，任何冲突立即结束本轮，零新写入。
            pending,reused=partition_existing(disease,selected)
            checkpoint.mark(selected,"in_progress",origin="selected")
            if reused:
                checkpoint.mark(reused,"done",origin="existing_exact_match")
            added=0
            for offset in range(0,len(pending),embedding_batch):
                batch=pending[offset:offset+embedding_batch]
                added += embed_and_add(disease,embedder,batch)
                checkpoint.mark(batch,"done",origin="embedded_added")
                # 每个已完成小批次提交检查点后，再输出进度。
                if pause_seconds and offset+embedding_batch<len(pending):
                    time.sleep(pause_seconds)
            after=rag_fingerprint(rag)
            _require(after==rag_fp,
                     "原 rag Collection 完整指纹在本轮后发生变化，"
                     "本程序不会自动修复，必须排查并发写入")
            _require(disease.count() <= expect_chunks,
                     "疾病 Collection 数量超过全量计划，可能存在额外 Chunk")
            added_now=int(added)
            reused_now=len(reused)
            total_new+=added_now
            total_reused+=reused_now
            total_selected+=len(selected)
            progress=checkpoint.stats()
            record={
                "pass_number":pass_number,
                "selected":len(selected),
                "new_chunks":added_now,
                "reused_exact_chunks":reused_now,
                "embedding_characters_selected":chars,
                "checkpoint_status":progress,
                "disease_collection_count":disease.count(),
                "rag_count_unchanged":rag_fp["count"],
                "timestamp_utc":now(),
            }
            passes.append(record)
            if pass_callback:
                pass_callback(record)
            else:
                print(
                    f"[P3-04.2] pass={pass_number:03d} "
                    f"selected={len(selected)} new={added_now} "
                    f"reused={reused_now} checkpoint_done={progress.get('done',0)}/"
                    f"{expect_chunks} disease_collection={disease.count()}",
                    flush=True
                )
            if not continuous:
                last_status="PASS_COMPLETED_RESUMABLE"
                break
            if not remaining:
                # 下一次检查选中所有内容，完整校验 checkpoint。
                if progress.get("done",0)==expect_chunks:
                    last_status="ALL_IMPORTED"
                    break
        else:
            last_status="PASS_LIMIT_REACHED"
        # 每个正常返回前再次复核源数据与原 rag 指纹。
        _require(sha256_file(source)==data["source_sha256"]
                 and sha256_file(chunks)==data["stage_sha256"],
                 "导入期间源文件或 staging 被修改，停止继续导入")
        _require(rag_fingerprint(rag)==rag_fp,
                 "结束时原 rag Collection 指纹变化")
        counts=checkpoint.stats()
        if counts.get("done",0)==expect_chunks:
            _require(disease.count()==expect_chunks,
                     "检查点全部完成，但 disease_knowledge 数量不一致；"
                     "不可宣称导入完成")
            last_status="ALL_IMPORTED_NEEDS_FULL_AUDIT"
        return {
            "stage":"P3-04.2",
            "mode":"separate_chroma_disease_import",
            "status":last_status,
            "timestamp_utc":now(),
            "disease_collection":DISEASE_COLLECTION,
            "drug_collection":DRUG_COLLECTION,
            "drug_collection_unchanged_fingerprint":rag_fp,
            "source_sha256":data["source_sha256"],
            "staging_sha256":data["stage_sha256"],
            "embedding_model":embedding_model,
            "total_planned":expect_chunks,
            "checkpoint_status":counts,
            "collection_count":disease.count(),
            "this_run_new_chunks":total_new,
            "this_run_reused_chunks":total_reused,
            "this_run_selected_chunks":total_selected,
            "passes":passes,
            "source_status":SOURCE_STATUS,
            "clinical_verified":False,
            "neo4j_disease_term_linked":False,
            "online_qa_connected":False,
        }
    finally:
        checkpoint.close()


def apply_job(**kwargs: Any) -> dict:
    """Protect the whole cloud-embedding campaign with a process-level lock."""
    state_path=kwargs.get("state_path")
    _require(isinstance(state_path,(str,Path)),
             "缺少独立检查点 state_path")
    with ImportFileLock(Path(state_path)):
        return _apply_job_impl(**kwargs)


def _audit_complete_job_impl(
    *, source: Path, chunks: Path, prepare_report: Path, audit_report: Path,
    chroma_path: Path, state_path: Path, client: Any,
    expect_chunks: int, expect_rag_count: int,
    page: int=200
) -> dict:
    data=validate_inputs(
        source,chunks,prepare_report,audit_report,expect_count=expect_chunks
    )
    status=read_checkpoint_status(state_path)
    identity=status["identity"]
    _require(status["status"]=="EXISTS" and isinstance(identity,dict),
             "缺少独立疾病检查点身份")
    _require(identity.get("pipeline")==PIPELINE
             and identity.get("chroma_path")==str(Path(chroma_path).resolve())
             and identity.get("stage_sha256")==data["stage_sha256"]
             and identity.get("source_sha256")==data["source_sha256"]
             and identity.get("stage_count")==expect_chunks,
             "检查点来源身份不符")
    rag=get_drug_collection(client,expect_count=expect_rag_count)
    rag_fp=rag_fingerprint(rag)
    _require(rag_fp==identity["rag_fingerprint"],
             "原药品 rag Collection 与首次导入时的指纹不同")
    names=_list_collection_names(client)
    _require(DISEASE_COLLECTION in names,
             "缺少 disease_knowledge Collection")
    disease=get_or_create_disease_collection(client,identity=identity)
    _require(disease.count()==expect_chunks,
             f"disease_knowledge 数量 {disease.count()} ≠ {expect_chunks}")
    _require(status["counts"].get("done",0)==expect_chunks
             and status["counts"].get("in_progress",0)==0,
             "checkpoint 尚未全部完成")
    examined=0
    block=[]
    with chunks.open("r",encoding="utf-8-sig") as stream:
        for line_no,line in enumerate(stream,1):
            record=json.loads(line)
            validate_staging_row(
                record,stage_source_sha=data["source_sha256"],
                source_basename=Path(source).name
            )
            block.append(record)
            if len(block)<page:
                continue
            exists=_get_rows(disease,[x["id"] for x in block])
            _require(len(exists)==len(block),
                     f"审计组 {line_no} 缺失部分 disease Chunk")
            for item in block:
                check_stored_chunk(item,exists[item["id"]])
            examined+=len(block)
            block.clear()
        if block:
            exists=_get_rows(disease,[x["id"] for x in block])
            _require(len(exists)==len(block),"尾部疾病 Chunk 缺失")
            for item in block:
                check_stored_chunk(item,exists[item["id"]])
            examined+=len(block)
    _require(examined==expect_chunks,
             "审计读取的 staging 数量不符合预期")
    _require(rag_fingerprint(rag)==rag_fp,
             "审计期间原药品 rag Collection 发生变化")
    _require(disease.count()==expect_chunks,
             "审计期间疾病 Collection 发生变化")
    return {
        "stage":"P3-04.2",
        "mode":"read_only_full_chroma_source_audit",
        "status":"PASS_FULL_DISEASE_CHROMA_PROVENANCE",
        "timestamp_utc":now(),
        "disease_collection":DISEASE_COLLECTION,
        "verified_chunks":examined,
        "rag_snapshot":rag_fp,
        "source_sha256":data["source_sha256"],
        "staging_sha256":data["stage_sha256"],
        "embedding_model":identity["embedding_model"],
        "source_status":SOURCE_STATUS,
        "medical_accuracy_reviewed":False,
        "neo4j_disease_term_linked":False,
        "online_qa_connected":False,
    }


def audit_complete_job(**kwargs: Any) -> dict:
    """Audit a consistent inactive import snapshot; block concurrent writers."""
    state_path=kwargs.get("state_path")
    _require(isinstance(state_path,(str,Path)),
             "缺少独立检查点 state_path")
    with ImportFileLock(Path(state_path)):
        return _audit_complete_job_impl(**kwargs)


def _save_new_report(path: Path | None, report: dict) -> None:
    if path is None:
        return
    path=path.resolve()
    _require(not path.exists(),f"输出报告已存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
        stream.write("\n")


def _cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status",action="store_true",
                        help="仅读取独立疾病 SQLite 检查点")
    parser.add_argument("--audit-full",action="store_true",
                        help="从 Chroma 逐条验证全部疾病 Chunk 和原药品 rag 指纹")
    parser.add_argument("--apply",action="store_true",
                        help="明确允许写入新 disease_knowledge Collection")
    parser.add_argument("--allow-cloud-embedding",action="store_true")
    parser.add_argument("--confirm-data-rights",default="")
    parser.add_argument("--continuous",action="store_true",
                        help="多轮连续导入；Ctrl+C 停止，下次从 checkpoint 恢复")
    parser.add_argument("--max-passes",type=int,default=250)
    parser.add_argument("--max-new-chunks",type=int,default=200)
    parser.add_argument("--embedding-batch",type=int,default=8)
    parser.add_argument("--max-embed-chars",type=int,default=220000)
    parser.add_argument("--pause-seconds",type=float,default=0.15)
    parser.add_argument("--expect-chunks",type=int,default=EXPECT_STAGE_CHUNKS)
    parser.add_argument("--expect-rag-count",type=int,default=EXPECT_RAG_CHUNKS)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--chunks",type=Path)
    parser.add_argument("--prepare-report",type=Path)
    parser.add_argument("--audit-report",type=Path)
    parser.add_argument("--state",type=Path,
                        default=Path("medical_disease_index_state_v117.sqlite3"))
    parser.add_argument("--chroma-path",type=Path,default=Path("chroma_db"))
    parser.add_argument("--report",type=Path)
    args=parser.parse_args(argv)
    _require(args.max_new_chunks>=1 and args.max_new_chunks<=MAX_SELECT,
             f"--max-new-chunks 必须为 1～{MAX_SELECT}")
    _require(1<=args.embedding_batch<=MAX_BATCH,
             f"--embedding-batch 必须为 1～{MAX_BATCH}")
    _require(args.max_new_chunks>=args.embedding_batch,
             "本轮新增 Chunk 上限必须不小于 embedding batch")
    _require(100<=args.max_embed_chars<=MAX_EMBED_CHARS,
             f"--max-embed-chars 必须为 100～{MAX_EMBED_CHARS}")
    _require(1<=args.max_passes<=MAX_PASSES,
             f"--max-passes 必须为 1～{MAX_PASSES}")
    _require(0<=args.pause_seconds<=30,
             "--pause-seconds 必须为 0～30")
    _require(args.expect_chunks>0 and args.expect_rag_count>=-1,
             "--expect-chunks 和 --expect-rag-count 不合法")
    _require(not (args.apply and args.audit_full),
             "--apply 与 --audit-full 不能同时使用")
    _require(not(args.continuous and not args.apply),
             "--continuous 必须与 --apply 同时使用")
    _require(not args.apply or (
        args.allow_cloud_embedding and
        args.confirm_data_rights==RIGHTS_TOKEN),
        "--apply 必须显式提供 --allow-cloud-embedding 和"
        " --confirm-data-rights I_CONFIRM_DATA_RIGHTS_AND_EMBEDDING_COSTS")
    if not args.status:
        for name in ("source","chunks","prepare_report","audit_report"):
            _require(getattr(args,name) is not None,
                     "必须指定 --source、--chunks、--prepare-report、--audit-report")
    return args


def main(argv: list[str] | None = None, *,
         inject_client: Any=None, inject_embedder: Any=None,
         inject_model: str="") -> int:
    args=_cli(argv)
    if args.status:
        result=read_checkpoint_status(args.state)
        print("[P3-04.2]",result["status"])
        print("Checkpoint 状态:",result["counts"])
        if result["identity"]:
            print("疾病计划数:",result["identity"].get("stage_count"))
            print("原 rag 快照:",result["identity"].get("rag_fingerprint"))
        return 0
    if not args.apply and not args.audit_full:
        data=validate_inputs(
            args.source,args.chunks,args.prepare_report,args.audit_report,
            expect_count=args.expect_chunks
        )
        result={
            "stage":"P3-04.2","mode":"DRY_RUN_NO_DB",
            "status":"INPUTS_VERIFIED_NO_DB_WRITES",
            "planned_chunks":data["count"],
            "source_sha256":data["source_sha256"],
            "stage_sha256":data["stage_sha256"],
            "target_collection":DISEASE_COLLECTION,
            "old_collection":DRUG_COLLECTION,
            "cloud_embedding_called":False,
            "new_checkpoint_created":False,
        }
        _save_new_report(args.report,result)
        print("[P3-04.2] DRY_RUN 输入审计、哈希、来源全部一致")
        print("计划疾病 Chunk:",data["count"])
        print("目标 Collection:",DISEASE_COLLECTION)
        print("未打开 Chroma，未创建检查点，未调用 Embedding。")
        return 0

    chroma_path=args.chroma_path.resolve()
    if inject_client is None:
        if args.audit_full:
            import chromadb
            client=chromadb.PersistentClient(path=str(chroma_path))
            embedder=None
            model=""
        else:
            client,embedder,model=_load_runtime(path=chroma_path)
    else:
        client=inject_client
        embedder=inject_embedder
        model=inject_model
    if args.audit_full:
        report=audit_complete_job(
            source=args.source,chunks=args.chunks,
            prepare_report=args.prepare_report,
            audit_report=args.audit_report,
            chroma_path=chroma_path,state_path=args.state,client=client,
            expect_chunks=args.expect_chunks,
            expect_rag_count=args.expect_rag_count
        )
    else:
        _require(embedder is not None and model,
                 "没有显式的项目 Embedding 对象/模型配置")
        report=apply_job(
            source=args.source,chunks=args.chunks,
            prepare_report=args.prepare_report,
            audit_report=args.audit_report,
            chroma_path=chroma_path,state_path=args.state,
            embedding_model=model,client=client,embedder=embedder,
            expect_chunks=args.expect_chunks,
            expect_rag_count=args.expect_rag_count,
            max_new_chunks=args.max_new_chunks,
            embedding_batch=args.embedding_batch,
            max_embed_chars=args.max_embed_chars,
            max_passes=(args.max_passes if args.continuous else 1),
            continuous=args.continuous,
            pause_seconds=args.pause_seconds
        )
    _save_new_report(args.report,report)
    print("[P3-04.2]",report["status"])
    if args.audit_full:
        print("全量疾病 Chunk 精确核验通过:",report["verified_chunks"])
        print("原药品 rag Collection SHA256:",
              report["rag_snapshot"]["document_metadata_sha256"])
    else:
        print("本次新增:",report["this_run_new_chunks"],
              "复用:",report["this_run_reused_chunks"])
        print("疾病检查点:",report["checkpoint_status"])
        print("原 rag Collection 完整指纹不变。")
    if args.report:
        print("报告:",args.report.resolve())
    return 0


if __name__=="__main__":
    try:
        if hasattr(sys.stdout,"reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
        if hasattr(sys.stderr,"reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8",errors="backslashreplace")
        sys.exit(main())
    except KeyboardInterrupt:
        print("[P3-04.2] 已中断；每个完成的 batch 已提交检查点。"
              "重新运行相同 --apply 命令即可继续。",file=sys.stderr)
        sys.exit(130)
    except (ImportBlocked, OSError, ValueError, RuntimeError,
            ImportError, sqlite3.Error) as exc:
        print("[P3-04.2] BLOCKED:",exc,file=sys.stderr)
        sys.exit(2)
