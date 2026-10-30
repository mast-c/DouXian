# -*- coding: utf-8 -*-
"""V1.5：已清洗药品 JSON → Neo4j 精确核验 → Chroma 分批增量入库。

默认 dry-run：不创建检查点、不调用云端、不写 Chroma/Neo4j。
--apply --allow-cloud-embedding 明确授权后，每批最多处理少量药品，并限额写入。
基于 medical_json_chunker.py 的 V2 稳定 ID，与单药品试点已导入的 6 Chunk 兼容。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_json_chunker import (
    DEFAULT_FIELDS, FIELD_LABELS, align_to_neo4j,
    extract_raw_chunks, make_chroma_record,
)

QUERY_SECTIONS = """
MATCH (d:Drug {name: $drug_name})-[:HAS_RECORD]->(r:SourceRecord)
      -[:HAS_SECTION]->(s:Section)
WHERE toLower(coalesce(s.field, '')) IN $fields
  AND s.text IS NOT NULL
RETURN d.id AS drug_id, d.name AS drug_name,
       r.id AS record_id, r.source_url AS source_url,
       coalesce(r.authority_status, r.source_status, 'not_verified') AS source_status,
       s.id AS section_id, toLower(s.field) AS field, s.text AS section_text
ORDER BY r.id, s.id
LIMIT $limit
"""

_TERMINAL = {"done", "no_chunks", "alignment_failed", "data_error", "chroma_conflict", "write_failed"}
_ERRORS = {"alignment_failed", "data_error", "chroma_conflict", "write_failed"}
# 这些 Metadata 是现有 JSON→Section 链接的一部分，变化不能视为纯向量复用。
_CRITICAL_META = (
    "chunk_id", "drug_id", "record_id", "section_id", "section_field",
    "json_path", "source_url", "section_sha256", "quote_start",
    "quote_end", "data_origin", "content_sha256", "chunk_index", "chunk_total",
    "source_status", "forced_cut",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> dict[str, dict]:
    """读取唯一清洗快照，不使用原始非清洗 JSON。"""
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    drugs = data.get("drugs") if isinstance(data, dict) else None
    if not isinstance(drugs, dict) or not all(isinstance(v, dict) for v in drugs.values()):
        raise ValueError("文件必须具有 drugs 对象，且每个药品内容均为 JSON 对象")
    return drugs


def build_identity(*, path: Path, digest: str, fields: tuple[str, ...],
                   chunk_size: int, overlap: int, collection: str,
                   persist_directory: str, neo4j_uri: str, database: str,
                   embedding_model: str = "") -> dict:
    return {
        "pipeline_version": "medical-json-bulk-v1.5",
        "json_path": str(path.resolve()),
        "json_sha256": digest,
        "fields": list(fields),
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "collection": collection,
        "persist_directory": str(Path(persist_directory).resolve()),
        "neo4j_uri": neo4j_uri,
        "neo4j_database": database,
        "embedding_model": embedding_model,
    }


class Checkpoint:
    """写入仅发生在 --apply；完成一个药品立即 commit，异常可重跑。"""

    def __init__(self, path: Path, identity: dict):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=10)
        self.conn.execute("PRAGMA busy_timeout=10000")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS manifest (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS drug_progress (
                drug_name TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                chunks INTEGER NOT NULL DEFAULT 0,
                new_chunks INTEGER NOT NULL DEFAULT 0,
                reused INTEGER NOT NULL DEFAULT 0,
                detail TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
        """)
        self.conn.commit()
        expected = json.dumps(identity, sort_keys=True, ensure_ascii=False)
        existing = self.conn.execute(
            "SELECT value FROM manifest WHERE key='identity'"
        ).fetchone()
        if existing is None:
            self.conn.execute(
                "INSERT INTO manifest(key,value) VALUES('identity',?)",
                (expected,),
            )
            self.conn.commit()
        elif existing[0] != expected:
            self.close()
            raise RuntimeError(
                "当前 JSON 快照、字段、切分参数、Chroma/Neo4j 目标与检查点不一致。"
                "为避免跨配置断点混用，请恢复原设置或另指定 --state 新文件。"
            )

    def get(self, drug_name: str) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM drug_progress WHERE drug_name=?",
            (drug_name,),
        ).fetchone()
        return str(row[0]) if row else None

    def set(self, drug_name: str, status: str, chunks: int = 0,
            new_chunks: int = 0, reused: int = 0, detail: str = "") -> None:
        self.conn.execute("""
            INSERT INTO drug_progress(drug_name, status, chunks, new_chunks, reused,
                                      detail, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(drug_name) DO UPDATE SET
                status=excluded.status,
                chunks=excluded.chunks,
                new_chunks=excluded.new_chunks,
                reused=excluded.reused,
                detail=excluded.detail,
                updated_at=excluded.updated_at
        """, (
            drug_name, status, chunks, new_chunks, reused,
            str(detail)[:400], now_iso()
        ))
        self.conn.commit()

    def counts(self) -> dict[str, int]:
        return {
            status: int(count) for status, count in self.conn.execute(
                "SELECT status, count(*) FROM drug_progress GROUP BY status"
            )
        }

    def failures(self, limit: int = 10) -> list[tuple[str, str, str]]:
        return self.conn.execute(
            "SELECT drug_name,status,detail FROM drug_progress "
            "WHERE status IN ('alignment_failed','data_error',"
            "'chroma_conflict','write_failed') "
            "ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def close(self) -> None:
        self.conn.close()


def select_drug_names(drugs: dict, *, limit: int, start_after: str,
                      names: list[str] | None,
                      progress: Checkpoint | None, retry_failed: bool) -> list[str]:
    """严格稳定排序，自动跳过已成功或已经标记为失败的药品。"""
    if names:
        unknown = [name for name in names if name not in drugs]
        if unknown:
            raise ValueError(f"JSON 中没有精确药名：{unknown[:5]}")
        ordered = sorted(set(names))
    else:
        ordered = sorted(name for name in drugs if name > start_after)
    chosen = []
    for name in ordered:
        state = progress.get(name) if progress else None
        if state in ("done", "no_chunks"):
            continue
        if state in _ERRORS and not retry_failed:
            continue
        chosen.append(name)
        if len(chosen) >= limit:
            break
    return chosen


def prepare_records(drug_name: str, drug: dict, *, fields: tuple[str, ...],
                    chunk_size: int, overlap: int,
                    rows: list[dict], json_filename: str) -> tuple[list[dict], dict]:
    chunks = extract_raw_chunks(
        drug_name, drug, fields, chunk_size=chunk_size, overlap=overlap
    )
    report = {
        "raw_chunks": len(chunks),
        "forced_cuts": sum(int(c.forced_cut) for c in chunks),
        "neo4j_section_rows": len(rows),
        "matched": 0,
        "problems": [],
    }
    if not chunks:
        return [], report
    linked, problems = align_to_neo4j(
        chunks, rows, str(drug.get("source") or "")
    )
    report["matched"] = len(linked)
    report["problems"] = problems[:6]
    if problems or len(linked) != len(chunks):
        return [], report
    records = [
        make_chroma_record(
            item, chunk_index=i, chunk_total=len(linked),
            json_filename=json_filename,
        )
        for i, item in enumerate(linked)
    ]
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("同一药品出现重复的 Chroma Chunk ID，拒绝写入")
    return records, report


def partition_existing(chroma: Any, records: list[dict]
                       ) -> tuple[list[dict], int]:
    """逐药预检：全部同 ID 内容一致才入库；任何冲突都停止该药品。"""
    if not records:
        return [], 0
    result = chroma.get(
        ids=[record["id"] for record in records],
        include=["documents", "metadatas"],
    )
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError("Chroma 返回的 IDs、documents、metadatas 数量不一致")
    old = dict(zip(ids, zip(documents, metadatas)))
    pending: list[dict] = []
    reused = 0
    for record in records:
        current = old.get(record["id"])
        if current is None:
            pending.append(record)
            continue
        old_text, old_meta = current
        if old_text != record["text"] or not isinstance(old_meta, dict):
            raise ValueError(
                f"既有 Chunk {record['id']} 正文或 Metadata 类型与计划冲突，拒绝覆盖"
            )
        new_meta = record["metadata"]
        differs = [field for field in _CRITICAL_META
                   if old_meta.get(field) != new_meta.get(field)]
        if differs:
            raise ValueError(
                f"既有 Chunk {record['id']} 来源/原文指纹不一致：{differs[:6]}"
            )
        reused += 1
    return pending, reused


def write_pending(chroma: Any, pending: list[dict], *,
                  batch_size: int, pause: float) -> int:
    completed = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        chroma.add_texts(
            texts=[row["text"] for row in batch],
            metadatas=[row["metadata"] for row in batch],
            ids=[row["id"] for row in batch],
        )
        completed += len(batch)
        print(f"    [EMBED] {completed}/{len(pending)} Chunk", flush=True)
        if pause > 0 and completed < len(pending):
            time.sleep(pause)
    return completed


def checkpoint_summary(path: Path) -> None:
    if not path.is_file():
        print("还没有检查点；请先预览，之后通过 --apply 创建检查点。")
        return
    # 只读 URI 避免意外修改用户状态；不依赖任何网络服务。
    from urllib.parse import quote
    uri = "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            "SELECT status, count(*), coalesce(sum(new_chunks),0) "
            "FROM drug_progress GROUP BY status ORDER BY status"
        ).fetchall()
        print("=== 批量入库检查点 ===")
        for status, count, inserted in rows:
            print(f"{status:18} 药品={count:5} 记录新增 Chunk={inserted}")
        errors = connection.execute(
            "SELECT drug_name,status,detail FROM drug_progress "
            "WHERE status IN ('alignment_failed','data_error',"
            "'chroma_conflict','write_failed') "
            "ORDER BY updated_at DESC LIMIT 8"
        ).fetchall()
        if errors:
            print("最近需要人工处理的失败：")
            for name, status, detail in errors:
                print(f"  {name} | {status} | {detail[:130]}")
    finally:
        connection.close()


def _neo4j_driver():
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("缺少 NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def read_sections(driver: Any, drug_name: str, fields: tuple[str, ...],
                  database: str, limit: int = 200) -> list[dict]:
    with driver.session(database=database) as session:
        rows = session.execute_read(lambda tx: [
            r.data() for r in tx.run(
                QUERY_SECTIONS,
                drug_name=drug_name,
                fields=list(fields),
                limit=limit,
            )
        ])
    if len(rows) >= limit:
        raise ValueError(
            "Section 数量达到安全上限，可能被 LIMIT 截断；"
            "本药品拒绝自动关联，请检查来源记录"
        )
    return rows


def run_batch(args: argparse.Namespace, *, drugs: dict, progress: Checkpoint | None,
              fetch_rows: Any, chroma: Any | None,
              chunk_size: int, overlap: int, collection: str) -> dict:
    """按药品作为原子检查点单元，逐药校验、控预算、写入或预览。"""
    names = select_drug_names(
        drugs, limit=args.limit_drugs, start_after=args.start_after,
        names=args.drug or None, progress=progress, retry_failed=args.retry_failed
    )
    report = {
        "mode": "apply" if args.apply else "verify" if args.verify_neo4j else "preview",
        "selected_drugs": len(names),
        "completed_drugs": 0,
        "no_chunks": 0,
        "failed_drugs": 0,
        "skipped_existing_chunks": 0,
        "added_chunks": 0,
        "planned_chunks": 0,
        "forced_cuts": 0,
        "embedding_characters": 0,
        "stopped_by_budget": False,
    }
    print(f"[PLAN] 选中药品={len(names)}，模式={report['mode']}，Collection={collection}")
    for pos, name in enumerate(names, 1):
        drug = drugs[name]
        print(f"[{pos}/{len(names)}] {name}", flush=True)
        try:
            raw = extract_raw_chunks(
                name, drug, args.fields,
                chunk_size=chunk_size, overlap=overlap,
            )
            report["planned_chunks"] += len(raw)
            report["forced_cuts"] += sum(int(c.forced_cut) for c in raw)
            if not raw:
                print("  [EMPTY] 所选字段均无有效原文")
                report["no_chunks"] += 1
                if progress:
                    progress.set(name, "no_chunks")
                continue

            forced = sum(int(chunk.forced_cut) for chunk in raw)
            if args.apply and forced and not args.allow_forced_cuts:
                report["failed_drugs"] += 1
                print(f"  [SKIP] {forced} 条强制截断 Chunk：需要检查语义边界；"
                      "默认不允许云端写入。")
                if progress:
                    progress.set(
                        name, "data_error", chunks=len(raw),
                        detail=f"{forced} 条 Chunk 在非自然边界被迫截断，"
                               "需要人工审查后才可 --allow-forced-cuts",
                    )
                continue

            if not args.verify_neo4j and not args.apply:
                print(f"  [PREVIEW] {len(raw)} Chunk，长文本强制截断="
                      f"{sum(c.forced_cut for c in raw)}")
                report["completed_drugs"] += 1
                continue

            rows = fetch_rows(name, args.fields)
            records, alignment = prepare_records(
                name, drug, fields=args.fields,
                chunk_size=chunk_size, overlap=overlap,
                rows=rows, json_filename=args.json.name
            )
            print(f"  [VERIFY] {alignment['matched']}/{alignment['raw_chunks']} "
                  f"原文精确匹配，图谱 Section={len(rows)}")
            if len(records) != len(raw):
                report["failed_drugs"] += 1
                why = "; ".join(alignment["problems"][:3]) or "精确原文未全部匹配"
                print("  [SKIP] 对齐失败：", why[:220])
                if progress:
                    progress.set(name, "alignment_failed", chunks=len(raw), detail=why)
                continue

            if not args.apply:
                report["completed_drugs"] += 1
                continue

            if chroma is None or progress is None:
                raise RuntimeError("写入模式缺少 Chroma 或检查点，停止执行")
            try:
                pending, reused = partition_existing(chroma, records)
            except ValueError as exc:
                report["failed_drugs"] += 1
                print("  [CONFLICT]", str(exc)[:210])
                progress.set(name, "chroma_conflict", chunks=len(records), detail=str(exc))
                continue

            new_chars = sum(len(row["text"]) for row in pending)
            if (report["added_chunks"] + len(pending) > args.max_new_chunks
                    or report["embedding_characters"] + new_chars > args.max_embed_chars):
                report["stopped_by_budget"] = True
                print(
                    "  [STOP] 本次 Embedding 上限不足以纳入下一个药品；"
                    f"需要新 Chunk={len(pending)}、字符={new_chars}。"
                    "未标记为完成，下次可以调整额度重跑。"
                )
                break

            progress.set(name, "in_progress", chunks=len(records),
                         reused=reused, detail="准备增量写入")
            try:
                saved = write_pending(
                    chroma, pending, batch_size=args.embedding_batch,
                    pause=args.pause_seconds,
                )
            except Exception as exc:
                report["failed_drugs"] += 1
                progress.set(name, "write_failed", chunks=len(records),
                             reused=reused, detail=f"{type(exc).__name__}: {exc}")
                print("  [WRITE FAILED] 可能已经部分写入；"
                      "下次用 --retry-failed 复核既有 Chunk 后再提交。", flush=True)
                if args.stop_on_write_error:
                    print("  [STOP] 写入失败，按安全默认停止后续网络请求。")
                    break
                continue

            progress.set(name, "done", chunks=len(records),
                         reused=reused, new_chunks=saved)
            report["completed_drugs"] += 1
            report["skipped_existing_chunks"] += reused
            report["added_chunks"] += saved
            report["embedding_characters"] += new_chars
            print(f"  [DONE] 总 Chunk={len(records)}，新增={saved}，复用={reused}")

        except (ValueError, TypeError) as exc:
            report["failed_drugs"] += 1
            print(f"  [DATA ERROR] {type(exc).__name__}: {str(exc)[:220]}")
            if progress:
                progress.set(name, "data_error", detail=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            report["failed_drugs"] += 1
            print(f"  [ERROR] {type(exc).__name__}: {str(exc)[:220]}")
            if progress:
                progress.set(name, "write_failed", detail=f"{type(exc).__name__}: {exc}")
            if args.apply and args.stop_on_write_error:
                break

    if progress:
        report["checkpoint"] = str(progress.path.resolve())
        report["checkpoint_counts"] = progress.counts()
    return report


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="医疗 JSON→Neo4j 校验→Chroma 可恢复、限额增量索引 V1.5"
    )
    parser.add_argument("--json", type=Path, help="完整清洗版 drug_db_v2_cleaned.json 路径")
    parser.add_argument("--status", action="store_true", help="仅读取断点状态；无需 --json")
    parser.add_argument("--state", type=Path, default=Path("medical_bulk_state.sqlite3"))
    parser.add_argument("--limit-drugs", type=int, default=20,
                        help="每次最多处理多少个药品；默认 20")
    parser.add_argument("--drug", action="append",
                        help="仅处理给定的精确药品名称；可传多次")
    parser.add_argument("--start-after", default="",
                        help="仅从字典序大于给定药名的位置开始")
    parser.add_argument("--retry-failed", action="store_true",
                        help="显式重新处理已记录的失败药品")
    parser.add_argument("--fields", nargs="+", choices=tuple(FIELD_LABELS),
                        default=list(DEFAULT_FIELDS))
    parser.add_argument("--verify-neo4j", action="store_true",
                        help="只读核验所选药品的原文 Section（不写入）")
    parser.add_argument("--apply", action="store_true",
                        help="明确执行分批 Chroma 入库")
    parser.add_argument("--allow-cloud-embedding", action="store_true",
                        help="确认有权向云端模型发送该批资料并接受调用费用")
    parser.add_argument("--expect-collection", default="rag")
    parser.add_argument("--expect-drugs", type=int, default=11953,
                        help="默认要求 11,953 款清洗版药品；不一致则拒绝实际写入")
    parser.add_argument("--allow-forced-cuts", action="store_true",
                        help="仅在人工审查长句语义后明确允许非自然边界切分的 Chunk 入库")
    parser.add_argument("--max-new-chunks", type=int, default=150,
                        help="本次运行最多新增多少 Chunk，默认 150")
    parser.add_argument("--max-embed-chars", type=int, default=150000,
                        help="本次运行最多发送多少文本字符用于 Embedding，默认 150000")
    parser.add_argument("--embedding-batch", type=int, default=8)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--stop-on-write-error", action="store_true", default=True)
    parser.add_argument("--continue-on-write-error", action="store_false",
                        dest="stop_on_write_error",
                        help="高级选项：失败时继续处理后续药品，不推荐初始阶段使用")
    parser.add_argument("--report", type=Path, default=Path("medical_bulk_last_report.json"))
    args = parser.parse_args(argv)

    if args.status:
        return args
    if not args.json or not args.json.is_file():
        parser.error("必须提供存在的 --json 清洗版文件")
    if args.apply and args.verify_neo4j:
        parser.error("--verify-neo4j 与 --apply 不能同时出现；--apply 本身会执行核验")
    if args.apply and not args.allow_cloud_embedding:
        parser.error("--apply 必须与 --allow-cloud-embedding 同时提供")
    if not 1 <= args.limit_drugs <= 200:
        parser.error("--limit-drugs 应在 1～200 之间")
    if not 1 <= args.max_new_chunks <= 5000:
        parser.error("--max-new-chunks 应在 1～5000 之间")
    if not 100 <= args.max_embed_chars <= 5_000_000:
        parser.error("--max-embed-chars 应在 100～5000000 之间")
    if not 1 <= args.embedding_batch <= 50:
        parser.error("--embedding-batch 应在 1～50 之间")
    if not 0 <= args.pause_seconds <= 60:
        parser.error("--pause-seconds 应在 0～60 之间")
    args.fields = tuple(dict.fromkeys(args.fields))
    return args


def main(argv: list[str] | None = None) -> dict | None:
    args = _args(argv)
    if args.status:
        checkpoint_summary(args.state)
        return None

    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    import config_data as config
    chunk_size = int(config.chunk_size)
    overlap = int(config.chunk_overlap)
    collection = str(config.collection_name)
    if collection != args.expect_collection:
        raise RuntimeError(
            f"Chroma Collection 配置={collection!r}，"
            f"与 --expect-collection={args.expect_collection!r} 不符，停止"
        )

    drugs = load_dataset(args.json)
    if len(drugs) != args.expect_drugs:
        if args.apply:
            raise RuntimeError(
                f"药品数 {len(drugs)} 与预期清洗版 {args.expect_drugs} 不符；"
                "拒绝自动写入，请核实是否使用了原始未清洗数据。"
            )
        print(
            f"[WARNING] 当前 JSON 的药品数 {len(drugs)} "
            f"与预期 {args.expect_drugs} 不符，仅允许预览。"
        )
    print("=== 医疗批量增量索引 ===")
    print("JSON 文件:", args.json.resolve())
    print("JSON 药品数:", len(drugs))
    print("JSON 结构优先切分参数:", f"chunk_size={chunk_size}, overlap={overlap}")
    print("来源：JSON / Neo4j Section（第三方资料，未经医学权威核验）")
    print("Chroma:", Path(config.persist_directory).resolve(), "/", collection)
    print("模式:", "受控写入" if args.apply else "原文校验预览" if args.verify_neo4j else "JSON 无网络预览")

    state = None
    driver = None
    try:
        if args.apply:
            if not os.getenv("DASHSCOPE_API_KEY") or not os.getenv("DASHSCOPE_EMBEDDING_MODEL"):
                raise RuntimeError("缺少 DASHSCOPE_API_KEY 或 DASHSCOPE_EMBEDDING_MODEL")
            identity = build_identity(
                path=args.json, digest=file_sha256(args.json),
                fields=args.fields, chunk_size=chunk_size, overlap=overlap,
                collection=collection, persist_directory=config.persist_directory,
                neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                database=os.getenv("NEO4J_DATABASE", "neo4j"),
                embedding_model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", ""),
            )
            state = Checkpoint(args.state, identity)
        if args.verify_neo4j or args.apply:
            driver = _neo4j_driver()

        # Chroma SDK 初始化只发生在明确授权写入时，预览完全不会创建 Collection。
        chroma = None
        if args.apply:
            from knowledge_base import KnowledgeBaseService
            service = KnowledgeBaseService()
            chroma = service.chroma
            print("[CHROMA] 初始 Chunk 数量:", chroma._collection.count())

        def fetch(name: str, fields: tuple[str, ...]) -> list[dict]:
            assert driver is not None
            return read_sections(
                driver, name, fields, database=os.getenv("NEO4J_DATABASE", "neo4j")
            )

        report = run_batch(
            args, drugs=drugs, progress=state,
            fetch_rows=fetch if driver else None, chroma=chroma,
            chunk_size=chunk_size, overlap=overlap, collection=collection,
        )
        if chroma is not None:
            report["chroma_count_after"] = chroma._collection.count()
        print("\n=== 本次汇总 ===")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.apply:
            # 保存报告时原子替换；报告本身不包含 API 密钥或药品正文。
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temp = args.report.with_suffix(args.report.suffix + ".tmp")
            temp.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            os.replace(temp, args.report)
            print("[REPORT]", args.report.resolve())
        return report
    finally:
        if driver is not None:
            driver.close()
        if state is not None:
            state.close()


if __name__ == "__main__":
    main()
