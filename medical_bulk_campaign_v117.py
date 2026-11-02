# -*- coding: utf-8 -*-
"""豆馅医疗 GraphRAG V1.17 / P3-02.2 连续增量扩容控制器。

不修改原有 medical_json_bulk_index.py 和 medical_json_chunker.py。
默认只检查，明确 --execute --confirm-data-rights 才调用现有索引器写入。
单次导入不超过现有 --limit-drugs=200 的限制，循环由 checkpoint 自动续接。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from urllib.parse import quote

APPROVAL_SENTENCE = "I_CONFIRM_DATA_RIGHTS_AND_EMBEDDING_COSTS"
EXPECTED_INDEXER = "medical_json_bulk_index.py"
EXPECTED_CHUNKER = "medical_json_chunker.py"
DEFAULT_FIELDS = [
    "indication", "contraindications", "adverse_reactions",
    "warnings", "drug_interactions"
]


class CampaignBlocked(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(meta: dict, text: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {"metadata": meta, "document": text}, ensure_ascii=False,
            sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def load_object(path: Path) -> dict:
    if not path.is_file():
        raise CampaignBlocked(f"文件不存在：{path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise CampaignBlocked(f"JSON 根元素不是对象：{path}")
    return value


def validate_plan(plan: dict, *, source: Path, indexer: Path,
                  chunker: Path) -> None:
    if (plan.get("stage") != "P3-02" or plan.get("mode") != "plan"
            or plan.get("project_version") != "V1.17"):
        raise CampaignBlocked("必须提供 V1.17 的 P3-02 plan 文件")
    if (plan.get("status") != "PILOT_PLANNED_NOT_AUTHORIZED"
            or plan.get("readiness_gate") != "PASS_PRECHECK"
            or plan.get("write_execution") != "NOT_EXECUTED"):
        raise CampaignBlocked("P3-02 plan 未处于通过深度预检的规划状态")
    if source.resolve() == indexer.resolve() or source.resolve() == chunker.resolve():
        raise CampaignBlocked("源 JSON 与索引器/分块器路径冲突")
    for f in (source, indexer, chunker):
        if not f.is_file():
            raise CampaignBlocked(f"文件不存在：{f}")
    if indexer.name != EXPECTED_INDEXER or chunker.name != EXPECTED_CHUNKER:
        raise CampaignBlocked("必须复用原有索引器与分块器文件名")
    checks = [
        ("药品 JSON", file_sha256(source), plan.get("source_json_sha256")),
        ("索引器", file_sha256(indexer), (plan.get("indexer") or {}).get("sha256")),
        ("分块器", file_sha256(chunker), (plan.get("chunker") or {}).get("sha256")),
    ]
    for label, current, expected in checks:
        if not expected or current != expected:
            raise CampaignBlocked(
                f"{label} SHA256 与 P3-02 规划不同；请重新规划，不要跨版本续接"
            )
    if int((plan.get("source") or {}).get("source_records", 0)) != 11953:
        raise CampaignBlocked("P3-02 药品总数不是 11,953；拒绝执行")
    if ((plan.get("chroma") or {}).get("collection") != "rag"
            or not (plan.get("chroma") or {}).get("before_signatures")):
        raise CampaignBlocked("Chroma 集合或初始 Chunk 基线缺失")


def load_local_config(root: Path):
    """与原索引器相同的 .env/config_data 解析，不连接云端。"""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise CampaignBlocked("缺少 python-dotenv，需使用项目原有 rag 环境") from exc
    load_dotenv(root / ".env", override=False)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        config = importlib.import_module("config_data")
    except Exception as exc:
        raise CampaignBlocked(
            f"无法读取项目 config_data.py：{type(exc).__name__}: {exc}"
        ) from exc
    return config


def check_target_config(plan: dict, config: object) -> dict:
    collection = str(config.collection_name)
    persist = str(Path(config.persist_directory).resolve())
    expected_path = str(Path(plan["chroma"]["path"]).resolve())
    if collection != plan["chroma"]["collection"]:
        raise CampaignBlocked(
            f"当前 collection={collection}，规划为 {plan['chroma']['collection']}"
        )
    if Path(persist).resolve() != Path(expected_path).resolve():
        raise CampaignBlocked(
            "当前 Chroma 物理目录与 P3-02 规划不一致："
            f"实际={persist}，规划={expected_path}"
        )
    size, overlap = int(config.chunk_size), int(config.chunk_overlap)
    if size <= 0 or overlap < 0:
        raise CampaignBlocked("当前分块参数异常")
    return {
        "collection": collection, "persist_directory": persist,
        "chunk_size": size, "chunk_overlap": overlap
    }


def read_checkpoint_state(state: Path, *, allow_new: bool) -> dict:
    if not state.is_file():
        if not allow_new:
            raise CampaignBlocked(
                f"未发现既有 checkpoint：{state}；请核对原 --state 路径，"
                "或经检查后明确使用 --allow-new-state"
            )
        return {"exists": False, "manifest": None, "statuses": {}}
    uri = "file:" + quote(state.resolve().as_posix(), safe="/:") + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=10)
        try:
            manifest_row = conn.execute(
                "SELECT value FROM manifest WHERE key='identity'"
            ).fetchone()
            if not manifest_row:
                raise CampaignBlocked("checkpoint 缺少 identity manifest，无法可靠续接")
            manifest = json.loads(manifest_row[0])
            statuses = {
                str(status): int(count)
                for status, count in conn.execute(
                    "SELECT status, COUNT(*) FROM drug_progress GROUP BY status"
                )
            }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise CampaignBlocked(
            f"checkpoint 读取失败/格式不符：{type(exc).__name__}: {exc}"
        ) from exc
    return {"exists": True, "manifest": manifest, "statuses": statuses}


def check_checkpoint_identity(state_info: dict, *, source: Path,
                              plan: dict, config: dict) -> None:
    manifest = state_info.get("manifest")
    if manifest is None:
        return
    expected = {
        "pipeline_version": "medical-json-bulk-v1.5",
        "json_path": str(source.resolve()),
        "json_sha256": plan["source_json_sha256"],
        "collection": config["collection"],
        "persist_directory": config["persist_directory"],
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "fields": DEFAULT_FIELDS,
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_database": os.getenv("NEO4J_DATABASE", "neo4j"),
        "embedding_model": os.getenv("DASHSCOPE_EMBEDDING_MODEL", ""),
    }
    mismatches = [
        key for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise CampaignBlocked(
            f"checkpoint 与当前源/模型/切分/目标不一致：{mismatches}；"
            "不得自动使用新 checkpoint 绕过身份保护"
        )


def check_original_chunks(plan: dict, *, chroma_path: Path) -> dict:
    """核对计划阶段旧 Chunk 的完整文字与 metadata，允许其后正常增量增长。"""
    if not (chroma_path / "chroma.sqlite3").is_file():
        raise CampaignBlocked(
            f"找不到已有 Chroma SQLite 文件：{chroma_path / 'chroma.sqlite3'}；"
            "拒绝创建新库"
        )
    try:
        import chromadb
    except ImportError as exc:
        raise CampaignBlocked("缺少 chromadb，无法验证旧 Chunk 基线") from exc
    client = chromadb.PersistentClient(path=str(chroma_path.resolve()))
    collection = client.get_collection(
        name=str(plan["chroma"]["collection"]), embedding_function=None
    )
    old = plan["chroma"]["before_signatures"]
    current_count = int(collection.count())
    if current_count < len(old):
        raise CampaignBlocked(
            f"Chroma 当前只有 {current_count} 条，少于原始 {len(old)} 条"
        )
    old_ids = list(old)
    for i in range(0, len(old_ids), 150):
        page = collection.get(
            ids=old_ids[i:i + 150], include=["metadatas", "documents"]
        )
        ids = page.get("ids") or []
        metas = page.get("metadatas") or []
        docs = page.get("documents") or []
        if not (len(ids) == len(metas) == len(docs)):
            raise CampaignBlocked("Chroma 原文快照返回的数据列不一致")
        received = set(ids)
        if received != set(old_ids[i:i + 150]):
            raise CampaignBlocked(
                "Chroma 存在被删除的原始 Chunk，停止批量扩容"
            )
        for cid, meta, doc in zip(ids, metas, docs):
            if not isinstance(meta, dict) or not isinstance(doc, str):
                raise CampaignBlocked(f"旧 Chunk {cid} 的内容类型异常")
            if fingerprint(meta, doc) != old[cid]:
                raise CampaignBlocked(
                    f"旧 Chunk {cid} 被改写；拒绝继续写入"
                )
    if int(collection.count()) != current_count:
        raise CampaignBlocked("Chroma 原始基线验证时发生并发数量变更")
    return {"collection_count": current_count, "old_chunks_verified": len(old)}


def make_import_command(args: argparse.Namespace, *, limit: int,
                        chunk_cap: int, char_cap: int,
                        report_path: Path) -> list[str]:
    """不使用 shell，也不修改现有索引器参数。"""
    if not (1 <= limit <= 200):
        raise CampaignBlocked("单次 --limit-drugs 必须在 1～200 之间")
    if not (1 <= chunk_cap <= 5000):
        raise CampaignBlocked("单次 --max-new-chunks 必须在 1～5000 之间")
    if not (100 <= char_cap <= 5_000_000):
        raise CampaignBlocked("单次 --max-embed-chars 必须在 100～5000000 之间")
    command = [
        str(Path(sys.executable).resolve()), str(args.indexer.resolve()),
        "--json", str(args.json.resolve()),
        "--state", str(args.state.resolve()),
        "--expect-collection", "rag",
        "--expect-drugs", "11953",
        "--limit-drugs", str(limit),
        "--max-new-chunks", str(chunk_cap),
        "--max-embed-chars", str(char_cap),
        "--embedding-batch", str(args.embedding_batch),
        "--pause-seconds", str(args.pause_seconds),
        "--report", str(report_path.resolve()),
        "--apply", "--allow-cloud-embedding",
    ]
    return command


def assess_pass(report: dict, *, limit: int, max_nonwrite_failures: int
                ) -> tuple[int, str]:
    if report.get("mode") != "apply":
        raise CampaignBlocked("子进程报告并非 apply 模式")
    completed = int(report.get("completed_drugs", 0))
    empty = int(report.get("no_chunks", 0))
    failed = int(report.get("failed_drugs", 0))
    selected = int(report.get("selected_drugs", -1))
    processed = completed + empty + failed
    if selected < 0 or selected > limit or processed > selected:
        raise CampaignBlocked("子进程计数不自洽，停止批次调度")
    added = int(report.get("added_chunks", -1))
    chars = int(report.get("embedding_characters", -1))
    if added < 0 or chars < 0:
        raise CampaignBlocked("子进程写入/字符计数异常")
    if selected == 0:
        return 0, "NO_MORE_ELIGIBLE"
    if failed > max_nonwrite_failures:
        return processed, "TOO_MANY_FAILED_DRUGS"
    if processed == 0:
        return 0, "NO_PROGRESS"
    return processed, "CONTINUE"


def new_campaign_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory = base / f"p302_scale_{name}_{os.getpid()}"
    directory.mkdir(exist_ok=False)
    return directory


def save_json_new(path: Path, obj: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_run_report(report_path: Path) -> dict:
    if not report_path.is_file():
        raise CampaignBlocked(
            f"索引器未输出批次报告：{report_path}；"
            "请查看日志及 SQLite checkpoint，勿盲目自动重试"
        )
    return load_object(report_path)


def run_campaign(args: argparse.Namespace, plan: dict, current: dict) -> dict:
    run_dir = new_campaign_dir(args.report_dir)
    summary = {
        "schema_version": 1, "stage": "P3-02.2", "project_version": "V1.17",
        "created_utc": utc_now(), "status": "STARTED",
        "plan_id": plan.get("plan_id"), "execute": True,
        "target_drugs_this_campaign": args.target_drugs,
        "limits": {
            "per_pass_drugs": args.per_pass_drugs,
            "max_new_chunks_per_pass": args.max_new_chunks_per_pass,
            "max_embed_chars_per_pass": args.max_embed_chars_per_pass,
            "max_total_new_chunks": args.max_total_new_chunks,
            "max_total_embed_chars": args.max_total_embed_chars,
            "max_nonwrite_failures_per_pass": args.max_nonwrite_failures_per_pass,
        },
        "processed_drugs": 0, "added_chunks": 0,
        "embedding_characters": 0, "failures": 0, "passes": [],
        "starting_chroma": current,
        "report_directory": str(run_dir.resolve()),
    }
    output = run_dir / "campaign_summary.json"
    try:
        for number in range(1, args.max_passes + 1):
            remaining_drugs = args.target_drugs - summary["processed_drugs"]
            chunks_left = args.max_total_new_chunks - summary["added_chunks"]
            chars_left = args.max_total_embed_chars - summary["embedding_characters"]
            if remaining_drugs <= 0:
                summary["status"] = "TARGET_PROCESSED"
                break
            if chunks_left < 1 or chars_left < 100:
                summary["status"] = "CAMPAIGN_BUDGET_REACHED"
                break
            limit = min(args.per_pass_drugs, remaining_drugs)
            chunk_cap = min(args.max_new_chunks_per_pass, chunks_left)
            char_cap = min(args.max_embed_chars_per_pass, chars_left)
            report_path = run_dir / f"indexer_pass_{number:04d}.json"
            log_path = run_dir / f"indexer_pass_{number:04d}.log"
            if report_path.exists() or log_path.exists():
                raise CampaignBlocked("批次日志或报告已存在；拒绝覆盖")
            command = make_import_command(
                args, limit=limit, chunk_cap=chunk_cap,
                char_cap=char_cap, report_path=report_path
            )
            print(f"[P3-02.2] pass {number}: "
                  f"最多 {limit} 种药品，新增上限 {chunk_cap} Chunk，"
                  f"字符上限 {char_cap}", flush=True)
            print(f"  实时日志路径：{log_path.resolve()}", flush=True)
            # Windows: stdout 重定向到文件时，Python 子进程可能默认使用 GBK。
            # 药品名称可包含 GBK 不支持的汉字；必须在子进程启动前强制 UTF-8。
            # 单纯给父进程的 log.open(encoding="utf-8") 无法改变子进程编码。
            child_env = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
            }
            with log_path.open("x", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=str(args.root.resolve()),
                    stdout=log, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    env=child_env,
                    check=False
                )
            if result.returncode != 0:
                raise CampaignBlocked(
                    f"第 {number} 轮索引器退出码 {result.returncode}；"
                    f"日志={log_path}；检查 checkpoint 后再决定是否重试"
                )
            report = load_run_report(report_path)
            processed, state = assess_pass(
                report, limit=limit,
                max_nonwrite_failures=args.max_nonwrite_failures_per_pass
            )
            added, chars = int(report["added_chunks"]), int(report["embedding_characters"])
            if added > chunk_cap or chars > char_cap:
                raise CampaignBlocked("索引器报告超过已设置的单次预算，立即停止")
            summary["passes"].append({
                "pass": number, "processed_drugs": processed,
                "failed_drugs": int(report["failed_drugs"]),
                "new_chunks": added, "embedding_characters": chars,
                "stopped_by_budget": bool(report.get("stopped_by_budget")),
                "report": str(report_path.resolve()),
                "log": str(log_path.resolve()), "pass_state": state,
            })
            summary["processed_drugs"] += processed
            summary["added_chunks"] += added
            summary["embedding_characters"] += chars
            summary["failures"] += int(report["failed_drugs"])
            print(f"  处理药品={processed} 新增 Chunk={added} "
                  f"Embedding 字符={chars} 失败={report['failed_drugs']}",
                  flush=True)
            # 每轮后再次核验历史 313 Chunk，避免旧证据被意外覆写。
            check_original_chunks(plan, chroma_path=Path(current["persist_directory"]))
            # 读 checkpoint，阻止写入失败后继续触发更多云端请求。
            latest = read_checkpoint_state(args.state, allow_new=False)
            statuses = latest["statuses"]
            if statuses.get("write_failed") or statuses.get("chroma_conflict"):
                summary["status"] = "BLOCKED_HARD_CHECKPOINT_ERROR"
                break
            if state == "NO_MORE_ELIGIBLE" and any(
                statuses.get(k, 0) > 0 for k in ("alignment_failed", "data_error")
            ):
                # 这些药品可能是安全拒绝关联，并非已经全部成功覆盖。
                summary["status"] = "EXHAUSTED_WITH_UNRESOLVED_SKIPS"
                break
            if state != "CONTINUE":
                summary["status"] = state
                break
        else:
            summary["status"] = (
                "TARGET_PROCESSED"
                if summary["processed_drugs"] >= args.target_drugs
                else "MAX_PASSES_REACHED"
            )
        summary["finished_utc"] = utc_now()
    except Exception as exc:
        summary["status"] = "BLOCKED_UNEXPECTED_ERROR"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["finished_utc"] = utc_now()
    finally:
        # 报告写入失败时不允许吞掉错误；下一轮应人工查 checkpoint。
        save_json_new(output, summary)
        print(f"[P3-02.2] {summary['status']} 报告：{output.resolve()}", flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V1.17 连续批次增量入库控制器；默认检查、不执行云端写入"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--indexer", type=Path, default=Path(EXPECTED_INDEXER))
    parser.add_argument("--chunker", type=Path, default=Path(EXPECTED_CHUNKER))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report-dir", type=Path, default=Path("stage3_reports"))
    parser.add_argument("--allow-new-state", action="store_true",
                        help="仅当确实不存在旧断点时允许创建全新 checkpoint")
    parser.add_argument("--target-drugs", type=int, default=100)
    parser.add_argument("--per-pass-drugs", type=int, default=100)
    parser.add_argument("--max-new-chunks-per-pass", type=int, default=1000)
    parser.add_argument("--max-embed-chars-per-pass", type=int, default=1_000_000)
    parser.add_argument("--max-total-new-chunks", type=int, default=6000)
    parser.add_argument("--max-total-embed-chars", type=int, default=6_000_000)
    parser.add_argument("--embedding-batch", type=int, default=8)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--max-nonwrite-failures-per-pass", type=int, default=5)
    parser.add_argument("--max-passes", type=int, default=80)
    parser.add_argument("--execute", action="store_true",
                        help="真正运行原有索引器，并向 Embedding API 发送数据")
    parser.add_argument("--confirm-data-rights", default="",
                        help=f"执行时需精确输入 {APPROVAL_SENTENCE}")
    args = parser.parse_args(argv)
    if not (1 <= args.target_drugs <= 11953):
        parser.error("--target-drugs 必须在 1～11953 之间")
    if not (1 <= args.per_pass_drugs <= 200):
        parser.error("--per-pass-drugs 必须在 1～200 之间")
    if not (1 <= args.max_new_chunks_per_pass <= 5000):
        parser.error("--max-new-chunks-per-pass 必须在 1～5000 之间")
    if not (100 <= args.max_embed_chars_per_pass <= 5_000_000):
        parser.error("--max-embed-chars-per-pass 必须在 100～5000000 之间")
    if args.max_total_new_chunks < 1 or args.max_total_embed_chars < 100:
        parser.error("批次总限额不合法")
    if not (1 <= args.embedding_batch <= 50):
        parser.error("--embedding-batch 必须在 1～50 之间")
    if not (0 <= args.pause_seconds <= 60):
        parser.error("--pause-seconds 必须在 0～60 之间")
    if args.max_nonwrite_failures_per_pass < 0:
        parser.error("--max-nonwrite-failures-per-pass 不能为负数")
    if not (1 <= args.max_passes <= 120):
        parser.error("--max-passes 应在 1～120 之间")
    if args.execute and args.confirm_data_rights != APPROVAL_SENTENCE:
        parser.error(
            "--execute 必须明确确认数据发送权限和 Embedding 费用；"
            f"添加 --confirm-data-rights {APPROVAL_SENTENCE}"
        )
    if not args.execute and args.confirm_data_rights:
        parser.error("默认检查模式不需要 --confirm-data-rights")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.root = args.root.resolve()
    if args.root != Path.cwd().resolve():
        raise CampaignBlocked(
            f"先 cd 到项目根目录再执行；当前={Path.cwd().resolve()}，"
            f"--root={args.root}"
        )
    for attribute in ("plan", "json", "state", "indexer", "chunker", "report_dir"):
        setattr(args, attribute, getattr(args, attribute).resolve())
    plan = load_object(args.plan)
    validate_plan(plan, source=args.json, indexer=args.indexer,
                  chunker=args.chunker)
    config_raw = load_local_config(args.root)
    config = check_target_config(plan, config_raw)
    initial_state = read_checkpoint_state(
        args.state, allow_new=args.allow_new_state
    )
    check_checkpoint_identity(initial_state, source=args.json, plan=plan,
                              config=config)
    old_check = check_original_chunks(
        plan, chroma_path=Path(config["persist_directory"])
    )
    if args.execute and (
        not os.getenv("DASHSCOPE_API_KEY")
        or not os.getenv("DASHSCOPE_EMBEDDING_MODEL")
    ):
        raise CampaignBlocked("缺少 DASHSCOPE_API_KEY 或 DASHSCOPE_EMBEDDING_MODEL")
    status = {
        "plan_id": plan.get("plan_id"),
        "source_sha256": plan["source_json_sha256"],
        "root": str(args.root), "config": config, "state": str(args.state),
        "checkpoint": initial_state, "chroma": old_check,
    }
    print("[P3-02.2] 检查通过：原始 Chunk 完整；JSON、索引器、分块器哈希一致")
    print(f"[P3-02.2] 目标药品={args.target_drugs}，单轮上限={args.per_pass_drugs}")
    print("[P3-02.2] SQLite 断点：", args.state)
    if not args.execute:
        print("[P3-02.2] DRY_RUN_ONLY：未运行索引器，未请求云端，未写数据库")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    result = run_campaign(
        args, plan, {**old_check, "persist_directory": config["persist_directory"]}
    )
    return 0 if result["status"] in ("TARGET_PROCESSED", "NO_MORE_ELIGIBLE") else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CampaignBlocked, OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"[P3-02.2] BLOCKED：{exc}", file=sys.stderr)
        sys.exit(2)
