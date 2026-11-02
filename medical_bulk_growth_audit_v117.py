# -*- coding: utf-8 -*-
"""豆馅医疗 V1.17 / P3-02.2 增量扩容后原文审计：Chroma → Neo4j Section。

默认抽样，--full 逐条核验全部新增 Chunk。
只读 Chroma/Neo4j；仅向全新 JSON 报告文件写出结果；不访问模型服务。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

ORIGIN = "json_matched_to_neo4j_section"


def sha(meta: dict, doc: str) -> str:
    return hashlib.sha256(
        json.dumps({"metadata": meta, "document": doc},
                   ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def select_audit_ids(new_ids: set[str], sample: int, *, full: bool) -> list[str]:
    order = sorted(new_ids, key=lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())
    return order if full else order[:sample]


def collect_ids(collection: Any, batch: int = 500) -> tuple[set[str], int]:
    count = int(collection.count())
    ids: set[str] = set()
    offset = 0
    while offset < count:
        page = collection.get(
            limit=min(batch, count - offset), offset=offset,
            include=["metadatas"]
        )
        got = [str(v) for v in (page.get("ids") or [])]
        if not got or len(got) != len(set(got)) or any(i in ids for i in got):
            raise RuntimeError("Chroma 分页返回空页或重复 Chunk ID")
        ids.update(got)
        offset += len(got)
    if len(ids) != count or int(collection.count()) != count:
        raise RuntimeError("Chroma 在审计期间发生变化，拒绝判定")
    return ids, count


def load_by_ids(collection: Any, ids: list[str],
                batch: int = 200) -> dict[str, tuple[dict, str]]:
    rows = {}
    for i in range(0, len(ids), batch):
        page = collection.get(
            ids=ids[i:i+batch], include=["metadatas", "documents"]
        )
        got_ids = page.get("ids") or []
        metas = page.get("metadatas") or []
        docs = page.get("documents") or []
        if not (len(got_ids) == len(metas) == len(docs)):
            raise RuntimeError("Chroma ids/metadata/document 长度不一致")
        for id_, meta, doc in zip(got_ids, metas, docs):
            id_ = str(id_)
            if id_ in rows or not isinstance(meta, dict) or not isinstance(doc, str):
                raise RuntimeError("Chroma 返回的同一 ID 重复或存在无效文档")
            rows[id_] = (meta, doc)
    if set(rows) != set(ids):
        raise RuntimeError("Chroma 部分 Chunk ID 缺失，停止原文审计")
    return rows


def audit_existing(
    plan: dict, collection: Any, fetch_parent: Callable,
    *, sample: int = 100, full: bool = False
) -> dict:
    if (plan.get("stage") != "P3-02" or plan.get("mode") != "plan"
            or plan.get("project_version") != "V1.17"
            or plan.get("readiness_gate") != "PASS_PRECHECK"):
        raise ValueError("P3-02 规划报告不符合审计条件")
    baseline = (plan.get("chroma") or {}).get("before_signatures")
    if not isinstance(baseline, dict) or not baseline:
        raise ValueError("P3-02 规划报告没有旧 Chunk 内容指纹")
    all_ids, count = collect_ids(collection)
    old_ids = set(baseline)
    missing = old_ids - all_ids
    if missing:
        raise RuntimeError(f"旧 Chunk 丢失 {len(missing)} 条，暂停审计")
    old_rows = load_by_ids(collection, sorted(old_ids))
    changed = [id_ for id_, (meta, doc) in old_rows.items()
               if sha(meta, doc) != baseline[id_]]
    if changed:
        raise RuntimeError(f"旧 Chunk 被改写 {len(changed)} 条，暂停审计")
    new_ids = all_ids - old_ids
    selected = select_audit_ids(new_ids, sample, full=full)
    samples = load_by_ids(collection, selected)
    from medical_evidence_bridge import verify_json_chunk
    counts = Counter()
    examples = []
    for chunk_id in selected:
        meta, doc = samples[chunk_id]
        if meta.get("data_origin") != ORIGIN:
            ok, reason = False, "invalid_data_origin"
        else:
            try:
                parent = fetch_parent(meta)
                ok, reason = verify_json_chunk(meta, doc, parent)
            except Exception as exc:
                ok, reason = False, f"lookup_{type(exc).__name__}"
        counts["verified" if ok else "failed"] += 1
        if not ok:
            counts[f"reason:{reason}"] += 1
            if len(examples) < 25:
                examples.append({
                    "chunk_id": chunk_id,
                    "drug_name": str(meta.get("drug_name") or "")[:120],
                    "reason": reason,
                })
    if int(collection.count()) != count:
        raise RuntimeError("Neo4j 核验期间 Chroma 数量改变，拒绝判定")
    if not new_ids:
        status = "NO_NEW_CHUNKS"
    elif counts["failed"]:
        status = "BLOCKED"
    elif full or len(selected) == len(new_ids):
        status = "PASS_FULL_PROVENANCE"
    else:
        status = "PASS_SAMPLED_PROVENANCE"
    return {
        "schema_version": 1, "stage": "P3-02.2",
        "mode": "growth_provenance_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "plan_id": plan.get("plan_id"),
        "chroma_count": count,
        "old_chunk_count": len(old_ids),
        "old_chunks_unchanged": len(old_ids),
        "new_chunks_since_plan": len(new_ids),
        "audit_is_full": len(selected) == len(new_ids) and bool(new_ids),
        "new_chunks_checked": len(selected),
        "verified": counts["verified"],
        "failed": counts["failed"],
        "failure_reasons": {
            k.removeprefix("reason:"): v for k, v in sorted(counts.items())
            if k.startswith("reason:")
        },
        "failure_examples": examples,
        "disclaimer": (
            "仅技术性原文溯源；不能证明药品批准适应症、临床证据或第三方来源权威性"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V1.17 大批量入库后 Neo4j 原文溯源审计")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--full", action="store_true",
                        help="对全部新增 Chunk 逐条执行 Neo4j Section 原文校验")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.sample <= 10000:
        parser.error("--sample 应在 1～10000 之间")
    if args.output.exists():
        parser.error(f"审计报告文件已存在，拒绝覆盖：{args.output}")
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict):
        parser.error("规划文件应为 JSON 对象")
    info = plan.get("chroma") or {}
    if info.get("collection") != "rag" or not info.get("path"):
        parser.error("规划目标必须是既有 rag Chroma Collection")
    from dotenv import load_dotenv
    load_dotenv(Path.cwd() / ".env", override=False)
    import chromadb
    from drug_graph_retriever import DrugGraphRetriever
    chroma_path = Path(info["path"]).resolve()
    if not (chroma_path / "chroma.sqlite3").is_file():
        parser.error(f"找不到已有 Chroma 数据目录：{chroma_path}")
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name="rag", embedding_function=None)
    retriever = DrugGraphRetriever()
    try:
        result = audit_existing(
            plan, collection, retriever.fetch_parent_section,
            sample=args.sample, full=args.full
        )
    finally:
        retriever.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as report:
        json.dump(result, report, indent=2, ensure_ascii=False)
        report.write("\n")
    print(f"[P3-02.2] {result['status']} "
          f"新增={result['new_chunks_since_plan']} "
          f"已审计={result['new_chunks_checked']} "
          f"通过={result['verified']} 失败={result['failed']}")
    print(f"报告：{args.output.resolve()}")
    return 0 if result["status"] in (
        "PASS_FULL_PROVENANCE", "PASS_SAMPLED_PROVENANCE", "NO_NEW_CHUNKS"
    ) else 2


if __name__ == "__main__":
    sys.exit(main())
