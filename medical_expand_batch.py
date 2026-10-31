# -*- coding: utf-8 -*-
"""豆馅 V1.12：固定名单的医疗知识库扩容调度器。

读取本包 expansion_manifests/round_XX.json 后调用现有
medical_json_bulk_index.py V1.5，绝不替代其 JSON/Neo4j/Chroma 实现。
相同的 --batch 即使再次运行，也只会处理该清单中的药品，
不会因为 SQLite checkpoint 已跳过 done 而滚动选中下一批。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_manifest(directory: Path, batch: int) -> dict[str, Any]:
    if not 1 <= batch <= 6:
        raise ValueError("--batch 仅支持 1～6")
    source = directory / f"round_{batch:02d}.json"
    with source.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("批次清单不是合法 JSON 对象")
    drugs = manifest.get("selected_drugs")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("batch") != batch
        or manifest.get("selection_policy") != "strict_fixed_names_no_auto_advance"
        or not isinstance(drugs, list)
        or len(drugs) != 50
        or not all(isinstance(name, str) and name.strip() for name in drugs)
        or drugs != sorted(set(drugs))
    ):
        raise ValueError("批次清单签名、药品数或顺序异常；拒绝执行")
    if drugs[-1] != manifest.get("last_drug"):
        raise ValueError("批次结束位置与药品清单不匹配")
    if any(name <= str(manifest.get("starting_after") or "") for name in drugs):
        raise ValueError("批次存在不在规定字典序范围内的药品")
    estimates = manifest.get("offline_estimates")
    if not isinstance(estimates, dict) or estimates.get("drug_count") != len(drugs):
        raise ValueError("批次清单离线统计不一致")
    return manifest


def build_bulk_args(
    *,
    manifest: dict[str, Any],
    mode: str,
    json_path: Path,
    state: Path,
    report: Path,
    allow_cloud_embedding: bool = False,
    retry_failed: bool = False,
    max_new_chunks: int = 300,
    max_embed_chars: int = 160000,
) -> list[str]:
    """生成可审计参数；固定药名避免重跑时自动滚动后续药品。"""
    if mode not in ("preview", "verify", "apply"):
        raise ValueError("mode 必须是 preview、verify 或 apply")
    if not 1 <= max_new_chunks <= 300:
        raise ValueError("安全上限 max_new_chunks 为 300")
    if not 100 <= max_embed_chars <= 160000:
        raise ValueError("安全上限 max_embed_chars 为 160000")
    if mode == "apply" and not allow_cloud_embedding:
        raise PermissionError(
            "实际写入需要同时指定 --mode apply --allow-cloud-embedding，"
            "并确认该批资料允许发送至 DashScope Embedding。"
        )

    args = [
        "--json", str(json_path),
        "--state", str(state),
        "--report", str(report),
        "--expect-collection", "rag",
        "--expect-drugs", "11953",
        "--limit-drugs", str(len(manifest["selected_drugs"])),
    ]
    for name in manifest["selected_drugs"]:
        args.extend(("--drug", name))
    if mode == "verify":
        args.append("--verify-neo4j")
    if mode == "apply":
        args.extend([
            "--max-new-chunks", str(max_new_chunks),
            "--max-embed-chars", str(max_embed_chars),
            "--embedding-batch", "8",
            "--pause-seconds", "0.5",
            "--apply", "--allow-cloud-embedding",
        ])
    if retry_failed:
        args.append("--retry-failed")
    return args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="固定批次医疗 JSON→Neo4j 核验→Chroma 增量扩容 V1.12"
    )
    parser.add_argument("--batch", type=int, default=1, choices=range(1, 7),
                        help="扩容批次 1～6，每批恰好固定 50 个药品")
    parser.add_argument("--mode", choices=["preview", "verify", "apply", "status"],
                        default="preview",
                        help="默认 preview 不联网、不写入；verify 只读 Neo4j")
    parser.add_argument("--json", type=Path,
                        help="已有 drug_db_v2_cleaned.json 的实际路径")
    parser.add_argument("--state", type=Path,
                        default=Path("medical_bulk_state.sqlite3"),
                        help="继续使用此前已写入 18 个药品的 SQLite checkpoint")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-cloud-embedding", action="store_true",
                        help="明确许可本批资料发送至云端、接受 API 费用并写入 Chroma")
    parser.add_argument("--retry-failed", action="store_true",
                        help="经人工核对后只在本固定批次中重新尝试失败药品")
    parser.add_argument("--max-new-chunks", type=int, default=300)
    parser.add_argument("--max-embed-chars", type=int, default=160000)
    args = parser.parse_args(argv)
    if args.mode == "status":
        from medical_json_bulk_index import main as import_main
        import_main(["--status", "--state", str(args.state)])
        return 0
    if not args.json or not args.json.is_file():
        parser.error("请指定存在的 --json 清洗版文件")
    if args.mode == "apply" and not args.allow_cloud_embedding:
        parser.error(
            "--mode apply 必须显式附加 --allow-cloud-embedding，"
            "表示你确认第三方原文的云端使用许可及调用费用"
        )
    if args.retry_failed and args.mode != "apply":
        parser.error("--retry-failed 仅用于 --mode apply")
    if not 1 <= args.max_new_chunks <= 300:
        parser.error("--max-new-chunks 需在 1～300 之间")
    if not 100 <= args.max_embed_chars <= 160000:
        parser.error("--max-embed-chars 需在 100～160000 之间")

    manifest = load_manifest(
        Path(__file__).resolve().parent / "expansion_manifests",
        args.batch
    )
    report = args.report or Path(
        f"medical_bulk_expansion_round_{args.batch:02d}.json"
    )
    estimates = manifest["offline_estimates"]
    print("=== 医疗扩容固定批次调度器 V1.12 ===")
    print(f"扩容批次：{args.batch}/6，固定药品：{len(manifest['selected_drugs'])}")
    print(f"范围：{manifest['selected_drugs'][0]} → {manifest['last_drug']}")
    print(f"清洗 JSON 离线预估：Chunk={estimates['raw_chunks']}，"
          f"forced_cuts={estimates['forced_cuts']}，"
          f"empty_drugs={estimates['empty_drugs']}")
    print("模式：", args.mode)
    print("检查点：", args.state.resolve())
    print("报告：", report.resolve())
    print("说明：后续不扩大固定药品范围；实际匹配结果以本机 Neo4j 为准。")
    print("本工具不修改原有知识库切分、Hybrid/RRF/Reranker 或 GraphRAG 代码。")

    argv_importer = build_bulk_args(
        manifest=manifest, mode=args.mode,
        json_path=args.json, state=args.state, report=report,
        allow_cloud_embedding=args.allow_cloud_embedding,
        retry_failed=args.retry_failed,
        max_new_chunks=args.max_new_chunks,
        max_embed_chars=args.max_embed_chars,
    )
    from medical_json_bulk_index import main as import_main
    result = import_main(argv_importer)
    if isinstance(result, dict):
        if int(result.get("failed_drugs") or 0):
            print(
                "本批存在核验/写入失败记录。"
                "后续批次可以先暂停，并查看报告的失败详情。"
            )
            return 2
        if result.get("stopped_by_budget"):
            print(
                "本批因预算上限提前停止，部分药品尚未执行；"
                "重复本批命令不会选到清单之外的药品。"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
