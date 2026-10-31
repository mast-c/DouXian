# -*- coding: utf-8 -*-
"""V1.17 阶段2：20题固定复测清单、真实 trace 完整性检查、前后人工基线对比。

只读用户指定的单轮导出与 V1.17 原始评测文件，不联网、不碰知识库，
且任何生成的报告都不能覆盖已有文件。人工标签不跨轮自动迁移。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

MAX_BYTES = 15 * 1024 * 1024


def canonical(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def load_questions(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if (not isinstance(obj, dict)
                    or not isinstance(obj.get("id"), str)
                    or not isinstance(obj.get("question"), str)
                    or not canonical(obj["question"])):
                raise ValueError(f"{path}:{line_number}: 非法题目")
            items.append(obj)
    names = [q["id"] for q in items]
    questions = [canonical(q["question"]) for q in items]
    if len(names) != len(set(names)) or len(questions) != len(set(questions)):
        raise ValueError("题号或题目重复，拒绝作为固定基线")
    return items


def plan_fingerprint(questions: list[dict]) -> str:
    raw = "\n".join(q["id"] + "\t" + canonical(q["question"]) for q in questions)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"已存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_plan(questions_file: Path, output_dir: Path) -> dict:
    questions = load_questions(questions_file)
    if len(questions) != 20:
        raise ValueError(f"正式计划必须有20道不同问题，当前只有{len(questions)}")
    if output_dir.exists():
        raise FileExistsError(f"计划目录已存在，请换一个新目录：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1, "campaign_version": "V1.17",
        "purpose": "阶段2质量回归；不是医学事实验证或官方说明书审核",
        "unique_questions": len(questions),
        "question_plan_sha256": plan_fingerprint(questions),
        "source_file": str(questions_file),
        "requirements": {
            "fresh_separate_chats": True,
            "one_trace_per_question": True,
            "do_not_reuse_old_review_labels": True,
            "no_automatic_qwen_calls": True,
        },
    }
    write_new(output_dir / "plan_manifest.json",
              json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    checklist = [
        "# V1.17 阶段2｜20题人工回归执行清单", "",
        f"问题集 SHA256：`{summary['question_plan_sha256']}`", "",
        "每题使用**新的独立会话**；完成回答后下载本轮 trace JSON。",
        "不要将旧 V1.17 trace 复制到这一轮；不要修改原始人工 CSV。", "",
        "| 进度 | ID | 测试问题 | 核验焦点 |",
        "|---|---|---|---|",
    ]
    for q in questions:
        checklist.append(
            f"| [ ] | {q['id']} | {q['question'].replace('|', '／')} "
            f"| {q.get('review_focus', '').replace('|', '／')} |"
        )
    checklist += [
        "", "## 检查标准", "",
        "- 引用编号能对应真实的本轮原文和 Section。",
        "- 未审核 DiseaseCandidate 不得升级为治疗或获批事实。",
        "- 儿童自行用药应明确非个体化的安全边界，不能直接给处方。",
        "- 区分“未检索到”“原文尚不明确”和“临床不存在风险”。",
        "- 仅在真实人工审核完成后填写 `review_status=reviewed`。", "",
    ]
    write_new(output_dir / "20题执行清单.md", "\n".join(checklist))
    return summary


def load_trace(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("不是普通文件或是符号链接")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_BYTES:
        raise ValueError("文件为空或超过15MB")
    binary = path.read_bytes()
    payload = json.loads(binary.decode("utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("is_synthetic") or payload.get("demo_warning"):
        raise ValueError("合成样本/非法数据不可进入真实评测")
    if (payload.get("schema_version") not in (None, 1)
            or not canonical(payload.get("question"))
            or not isinstance(payload.get("answer"), str)
            or not payload["answer"].strip()
            or not isinstance(payload.get("trace"), dict)
            or not isinstance(payload["trace"].get("knowledge_evidence"), list)):
        raise ValueError("不符合V1.16单轮导出结构")
    return payload, hashlib.sha256(binary).hexdigest()


def audit_traces(plan_file: Path, input_dir: Path) -> dict:
    questions = load_questions(plan_file)
    expected = {canonical(q["question"]): q["id"] for q in questions}
    if not input_dir.is_dir():
        raise FileNotFoundError(f"没有找到新批次 trace 目录：{input_dir}")

    paths = sorted(input_dir.glob("medical_trace_*.json"))
    by_id: dict[str, list[dict]] = defaultdict(list)
    unexpected, invalid, duplicate_hash = [], [], []
    seen_hashes = set()
    chroma_counts = set()
    for path in paths:
        try:
            payload, checksum = load_trace(path)
            if checksum in seen_hashes:
                duplicate_hash.append(path.name)
                continue
            seen_hashes.add(checksum)
            q = canonical(payload["question"])
            case_id = expected.get(q)
            if not case_id:
                unexpected.append({"file": path.name, "question": payload["question"]})
                continue
            trace = payload["trace"]
            pipeline = trace.get("retrieval_pipeline") or {}
            chroma = pipeline.get("chroma_hybrid") or {}
            count = chroma.get("indexed_chunks")
            if isinstance(count, int):
                chroma_counts.add(count)
            evidence = trace["knowledge_evidence"]
            by_id[case_id].append({
                "file": path.name, "sha256": checksum,
                "evidence_cards": len(evidence),
                "engineering_status": "未在此脚本评估，请执行V1.17原版collect",
                "guard_decision": (trace.get("evidence_coverage") or {}).get("decision"),
                "self_rag_status": (trace.get("self_rag_retry") or {}).get("status"),
                "safety_notice_applied": (trace.get("medical_safety_guidance") or {}).get("applied"),
                "total_seconds": trace.get("total_seconds"),
            })
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            invalid.append({"file": path.name, "reason": str(exc)[:250]})

    missing = [
        {"id": q["id"], "question": q["question"]}
        for q in questions if q["id"] not in by_id
    ]
    repeated = {
        qid: [x["file"] for x in entries]
        for qid, entries in by_id.items() if len(entries) > 1
    }
    complete = (
        not missing and not repeated and not unexpected
        and not invalid and not duplicate_hash and len(by_id) == len(questions)
    )
    return {
        "schema_version": 1,
        "status": "READY_FOR_COLLECT" if complete else "INCOMPLETE_OR_NEEDS_ATTENTION",
        "question_plan_sha256": plan_fingerprint(questions),
        "expected_unique_questions": len(questions),
        "covered_unique_questions": len(by_id),
        "missing_questions": missing,
        "repeated_question_files": repeated,
        "unexpected_questions": unexpected,
        "invalid_files": invalid,
        "duplicate_export_bytes": duplicate_hash,
        "chroma_count_snapshots_observed": sorted(chroma_counts),
        "snapshot_warning": (
            "本批 Chroma Chunk 计数不同；应记录数据库变更与评测日期"
            if len(chroma_counts) > 1 else ""
        ),
        "cases": {qid: entries for qid, entries in sorted(by_id.items())},
        "note": (
            "此脚本只核对新批次 trace 与固定题集，不评审医学内容。"
            "请继续执行 medical_quality_campaign.py collect 和人工审核/score。"
        ),
    }


def compare_reviews(old_manifest: Path, old_csv: Path,
                    new_manifest: Path, new_csv: Path) -> dict:
    # 复用 V1.17 自带的签名核验，不重复实现 SHA / 医学标签校验。
    from medical_quality_campaign import load_manifest, parse_reviews
    old_m, new_m = load_manifest(old_manifest), load_manifest(new_manifest)
    old_rows, _ = parse_reviews(old_csv, old_m)
    new_rows, _ = parse_reviews(new_csv, new_m)

    def index(manifest, rows):
        case_questions = {c["case_id"]: canonical(c["question"]) for c in manifest["cases"]}
        by_question: dict[str, dict] = {}
        for row in rows:
            question = case_questions[row["case_id"]]
            if question in by_question:
                raise ValueError("发现同一问题多次复测，请先选择唯一运行作为正式样本")
            by_question[question] = row
        return by_question

    old_by_question, new_by_question = index(old_m, old_rows), index(new_m, new_rows)
    compared, missing, awaiting_review = [], [], []
    for question, old in old_by_question.items():
        new = new_by_question.get(question)
        if not new:
            missing.append(question)
            continue
        if old["review_status"] != "reviewed" or new["review_status"] != "reviewed":
            awaiting_review.append(question)
            continue
        compared.append({
            "question": question,
            "old_source_sha256": old["source_sha256"],
            "new_source_sha256": new["source_sha256"],
            "old_engineering_status": old["engineering_status"],
            "new_engineering_status": new["engineering_status"],
            "old_evidence_support": old["evidence_support"],
            "new_evidence_support": new["evidence_support"],
            "old_relation_safety": old["relation_safety"],
            "new_relation_safety": new["relation_safety"],
            "old_medical_safety": old["medical_safety"],
            "new_medical_safety": new["medical_safety"],
        })
    safety_transitions = Counter(
        f"{case['old_medical_safety']} -> {case['new_medical_safety']}"
        for case in compared
    )
    relation_transitions = Counter(
        f"{case['old_relation_safety']} -> {case['new_relation_safety']}"
        for case in compared
    )
    return {
        "old_runs": len(old_rows), "new_runs": len(new_rows),
        "matched_reviewed_questions": len(compared),
        "missing_old_questions_in_new_run": missing,
        "not_yet_reviewed_on_both_sides": awaiting_review,
        "medical_safety_transitions": dict(safety_transitions),
        "relation_safety_transitions": dict(relation_transitions),
        "per_question": compared,
        "warning": (
            "此比较只使用两个 Manifest 与对应审核 CSV 的校验后记录；"
            "不自动复制旧人工结论，也不能代表官方医学验证。"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V1.17 阶段2：固定题集/回归完整性/人工前后比较")
    cmd = parser.add_subparsers(dest="cmd", required=True)
    prepare = cmd.add_parser("prepare", help="新建不覆盖的20题执行计划")
    prepare.add_argument("--questions", type=Path, default=Path("evaluation_questions_v117_phase2_20.jsonl"))
    prepare.add_argument("--output-dir", type=Path, required=True)
    audit = cmd.add_parser("audit", help="检查本轮20题真实trace覆盖，不调用任何模型")
    audit.add_argument("--questions", type=Path, default=Path("evaluation_questions_v117_phase2_20.jsonl"))
    audit.add_argument("--input-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    compare = cmd.add_parser("compare", help="读取旧/新人工复核表并按问题对比")
    compare.add_argument("--old-manifest", type=Path, required=True)
    compare.add_argument("--old-reviews", type=Path, required=True)
    compare.add_argument("--new-manifest", type=Path, required=True)
    compare.add_argument("--new-reviews", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "prepare":
            result = create_plan(args.questions, args.output_dir)
            print(f"[PLAN] {result['unique_questions']}道不同问题，计划已创建：{args.output_dir}")
        elif args.cmd == "audit":
            result = audit_traces(args.questions, args.input_dir)
            write_new(args.output, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(f"[TRACE] {result['covered_unique_questions']}/{result['expected_unique_questions']}")
            print(f"[STATUS] {result['status']}")
            print(f"[MISSING] {', '.join(q['id'] for q in result['missing_questions']) or 'none'}")
            print(f"[REPORT] {args.output}")
        else:
            result = compare_reviews(
                args.old_manifest, args.old_reviews, args.new_manifest, args.new_reviews
            )
            write_new(args.output, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(f"[COMPARE] 已完成两轮人工审核且可比较的问题：{result['matched_reviewed_questions']}")
            print(f"[REPORT] {args.output}")
        return 0
    except (ValueError, OSError, FileExistsError, ImportError, json.JSONDecodeError) as exc:
        parser.exit(2, f"失败：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
