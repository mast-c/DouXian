"""给本地 rag.py 增量安装 Evidence Sufficiency V1。

特点：
- 不覆盖或重置你已有的提示词、查询分解、重排配置。
- 修改前保存一个时间戳备份。
- 找不到预期代码锚点就停止，不强行修改。
- 可用 --dry-run 先检查兼容性。
"""

import argparse
import ast
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


IMPORT_MARK = "from evidence_guard import assess_evidence\n"
BEGIN = "# BEGIN EVIDENCE_GUARD_V1"
END = "# END EVIDENCE_GUARD_V1"


def replace_once(source: str, old: str, new: str, desc: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"无法定位 {desc}（匹配 {count} 次），文件未修改。")
    return source.replace(old, new, 1)


def patch_rag(source: str) -> tuple[str, bool]:
    if BEGIN in source:
        if END not in source or IMPORT_MARK not in source:
            raise RuntimeError("发现不完整的 Evidence Guard 标记，请人工检查。")
        ast.parse(source)
        return source, False

    mode = '            elif mode == "知识库":\n'
    mode_start = source.find(mode)
    if mode_start < 0:
        raise RuntimeError("没有找到知识库模式，文件未修改。")
    end_marker = '            else:\n                messages = ['
    mode_end = source.find(end_marker, mode_start)
    if mode_end < 0:
        raise RuntimeError("没有找到知识库模式末尾，文件未修改。")

    head = source[:mode_start]
    section = source[mode_start:mode_end]
    tail = source[mode_end:]

    anchor = '                        kb_text = "\\n\\n".join(evidence)\n'

    added_guard = """                        # BEGIN EVIDENCE_GUARD_V1
                        # 先审核原始问题的每个方面；子查询仅作为检索线索。
                        guard_excerpts = []
                        for original_item in excerpts:
                            if not isinstance(original_item, dict):
                                continue
                            original_content = original_item.get("content")
                            if not isinstance(original_content, str) or not original_content.strip():
                                continue
                            guard_excerpts.append({
                                "id": len(guard_excerpts) + 1,
                                "source": original_item.get("source", ""),
                                "section_path": original_item.get("section_path", ""),
                                "content": original_content,
                            })

                        coverage = assess_evidence(
                            model=self.chat_model,
                            question=question,
                            excerpts=guard_excerpts,
                            retrieval_queries=kb_data.get("subqueries"),
                        )
                        trace["evidence_coverage"] = coverage
                        print(
                            "\\n========== 证据充分性校验 ==========",
                            coverage["decision"],
                        )
                        for aspect in coverage["aspects"]:
                            print(
                                aspect["status"],
                                aspect["question"],
                                aspect["evidence_ids"],
                            )

                        if coverage["decision"] in ("refuse", "error"):
                            if coverage["decision"] == "error":
                                reply = (
                                    "证据充分性校验暂未完成，为避免生成无依据回答，"
                                    "本次暂不作答，请稍后重试。"
                                )
                            else:
                                reply = (
                                    "当前检索到的知识库证据不足，"
                                    "无法可靠回答该问题。"
                                )
                            answer_parts.append(reply)
                            yield reply
                            finished = True
                            return

                        # 硬约束：生成模型只能看到已验证引文支持的 Chunk。
                        accepted_ids = set(coverage["evidence_ids"])
                        evidence = [
                            text for i, text in enumerate(evidence, 1)
                            if i in accepted_ids
                        ]
                        kb_text = "\\n\\n".join(evidence)

                        # 只要求回答真正被用户询问、且审核通过的方面；
                        # 背景检索子查询不能变成新的必答题。
                        approved = [
                            ("需纠正错误前提：" if a["status"] == "refuted" else "")
                            + a["question"]
                            for a in coverage["aspects"]
                            if a["status"] in ("supported", "refuted")
                        ]
                        coverage_prompt = (
                            "本轮已通过程序级证据预检查。仅回答以下方面：\\n"
                            + "\\n".join("- " + q for q in approved)
                            + "\\n仅使用审核通过的证据编号："
                            + ", ".join(f"[{i}]" for i in coverage["evidence_ids"])
                            + "。不要回答无证据的方面，也不要靠常识补充事实。"
                            + "如果用户前提被证据反驳，请明确纠正。"
                        )
                        # END EVIDENCE_GUARD_V1
"""

    section = replace_once(
        section, anchor, anchor + added_guard,
        "知识库正文证据准备处",
    )

    # 追加在原有同一条 SystemMessage 中，避免模型接口对多个 system 消息的兼容差异。
    section = replace_once(
        section,
        '                            """\n                            ),\n                            *history,\n',
        '                            """ + "\\n" + coverage_prompt\n'
        '                            ),\n                            *history,\n',
        "知识库原有的 SystemMessage 结尾",
    )

    refusal_insert = """                        # 对无证据的方面追加确定性部分拒答，避免仅依赖生成模型。
                        if coverage["decision"] == "partial":
                            refused_note = (
                                "\\n\\n**当前知识库证据不足，暂无法可靠回答：**\\n"
                                + "\\n".join(
                                    "- " + q
                                    for q in coverage["refused_questions"]
                                )
                            )
                            answer_parts.append(refused_note)
                            yield refused_note

"""
    section = replace_once(
        section,
        "                        # 3. 模型回答结束后，只生成一次参考资料\n",
        refusal_insert
        + "                        # 3. 模型回答结束后，只生成一次参考资料\n",
        "流式生成后的参考资料位置",
    )

    patched = head + section + tail
    if IMPORT_MARK not in patched:
        patched = replace_once(
            patched,
            "from knowledge_base import KnowledgeBaseService\n",
            "from knowledge_base import KnowledgeBaseService\n"
            + IMPORT_MARK,
            "KnowledgeBaseService 导入",
        )

    ast.parse(patched, filename="rag.py")
    return patched, True


def main() -> None:
    cli = argparse.ArgumentParser(description="豆馅知识库证据校验增量安装")
    cli.add_argument(
        "--path", type=Path, default=Path("rag.py"),
        help="需要修改的 rag.py（默认当前目录）",
    )
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args()

    target = args.path.resolve()
    module = Path(__file__).resolve().parent / "evidence_guard.py"
    if not target.is_file():
        raise SystemExit(f"找不到 {target}")
    if not module.is_file():
        raise SystemExit("请先把 evidence_guard.py 复制到本脚本同一目录。")

    source = target.read_text(encoding="utf-8")
    try:
        output, changed = patch_rag(source)
    except (RuntimeError, SyntaxError) as exc:
        raise SystemExit(str(exc))

    if not changed:
        print("✅ 已经安装过 Evidence Guard，无需重复修改。")
        return

    if args.dry_run:
        print("✅ dry-run 通过：检测到 3 个集成位置且 Python 语法正确。")
        print("未写入文件，也没有调用 Embedding 或 Qwen。")
        return

    backup = target.with_name(
        target.name + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    shutil.copy2(target, backup)

    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8",
            dir=str(target.parent), suffix=".tmp",
            delete=False,
        ) as temporary:
            tmp_name = temporary.name
            temporary.write(output)

        os.replace(tmp_name, target)
    finally:
        if tmp_name and Path(tmp_name).exists():
            Path(tmp_name).unlink()

    print("✅ 已安装：", target)
    print("✅ 原版已备份：", backup)
    print("不会修改知识库索引，不会重新 Embedding。")


if __name__ == "__main__":
    main()
