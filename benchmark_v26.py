# -*- coding: utf-8 -*-
"""豆馅 V2.6 跨问题回归基准：不修改任何项目源文件。

安全默认：只列出用例；只有显式 --run 配合 --case/--all 才会调用
真实 DashScope/Qwen 和 Reranker（可能产生计费）。
"""
import argparse
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import uuid


@dataclass(frozen=True)
class Case:
    id: str
    topic: str
    question: str
    review_points: tuple[str, ...]
    negative_control: bool = False


CASES = (
    Case(
        "B01", "概率图模型 / 基线",
        "贝叶斯网有哪些精确推断和近似推断方法？它们分别采用什么基本思路？",
        (
            "精确推断与近似推断都覆盖，并区分变量消去/信念传播/MCMC/变分推断",
            "Gibbs 应是 MCMC 的具体方法，而非平级大类",
            "不将 MCMC 样本表述为必定相互独立",
            "有引文对应的真实知识库章节，逐句检查引用支持关系",
        ),
    ),
    Case(
        "B02", "集成学习 / 方法关系",
        "Bagging与随机森林在训练方式和随机性上有什么关系？随机森林相对Bagging增加了哪一项随机选择？",
        (
            "随机森林与 Bagging 的联系表述准确",
            "解释随机森林引入随机特征或属性选择的机制",
            "不把 Bagging、随机森林及其子方法混为同一概念",
            "所有机制说明都有实际原文支持，缺乏证据时应明确说不足",
        ),
    ),
    Case(
        "B03", "神经网络 / 计算机制",
        "神经网络中的反向传播算法如何利用链式法则计算各层参数的梯度？请说明基本步骤。",
        (
            "解释梯度从输出端向前层传播以及链式法则如何发挥作用",
            "区分梯度计算与优化器更新参数这两件事",
            "不生成教材没有介绍的具体公式或实验数据",
            "确认引文支持所列计算步骤；若资料不足应说明",
        ),
    ),
    Case(
        "B04", "卷积网络 / 结构与作用",
        "卷积神经网络中，卷积层与池化层分别完成什么运算？它们怎样影响特征表示？",
        (
            "卷积层提取特征的操作与池化层汇聚/下采样的操作分开说明",
            "不把池化说成必然提升准确率或必然防止过拟合",
            "各项操作与结果的因果描述有教材原文支持",
            "若当前资料没有相关章节，不应凭模型常识冒充知识库证据",
        ),
    ),
    Case(
        "B05", "负例 / 证据缺失处理",
        "请仅根据当前知识库给出豆馅项目在2026年10月20日的正式基准测试准确率和实验编号；没有原始记录请明确说明证据不足，不要推测。",
        (
            "检查当前知识库是否真的含有被引用的该日期项目测试报告",
            "缺乏原始记录时应明确答复无法确定，不能编造百分比或实验编号",
            "即使检索到相关算法章节，也不能用无关引文支持测试成绩",
        ),
        negative_control=True,
    ),
)

CONFIG = {
    "DOUXIAN_KB_MODE": "fast",
    "DOUXIAN_GUARD_FAST": "0",
    "DOUXIAN_FAST_QUERY_PLAN": "1",
    "RERANK_DOC_CHARS": "600",
    "DOUXIAN_FAST_QUALITY_PROMPT": "1",
}

# 40 秒是用户在 Streamlit 中期望的上限；CLI 数值不包括界面渲染与其额外开销。
TARGET_SECONDS = 40.0


def list_cases() -> None:
    print("豆馅 V2.6 回归题目（仅浏览，不调用 API）：")
    for item in CASES:
        kind = "负例" if item.negative_control else "常规"
        print(f"\n  [{item.id}] {item.topic} | {kind}\n  {item.question}")
        for check in item.review_points:
            print(f"    - {check}")


def citation_ids(text: str) -> list[int]:
    # 过滤自动生成的参考资料列表，避免将 footer 视为正文的真实引文。
    body = text.split("\n参考资料：", 1)[0]
    return sorted({int(value) for value in re.findall(r"\[(\d+)\]", body)})


def extract_signals(answer: str, trace: dict, logs: str, time_s: float) -> dict:
    evidence = trace.get("knowledge_evidence")
    evidence = evidence if isinstance(evidence, list) else []
    evidence_ids = {
        int(doc["id"]) for doc in evidence
        if isinstance(doc, dict) and str(doc.get("id", "")).isdigit()
    }
    refs = set(citation_ids(answer))
    quality = trace.get("fast_quality_check")
    quality = quality if isinstance(quality, dict) else {}
    tokens = [int(n) for n in re.findall(
        r"\[Rerank成本\][^\n]*官方total_tokens=(\d+)", logs
    )]
    poor_evidence_words = (
        "证据不足", "无法确定", "无法可靠", "未提供", "未找到",
        "没有相关", "没有明确", "未包含", "缺少", "无法核实",
        "没有检索到", "不能确认", "没有记录", "无法从",
    )
    return {
        "elapsed_under_40s": time_s <= TARGET_SECONDS,
        "evidence_count": len(evidence),
        "reference_ids_in_body": sorted(refs),
        "valid_reference_ids": sorted(refs & evidence_ids),
        "unknown_reference_ids": sorted(refs - evidence_ids),
        "quality_rules_status": quality.get("status", "not_available"),
        "quality_rules_warnings": quality.get("warnings", []),
        "rerank_total_tokens": sum(tokens) if tokens else None,
        "rerank_calls_with_official_usage": len(tokens),
        "python_fast_planner_hit": "[Query Planner] fast_path=on" in logs,
        "answer_states_insufficient_evidence": any(
            keyword in answer for keyword in poor_evidence_words
        ),
        "human_fact_check": "待人工核验",
        "human_citation_support_check": "待人工核验",
    }


def safe_md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def make_report(config: dict, data: list[dict], startup_sec: float) -> str:
    output = [
        "# 豆馅 V2.6：五题回归报告（待人工审核）",
        "",
        "**报告中的规则检查、合法引用编号不等于事实或引用充分性审核。**",
        "CLI 耗时是问答链计时，不包含 Streamlit 界面渲染；",
        "本脚本对 40 秒的比较仅作为初步指标。",
        "",
        f"RagService 启动：{startup_sec:.2f}s（不计入单题耗时）",
        "",
        "## 本次配置",
    ]
    for key, val in config.items():
        output.append(f"- `{key}={val}`")
    output.extend([
        "",
        "## 自动记录的指标",
        "",
        "| 题号 | 问答时间(s) | 首字(s) | 40秒内 | 证据数 | 有效引用数 | 无效编号数 | Rerank Tokens | 规则提醒 |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---|",
    ])
    for item in data:
        signals = item.get("signals", {})
        status = item.get("status")
        if status != "completed":
            output.append(f"| {item['id']} | 失败 | — | — | — | — | — | — | {safe_md(item.get('error',''))} |")
            continue
        quality = signals.get("quality_rules_status", "not_available")
        warnings = len(signals.get("quality_rules_warnings", []))
        tokens = signals.get("rerank_total_tokens")
        output.append(
            f"| {item['id']} | {item['elapsed_seconds']:.2f} | "
            f"{item['first_char_seconds']:.2f} | "
            f"{'是' if signals['elapsed_under_40s'] else '否'} | "
            f"{signals['evidence_count']} | "
            f"{len(signals['valid_reference_ids'])} | "
            f"{len(signals['unknown_reference_ids'])} | "
            f"{tokens if tokens is not None else '未返回'} | {quality} ({warnings}) |"
        )
    output.extend([
        "",
        "## 人工逐题核验（非常重要）",
        "",
        "每一题请打开同名 `.json` 的 `evidence` 和 `answer`，",
        "按下面的问题检查。未读原始 Chunk 前，不应将任何题判为“证据支持”。",
    ])
    for item in data:
        output.extend([
            "",
            f"### {item['id']}｜{item['topic']}",
            "",
            f"> {item['question']}",
            "",
        ])
        case = next(x for x in CASES if x.id == item["id"])
        for note in case.review_points:
            output.append(f"- [ ] {note}")
        output.extend([
            "- [ ] 每条关键论断与实际原文一致（逐个引用核对）",
            "- [ ] 该题本轮检索的资料确实足够回答（不足时允许明确拒答）",
            "- [ ] 答案清楚、没有无依据结论或张冠李戴的引用",
            f"- [ ] CLI 耗时不超过 {TARGET_SECONDS:.0f} 秒（并用 Streamlit 复核用户端体感）",
            "- **人工结论：待定（合格／需改进／证据缺失，三选一）**",
        ])
        if item["id"] == "B05":
            output.append("  - 负例：必须检查是否编造了不存在的测试百分比或实验编号。")
    output.extend([
        "",
        "## 测试解读注意事项",
        "",
        "1. 首题可能包含 BM25 建索引、Chroma 初始化等冷启动成本；后续通常是热运行。",
        "2. V2.3 Python 查询快速路径只覆盖匹配规则的问法，其他题可能回落到 Qwen 查询分解，耗时会更长。",
        "3. 改写、Rerank 和答案生成均可能产生变化；不要仅用一次运行判断所有问题的稳定性。",
        "4. 规则 `pass` 只是没有匹配到已知问题，不是事实验证成功。",
        "5. 负例能否识别证据缺失，尤其值得检查：快速模式不会启动 Evidence Guard。",
        "6. 若其中一题证据不足，优先诊断召回数据和原文，不要要求模型凭常识补齐。",
        "7. 本地结果文件可能包含知识库原文，请勿公开发布含内部材料的完整日志。",
        "",
    ])
    return "\n".join(output)


def run_cases(selected: list[Case]) -> Path:
    root = Path(__file__).resolve().parent
    rag_source = root / "rag.py"
    if not rag_source.exists():
        raise RuntimeError("没有找到同目录的 rag.py；请把脚本放入 data 项目目录。")
    code = rag_source.read_text(encoding="utf-8")
    if not all(x in code for x in ("DOUXIAN_FAST_QUALITY_PROMPT",
                                   "inspect_fast_kb_answer")):
        raise RuntimeError("rag.py 不是预期的 V2.6 版本，请先确认版本。")

    for name, value in CONFIG.items():
        os.environ[name] = value  # 仅当前 Python 测试进程，不修改 .env 或系统变量。

    output_dir = root / "benchmark_v26_results" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_"
        + uuid.uuid4().hex[:6]
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    print("\n即将调用真实 Qwen/DashScope API，可能产生 Token 费用。")
    print("本次仅测试", ", ".join(x.id for x in selected))
    for k,v in CONFIG.items():
        print(" ",k,"=",v)
    print("结果目录：", output_dir)
    print("\n[RagService] 初始化...")
    start_init = time.perf_counter()
    from rag import RagService  # 只有显式 --run 时才导入可能产生 API 操作的应用模块。
    service = RagService()
    startup = time.perf_counter() - start_init
    print(f"初始化完成：{startup:.2f}s")

    results = []
    for index, case in enumerate(selected, 1):
        session_id = "benchmark-v26-" + uuid.uuid4().hex
        trace: dict = {"version": 1, "events": [], "sources": []}
        answer_parts = []
        first_char_time = None
        started = time.perf_counter()
        logfile = output_dir / f"{case.id}.log"
        error = None
        print(f"\n[{index}/{len(selected)}] {case.id}: {case.topic} 开始检索和生成...")
        # 原始日志可能包含完整检索 Chunk；写入本地，不在终端刷屏。
        with logfile.open("w", encoding="utf-8") as output_log:
            try:
                with contextlib.redirect_stdout(output_log), \
                     contextlib.redirect_stderr(output_log):
                    for chunk in service.chain.stream(
                        {
                            "input": case.question,
                            "mode": "知识库",
                            "history": [],  # 每题新会话，避免会话记忆污染。
                            "trace": trace,
                        },
                        {"configurable": {"session_id": session_id}},
                    ):
                        if not isinstance(chunk, str):
                            chunk = str(chunk)
                        if first_char_time is None and chunk.strip():
                            first_char_time = time.perf_counter() - started
                        answer_parts.append(chunk)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc(file=output_log)

        elapsed = time.perf_counter() - started
        answer = "".join(answer_parts)
        log_text = logfile.read_text(encoding="utf-8")
        result = {
            "id": case.id,
            "topic": case.topic,
            "question": case.question,
            "negative_control": case.negative_control,
            "review_points": list(case.review_points),
            "status": "failed" if error else "completed",
            "error": error,
            "cold_search_run": index == 1,
            "elapsed_seconds": round(elapsed, 3),
            "first_char_seconds": round(first_char_time or elapsed, 3),
            "answer": answer,
            "evidence": trace.get("knowledge_evidence") or [],
            "trace": trace,
            "signals": extract_signals(answer, trace, log_text, elapsed),
            "log_file": logfile.name,
        }
        (output_dir / f"{case.id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(result)
        signal = result["signals"]
        print(
            f"  {'OK' if not error else 'ERROR'} | "
            f"{elapsed:.2f}s（首字 {result['first_char_seconds']:.2f}s） | "
            f"证据 {signal['evidence_count']} 条 | "
            f"Rerank 官方 Tokens {signal['rerank_total_tokens']} | "
            f"规则={signal['quality_rules_status']}"
        )
        if error:
            print(f"  异常：{error}；完整细节见 {logfile.name}")

    (output_dir / "report.md").write_text(
        make_report(CONFIG, results, startup), encoding="utf-8"
    )
    (output_dir / "results_summary.json").write_text(
        json.dumps({
            "config": CONFIG,
            "startup_seconds": startup,
            "metrics_scope": "CLI RAG chain, excludes Streamlit UI",
            "cases": [
                {key: value for key,value in x.items()
                 if key not in ("answer", "evidence", "trace")}
                for x in results
            ],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n测试完成。报告：", output_dir / "report.md")
    print("可把 results_summary.json 和 report.md 发给我分析；")
    print("如果要复查引文支持，请在检查不涉敏后再提供相关 Bxx.json。")
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="列出全部题目（无需模型）")
    parser.add_argument("--run", action="store_true", help="调用真实模型进行测试，产生费用")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--all", action="store_true", help="执行全部 5 题")
    selector.add_argument("--case", action="append", metavar="B01",
                          help="执行某个案例，可重复指定")
    args = parser.parse_args(argv)
    if args.list or not args.run:
        list_cases()
        if args.all or args.case:
            print("\n当前仅预览；加 --run 才会真正执行并产生 API 费用。")
        return 0
    by_id = {case.id: case for case in CASES}
    if not args.all and not args.case:
        parser.error("为避免意外计费，请明确使用 --run --case B01 或 --run --all")
    case_ids = list(by_id) if args.all else list(dict.fromkeys(args.case))
    for case_id in case_ids:
        if case_id not in by_id:
            parser.error(f"未知 case: {case_id}；有效值：{', '.join(by_id)}")
    try:
        run_cases([by_id[case_id] for case_id in case_ids])
    except Exception as exc:
        print("基准运行失败：", type(exc).__name__, str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
