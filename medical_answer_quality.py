# -*- coding: utf-8 -*-
"""豆馅医疗问答显示整洁度工具。

职责：清理模型产生的空 Markdown 项与本地伪引用；按同一来源合并参考资料。
不修改证据正文、不推断医学关系、不把审核状态升级为医学事实。
"""
from __future__ import annotations

from collections import OrderedDict
import re
from typing import Any

_FAKE_LOCAL_LINK = re.compile(
    r"\[svg\]\(\s*https?://(?:localhost|127\.0\.0\.1)"
    r"(?::\d+)?/#[0-9]+\s*\)",
    re.IGNORECASE,
)
_EMPTY_BULLET = re.compile(r"^\s*[-*+]\s*$")
_REFERENCE_MARKERS = re.compile(r"\[([0-9]+)\]")
# LLM可能模仿历史回答，自行写出下方这种尾注；程序已有唯一的真实引用表。
# 支持“参考资料（同一来源的不同片段合并显示）”：以及 Markdown 标题。
_MODEL_FOOTER_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*{1,2})?"
    r"(?:参考资料|参考文献|引用来源)"
    r"(?:（[^）\n]{0,90}）|\([^)]{0,90}\))?"
    r"(?:\*{1,2})?\s*(?:[:：]\s*(?:\*{0,2})?|(?=\s*$))",
    re.IGNORECASE,
)


def _clean_line(line: str) -> str | None:
    """只修复可证明无意义的本地 SVG 链接与空列表行。"""
    if _EMPTY_BULLET.fullmatch(line):
        return None
    # 仅修复此前模型实际出现过的医学审核措辞错误，不改变来源文档。
    # 通过的是程序级文本核对，并非医学专家审批。
    line = line.replace(
        "根据审核通过的证据",
        "根据检索到并完成程序级原文核对的资料",
    )
    line = line.replace(
        "依据审核通过的证据",
        "依据检索到并完成程序级原文核对的资料",
    )
    return _FAKE_LOCAL_LINK.sub("", line).rstrip()


class MedicalAnswerStreamFilter:
    """流式小缓存：分行清理垃圾 Markdown，不吞掉合法的 [1] 证据编号。

    每次取得完整行立即放行。特别长的单行分段输出，以免让用户等整个
    回答生成完毕；为跨 chunk 的 [svg](localhost/#N) 保留安全缓冲区。
    """

    def __init__(self, long_line_threshold: int = 720, reserve: int = 240):
        if reserve < 140 or long_line_threshold <= reserve:
            raise ValueError("long_line_threshold 必须大于 reserve，reserve 至少 140")
        self._buffer = ""
        self._threshold = int(long_line_threshold)
        self._reserve = int(reserve)
        self._footer_detected = False

    def feed(self, piece: str) -> str:
        if not isinstance(piece, str) or not piece or self._footer_detected:
            return ""
        self._buffer += piece
        output = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if _MODEL_FOOTER_HEADING.match(line):
                # 一旦模型自己开始了参考资料块，后续正文全部丢弃。
                # rag.py 会在模型输出结束后追加且只追加一次正式引用。
                self._footer_detected = True
                self._buffer = ""
                break
            clean = _clean_line(line)
            if clean is not None:
                output.append(clean + "\n")

        # 只在未换行的超长句子中保留 240 字符尾部，保持流式输出。
        # 避免空项和 [svg](...) 恰好跨流式分块时被提前显示。
        if len(self._buffer) > self._threshold:
            # 不输出本身已经以“参考资料（...）”开头的长尾注。
            if _MODEL_FOOTER_HEADING.match(self._buffer):
                self._footer_detected = True
                self._buffer = ""
            else:
                safe_cut = len(self._buffer) - self._reserve
                prefix, self._buffer = self._buffer[:safe_cut], self._buffer[safe_cut:]
                output.append(_FAKE_LOCAL_LINK.sub("", prefix))
        return "".join(output)

    def finish(self) -> str:
        if self._footer_detected or _MODEL_FOOTER_HEADING.match(self._buffer):
            self._buffer = ""
            return ""
        last = _clean_line(self._buffer)
        self._buffer = ""
        return last if last is not None else ""


def render_compact_references(
    answer: str,
    reference_map: dict[int, tuple[str, str]],
) -> str:
    """对本轮真实存在、且正文引用到的编号生成参考资料。

    相同“来源链接/文件”只展示一次；不同引用编号仍然保留，
    让读者看清是同一份材料里的不同证据片段，而非独立核验。
    """
    cited = sorted(set(int(x) for x in _REFERENCE_MARKERS.findall(answer or "")))
    known = set(reference_map)
    valid = [x for x in cited if x in known]
    invalid = [x for x in cited if x not in known]
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for num in valid:
        display, section = reference_map[num]
        source_key = " ".join(str(display or "").split())[:2048]
        section = " ".join(str(section or "").split())[:120]
        group = grouped.setdefault(source_key, {"numbers": [], "sections": []})
        group["numbers"].append(num)
        if section and section not in group["sections"]:
            group["sections"].append(section)

    blocks = []
    if grouped:
        lines = []
        for source, values in grouped.items():
            nums = "、".join(f"[{num}]" for num in values["numbers"])
            sections = "；".join(values["sections"][:3])
            if len(values["sections"]) > 3:
                sections += " 等"
            lines.append(f"{nums} {source}" + (f" · {sections}" if sections else ""))
        blocks.append("参考资料（同一来源的不同片段合并显示）：\n\n"
                      + "\n\n".join(lines))
    if invalid:
        blocks.append("引用核对提醒："
                      + "、".join(f"[{num}]" for num in invalid)
                      + " 未对应本轮可用证据。")
    return "\n\n" + "\n\n".join(blocks) if blocks else ""


def requested_refusal_questions(question: str, refused_questions: Any) -> list[str]:
    """只展示与用户原问题实际提问维度相关的缺证据提醒。

    Evidence Guard 可能把检索阶段联想到的“机制、剂量、审批”
    当作问题缺失方面；这些不应被加到用户只问“有关联吗”的回答结尾。
    对未识别的问题类型保持原样，不强行丢失真实用户诉求。
    """
    if not isinstance(refused_questions, list):
        return []
    asked = str(question or "")
    dimension_patterns = (
        r"作用机制|药理机制|药理作用|为什么有效|如何发挥作用|作用原理|确切临床效果|临床效果",
        r"剂量|用量|给药|怎么服|怎么用|一次吃|几次服|用药方案",
        r"获批|批准|审批|注册|官方核准|监管部门",
        r"药物相互作用|联用|合用|一起吃|配伍",
        r"不良反应|副作用|不适反应",
        r"禁忌|禁用|慎用|注意事项",
    )
    triggers = (
        r"作用机制|原理|为什么|药理|临床效果|疗效",
        r"剂量|用量|给药|服用|怎么用|怎么吃|用药方案",
        r"获批|批准|审批|注册|核准|监管|官方",
        r"相互作用|联用|合用|一起吃|配伍",
        r"不良反应|副作用|不适反应",
        r"禁忌|禁用|慎用|注意事项",
    )
    selected = []
    for raw in refused_questions:
        if not isinstance(raw, str) or not raw.strip():
            continue
        item = raw.strip()
        if any(re.search(pattern, item) and not re.search(trigger, asked)
               for pattern, trigger in zip(dimension_patterns, triggers)):
            continue
        selected.append(item)
    return selected
