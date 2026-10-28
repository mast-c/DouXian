# -*- coding: utf-8 -*-
"""双方面问题规划：可控的双层结构、双方面问题快速检索规划。

只识别明确的“卷积层 + 池化/汇聚层，分别完成什么运算、怎样影响特征表示”
句式；不根据模型常识补充算法机制、不接触 DashScope、不改 RAG 正文。
不匹配返回 []，由原有 Qwen 查询分解继续处理。
"""
import re


_COMPONENTS = ("卷积层", "池化层", "汇聚层")
_EFFECTS = ("特征表示", "特征图", "特征表达", "特征的表示")

# fullmatch 防止截掉用户额外的约束或任务；仅支持这一类已定义的双方面问法。
_PATTERN = re.compile(
    r"\A"
    r"(?:(?:请问|请说明|请解释|请介绍)(?:一下)?[，,:：]?)?"
    r"(?:(?:在)?(?P<scope>卷积神经网络|CNN)(?:中|内|里)?[，,:：]?)?"
    r"(?P<first>卷积层|池化层|汇聚层)"
    r"(?:和|与|及|、)"
    r"(?P<second>卷积层|池化层|汇聚层)"
    r"(?:分别|各自)"
    r"(?:完成|执行|进行|做)"
    r"(?:(?:什么|哪些)(?:运算|操作|计算)|"
    r"(?:什么|哪些)(?:计算步骤|操作步骤))"
    r"[？?，,；;。]*"
    r"(?:(?:它们|二者|两者|这两层)(?:分别)?|各自|分别)?"
    r"(?:"
    r"(?:怎样|如何|会怎样|会如何)(?:影响|改变)"
    r"(?P<effect1>特征表示|特征图|特征表达|特征的表示)"
    r"|"
    r"对(?P<effect2>特征表示|特征图|特征表达|特征的表示)"
    r"(?:有何|有什么|产生什么|造成什么)(?:影响|作用|变化)"
    r")"
    r"[？?！!。]*"
    r"\Z"
)


def plan_two_layer_aspects(question: str) -> list[str]:
    """返回四条检索子问题或 []，不进行外部调用。

    仅复述用户已经明确要求的“运算”和“特征影响”两个方面。
    不加入池化类型、卷积公式、感受野等用户未指明的机制或结论。
    """
    if not isinstance(question, str):
        return []
    compact = re.sub(r"\s+", "", question).strip()
    if not 15 <= len(compact) <= 140:
        return []
    match = _PATTERN.fullmatch(compact)
    if not match:
        return []

    first, second = match.group("first"), match.group("second")
    if first not in _COMPONENTS or second not in _COMPONENTS:
        return []
    # 两个节点应分别是卷积和池化；不能将“池化层/汇聚层”错认成两个独立概念。
    if "卷积层" not in (first, second):
        return []
    if not ({"池化层", "汇聚层"} & {first, second}):
        return []
    effect = match.group("effect1") or match.group("effect2")
    if effect not in _EFFECTS:
        return []
    scope = "卷积神经网络中，" if match.group("scope") else ""

    # 固定成 4 次检索（与 B04 原有 Qwen 规划的 4 个子问题数量相同）。
    return [
        f"{scope}{first}完成什么运算？",
        f"{scope}{first}怎样影响{effect}？",
        f"{scope}{second}完成什么运算？",
        f"{scope}{second}怎样影响{effect}？",
    ]
