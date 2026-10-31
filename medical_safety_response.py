# -*- coding: utf-8 -*-
"""儿童自行用药问题的一般安全提示（与数据库事实引用明确分离）。

不判断具体患者的适用性、剂量、疗程或监管批准状态。
只在用户明确询问儿童自行服药，且本轮药品证据出现抗菌药类别时触发。
"""
from __future__ import annotations

import re
from typing import Any, Iterable

_CHILD = re.compile(r"儿童|小儿|婴幼儿|婴儿|孩子|宝宝|新生儿")
_SELF_MED = re.compile(
    r"自己.{0,10}(?:服|用|吃)|自行.{0,10}(?:服|用|吃)|"
    r"未经.{0,10}(?:医嘱|医生|医师)|"
    r"无.{0,6}(?:医生|医师).{0,6}指导|"
    r"不(?:用|必).{0,6}(?:看医生|找医生)"
)
_MEDICINE = re.compile(r"服|药|吃|注射")
_ANTIBIOTIC = re.compile(r"抗生素|抗菌药|抗菌素|大环内酯")


def is_selfmedication_aspect(text: str) -> bool:
    """是否实际讨论自行/未经医生指导服药，而非仅谈剂量或适应症。"""
    return bool(_SELF_MED.search(str(text or "")) and _MEDICINE.search(str(text or "")))


def pediatric_antibiotic_selfmedication_notice(
    question: str, evidence: Iterable[dict[str, Any]]
) -> str:
    """安全建议是一般指导，不能冒充知识库引用或医师的个体处方。"""
    question = str(question or "")
    if not (_CHILD.search(question) and is_selfmedication_aspect(question)):
        return ""

    # 有药品名称和实际文本时才判断；不把仅凭疾病词推测出的药物当抗菌药。
    if not any(
        isinstance(item, dict)
        and str(item.get("drug_name") or "").strip()
        and _ANTIBIOTIC.search(str(item.get("content") or ""))
        for item in (evidence or [])
    ):
        return ""

    return (
        "**用药安全提醒（一般性建议，不是本轮药品说明书的引文）："
        "不建议在没有医生评估和指导的情况下自行给儿童使用抗菌药。**"
        "是否需要抗菌治疗，应由医生根据病因、年龄、体重、过敏史等判断。"
        "即使资料中列出了小儿剂量，也不能据此判断某个孩子适合用药，"
        "更不能将该剂量直接作为个人用药方案。"
    )
