# -*- coding: utf-8 -*-
"""遍历 RAG知识库 目录，批量清洗所有 Markdown 文档中的数学公式与乱码"""
from pathlib import Path
import re
import sys

# 核心逻辑完全复用你现有的 _fix_math.py 脚本
# 注意：_fix_math.py 必须位于当前的项目根目录下
sys.path.append(".")
from _fix_math import compact_math, repl_display, repl_inline

# ==========================================
# 以下是专门为遍历目录新增的逻辑
# ==========================================

KB_DIR = Path(r"../RAG知识库")  # 目标目录
R = "\ufffd"

def clean_md_file(file_path: Path):
    """
    复用现有逻辑，清洗单个 Markdown 文件。
    返回该文件处理的统计信息。
    """
    global n_eq, n_inline, n_phrase
    
    text = file_path.read_text(encoding="utf-8")
    original_text = text

    # 1. 处理 <eq>...</eq> 标签
    n_eq = 0
    def repl_eq(m):
        global n_eq
        n_eq += 1
        return "$" + compact_math(m.group(1)) + "$"
    text = re.sub(r"<eq>(.*?)</eq>", repl_eq, text, flags=re.S)

    # 2. 处理行间公式 $$...$$
    n_disp = len(re.findall(r"\$\$(.*?)\$\$", text, flags=re.S))
    text = re.sub(r"\$\$(.*?)\$\$", repl_display, text, flags=re.S)

    # 3. 处理行内公式 $...$
    parts = []
    i = 0
    n_inline = 0
    n = len(text)
    while i < n:
        if text.startswith("$$", i):
            j = text.find("$$", i + 2)
            if j == -1:
                parts.append(text[i:])
                break
            parts.append(text[i : j + 2])
            i = j + 2
            continue
        if text[i] == "$":
            j = text.find("$", i + 1)
            if j == -1:
                parts.append(text[i:])
                break
            inner = text[i + 1 : j]
            if "\n" in inner and inner.count("\n") > 4:
                parts.append(text[i])
                i += 1
                continue
            n_inline += 1
            parts.append("$" + compact_math(inner) + "$")
            i = j + 1
            continue
        parts.append(text[i])
        i += 1
    text = "".join(parts)

    # 4. 粗体向量标记恢复
    text = re.sub(r"\$_\s*\{\s*([A-Za-z])\s*\}\s*\$", r"$\\boldsymbol{\1}$", text)
    text = re.sub(r"\$_\s*([A-Za-z])\s*\$", r"$\\boldsymbol{\1}$", text)

    # 5. 替换丢失的替换字符（复用原脚本的映射表）
    phrase_repls = [
        (f"ℝ<sup>|{R}|</sup>", r"$\mathbb{R}^{|\mathcal{V}|}$"),
        (f"R<sup>|{R}|</sup>", r"$\mathbb{R}^{|\mathcal{V}|}$"),
        (f"ℝ<sup>{R}</sup>", r"$\mathbb{R}^{D}$"),
        (f"输入空间{R}和输出空间{R}", r"输入空间$\mathcal{X}$和输出空间$\mathcal{Y}$"),
        (f"其中{R}(⋅)为指示函数", r"其中$I(\cdot)$为指示函数"),
        (f"其中{R}(⋅)是指示函数", r"其中$I(\cdot)$是指示函数"),
        (f"第{R}层神经元", r"第$l$层神经元"),
        (f"第{R}层的", r"第$l$层的"),
        (f"第{R}层神", r"第$l$层神"),
        (f"第{R}次迭代", r"第$t$次迭代"),
        (f"第{R}次更新", r"第$t$次更新"),
        (f"第{R}时刻", r"第$t$时刻"),
        (f"第{R}步时", r"第$t$步时"),
        (f"第{R}步的", r"第$t$步的"),
        (f"第{R}−1", r"第$l-1$"),
        (f"第{R}-1", r"第$l-1$"),
        (f"第{R}+1", r"第$l+1$"),
        (f"第{R} +1", r"第$l+1$"),
        (f"第{R} + 1", r"第$l+1$"),
        (f"训练集{R}", r"训练集$\mathcal{D}$"),
        (f"{R}个样本", r"$N$个样本"),
        (f"有{R}个样", r"有$N$个样"),
        (f"学习算法{R}", r"学习算法$\mathcal{A}$"),
        (f"词表{R}", r"词表$\mathcal{V}$"),
        (f"在{R}分类", r"在$C$分类"),
        (f"一个{R}维向量", r"一个$D$维向量"),
        (f"个{R}维向量", r"个$D$维向量"),
        (f"个{R}维的", r"个$D$维的"),
        (f"个{R}维随", r"个$K$维随"),
        (f"是{R}维的", r"是$D$维的"),
        (f"|{R}|维", r"$|\mathcal{V}|$维"),
        (f"参数为{R}的函数", r"参数为$\theta$的函数"),
        (f"{R}为参数的数量", r"$D$为参数的数量"),
        (f"权重向量{R}和偏置{R}", r"权重向量$\boldsymbol{w}$和偏置$b$"),
        (f"{R}(⋅)", r"$I(\cdot)$"),
    ]
    
    n_phrase = 0
    for a, b in phrase_repls:
        c = text.count(a)
        if c:
            text = text.replace(a, b)
            n_phrase += c

    # 只有内容发生变化时才覆写保存文件
    if text != original_text:
        file_path.write_text(text, encoding="utf-8")
        return True, n_eq, n_disp, n_inline, n_phrase, text.count(R)
    return False, 0, 0, 0, 0, text.count(R)


def main():
    if not KB_DIR.exists():
        print(f"[错误] 找不到知识库目录: {KB_DIR.absolute()}")
        return

    print(f"[*] 正在扫描目录: {KB_DIR.absolute()}")
    files = list(KB_DIR.rglob("*.md"))
    print(f"[*] 发现 {len(files)} 个 Markdown 文件\n")

    total_changes = 0
    for f in files:
        changed, n1, n2, n3, n4, left = clean_md_file(f)
        if changed:
            total_changes += 1
            print(f"[已更新] {f.name} (eq={n1}, display={n2}, inline={n3}, phrases={n4}, 残留乱码={left})")
        else:
            print(f"[无需修改] {f.name} (残留乱码={left})")

    print(f"\n{'='*40}")
    print(f"[*] 处理完毕！共更新了 {total_changes} / {len(files)} 个文件。")


if __name__ == "__main__":
    main()