# -*- coding: utf-8 -*-
"""Clean MinerU-extracted LaTeX and recover high-confidence OCR math glyphs."""
from pathlib import Path
import re

path = Path(r"e:\VsCode项目\DockerTest_Agent\RAG项目\RAG知识库\神经网络与深度学习.md")
text = path.read_text(encoding="utf-8")
R = "\ufffd"


def collapse_spaced_ident(inner: str) -> str:
    s = inner.strip()
    if re.fullmatch(r"[A-Za-z](?:[ \t]+[A-Za-z])+", s):
        return s.replace(" ", "").replace("\t", "")
    return inner


def compact_math(s: str) -> str:
    s = s.replace("\r\n", "\n")
    # \left / \big before delimiters
    s = re.sub(r"\\(left|right|big|Big|bigg|Bigg|bigl|bigr|Bigl|Bigr)\s+", r"\\\1", s)
    s = re.sub(r"\\([a-zA-Z]+)\s+\{", r"\\\1{", s)
    s = re.sub(r"\\([a-zA-Z]+)\s+", r"\\\1 ", s)

    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\{\s+", "{", s)
        s = re.sub(r"\s+\}", "}", s)

    s = re.sub(r"\s*_\s*", "_", s)
    s = re.sub(r"\s*\^\s*", "^", s)

    def braces(m):
        inner = m.group(1)
        collapsed = collapse_spaced_ident(inner)
        return "{" + collapsed + "}"

    s = re.sub(r"\{([^{}]+)\}", braces, s)

    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    s = re.sub(r"\\\{ ", r"\\{", s)
    s = re.sub(r" \\\}", r"\\}", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s*=\s*", " = ", s)
    s = re.sub(r"\s*&\s*", " & ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def repl_display(m):
    inner = compact_math(m.group(1))
    return "$$\n" + inner + "\n$$"


def repl_inline(m):
    inner = compact_math(m.group(1))
    return "$" + inner + "$"


# 1) <eq>...</eq> -> $...$
n_eq = 0

def repl_eq(m):
    global n_eq
    n_eq += 1
    return "$" + compact_math(m.group(1)) + "$"


text = re.sub(r"<eq>(.*?)</eq>", repl_eq, text, flags=re.S)

# 2) display math
n_disp = len(re.findall(r"\$\$(.*?)\$\$", text, flags=re.S))
text = re.sub(r"\$\$(.*?)\$\$", repl_display, text, flags=re.S)

# 3) inline math (avoid $$)
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

# 4) $_x$ / $_ y$ / $_{x}$ style bold vectors
text = re.sub(r"\$_\s*\{\s*([A-Za-z])\s*\}\s*\$", r"$\\boldsymbol{\1}$", text)
text = re.sub(r"\$_\s*([A-Za-z])\s*\$", r"$\\boldsymbol{\1}$", text)

# 5) high-confidence replacement-char recoveries in body text
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

path.write_text(text, encoding="utf-8")
left = text.count(R)
print(
    f"eq={n_eq} display={n_disp} inline={n_inline} phrases={n_phrase} fffd_left={left}"
)
