
"""将 PDF 转为 Markdown，保留原始 PDF 文件。"""

from pathlib import Path
import argparse
import pymupdf4llm


def convert_pdf_to_md(
    pdf_path: str,
    output_path: str | None = None,
) -> Path:

    source = Path(pdf_path).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"PDF 文件不存在：{source}")

    if source.suffix.lower() != ".pdf":
        raise ValueError("输入文件必须是 PDF")

    # 默认生成与 PDF 同名的 Markdown 文件
    target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else source.with_suffix(".md")
    )

    if target == source:
        raise ValueError("输出文件不能覆盖原始 PDF")

    # 核心：PDF -> Markdown
    markdown = pymupdf4llm.to_markdown(str(source))

    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError(
            "未提取到有效文字，请检查是否为扫描版 PDF 或需要 OCR"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")

    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 PDF 转换成 Markdown"
    )

    parser.add_argument("pdf", help="PDF 文件路径")
    parser.add_argument(
        "-o", "--output",
        help="输出 .md 文件路径（可选）"
    )

    args = parser.parse_args()

    output = convert_pdf_to_md(args.pdf, args.output)
    print(f"转换成功：{output}")