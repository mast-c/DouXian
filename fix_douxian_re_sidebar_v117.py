# -*- coding: utf-8 -*-
"""P3-06.3C · Fix NameError: re, disable separate Streamlit multipage link.

By default performs a READ-ONLY dry-run. Use --apply to modify the local
豆馅.py with a one-line import and disable ONLY pages/06_Multihop_Evidence.py.

Preserves every other line and keeps timestamped backups for full rollback.
No Neo4j, Chroma, user conversation database, or rag.py modifications.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from datetime import datetime


class FixBlocked(RuntimeError):
    pass


SIDEBAR_PAGE = "06_Multihop_Evidence.py"
FEATURE_MARKERS = (
    "def trace_details_html(trace):",
    "def _mh_display_config(trace=None):",
    "def multihop_path_evidence_html(trace):",
)


def _find_patch(raw: bytes) -> tuple[bytes, str]:
    bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        source = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise FixBlocked("豆馅.py 必须是 UTF-8 编码") from exc

    if any(marker not in source for marker in FEATURE_MARKERS):
        raise FixBlocked(
            "当前豆馅.py 未找到 P3-06.3B 的完整多跳证据展示标记；"
            "为了保护你可能修改过的版本，本脚本拒绝盲目覆盖"
        )
    tree = ast.parse(source)
    # Check a real top-level import, not an occurrence inside a comment.
    has_re = any(
        isinstance(node, ast.Import)
        and any(alias.name == "re" and alias.asname is None for alias in node.names)
        for node in tree.body
    )
    if has_re:
        return raw, "IMPORT_ALREADY_PRESENT"

    # Insert after existing top-level `import os`, retaining BOM + CRLF.
    os_imports = [
        node for node in tree.body
        if isinstance(node, ast.Import)
        and any(alias.name == "os" for alias in node.names)
    ]
    if len(os_imports) != 1:
        raise FixBlocked("未找到唯一的顶层 import os 位置，拒绝猜测注入")
    anchor = os_imports[0]
    lines = source.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in source else "\n"
    lines.insert(anchor.end_lineno, f"import re{newline}")
    updated = "".join(lines)
    ast.parse(updated)
    data = updated.encode("utf-8-sig" if bom else "utf-8")
    return data, "IMPORT_RE_INSERTED"


def _atomic_replace(target: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(
        prefix="douxian_re_fix_", suffix=".tmp",
        dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as writer:
            writer.write(data)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def install(root: Path, apply: bool = False) -> dict:
    root = Path(root).resolve()
    target = root / "豆馅.py"
    sidebar = root / "pages" / SIDEBAR_PAGE
    if not target.is_file():
        raise FixBlocked(f"缺少主前端文件：{target}")
    original = target.read_bytes()
    fixed, import_status = _find_patch(original)
    needs_import = fixed != original
    needs_sidebar = sidebar.is_file()
    status = (
        "READY_TO_APPLY" if (needs_import or needs_sidebar) and not apply
        else "ALREADY_FIXED" if not (needs_import or needs_sidebar)
        else "APPLIED"
    )
    result = {
        "status": status,
        "frontend": str(target),
        "import": import_status,
        "separate_page_found": needs_sidebar,
        "saved_frontend": None,
        "saved_sidebar_page": None,
        "frontend_sha256_before": hashlib.sha256(original).hexdigest(),
        "frontend_sha256_after": hashlib.sha256(fixed).hexdigest(),
        "neo4j_writes": 0,
        "chroma_writes": 0,
        "rag_py_changed": False,
    }
    if not apply or not (needs_import or needs_sidebar):
        return result

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = root / f"豆馅.py.pre_re_sidebar_fix_{stamp}.bak"
    shutil.copy2(target, backup)
    if backup.read_bytes() != original:
        raise FixBlocked("主前端备份验证不通过，拒绝覆盖")
    result["saved_frontend"] = str(backup)

    moved_sidebar = None
    wrote_frontend = False
    try:
        if needs_import:
            # Double-check no external edits occurred between validation
            # and replacement. Never overwrite an unexpected version.
            if target.read_bytes() != original:
                raise FixBlocked("豆馅.py 在操作期间被其他程序更改")
            _atomic_replace(target, fixed)
            wrote_frontend = True

        if needs_sidebar:
            archive_dir = root / "disabled_streamlit_pages"
            archive_dir.mkdir(exist_ok=True)
            moved_sidebar = archive_dir / (
                f"06_Multihop_Evidence_{stamp}.py.bak"
            )
            shutil.move(str(sidebar), str(moved_sidebar))
            result["saved_sidebar_page"] = str(moved_sidebar)

        ast.parse(target.read_text(encoding="utf-8-sig"))
        if target.read_bytes() != fixed:
            raise FixBlocked("替换后主前端 SHA256 与预期不符")
        if sidebar.exists():
            raise FixBlocked("独立多跳页面仍在 Streamlit pages 目录")
        return result
    except Exception:
        # Best-effort reversal of only this operation.
        if wrote_frontend:
            shutil.copy2(backup, target)
        if moved_sidebar is not None and moved_sidebar.exists():
            shutil.move(str(moved_sidebar), str(sidebar))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parent,
        help="主前端所在项目目录（默认脚本自身目录）"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="明确确认后才写入一行 import re 并停用旧页面"
    )
    args = parser.parse_args()
    try:
        summary = install(args.root, args.apply)
        print("[P3-06.3C]", summary["status"])
        for key, value in summary.items():
            if key != "status":
                print(f"{key}: {value}")
        if summary["status"] == "APPLIED":
            print(
                "已修复；请完整重启 Streamlit。"
                "多跳路径继续在原知识库证据区显示。"
            )
        return 0
    except (FixBlocked, OSError, SyntaxError, UnicodeError) as exc:
        print("[P3-06.3C] BLOCKED:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
