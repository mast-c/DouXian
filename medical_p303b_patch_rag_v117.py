# -*- coding: utf-8 -*-
"""P3-03B 安全安装脚本：只向现有 rag.py 注入四处局部改动。

默认 dry-run。--apply 才备份 rag.py、校验语法并原子替换。
不修改 Neo4j/Chroma/SQLite，也不创建任何新临床关系。
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile


class PatchNotApplicable(RuntimeError):
    pass


ANCHOR_IMPORT = """from medical_evidence_bridge import (
    prepare_medical_documents, collapse_duplicate_parent_sections,
)
"""
NEW_IMPORT = """from medical_p303b_disease_runtime_v117 import (
    choose_links_file, resolve_disease_links, merge_disease_evidence,
)
"""
ANCHOR_REFERENCE = """    if kind == "json_neo4j_text_verified" and item.get("parent_verified"):
        return section + " · Chroma Hybrid 召回，Neo4j Section 原文核对"
"""
REPLACEMENT_REFERENCE = """    if kind == "json_neo4j_text_verified" and item.get("disease_link_verified"):
        return section + " · 疾病候选直连 Chroma ID，Neo4j Section 原文核对（非临床验证）"
""" + ANCHOR_REFERENCE

ANCHOR_ROUTE = """                # Chroma 中已逐字回溯到父级 Section 的内容，
"""
INSERT_ROUTE = """                # P3-03B: 在既有 Graph + Hybrid 合流处，把疾病提及候选按
                # P3-03A 精确映射直连 Chroma Chunk ID，并逐条复核 Neo4j 原文。
                # 不调用 Embedding、不修改旧检索算法、不创建治疗关系。
                _disease_candidates = [
                    item for item in merged_excerpts
                    if item.get("evidence_type") == "unverified_drug_disease_candidate"
                ]
                if _disease_candidates:
                    try:
                        _link_file = choose_links_file(Path(__file__).resolve().parent)
                        _native_chroma = getattr(
                            getattr(self.knowledge_service, "chroma", None),
                            "_collection", None
                        )
                        if _link_file is None:
                            trace["disease_leaflet_links"] = {
                                "status": (
                                    "DISABLED" if os.getenv(
                                        "MEDICAL_DISEASE_LINKS_ENABLED", "1"
                                    ).strip().lower() in ("0", "false", "no", "off")
                                    else "NO_AUDIT_LINK_FILE"
                                ),
                            }
                        elif _native_chroma is None:
                            trace["disease_leaflet_links"] = {
                                "status": "CHROMA_NOT_READY",
                            }
                        else:
                            _direct_docs, _link_trace = resolve_disease_links(
                                _disease_candidates,
                                links_path=_link_file,
                                native_chroma=_native_chroma,
                                fetch_parent=self._get_medical_parent_section,
                                max_documents=min(
                                    4 if overview_question else 2, evidence_cap
                                ),
                            )
                            merged_excerpts, _merge_trace = merge_disease_evidence(
                                merged_excerpts, _direct_docs, cap=evidence_cap,
                            )
                            trace["disease_leaflet_links"] = {
                                **_link_trace, **_merge_trace,
                                "source_file": _link_file.name,
                            }
                    except Exception as _link_exc:
                        # Fail closed: 不把映射文件异常当作已验证证据；
                        # 其余 Neo4j/Hybrid 现有路径保持可用。
                        trace["disease_leaflet_links"] = {
                            "status": "LINKAGE_REJECTED",
                            "reason_type": type(_link_exc).__name__,
                        }
                        print("[P3-03B] 直连校验未通过：", type(_link_exc).__name__)
                else:
                    trace["disease_leaflet_links"] = {
                        "status": "NO_DISEASE_CANDIDATES",
                    }

"""
ANCHOR_TRACE = """                        "parent_verified": bool(item.get("parent_verified") or False),
"""
REPLACEMENT_TRACE = ANCHOR_TRACE + """                        "disease_link_verified": bool(
                            item.get("disease_link_verified") or False
                        ),
                        "disease_association_type": str(
                            item.get("association_type") or ""
                        ),
                        "linked_candidate_ids": [
                            str(v) for v in item.get("linked_candidate_ids", [])
                            if isinstance(v, str)
                        ][:10],
                        "link_retrieval_route": str(
                            item.get("link_retrieval_route") or ""
                        ),
"""


def transform_rag(source: str) -> str:
    if NEW_IMPORT in source:
        raise PatchNotApplicable(
            "rag.py 已出现 P3-03B import；为防止部分补丁重复应用，拒绝覆盖"
        )
    modifications = (
        (ANCHOR_IMPORT, ANCHOR_IMPORT + NEW_IMPORT),
        (ANCHOR_REFERENCE, REPLACEMENT_REFERENCE),
        (ANCHOR_ROUTE, INSERT_ROUTE + ANCHOR_ROUTE),
    )
    for before, after in modifications:
        count = source.count(before)
        if count != 1:
            raise PatchNotApplicable(
                f"rag.py 目标锚点出现 {count} 次（预期恰好 1），"
                "可能不是当前 V1.17 基线；不做任何改动"
            )
        source = source.replace(before, after, 1)

    # rag.py 的最终证据与自检重试分支各有一处 parent_verified；
    # 必须保持原有缩进同时更新两处，不能静默漏掉任何证据展示入口。
    pattern = re.compile(
        r'(?m)^(?P<indent> +)"parent_verified": '
        r'bool\(item\.get\("parent_verified"\) or False\),$'
    )
    def add_trace_fields(match: re.Match) -> str:
        indent = match.group("indent")
        added = REPLACEMENT_TRACE[len(ANCHOR_TRACE):]
        added = added.replace("                        ", indent)
        return match.group(0) + "\n" + added.rstrip("\n")

    source, n = pattern.subn(add_trace_fields, source)
    if n != 2:
        raise PatchNotApplicable(
            f"rag.py 原文回溯证据输出入口发现 {n} 处（预期 2），拒绝修改"
        )
    ast.parse(source, filename="rag.py")
    return source


def apply_patch(target: Path, *, apply: bool,
                expected_module: Path | None = None) -> dict:
    target = Path(target).resolve()
    if not target.is_file():
        raise PatchNotApplicable(f"找不到现有 rag.py：{target}")
    module = (
        Path(expected_module).resolve()
        if expected_module is not None
        else target.with_name("medical_p303b_disease_runtime_v117.py")
    )
    if not module.is_file():
        raise PatchNotApplicable(
            f"同级必须先有 P3-03B runtime 文件：{module}"
        )
    module_src = module.read_text(encoding="utf-8-sig")
    ast.parse(module_src, filename=module.name)
    with target.open("r", encoding="utf-8-sig", newline="") as stream:
        raw = stream.read()
    newline = "\r\n" if "\r\n" in raw else "\n"
    original = raw.replace("\r\n", "\n")
    candidate = transform_rag(original)
    old_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    output = {
        "target": str(target), "mode": "apply" if apply else "dry_run",
        "old_sha256": old_sha,
        "changed_sections": [
            "runtime_import", "reference_label", "graph_hybrid_merge",
            "trace_evidence_schema"
        ],
        "backup": None,
    }
    if not apply:
        return output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    backup = target.with_name(target.name + f".pre_p303b_{timestamp}.bak")
    shutil.copy2(target, backup)
    if not backup.is_file() or hashlib.sha256(backup.read_bytes()).hexdigest() != old_sha:
        raise PatchNotApplicable("原始 rag.py 备份校验失败，拒绝安装")
    fd, temp = tempfile.mkstemp(
        prefix="rag.p303b_", suffix=".tmp", dir=str(target.parent)
    )
    try:
        output_text = candidate.replace("\n", newline)
        has_bom = target.read_bytes().startswith(b"\xef\xbb\xbf")
        encoding = "utf-8-sig" if has_bom else "utf-8"
        with os.fdopen(fd, "w", encoding=encoding, newline="") as file:
            file.write(output_text)
            file.flush()
            os.fsync(file.fileno())
        ast.parse(
            Path(temp).read_text(encoding="utf-8-sig"), filename="rag.py"
        )
        # 防止其他进程恰好在补丁准备阶段改动 rag.py。
        if hashlib.sha256(target.read_bytes()).hexdigest() != old_sha:
            raise PatchNotApplicable("rag.py 在备份后发生变化，拒绝覆盖")
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    output.update({
        "backup": str(backup),
        "new_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    })
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="V1.17 P3-03B 疾病候选证据直连补丁；默认仅预检查"
    )
    parser.add_argument("--target", type=Path, default=Path("rag.py"))
    parser.add_argument("--apply", action="store_true",
                        help="先备份再将经过语法验证的四处补丁原子写入 rag.py")
    args = parser.parse_args(argv)
    try:
        result = apply_patch(args.target, apply=args.apply)
    except (OSError, ValueError, SyntaxError, PatchNotApplicable) as exc:
        print("[P3-03B] BLOCKED:", exc, file=sys.stderr)
        return 2
    print("[P3-03B]", result["mode"])
    for key, value in result.items():
        if key != "mode":
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
