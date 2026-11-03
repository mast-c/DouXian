# -*- coding: utf-8 -*-
"""P3-04.3 local end-to-end preflight for audited disease encyclopedia.

Read-only Chroma .get and original medical.json; no Embedding and no DB writes.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from medical_p3043_disease_encyclopedia_runtime_v117 import (
    DiseaseEvidenceBlocked, read_only_preflight
)


def execute(root: Path, *, audit: Path | None=None,
            source: Path | None=None, report: Path | None=None,
            require_rag_patch: bool=False, collection=None) -> dict:
    root=Path(root).resolve()
    result=read_only_preflight(
        root,audit_path=audit,source_path=source,collection=collection
    )
    result["stage"]="P3-04.3"
    result["mode"]="read_only_online_disease_search_preflight"
    result["created_utc"]=datetime.now(timezone.utc).isoformat(timespec="seconds")
    if require_rag_patch:
        file_path=root/"rag.py"
        if not file_path.is_file():
            raise DiseaseEvidenceBlocked("找不到 rag.py")
        current=file_path.read_text(encoding="utf-8-sig")
        must=[
            "search_disease_encyclopedia(",
            'trace["disease_encyclopedia"]',
            "merge_into_kb("
        ]
        if not all(flag in current for flag in must):
            raise DiseaseEvidenceBlocked("rag.py 尚未安装 P3-04.3 完整在线接入补丁")
        result["rag_patch"]="INSTALLED"
    if result["status"]!="PRECHECK_OK":
        raise DiseaseEvidenceBlocked(
            "疾病百科全量审计/原文检索未通过："
            +str(result["trace"].get("status"))
        )
    if report:
        report=Path(report).resolve()
        if report.exists():
            raise FileExistsError("报告目标存在，拒绝覆盖")
        report.parent.mkdir(parents=True,exist_ok=True)
        with report.open("x",encoding="utf-8") as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2)
            stream.write("\n")
    return result


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--audit",type=Path)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--require-rag-patch",action="store_true")
    parser.add_argument("--report",type=Path)
    args=parser.parse_args(argv)
    try:
        result=execute(
            args.root,audit=args.audit,source=args.source,
            report=args.report,require_rag_patch=args.require_rag_patch
        )
        print("[P3-04.3]",result["status"])
        print("真实原始疾病词条测试问题：",result["test_question"])
        print("逐条原文验证通过：",result["verified_excerpts"])
        print("检索路由：",result["trace"]["route"])
        print("检索状态：",result["trace"]["status"])
        print("引用示例：",result["sample_excerpt"])
        if args.report: print("新报告：",args.report.resolve())
        return 0
    except (OSError,ValueError,RuntimeError,DiseaseEvidenceBlocked) as exc:
        print("[P3-04.3] PRECHECK_BLOCKED:",exc,file=sys.stderr)
        return 2


if __name__=="__main__":
    sys.exit(main())
