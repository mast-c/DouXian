# -*- coding: utf-8 -*-
"""P3-03B local preflight: validate audited link map and patched rag import, no DB calls."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from medical_p303b_disease_runtime_v117 import (
    choose_links_file, load_audited_index
)


def preflight(project_dir: Path, *, links: Path | None = None) -> dict:
    root=Path(project_dir).resolve()
    file_path=Path(links).resolve() if links is not None else choose_links_file(root)
    if file_path is None:
        raise FileNotFoundError(
            "找不到 P3-03A JSONL/JSON 配对审计文件，"
            "可设置 MEDICAL_DISEASE_LINKS_PATH 指向原审计 jsonl"
        )
    index, info=load_audited_index(file_path)
    source=(root/"rag.py").read_text(encoding="utf-8-sig")
    if (
        "from medical_p303b_disease_runtime_v117 import (" not in source
        or "resolve_disease_links(" not in source
        or 'trace["disease_leaflet_links"]' not in source
    ):
        raise RuntimeError(
            "rag.py 还没有完整安装 P3-03B 接入补丁"
        )
    examples=[]
    seen_drugs=set()
    for row in index.values():
        drug=row["drug_name"]
        disease=row["disease_name"]
        if len(drug) < 3 or len(disease) < 3 or drug in seen_drugs:
            continue
        seen_drugs.add(drug)
        examples.append(
            f"请核对{drug}与{disease}在说明书中的原文提及关系，并说明证据局限。"
        )
        if len(examples) >= 3:
            break
    return {
        "status":"PRECHECK_OK",
        "links_file":str(file_path),
        "report_file":info["source_report"],
        "audited_pairs":info["candidate_pairs"],
        "linked_pairs":len(index),
        "sample_queries":examples,
    }


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description="P3-03B 本地只读启用预检查")
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--links",type=Path)
    args=parser.parse_args(argv)
    try:
        summary=preflight(args.root,links=args.links)
    except (ValueError,OSError,RuntimeError) as exc:
        print("[P3-03B] PRECHECK_FAILED:",exc,file=sys.stderr)
        return 2
    print("[P3-03B] PRECHECK_OK")
    print("映射文件：",summary["links_file"])
    print("完整候选对：",summary["audited_pairs"])
    print("审计精确链接：",summary["linked_pairs"])
    print("可在 GraphRAG 页面测试的问题：")
    for question in summary["sample_queries"]:
        print(" -",question)
    print("注意：仅测试候选原文关联，不代表药品对疾病有效或已获批准。")
    return 0


if __name__=="__main__":
    sys.exit(main())
