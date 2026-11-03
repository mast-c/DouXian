# -*- coding: utf-8 -*-
"""Read-only 5-8 hop Neo4j source-path inspection and optional online preview."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import sys

from medical_p3061_multihop_graph_v117 import (
    retrieve_multihop,
    search_multihop_enhanced_disease_encyclopedia,
    MultiHopBlocked,
)


def _safe_path(row:dict) -> dict:
    """Preserve useful verified identity and hops, avoid exporting full Section."""
    safe={k:v for k,v in row.items() if k not in (
        "section_text","candidate_quote"
    )}
    quote=row.get("candidate_quote","")
    body=row.get("section_text")
    safe["candidate_quote_preview"]=quote[:240]
    safe["section_text_preview"]=body[:280] if isinstance(body,str) else ""
    safe["section_text_sha256"]=(
        hashlib.sha256(body.encode("utf-8")).hexdigest()
        if isinstance(body,str) else None
    )
    return safe


def inspect(question:str, *,root:Path,
            source:Path|None=None,graph_audit:Path|None=None,
            plan:Path|None=None,plan_report:Path|None=None,
            min_hops:int=5,max_hops:int=8,
            max_paths:int=12,per_template:int=4,
            online_preview:bool=False,
            path_provider=None,
            collection=None) -> dict:
    from dotenv import load_dotenv
    load_dotenv(Path(root).resolve()/".env",override=False)
    paths,trace=retrieve_multihop(
        question,project_root=root,source_path=source,
        graph_audit_path=graph_audit,
        plan_path=plan,plan_report_path=plan_report,
        min_hops=min_hops,max_hops=max_hops,max_paths=max_paths,
        per_template=per_template,path_provider=path_provider,
        require_enabled=False
    )
    result={
        "stage":"P3-06.1",
        "mode":"read_only_fixed_schema_bounded_real_graph_traversal",
        "created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question":question,
        "graph_path_status":trace["status"],
        "graph_trace":trace,
        "paths":[_safe_path(x) for x in paths],
        "actual_hops":sorted({x["hop_count"] for x in paths}),
        "graph_writes":0,
        "chroma_writes":0,
        "clinical_verified":False,
        "treatment_relation_inferred":False,
        "source_status":"third_party_unverified",
        "online_preview":None,
    }
    if online_preview:
        import os
        old=os.environ.get("MEDICAL_P3061_MULTIHOP_ENABLED")
        os.environ["MEDICAL_P3061_MULTIHOP_ENABLED"]="1"
        try:
            docs,report=search_multihop_enhanced_disease_encyclopedia(
                question,project_root=root,
                source_path=source,collection=collection,
                graph_audit_path=graph_audit,
                plan_path=plan,plan_report_path=plan_report,
                min_hops=min_hops,max_hops=max_hops,
                path_provider=path_provider,
            )
        finally:
            if old is None:
                os.environ.pop("MEDICAL_P3061_MULTIHOP_ENABLED",None)
            else:
                os.environ["MEDICAL_P3061_MULTIHOP_ENABLED"]=old
        result["online_preview"]={
            "status":report.get("multi_hop_graph",{}).get("status"),
            "base_disease_status":report.get("status"),
            "source_excerpts":len(docs),
            "multihop_section_excerpts":sum(
                item.get("multi_hop_verified") is True for item in docs
            ),
            "evidence_ids":[x.get("chunk_id") for x in docs
                            if x.get("chunk_id")],
            "graph_trace":report.get("multi_hop_graph",{}),
        }
    return result


def _safe_report(out:dict,path:Path) -> None:
    path=Path(path).resolve()
    if path.exists():
        raise MultiHopBlocked("目标报告已存在，拒绝覆盖")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as fp:
        json.dump(out,fp,ensure_ascii=False,indent=2)
        fp.write("\n")


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--question",default="哪些药品说明书提到了过敏性鼻炎？")
    parser.add_argument("--source",type=Path)
    parser.add_argument("--graph-audit",type=Path)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--plan-report",type=Path)
    parser.add_argument("--min-hops",type=int,default=5)
    parser.add_argument("--max-hops",type=int,default=8)
    parser.add_argument("--max-paths",type=int,default=12)
    parser.add_argument("--per-template",type=int,default=4)
    parser.add_argument("--online-preview",action="store_true")
    parser.add_argument("--require-path",action="store_true")
    parser.add_argument("--report",type=Path)
    args=parser.parse_args(argv)
    try:
        result=inspect(
            args.question,root=args.root,source=args.source,
            graph_audit=args.graph_audit,
            plan=args.plan,plan_report=args.plan_report,
            min_hops=args.min_hops,max_hops=args.max_hops,
            max_paths=args.max_paths,per_template=args.per_template,
            online_preview=args.online_preview,
        )
        if args.report:
            _safe_report(result,args.report)
        trace=result["graph_trace"]
        print("[P3-06.1]",trace["status"])
        print("Real query count:",trace["query_count"])
        print("Returned paths:",trace["returned_paths"])
        print("Actual hop depths:",result["actual_hops"])
        print("Per hop:",trace["found_per_hops"])
        for item in result["paths"][:5]:
            print("Path:",item["hop_count"],"hops |",
                  item["drug_name"],"|",
                  item["disease_name"],"->",
                  item.get("related_disease_name") or "Section")
        if result["online_preview"]:
            print("Online preview:",result["online_preview"]["status"])
            print("New Section evidence:",
                  result["online_preview"]["multihop_section_excerpts"])
        if args.report:print("Report:",args.report.resolve())
        if args.require_path and not result["paths"]:
            print("[P3-06.1] BLOCKED: 当前 Neo4j 没有实际可核验的多跳路径",
                  file=sys.stderr)
            return 2
        return 0
    except (OSError,ValueError,RuntimeError,ImportError,TypeError) as exc:
        print("[P3-06.1] BLOCKED:",str(exc),file=sys.stderr)
        return 2


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
    sys.exit(main())
