# -*- coding: utf-8 -*-
"""P3-07.1 research preflight: drug-drug interaction source-chain inspection.

A report saves ONLY small source excerpts and source hashes, not full sections.
Reports are exclusive-created; no graph/Chroma writes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from medical_p3071_drug_interaction_v117 import inspect_interaction


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--question",
                        default="服用华法林期间能不能吃布洛芬？")
    parser.add_argument("--max-per-drug",type=int,default=6)
    parser.add_argument("--require-local-evidence",action="store_true")
    parser.add_argument("--report",type=Path)
    args=parser.parse_args(argv)
    try:
        report=inspect_interaction(
            args.question,root=args.root,max_per_drug=args.max_per_drug
        )
        if args.report:
            filename=args.report.resolve()
            filename.parent.mkdir(parents=True,exist_ok=True)
            with filename.open("x",encoding="utf-8") as fh:
                json.dump(report,fh,ensure_ascii=False,indent=2)
                fh.write("\n")
        print("[P3-07.1]",report["status"])
        print("Pair:",report["pair"].get("pair_display"))
        print("Neo4j read queries:",report["query_count"])
        print("Local graph source paths:",len(report["local_evidence"]))
        counts={}
        for row in report["local_evidence"]:
            status=row["classification"]["status"]
            counts[status]=counts.get(status,0)+1
        print("Literal mention types:",counts)
        for row in report["local_evidence"][:6]:
            print("Evidence:",row["drug_name"],"|",row["section_field"],
                  "|",row["classification"]["status"],
                  "| graph 2 hops")
        print("Official label reference URLs:",len(report["official_references"]))
        print("Clinical evidence independently reviewed here: NO")
        if args.report:
            print("Report:",args.report.resolve())
        if args.require_local_evidence and not report["local_evidence"]:
            print("BLOCKED: 本地 Neo4j 没有返回可用相互作用原文",
                  file=sys.stderr)
            return 2
        return 0
    except (OSError,ValueError,RuntimeError,ImportError,TypeError) as ex:
        print("[P3-07.1] BLOCKED:",str(ex),file=sys.stderr)
        return 2


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
    sys.exit(main())
