# -*- coding: utf-8 -*-
"""Full graph drug catalog build + generic pair interaction evidence preflight.

Subcommands:
  build   : one-time all Drug + safety Section index, local derived SQLite ONLY.
  inspect : any two drugs from index; no source writes.
  audit   : inventory coverage / source category limits.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from medical_p3072_corpus_index_v117 import (
    CorpusBlocked,Neo4jReadPager,build_snapshot,open_snapshot,
)
from medical_p3072_pair_retrieval_v117 import (
    inspect_pair,index_path
)


def _json_print(obj)->None:
    print(json.dumps(obj,ensure_ascii=False,indent=2))


def _report_write(path:Path,obj:dict)->None:
    path=Path(path).resolve()
    if not(path.name.startswith("p3072_") and path.suffix==".json"):
        raise CorpusBlocked("报告文件必须命名为 p3072_*.json，且不能覆盖既有文件")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as out:
        json.dump(obj,out,ensure_ascii=False,indent=2)
        out.write("\n")


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    command=parser.add_subparsers(dest="mode",required=True)
    build=command.add_parser("build",help="Neo4j全药品/安全栏目只读全库快照")
    build.add_argument("--root",type=Path,default=Path.cwd())
    build.add_argument("--index",type=Path)
    build.add_argument("--batch",type=int,default=256)
    build.add_argument("--replace",action="store_true")
    build.add_argument("--report",type=Path)
    audit=command.add_parser("audit",help="只读索引覆盖审计")
    audit.add_argument("--root",type=Path,default=Path.cwd())
    audit.add_argument("--index",type=Path)
    audit.add_argument("--report",type=Path)
    inspect=command.add_parser("inspect",help="单题双药相互作用来源检索")
    inspect.add_argument("--root",type=Path,default=Path.cwd())
    inspect.add_argument("--index",type=Path)
    inspect.add_argument("--question",type=str,required=True)
    inspect.add_argument("--curated-map",type=Path)
    inspect.add_argument("--max-sections-per-drug",type=int,default=400)
    inspect.add_argument("--max-evidence",type=int,default=16)
    inspect.add_argument("--report",type=Path)
    inspect.add_argument("--require-pair",action="store_true")
    args=parser.parse_args(argv)
    try:
        root=args.root.resolve()
        idx=index_path(root,args.index)
        if args.mode=="build":
            pager=Neo4jReadPager(root)
            try:
                result=build_snapshot(
                    idx,pager=pager,replace=args.replace,batch=args.batch,
                    progress=lambda x:print(
                        "[BUILD] 药品:",x["drugs_scanned"],
                        "章节:",x["sections_scanned"],
                        "跳过超长:",x["sections_overlong_skipped"],
                        flush=True
                    )
                )
            finally:
                pager.close()
            print("[P3-07.2] FULL_CORPUS_INDEX_BUILT")
            print("Drug nodes:",result["drugs_indexed"])
            print("Drug nodes with safety Sections:",
                  result["drugs_with_safety_sections"])
            print("Safety Sections indexed:",result["sections_indexed"])
            print("Drug coverage %:",result["coverage_pct"])
            print("Overlong Sections not indexed:",
                  result["sections_overlong_skipped"])
            print("Orphan Sections:",result["orphan_sections_skipped"])
            print("Generated derived index:",result["index_path"])
        elif args.mode=="audit":
            conn,meta=open_snapshot(idx)
            try:
                observed={
                    "drugs":conn.execute(
                        "SELECT COUNT(*) FROM drugs").fetchone()[0],
                    "sections":conn.execute(
                        "SELECT COUNT(*) FROM sections").fetchone()[0],
                    "aliases":conn.execute(
                        "SELECT COUNT(*) FROM aliases").fetchone()[0],
                    "drugs_with_safety_sections":conn.execute(
                        "SELECT COUNT(DISTINCT drug_id) FROM sections"
                    ).fetchone()[0],
                }
                bad=conn.execute("PRAGMA foreign_key_check").fetchall()
                if bad:
                    raise CorpusBlocked("派生索引存在断开的Drug->Section引用")
                hashes_checked=0
                for section_text,saved_sha in conn.execute(
                    "SELECT section_text,section_sha256 FROM sections"
                ):
                    current=hashlib.sha256(
                        section_text.encode("utf-8")
                    ).hexdigest()
                    if current!=saved_sha:
                        raise CorpusBlocked(
                            "派生索引包含来源文本哈希不一致的Section"
                        )
                    hashes_checked+=1
                observed["section_sha256_verified"]=hashes_checked
            finally:
                conn.close()
            if (
                observed["drugs"]!=meta["drugs_indexed"]
                or observed["sections"]!=meta["sections_indexed"]
                or observed["aliases"]!=meta["aliases_indexed"]
                or observed["drugs_with_safety_sections"]!=meta["drugs_with_safety_sections"]
            ):
                raise CorpusBlocked("快照清单数量与 SQLite 内容不符")
            result={
                "stage":"P3-07.2","status":"FULL_CORPUS_INDEX_AUDIT_OK",
                "index":str(idx),"metadata":meta,"observed":observed,
                "scope":"all Neo4j Drug nodes and the 3 safety Section fields",
                "chroma_52104_chunks_independently_audited":False,
                "drug_pair_interactions_automatically_verified":False,
                "clinical_reviewed":False,
            }
            print("[P3-07.2] FULL_CORPUS_INDEX_AUDIT_OK")
            print("Drug nodes:",observed["drugs"])
            print("Safety sections:",observed["sections"])
            print("Section SHA256 verified:",
                  observed["section_sha256_verified"])
            print("Drug nodes with these fields:",
                  observed["drugs_with_safety_sections"])
        else:
            mapping=args.curated_map
            if mapping and not mapping.is_absolute():
                mapping=root/mapping
            result=inspect_pair(
                args.question,root=root,index=idx,curated_map=mapping,
                max_sections_per_group=args.max_sections_per_drug,
                max_evidence=args.max_evidence
            )
            print("[P3-07.2]",result["status"])
            names=result["name_resolution"].get("matches",[])
            print("Matched drug groups:",[
                r.get("canonical_key") for r in names
            ])
            print("Indexed drugs:",
                  result["index_snapshot"]["drugs_indexed"])
            print("Safety sections:",
                  result["index_snapshot"]["sections_indexed"])
            print("Scanned candidate sections:",
                  result["source_rows_scanned"])
            print("Total corresponding sections:",
                  result["source_rows_total"])
            print("Displayed evidence chains:",
                  len(result["source_evidence"]))
            print("Budget truncated:",result["search_budget_truncated"])
            print("Literal evidence summary:",result["evidence_summary"])
            print("This is NOT a clinically approved interaction or a safe-combination decision.")
        if args.report:
            _report_write(args.report,result)
            print("Report:",args.report.resolve())
        if args.mode=="inspect" and args.require_pair and (
            result["name_resolution"]["status"]!=
            "EXACTLY_TWO_CATALOG_GROUPS_DETECTED"
        ):
            return 2
        return 0
    except (CorpusBlocked,OSError,ValueError,TypeError,ImportError,
            json.JSONDecodeError) as exc:
        print("[P3-07.2] BLOCKED:",str(exc),file=sys.stderr)
        return 2


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="backslashreplace")
    raise SystemExit(main())
