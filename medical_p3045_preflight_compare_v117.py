# -*- coding: utf-8 -*-
"""P3-04.5 graph-to-Chroma retrieval preflight and within-query comparison.

Both paths read only the same audited medical.json/Chroma knowledge source.
The graph path is gated by a REAL P3-04.4B --audit PASS report.
No LLM answer, no generation, no Embedding; timings measure retrieval only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from medical_p3043_disease_encyclopedia_runtime_v117 import (
    search_disease_encyclopedia
)
from medical_p3045_graph_disease_retrieval_v117 import (
    LINK_TYPE, search_graph_enhanced_disease_encyclopedia
)


def _as_secs(t:float) -> float:
    return round(t*1000,3)


def compare(
    root:Path,question:str, *,
    collection=None,source_path:Path|None=None,
    audit_path:Path|None=None,
    graph_audit_path:Path|None=None,
    plan_path:Path|None=None,
    plan_report_path:Path|None=None,
    link_provider=None,
    enforce_real_graph:bool=True,
    require_rag_patch:bool=False,
) -> dict:
    """Read-only source-to-graph evidence comparison (not answer-quality scoring)."""
    root=Path(root).resolve()
    if require_rag_patch:
        rag=root/"rag.py"
        if not rag.is_file():
            raise RuntimeError("缺少原项目 rag.py")
        text=rag.read_text(encoding="utf-8-sig")
        if ("from medical_p3045_graph_disease_retrieval_v117 import ("
                not in text
                or "search_graph_enhanced_disease_encyclopedia(" not in text):
            raise RuntimeError("P3-04.5 尚未安装到 rag.py")
    root_kwargs={
        "project_root":root,
        "collection":collection,
        "source_path":source_path,
        "audit_path":audit_path,
        "max_documents":4,
    }
    start=time.perf_counter()
    baseline,old_trace=search_disease_encyclopedia(question,**root_kwargs)
    old_ms=_as_secs(time.perf_counter()-start)
    # The preflight intentionally enables the graph option for THIS process
    # only, independent of the .env config used by the running Streamlit app.
    old_env=os.environ.get("MEDICAL_P3045_GRAPH_ENABLED")
    os.environ["MEDICAL_P3045_GRAPH_ENABLED"]="1"
    try:
        start=time.perf_counter()
        changed,new_trace=search_graph_enhanced_disease_encyclopedia(
            question,**root_kwargs,
            graph_audit_path=graph_audit_path,
            plan_path=plan_path,
            plan_report_path=plan_report_path,
            link_provider=link_provider,
        )
        graph_ms=_as_secs(time.perf_counter()-start)
    finally:
        if old_env is None:
            os.environ.pop("MEDICAL_P3045_GRAPH_ENABLED",None)
        else:
            os.environ["MEDICAL_P3045_GRAPH_ENABLED"]=old_env

    new_gate=new_trace["graph_disease_bridge"]
    mode=new_gate["status"]
    success=mode in (
        "GRAPH_EXACT_OID_PROVENANCE_VERIFIED",
        "GRAPH_PARTIAL_WITH_ENCYCLOPEDIA_FALLBACK"
    )
    if enforce_real_graph and not success:
        raise RuntimeError(
            f"Neo4j 来源关系没有完成有效的图谱→Chroma 原文回溯：{mode}；"
            "请先确认 P3-04.4B 全量建图与 --audit PASS，"
            "并核对 Neo4j 连接及报告路径"
        )
    for x in changed:
        if (x.get("source_status")!="third_party_unverified"
                or x.get("clinical_verified") is not False
                or x.get("source_url")!=""):
            raise RuntimeError("返回文档没有保留第三方未核验来源边界")
    old_ids=[x["chunk_id"] for x in baseline]
    new_ids=[x["chunk_id"] for x in changed]
    return {
        "stage":"P3-04.5",
        "status":"GRAPH_ONLINE_PREFLIGHT_OK" if success
                 else "GRAPH_NOT_READY_BASELINE_AVAILABLE",
        "created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query":question,
        "baseline":{
            "route":old_trace["route"],
            "status":old_trace["status"],
            "evidence_chunk_ids":old_ids,
            "evidence_count":len(old_ids),
            "retrieval_ms":old_ms,
        },
        "graph_enhanced":{
            "route":new_trace["route"],
            "status":new_trace["status"],
            "graph_status":mode,
            "graph_verified_diseases":new_gate["graph_verified_names"],
            "graph_evidence_count":new_gate["graph_linked_evidence"],
            "evidence_chunk_ids":new_ids,
            "evidence_count":len(new_ids),
            "retrieval_ms":graph_ms,
            "graph_query_ran":new_gate["graph_query_ran"],
            "graph_writes":0,
            "chroma_writes":0,
            "query_embedding_called":False,
            "relationship_type":LINK_TYPE,
        },
        "comparison":{
            "shared_chunk_ids":sorted(set(old_ids)&set(new_ids)),
            "shared_chunk_count":len(set(old_ids)&set(new_ids)),
            "graph_only_chunk_ids":sorted(set(new_ids)-set(old_ids)),
            "baseline_only_chunk_ids":sorted(set(old_ids)-set(new_ids)),
            "method":"single_query_retrieval_trace_comparison",
            "does_not_measure_clinical_accuracy":True,
            "does_not_measure_generation_accuracy":True,
            "timing_note":(
                "单次检索耗时，含 Python/数据库 IO/缓存差异，"
                "不是多次采样的科学性能显著性结论"
            )
        },
        "source_status":"third_party_unverified",
        "medical_accuracy_reviewed":False,
    }


def main(argv=None, *,injected_collection=None,
         injected_link_provider=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--question",default="百日咳有哪些症状？")
    parser.add_argument("--source",type=Path)
    parser.add_argument("--chroma-audit",type=Path)
    parser.add_argument("--graph-audit",type=Path)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--plan-report",type=Path)
    parser.add_argument("--report",type=Path)
    parser.add_argument("--require-rag-patch",action="store_true")
    parser.add_argument("--allow-baseline-fallback",action="store_true",
                        help="没有 Graph PASS 也记录对照；不代表 P3-04.5 验收通过")
    args=parser.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(args.root).resolve()/".env",override=False)
        if args.report and args.report.exists():
            raise FileExistsError("同名验收报告已存在，禁止覆盖")
        output=compare(
            args.root,args.question,
            collection=injected_collection,
            source_path=args.source,
            audit_path=args.chroma_audit,
            graph_audit_path=args.graph_audit,
            plan_path=args.plan,
            plan_report_path=args.plan_report,
            link_provider=injected_link_provider,
            enforce_real_graph=not args.allow_baseline_fallback,
            require_rag_patch=args.require_rag_patch,
        )
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            with args.report.open("x",encoding="utf-8") as fp:
                json.dump(output,fp,ensure_ascii=False,indent=2)
                fp.write("\n")
        print("[P3-04.5]",output["status"])
        print("Question:",output["query"])
        print("Old route:",output["baseline"]["route"],
              "| old evidence:",output["baseline"]["evidence_count"])
        print("Graph status:",output["graph_enhanced"]["graph_status"])
        print("Graph evidence:",output["graph_enhanced"]["graph_evidence_count"])
        print("Shared Chunk IDs:",output["comparison"]["shared_chunk_count"])
        print("Baseline retrieval ms:",output["baseline"]["retrieval_ms"])
        print("Graph retrieval ms:",output["graph_enhanced"]["retrieval_ms"])
        if args.report: print("Report:",args.report.resolve())
        return 0
    except (OSError,ValueError,RuntimeError,ImportError) as exc:
        print("[P3-04.5] PREFLIGHT_BLOCKED:",str(exc),file=sys.stderr)
        return 2


if __name__=="__main__":
    sys.exit(main())
