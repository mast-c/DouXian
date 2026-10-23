# -*- coding: utf-8 -*-
"""DouXian V2.16 Graph Assist planning evaluation: offline-first.

--offline   : 11 source-grounded fixed queries; zero embedding/rerank calls.
--live      : original unaugmented Hybrid Top20 and an optional SECOND
              targeted Hybrid retrieval per *missing* evidence Chunk.
              Only `--run-branches` explicitly authorizes extra embedding
              queries. Both are READ ONLY. No automatic reranker.

Any evidence recovered is a CANDIDATE uplift, never an inferred answer
improvement or a "verified" newly promoted graph claim.
"""
from __future__ import annotations
from collections import Counter
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import time

from .graph_assist_planner_v216 import (
    load_graph,plan,key,sha256_file
)
from .verify_machine_graph_v215 import HERE

CASES=HERE/"graph_reports/assist_v216/fixed_eval_cases_v216.json"
LEGACY=HERE/"graph_tools/graph_assist_v28.py"

def _require(cond:bool,description:str):
    if not cond:raise RuntimeError(description)

def read_cases()->list[dict]:
    obj=json.loads(CASES.read_text(encoding="utf-8"))
    records=obj.get("queries")
    _require(obj.get("schema")=="douxian-v216-assist-controlled-eval-1"
             and isinstance(records,list) and len(records)==11,
             "V2.16 frozen 11-case pilot changed")
    ids=[c.get("id") for c in records]
    _require(len(ids)==len(set(ids)) and all(isinstance(i,str) and i for i in ids),
             "V2.16 fixed test IDs invalid/duplicate")
    _require(all(isinstance(c.get("question"),str) and c["question"].strip()
                 for c in records),"Invalid fixed test questions")
    return records

def legacy_relation_gate(legacy_file:Path=LEGACY)->tuple[re.Pattern|None,str|None]:
    """AST-extract ONLY old V2.8 RELATION_INTENT regex (without importing it).

    This is NOT a full legacy retrieval result: valid V2.7 graph and other
    old gates are also required to actually expand a query.
    """
    if not legacy_file.is_file(): return None,None
    tree=ast.parse(legacy_file.read_text(encoding="utf-8"))
    patterns=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(
            isinstance(target,ast.Name) and target.id=="RELATION_INTENT"
            for target in node.targets
        ):
            value=node.value
            if (isinstance(value,ast.Call) and
                isinstance(value.func,ast.Attribute) and
                value.func.attr=="compile" and value.args and
                isinstance(value.args[0],ast.Constant) and
                isinstance(value.args[0].value,str)):
                patterns.append(value.args[0].value)
    _require(len(patterns)==1,
             "Expected exactly one old V2.8 RELATION_INTENT regex; "
             "refusing to report an unsupported legacy comparison")
    return re.compile(patterns[0]),sha256_file(legacy_file)

def evaluate_offline(include_pending:bool)->dict:
    tests=read_cases()
    graph=load_graph(include_pending=include_pending)
    legacy,legacy_sha=legacy_relation_gate()
    rows=[]
    for case in tests:
        q=case["question"]
        report=plan(q,[q],graph,max_branches=2)
        rows.append({
           "case_id":case["id"],
           "question":q,
           "group":case["group"],
           "old_v28_relation_intent_regex_matched":(
               bool(legacy.search(q)) if legacy is not None else None
           ),
           "old_v28_full_suggest_retrieval_queries_executed":False,
           "v216_intent":report["intent"],
           "v216_status":report["status"],
           "matched_anchors":report["matched_anchors"],
           "v216_planned_branch_count":len(report["branches"]),
           "branches":report["branches"],
           "diagnostics":report["diagnostics"],
           "original_query_modified":False,
        })
    negatives=[x for x in rows if x["group"].startswith("negative_")]
    _require(all(not x["branches"] for x in negatives),
             "Negative control unexpectedly triggered graph branching")
    _require(all(x["original_query_modified"] is False for x in rows),
             "Planner must NEVER rewrite the original Hybrid query")
    pending=[r for r in rows
             if r["group"] in ("positive_pending_crossbook",
                               "exploratory_pending_singlebook")]
    if not include_pending:
        _require(all(x["v216_planned_branch_count"]==0 for x in pending),
                 "Unreviewed new-book paths must be inaccessible by default")
    return {
       "experiment":"DouXian_V2.16_Graph_Assist_ReadOnly_Plan_Study",
       "phase":"offline_planning_no_external_calls",
       "run_utc":datetime.now(timezone.utc).isoformat(),
       "old_v28_gate_comparison":"RELATION_INTENT regex necessary gate ONLY; "
                "not actual legacy graph evidence retrieval",
       "legacy_v28_source_sha256":legacy_sha,
       "fixed_case_file_sha256":sha256_file(CASES),
       "mode":"research_include_pending" if include_pending else "verified_only",
       "source_checksums":graph["source_checksums"],
       "verified_old_relations":graph["verified_count"],
       "pending_relations_explorable":graph["new_pending_count"],
       "total_questions":len(rows),
       "negative_control_questions":len(negatives),
       "old_v28_regex_match_count":sum(x["old_v28_relation_intent_regex_matched"]
                                      for x in rows
                                      if x["old_v28_relation_intent_regex_matched"] is not None),
       "v216_planned_query_count":sum(x["v216_planned_branch_count"] for x in rows),
       "questions_with_plan":sum(bool(x["branches"]) for x in rows),
       "old_original_queries_modified":0,
       "embedding_calls":0,
       "reranker_calls":0,
       "answer_generation_calls":0,
       "new_relations_promoted":0,
       "results":rows,
       "limitations":[
         "This only tests intent, directed path planning and evidence provenance.",
         "The V2.8 comparison is a regex gate, NOT actual retrieval/output accuracy.",
         "Eleven deliberately chosen fixed queries do not measure overall accuracy.",
         "No Hybrid/Chroma search, Qwen Reranker, final evidence or answer A/B.",
         "V2.15 new-book semantic edges remain offline pending human review.",
         "A planned query does not prove it can retrieve a useful missing Chunk."
       ],
    }

def _cid(doc)->str:
    meta=getattr(doc,"metadata",None)
    return str(meta.get("chunk_id") or "").strip() if isinstance(meta,dict) else ""

def _source(doc)->str:
    meta=getattr(doc,"metadata",None)
    return str(meta.get("source") or "") if isinstance(meta,dict) else ""

def rank_and_duplicates(chunks:list, target_id:str, target_body:str,
                        target_source:str)->dict:
    """Pure exact-ID and identical-body/source overlap test."""
    from .graph_retriever_v213 import canonical_source
    ranks={}
    for rank,doc in enumerate(chunks,1):
        cid=_cid(doc)
        _require(cid and cid not in ranks,
                 "Hybrid has missing or duplicate original Chunk IDs")
        ranks[cid]=rank
    target_rank=ranks.get(target_id)
    duplicates=[]
    for doc in chunks:
        cid=_cid(doc)
        if cid==target_id:continue
        if (getattr(doc,"page_content",None)==target_body
            and canonical_source(_source(doc))==
                canonical_source(target_source)):
            duplicates.append({"chunk_id":cid,"rank":ranks[cid]})
    return {
        "rank":target_rank,
        "same_content_same_source_other_ids":duplicates,
        "novel_in_top10":(
            (target_rank is None or target_rank>10)
            and not any(x["rank"]<=10 for x in duplicates)
        ),
        "novel_in_top20":(
            target_rank is None and not any(x["rank"]<=20 for x in duplicates)
        ),
    }

def evaluate_live(
    cases:list[str],include_pending:bool=False,run_branches:bool=False
)->dict:
    """Original KnowledgeBaseService calls only; no monkeypatch/production edits."""
    from .graph_retriever_v213 import (
        load_verified_bundle,_live_graph_documents
    )
    from knowledge_base import KnowledgeBaseService
    import config_data as config

    bank=read_cases()
    by_id={c["id"]:c for c in bank}
    _require(bool(cases) and len(cases)<=3 and len(set(cases))==len(cases)
             and all(c in by_id for c in cases),
             "--live must explicitly select 1..3 distinct fixed case IDs")
    bundle=load_graph(include_pending=include_pending)
    original=load_verified_bundle()
    _require(original.total_verified==29,
             "Old V2.12 formal relation count drifted")
    db_root=Path(config.persist_directory).expanduser()
    _require(db_root.is_dir(),
             "Original Chroma persistent folder absent; refusing DB initialization")
    service=KnowledgeBaseService()
    count=service.chroma._collection.count()
    _require(count==1953,"Live Chroma count differs from 1953 frozen snapshot")
    total=service.build_bm25_index()
    _require(total==1953,"Live BM25 index row count does not match frozen 1953")
    service.build_hybrid_retriever()

    prepared=[]
    for case_id in cases:
        test=by_id[case_id]
        question=test["question"]
        p=plan(question,[question],bundle,max_branches=2)
        for b in p["branches"]:
            _require(b["target_chunk_id"] in original.chunk_rows,
                     "Planned evidence Chunk absent from formal verified snapshot")
        prepared.append((test,p))
    targets=list(dict.fromkeys(
       b["target_chunk_id"]
       for _,p in prepared for b in p["branches"]
    ))
    hydrated=dict(zip(targets,
        _live_graph_documents(service,targets,original) if targets else []))
    # _live_graph_documents independently checks exact Chroma body and metadata
    # against the frozen snapshot even for the new textbook.
    results=[]
    embed_queries=0
    for case,planned in prepared:
        question=case["question"]
        started=time.perf_counter()
        base=service.search_hybrid(question,k=20)
        embed_queries+=1
        _require(bool(base) and all(_cid(d) for d in base),
                 "Original raw Hybrid returned empty/invalid candidate list")
        elapsed=round(time.perf_counter()-started,3)
        tests=[]
        for branch in planned["branches"]:
            cid=branch["target_chunk_id"]
            content,metadata=hydrated[cid]
            same=rank_and_duplicates(base,cid,content,metadata.get("source",""))
            record={
                "planner_query":branch["query"],
                "evidence_chunk_id":cid,
                "path_claim_ids":branch["path_claim_ids"],
                "path_kind":branch["path_kind"],
                "risk":branch["source_risk"],
                "baseline_rank":same["rank"],
                "already_in_baseline_top10":not same["novel_in_top10"],
                "novel_bound_evidence_vs_baseline_top20":same["novel_in_top20"],
                "identical_content_source_other_ids":same["same_content_same_source_other_ids"],
                "branch_executed":False,
                "branch_target_rank":None,
                "branch_recovered_absent_baseline_target":False,
                "skip_reason":"",
                "manual_query_relevance_reviewed":False,
            }
            if not same["novel_in_top20"]:
                record["skip_reason"]="target_evidence_already_in_original_hybrid_top20"
            elif not run_branches:
                record["skip_reason"]="extra_embedding_not_explicitly_authorized"
            else:
                search=service.search_hybrid(branch["query"],k=20)
                embed_queries+=1
                recovered=rank_and_duplicates(search,cid,content,metadata.get("source",""))
                record["branch_executed"]=True
                record["branch_target_rank"]=recovered["rank"]
                record["branch_recovered_absent_baseline_target"]=(
                    recovered["rank"] is not None
                )
                record["same_content_in_branch_other_id"]=recovered[
                    "same_content_same_source_other_ids"]
            tests.append(record)
        results.append({
            "case_id":case["id"],
            "question":question,
            "graph_intent":planned["intent"],
            "planner_branches":len(planned["branches"]),
            "raw_hybrid_seconds":elapsed,
            "raw_hybrid_top20_original_ids":[_cid(doc) for doc in base],
            "target_evidence_audits":tests,
            "new_target_evidence_candidates_vs_hybrid_top20":sum(
                x["novel_bound_evidence_vs_baseline_top20"] for x in tests),
            "branch_queries_executed":sum(x["branch_executed"] for x in tests),
            "recovered_omitted_evidence_chunks":sum(
                x["branch_recovered_absent_baseline_target"] for x in tests
            ),
        })
        print("CASE",case["id"],
              "planned",len(tests),
              "novel source-backed IDs",results[-1]["new_target_evidence_candidates_vs_hybrid_top20"],
              "branch requests",results[-1]["branch_queries_executed"])
    return {
        "experiment":"DouXian_V2.16_ReadOnly_Hybrid_Query_Reformulation_Study",
        "phase":"live_unaugmented_hybrid_with_optional_independent_planned_queries",
        "run_utc":datetime.now(timezone.utc).isoformat(),
        "mode":"research_include_pending" if include_pending else "verified_only",
        "live_chroma_count":count,
        "bm25_in_memory_count":total,
        "old_v212_verified_relations":29,
        "v215_new_relations_promoted":0,
        "fixed_queries":True,
        "query_decomposition_performed":False,
        "raw_hybrid_top_k":20,
        "original_embeddings_and_new_embeddings_calls":embed_queries,
        "run_optional_branches":run_branches,
        "reranker_calls":0,
        "answer_generation_calls":0,
        "tested_cases":cases,
        "new_target_evidence_candidates_vs_hybrid_top20":sum(
            x["new_target_evidence_candidates_vs_hybrid_top20"] for x in results),
        "recovered_omitted_evidence_chunks":sum(
            x["recovered_omitted_evidence_chunks"] for x in results),
        "results":results,
        "limitations":[
            "Only a few fixed, author-chosen queries; not a retrieval benchmark.",
            "Per-branch target is a source-bound existing Chunk, not a measured relevant/gold passage.",
            "Planned query may miss its targeted Chunk or retrieve extra unrelated Chunks.",
            "Hybrid expansion is NOT integrated into production RAG and no reranker calls made.",
            "V2.15 pending edges not human approved and never promoted.",
            "Baseline+target discovery or recovery alone is not answer quality improvement."
        ]
    }

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline",action="store_true",
                      help="Zero external API, frozen 11-question diagnostics")
    mode.add_argument("--live",action="store_true",
                      help="Live original Hybrid only, may call paid Embedding API")
    p.add_argument("--case",action="append",default=[],
                   help="Explicit fixed test case ID; required for --live (1 to 3)")
    p.add_argument("--explore-pending",action="store_true",
                   help="ONLY OFFLINE RESEARCH; V2.15 new edges remain unreviewed")
    p.add_argument("--run-branches",action="store_true",
                   help="LIVE ONLY: allow extra Hybrid embedding queries for missing evidence")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    out=args.out.expanduser().resolve()
    _require(out.parent==(HERE/"graph_reports/assist_v216").resolve()
             and out.suffix.lower()==".json",
             "Output must be a new JSON under graph_reports/assist_v216")
    _require(not out.exists(),
             "Refusing to overwrite a prior evaluation JSON")
    if args.offline:
        _require(not args.case and not args.run_branches,
                 "--case/--run-branches require explicit --live mode")
        obj=evaluate_offline(include_pending=args.explore_pending)
    else:
        obj=evaluate_live(cases=args.case,
            include_pending=args.explore_pending,
            run_branches=args.run_branches)
    with out.open("x",encoding="utf-8") as fd:
        json.dump(obj,fd,ensure_ascii=False,indent=2)
        fd.write("\n")
    print("[V2.16] Saved",out)
    print("[V2.16] No existing RAG/Chroma files modified; no graph claims promoted.")

if __name__=="__main__":
    main()
