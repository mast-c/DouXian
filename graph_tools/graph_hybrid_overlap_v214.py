# -*- coding: utf-8 -*-
"""DouXian V2.14 read-only live-Hybrid overlap audit.

Reads V2.14 *offline* report and frozen V2.12 graph/verified Chroma bindings.
Uses original KnowledgeBaseService.search_hybrid directly, not the Graph wrapper.
NO Reranker, no LLM, no index uploads/deletes, and no rag.py changes.

Example (from data/):
    python -m graph_tools.graph_hybrid_overlap_v214 \
      --report graph_reports/graph_v214_local_run01.json \
      --out graph_reports/graph_v214_hybrid_overlap_run01.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone

def _require(ok: bool, message: str):
    if not ok:
        raise RuntimeError(message)

def _hash_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def _cid(doc) -> str:
    meta=getattr(doc,"metadata",None)
    return str(meta.get("chunk_id") or "").strip() if isinstance(meta,dict) else ""

def _source(doc) -> str:
    meta=getattr(doc,"metadata",None)
    return str(meta.get("source") or "") if isinstance(meta,dict) else ""

def _rank_map(documents: list) -> dict[str,int]:
    result={}
    for pos,doc in enumerate(documents,1):
        cid=_cid(doc)
        _require(bool(cid),f"Hybrid rank {pos} has no reliable metadata.chunk_id; refusing an ID comparison")
        _require(cid not in result,f"Hybrid contains duplicate Chunk ID: {cid}")
        result[cid]=pos
    return result

def analyze_hybrid_overlap(hybrid_docs, graph_docs_by_id: dict, v213_ids: list[str],
                           v214_ids: list[str], v214_new_ids: list[str]) -> dict:
    """Pure function: exact identity and same-content/same-source stay separate.

    graph_docs_by_id values: (actual_live_content, actual_live_metadata).
    `hybrid_docs` are the *unaugmented* ranked Hybrid candidates.
    """
    rank=_rank_map(hybrid_docs)
    _require(len(hybrid_docs) > 0, "Hybrid returned zero candidates; no valid comparison")
    _require(all(cid in graph_docs_by_id for cid in v214_ids + v213_ids),
             "Missing live Graph chunk hydration")
    _require(set(v214_new_ids) == (set(v214_ids)-set(v213_ids)),
             "Report new IDs mismatch between candidate lists")

    from graph_tools.graph_chunk_linker_v211 import canonical_source
    def same_body_source(graph_id: str, top_k: int) -> list[str]:
        body, meta = graph_docs_by_id[graph_id]
        source_key = canonical_source(meta.get("source"))
        hits=[]
        for doc in hybrid_docs[:top_k]:
            if (_cid(doc) != graph_id
                and getattr(doc,"page_content",None) == body
                and canonical_source(_source(doc)) == source_key):
                hits.append(_cid(doc))
        return hits

    rows=[]
    for cid in v214_new_ids:
        r=rank.get(cid)
        equivalent10=same_body_source(cid,10)
        equivalent20=same_body_source(cid,20)
        rows.append({
            "chunk_id": cid,
            "hybrid_rank_top20": r,
            "exact_in_hybrid_top10": r is not None and r<=10,
            "exact_in_hybrid_top20": r is not None and r<=20,
            "same_content_other_id_top10": equivalent10,
            "same_content_other_id_top20": equivalent20,
            "candidate_novel_vs_hybrid_top10": (r is None or r>10) and not equivalent10,
            "candidate_novel_vs_hybrid_top20": r is None and not equivalent20,
        })
    return {
        "hybrid_returned":len(hybrid_docs),
        "hybrid_top10_chunk_ids":[_cid(doc) for doc in hybrid_docs[:10]],
        "hybrid_top20_chunk_ids":[_cid(doc) for doc in hybrid_docs[:20]],
        "v213_graph_candidate_id_ranks":{cid:rank.get(cid) for cid in v213_ids},
        "v214_graph_candidate_id_ranks":{cid:rank.get(cid) for cid in v214_ids},
        "v214_new_candidate_audit":rows,
        "v214_new_candidate_count":len(rows),
        "v214_new_vs_hybrid_top10":sum(r["candidate_novel_vs_hybrid_top10"] for r in rows),
        "v214_new_vs_hybrid_top20":sum(r["candidate_novel_vs_hybrid_top20"] for r in rows),
        "not_measured":["rerank_survival","final_evidence","answer_quality",
                        "citation_correctness","reranker_tokens"],
    }

def load_and_validate_report(report_path:Path):
    from graph_tools.graph_path_probe_v214 import load_offline_bundle, probe_query
    obj=json.loads(report_path.read_text(encoding="utf-8"))
    _require(isinstance(obj,dict) and obj.get("phase")=="offline_candidates_only",
             "Not a V2.14 offline candidates report")
    bundle=load_offline_bundle()
    _require(obj.get("source_artifact_sha256")==bundle.file_hashes,
             "Local frozen graph SHA does not match the uploaded offline report")
    queries=obj.get("queries")
    _require(isinstance(queries,list) and queries,"Report has no queries")
    seen=set()
    for row in queries:
        query=row.get("query")
        budget=row.get("budget")
        _require(isinstance(query,str) and query.strip() and query not in seen,
                 "Missing or repeated query in offline report")
        _require(type(budget) is int and 1<=budget<=3,
                 "Invalid candidate budget")
        seen.add(query)
        reproduced=probe_query(query,bundle,max_graph_candidates=budget)
        keys=("v213_candidate_ids","v214_budgeted_candidate_ids",
              "new_ids_in_budgeted_v214")
        for key in keys:
            _require(row.get(key)==reproduced[key],
                     "Offline report changed from reproducible graph candidate outputs: "+key)
    return obj

def run_audit(report_path:Path, only_new:bool=True):
    from graph_tools.graph_retriever_v213 import (
        load_verified_bundle, _live_graph_documents, GraphFiles
    )
    from knowledge_base import KnowledgeBaseService
    import config_data as config

    loaded=load_and_validate_report(report_path)
    subset=[row for row in loaded["queries"]
            if not only_new or row.get("new_ids_in_budgeted_v214")]
    _require(bool(subset),
             "No novel-vs-V2.13 candidate queries; use --all or supply a different report")

    # Original V2.13 verifies the graph, source Markdown, 1,953-row snapshot
    # and gate report. Never guess or weaken its verification rules.
    graph_bundle=load_verified_bundle()
    paths=GraphFiles()
    exact_source_hash={
        "v212_graph":_hash_file(paths.graph),
        "v212_links":_hash_file(paths.links),
        "v212_aliases":_hash_file(paths.aliases),
        "v212_snapshot":_hash_file(paths.chunks),
        "v212_gate":_hash_file(paths.gate),
        "textbook":_hash_file(paths.source),
    }
    # Avoid KnowledgeBaseService's os.makedirs creating an empty persistent DB.
    chroma_dir=Path(config.persist_directory).expanduser()
    _require(chroma_dir.is_dir(),
             "Configured Chroma persist directory does not exist; refusing initialization")
    started=time.perf_counter()
    kb=KnowledgeBaseService()  # bare/original class: Graph OFF regardless of environment flags
    local_collection_count=kb.chroma._collection.count()
    _require(local_collection_count > 0, "Empty Chroma collection; refusing candidate audit")
    bm25_count=kb.build_bm25_index()
    _require(bm25_count>0, "Empty BM25 index")
    kb.build_hybrid_retriever()
    index_seconds=round(time.perf_counter()-started,3)

    audited=[]
    for row in subset:
        query=row["query"]
        ids=list(dict.fromkeys(row["v213_candidate_ids"]+
                               row["v214_budgeted_candidate_ids"]))
        _require(all(cid in graph_bundle.chunk_rows for cid in ids),
                 "A candidate ID was not in the original verified Chroma snapshot")
        # Exactly the same V2.13 live Chroma content+metadata safety gate.
        hydrated=_live_graph_documents(kb, ids, graph_bundle)
        live_by_id=dict(zip(ids,hydrated))
        start=time.perf_counter()
        # The underlying V2.13 Hybrid source uses k=20 vector, k=20 BM25,
        # RRF, and returns a ranked list. The top 10 is its first 10.
        hybrid=kb.search_hybrid(query,k=20)
        seconds=round(time.perf_counter()-start,3)
        metrics=analyze_hybrid_overlap(
            hybrid, live_by_id,
            row["v213_candidate_ids"],
            row["v214_budgeted_candidate_ids"],
            row["new_ids_in_budgeted_v214"],
        )
        audited.append({
            "query":query,
            "intent_gates":row.get("intent_gates",[]),
            "budget":row["budget"],
            "v213_graph_candidate_ids":row["v213_candidate_ids"],
            "v214_budgeted_graph_candidate_ids":row["v214_budgeted_candidate_ids"],
            "v214_new_vs_v213_graph_ids":row["new_ids_in_budgeted_v214"],
            "matched_two_hop_claim_paths":[
                [e["claim_id"] for e in path["edges"]]
                for path in row.get("two_hop_paths",[])
            ],
            "candidate_validation":"live_chroma_equal_to_verified_v212_snapshot",
            "raw_hybrid_query_seconds":seconds,
            **metrics
        })
    return {
        "experiment":"DouXian_V2.14_Live_Hybrid_Overlap_Audit",
        "phase":"read_only_live_hybrid_candidates_only",
        "run_utc":datetime.now(timezone.utc).isoformat(),
        "original_offline_report":str(report_path.resolve()),
        "original_offline_report_sha256":_hash_file(report_path),
        "source_artifact_sha256":exact_source_hash,
        "chroma_collection_count":local_collection_count,
        "bm25_indexed_chunks":bm25_count,
        "index_build_seconds":index_seconds,
        "fixed_subqueries":True,
        "query_decomposition_performed":False,
        "raw_hybrid_top_k":20,
        "reranker_calls":0,
        "answer_generation_calls":0,
        "graph_candidate_budget":sorted(set(row["budget"] for row in subset)),
        "queries_checked":len(audited),
        "total_v214_new_vs_v213_graph_ids":sum(
            x["v214_new_candidate_count"] for x in audited),
        "total_truly_novel_vs_raw_hybrid_top10":sum(
            x["v214_new_vs_hybrid_top10"] for x in audited),
        "total_truly_novel_vs_raw_hybrid_top20":sum(
            x["v214_new_vs_hybrid_top20"] for x in audited),
        "results":audited,
        "limitations":[
            "New compared with V2.13 graph != new compared with Hybrid.",
            "This uses original raw Hybrid on fixed full question; does not reproduce LLM query decomposition.",
            "Chroma embedding lookups may incur embedding API calls and latency.",
            "No Reranker, citations, final answer, or answer quality evaluation.",
            "Novelty relative to top K is not proof no deeper Hybrid rank exists."
        ]
    }

def main(argv=None):
    from graph_tools.graph_paths import REPORTS_DIR
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",type=Path,
                   default=REPORTS_DIR/"graph_v214_local_run01.json")
    p.add_argument("--out",type=Path,
                   default=REPORTS_DIR/"graph_v214_hybrid_overlap_run01.json")
    p.add_argument("--all",action="store_true",
                   help="Audit all offline questions, not only queries with V2.14-new IDs")
    args=p.parse_args(argv)
    report_path=args.report.expanduser().resolve()
    dest=args.out.expanduser().resolve()
    _require(report_path.is_file(),"Offline report not found: "+str(report_path))
    _require(dest.parent==REPORTS_DIR.resolve() and dest.suffix.lower()==".json",
             "--out must be a JSON file directly in graph_reports")
    _require(not dest.exists(),
             "Output exists: choose a new report name, never overwrite")
    results=run_audit(report_path,only_new=not args.all)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("x",encoding="utf-8") as file:
        json.dump(results,file,ensure_ascii=False,indent=2)
        file.write("\n")
    print("[Graph V2.14] saved",dest)
    print("[Graph V2.14] raw Hybrid novelty: new-vs-V2.13=",
          results["total_v214_new_vs_v213_graph_ids"],
          "new-vs-Hybrid-top10=",results["total_truly_novel_vs_raw_hybrid_top10"],
          "new-vs-Hybrid-top20=",results["total_truly_novel_vs_raw_hybrid_top20"])
    for entry in results["results"]:
        print("QUESTION:",entry["query"])
        for row in entry["v214_new_candidate_audit"]:
            print("  Chunk",row["chunk_id"][:16]+"...",
                  "Hybrid rank",row["hybrid_rank_top20"],
                  "content-equal IDs",row["same_content_other_id_top20"],
                  "novel top10",row["candidate_novel_vs_hybrid_top10"],
                  "novel top20",row["candidate_novel_vs_hybrid_top20"])

if __name__=="__main__":
    main()
