# -*- coding: utf-8 -*-
"""DouXian V2.15.1: READ-ONLY multi-textbook Hybrid candidate overlap audit.

Investigates whether V2.15 exact-label *offline* graph candidates in the
six-question pilot are actually absent from raw Hybrid Top10 / Top20.

This does NOT promote pending semantic relations and does NOT prove relevance,
improved citations or answer quality. No Reranker or LLM generation is called.
The query-embedding backend may be invoked; costs may apply.

Run as module in the existing project `data` directory:
 python -m graph_tools.audit_multisource_hybrid_v2151 \
    --lookup graph_reports/expansions/multi_source_lookup_run01.json \
    --out graph_reports/expansions/multisource_hybrid_overlap_run01.json
"""
from __future__ import annotations
from collections import defaultdict, Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def digest(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            h.update(block)
    return h.hexdigest()

def _chunk_id(doc):
    meta=getattr(doc,"metadata",None)
    return str(meta.get("chunk_id") or "").strip() if isinstance(meta,dict) else ""

def _source(doc):
    meta=getattr(doc,"metadata",None)
    return str(meta.get("source") or "") if isinstance(meta,dict) else ""

def check_lookup_report(lookup_path:Path, index_obj:dict)->list[dict]:
    """No user-edited `Chunk ID` may silently influence this experiment."""
    from graph_tools.probe_machine_multisource_v215 import run_query
    obj=json.loads(Path(lookup_path).read_text(encoding="utf-8-sig"))
    require(obj.get("phase")=="v215_offline_exact_label_lookup_only",
            "This is not a V2.15 offline lookup report")
    results=obj.get("results")
    require(isinstance(results,list) and len(results)==6,
            "Expected six original V2.15 pilot questions")
    expected_questions=(
        "吉布斯采样和MCMC有什么关系？",
        "马尔可夫随机场的证据来自哪本教材？",
        "变分推断属于哪类推断方法？",
        "随机森林使用哪些方法训练？",
        "隐马尔可夫模型是什么？",
        "条件随机场与马尔可夫随机场有什么关联？",
    )
    require(tuple(r.get("query") for r in results)==expected_questions,
            "Pilot question set/order differs from the frozen V2.15 six-query experiment")
    for i,r in enumerate(results,1):
        rebuilt=run_query(index_obj,r["query"])
        require(r==rebuilt,
                f"V2.15 query #{i} did not reproduce from the original source index; "
                "abort before accessing external embedding services")
        require(r["truncated"] is False and
                r["candidate_count"]==len(r["chunks"]),
                "Truncated/missing candidate set cannot be compared fairly")
        require(len({c["chunk_id"] for c in r["chunks"]})==len(r["chunks"]),
                "Repeated graph candidate ID")
    return results

def analyze_query(
    lookup_row:dict,
    hybrid_docs:list,
    hydrated_by_id:dict[str,tuple[str,dict]]
)->dict:
    """A pure candidate-audit function, separately unit-tested.

    hydrated_by_id[cid] = (page_content, metadata) from verified live Chroma.
    No reranking, graph priority boost, or fabricated direct relation.
    """
    from graph_tools.graph_retriever_v213 import canonical_source
    if not hybrid_docs:
        raise RuntimeError("Original Hybrid returned no candidates")
    rank={}
    for position,doc in enumerate(hybrid_docs,1):
        cid=_chunk_id(doc)
        require(bool(cid) and cid not in rank,
                "Raw Hybrid result contains a missing or duplicate original Chunk ID")
        rank[cid]=position

    candidate_rows=[]
    for entry in lookup_row["chunks"]:
        cid=entry["chunk_id"]
        require(cid in hydrated_by_id,"A graph candidate was not hydrated from live Chroma")
        body,metadata=hydrated_by_id[cid]
        expected_source=entry["source_file"]
        require(canonical_source(metadata.get("source"))==
                canonical_source(expected_source),
                "Graph candidate's source changed in live Chroma: "+cid)
        content_equivalent=[]
        for other in hybrid_docs:
            if (_chunk_id(other)!=cid
                and getattr(other,"page_content",None)==body
                and canonical_source(_source(other))==
                    canonical_source(expected_source)):
                content_equivalent.append((_chunk_id(other),rank[_chunk_id(other)]))
        exact_rank=rank.get(cid)
        equivalents10=[{"chunk_id":x,"hybrid_rank":pos}
                       for x,pos in content_equivalent if pos<=10]
        equivalents20=[{"chunk_id":x,"hybrid_rank":pos}
                       for x,pos in content_equivalent if pos<=20]
        candidate_rows.append({
            "chunk_id":cid,
            "source_file":entry["source_file"],
            "section_path":entry["chapter"],
            "status":entry["status"],
            "semantic_review_pending":entry["status"]!="verified",
            "matched_claim_ids":entry["claim_ids"],
            "hybrid_rank_top20":exact_rank,
            "exact_in_hybrid_top10":exact_rank is not None and exact_rank<=10,
            "exact_in_hybrid_top20":exact_rank is not None and exact_rank<=20,
            "same_body_same_source_other_id_top10":equivalents10,
            "same_body_same_source_other_id_top20":equivalents20,
            "novel_candidate_vs_hybrid_top10":(
                (exact_rank is None or exact_rank>10) and not equivalents10),
            "novel_candidate_vs_hybrid_top20":(
                exact_rank is None and not equivalents20),
        })

    by_book={}
    for book in ("神经网络与深度学习.md","机器学习.md"):
        group=[x for x in candidate_rows if x["source_file"]==book]
        by_book[book]={
            "graph_candidate_chunks":len(group),
            "missing_in_hybrid_top10":sum(x["novel_candidate_vs_hybrid_top10"] for x in group),
            "missing_in_hybrid_top20":sum(x["novel_candidate_vs_hybrid_top20"] for x in group),
            "pending_semantic_review":sum(x["semantic_review_pending"] for x in group),
            "not_reviewed_for_question_relevance":True,
        }
    return {
        "query":lookup_row["query"],
        "matched_labels":lookup_row["matched_labels"],
        "graph_matched_claims":lookup_row["matched_claim_count"],
        "graph_candidate_chunk_count":len(candidate_rows),
        "raw_hybrid_returned":len(hybrid_docs),
        "hybrid_top10_ids":[_chunk_id(x) for x in hybrid_docs[:10]],
        "hybrid_top20_ids":[_chunk_id(x) for x in hybrid_docs[:20]],
        "per_book":by_book,
        "candidate_audit":candidate_rows,
        "total_missing_in_hybrid_top10":sum(x["novel_candidate_vs_hybrid_top10"]
                                           for x in candidate_rows),
        "total_missing_in_hybrid_top20":sum(x["novel_candidate_vs_hybrid_top20"]
                                           for x in candidate_rows),
        "not_measured":[
            "manual_query_specific_evidence_relevance",
            "relation_path_retrieval_value",
            "reranker_top3_survival",
            "final_cited_evidence",
            "answer_quality",
            "reranker_tokens",
        ]
    }

def run_live(lookup_path:Path)->dict:
    from graph_tools.verify_machine_graph_v215 import (
        verify_only,HERE,INDEX_REL,OLD_GRAPH,OLD_LINKS,OLD_ALIASES,SNAPSHOT,SRC
    )
    from graph_tools.graph_retriever_v213 import (
        load_verified_bundle,_live_graph_documents,GraphFiles
    )
    from knowledge_base import KnowledgeBaseService
    import config_data as config

    # First gate: reconstruct all V2.15 graph, links, aliases and lookup
    # from the SHA-pinned uploaded textbook and 1,953-row snapshot.
    evidence_gate=verify_only()
    index_path=HERE/INDEX_REL
    index_obj=json.loads(index_path.read_text(encoding="utf-8"))
    lookup_rows=check_lookup_report(lookup_path,index_obj)

    # Original V2.13 gate checks old textbook, 29 old verified edges,
    # graph SHA, V2.12 gate, snapshot and alias registry.
    verified_bundle=load_verified_bundle()
    paths=GraphFiles()
    require(verified_bundle.total_verified==29,
            "Original V2.12 verified relation count drifted")
    require(evidence_gate["snapshot_sha256"]==digest(paths.chunks),
            "V2.15 snapshot differs from the original verified V2.12 snapshot")
    require(evidence_gate["baseline_graph_sha256"]==digest(paths.graph)
            and evidence_gate["baseline_links_sha256"]==digest(paths.links)
            and evidence_gate["baseline_aliases_sha256"]==digest(paths.aliases),
            "V2.15 source baselines do not match the original verified V2.12 graph")

    # Do not let KnowledgeBaseService.__init__ silently create an empty
    # new Chroma directory at a misspelled configured path.
    persistent=Path(config.persist_directory).expanduser()
    require(persistent.is_dir(),
            "Configured Chroma persist directory does not exist; no DB initialized")
    started=time.perf_counter()
    kb=KnowledgeBaseService()  # original, unaugmented class, bypasses Graph wrappers
    collection_count=kb.chroma._collection.count()
    require(collection_count==1953,
            "Live Chroma collection count differs from frozen 1953-row snapshot")
    index_count=kb.build_bm25_index()
    require(index_count==1953,
            "In-memory BM25 document count differs from original 1953")
    kb.build_hybrid_retriever()
    seconds_index=round(time.perf_counter()-started,3)

    all_candidate_ids=list(dict.fromkeys(
        c["chunk_id"] for row in lookup_rows for c in row["chunks"]))
    require(len(all_candidate_ids)==16,
            "V2.15 pilot source Chunk ID distribution changed")
    require(set(all_candidate_ids).issubset(verified_bundle.chunk_rows),
            "Offline V2.15 source evidence candidate missing from old Chroma snapshot")

    # V2.13's original safe Chroma hydration handles all 1953 rows, even
    # new-book candidates not present in its old 29 semantic edges.
    safe_rows=_live_graph_documents(kb,all_candidate_ids,verified_bundle)
    hydrated=dict(zip(all_candidate_ids,safe_rows))
    require(len(hydrated)==len(all_candidate_ids),
            "Unable to hydrate and validate all original native Chroma Chunk IDs")

    results=[]
    for row in lookup_rows:
        start=time.perf_counter()
        hybrid=kb.search_hybrid(row["query"],k=20)
        seconds=round(time.perf_counter()-start,3)
        entry=analyze_query(row,hybrid,hydrated)
        entry["raw_hybrid_query_seconds"]=seconds
        results.append(entry)
        print("Q:",row["query"])
        print("  Graph candidate chunks:",entry["graph_candidate_chunk_count"],
              "NEW vs Hybrid Top10:",entry["total_missing_in_hybrid_top10"],
              "NEW vs Hybrid Top20:",entry["total_missing_in_hybrid_top20"])
        for book,detail in entry["per_book"].items():
            if detail["graph_candidate_chunks"]:
                print("   ",book,detail["graph_candidate_chunks"],
                      "chunks; Top20-novel:",detail["missing_in_hybrid_top20"])
    new_top20_ids={
        candidate["chunk_id"]
        for result in results for candidate in result["candidate_audit"]
        if candidate["novel_candidate_vs_hybrid_top20"]
    }
    return {
        "experiment":"DouXian_V2.15.1_MultiSource_ReadOnly_RawHybrid_Overlap",
        "phase":"original_hybrid_top20_candidate_overlap_only",
        "run_utc":datetime.now(timezone.utc).isoformat(),
        "offline_lookup_sha256":digest(lookup_path),
        "v215_index_sha256":digest(index_path),
        "v212_graph_sha256":digest(paths.graph),
        "v212_links_sha256":digest(paths.links),
        "v212_snapshot_sha256":digest(paths.chunks),
        "v215_source_sha256":evidence_gate["source_sha256"],
        "chroma_live_chunk_count":collection_count,
        "bm25_in_memory_count":index_count,
        "index_build_seconds":seconds_index,
        "fixed_subqueries":True,
        "query_decomposition_performed":False,
        "raw_hybrid_top_k":20,
        "reranker_calls":0,
        "generation_calls":0,
        "old_29_verified_relations_unchanged":True,
        "new_54_relations_semantically_promoted":False,
        "pilot_queries":len(results),
        "total_distinct_graph_candidate_ids":len(all_candidate_ids),
        "total_distinct_graph_candidate_ids_novel_vs_hybrid_top20":
            len(new_top20_ids),
        "total_query_candidate_novelties_top20":sum(
            r["total_missing_in_hybrid_top20"] for r in results),
        "queries":results,
        "limitations":[
            "This measures exact-label lookup candidate novelty, NOT a multi-hop "
            "graph traversal experiment.",
            "A novel candidate may be irrelevant to its query.",
            "V2.15 new-book relations are evidence-bound but pending independent "
            "semantic review and cannot be used online.",
            "Query embedding calls may incur costs.",
            "No production query decomposition or Qwen Reranker used.",
            "No generated answers, citations, or quality comparisons.",
            "Missing from raw Top20 does not mean missing from the entire collection."
        ],
    }

def main(argv=None):
    from graph_tools.graph_paths import REPORTS_DIR
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lookup",type=Path,
                   default=REPORTS_DIR/"expansions/multi_source_lookup_run01.json")
    p.add_argument("--out",type=Path,
                   default=REPORTS_DIR/"expansions/multisource_hybrid_overlap_run01.json")
    args=p.parse_args(argv)
    report=args.lookup.expanduser().resolve()
    dest=args.out.expanduser().resolve()
    require(report.is_file(),"Expected V2.15 offline lookup JSON: "+str(report))
    require(dest.parent==(REPORTS_DIR/"expansions").resolve()
            and dest.suffix.lower()==".json",
            "Output must be a new JSON directly in graph_reports/expansions")
    require(not dest.exists(),"Output already exists: never overwrite prior report")
    obj=run_live(report)
    with dest.open("x",encoding="utf-8") as fd:
        json.dump(obj,fd,ensure_ascii=False,indent=2)
        fd.write("\n")
    print("[V2.15.1] read-only Hybrid overlap saved:",dest)
    print("[V2.15.1] unique Graph candidate IDs missing from each relevant "
          "raw Hybrid Top20 union:",obj["total_distinct_graph_candidate_ids_novel_vs_hybrid_top20"])
    print("[V2.15.1] NO NEW GRAPH RELATIONS PROMOTED; NO ONLINE CODE MODIFIED")

if __name__=="__main__":
    main()
