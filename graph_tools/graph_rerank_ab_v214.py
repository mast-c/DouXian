# -*- coding: utf-8 -*-
"""V2.14 Reranker A/B, read-only offline/online hybrid candidate test.

The *only* potential enhancement tested is adding one verified new
two-hop Chunk to the same original Hybrid Top10 candidate set. Exact
original KnowledgeBaseService.search_with_rerank is reused.

NO RAG integration, no Chroma writes, no query decomposition, no
LLM answer generation and no automatic claim that a selected
Chunk is actually cited or improves the answer.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from typing import Any

def _require(cond: bool, explanation: str) -> None:
    if not cond:
        raise RuntimeError(explanation)

def _sha_file(path:Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def _doc_id(doc) -> str:
    meta=getattr(doc,"metadata",None)
    return str(meta.get("chunk_id") or "").strip() if isinstance(meta,dict) else ""

def _meta(doc, field) -> str:
    meta=getattr(doc,"metadata",None)
    return str(meta.get(field) or "") if isinstance(meta,dict) else ""

def _prepare_frozen_service(base_class, documents):
    """Use INHERITED production reranker with only search_hybrid substituted.

    Does not call parent __init__ or create a second DB. This is a
    transient test-only instance, not a production monkeypatch.
    """
    pool=list(documents)
    class FrozenCandidates(base_class):
        def __init__(self): # deliberately no Chroma writes/reinitialization
            self._frozen_candidates=pool
        def search_hybrid(self, query:str, k:int=5):
            return list(self._frozen_candidates[:k])
    return FrozenCandidates()

def _parse_rerank_cost(log: str) -> dict:
    match = re.search(
        r"\[Rerank成本\]\s*候选=(\d+)\s*原文字符=(\d+)\s*"
        r"发送字符=(\d+)\s*官方total_tokens=(\d+|未返回)\s*压缩上限=([^\n]+)",
        log
    )
    _require(match is not None,
             "Production reranker did not emit expected cost trace; abort")
    tokens=match.group(4)
    return {
        "candidate_count":int(match.group(1)),
        "full_document_chars":int(match.group(2)),
        "sent_document_chars":int(match.group(3)),
        "official_total_tokens":int(tokens) if tokens.isdecimal() else None,
        "rerank_doc_chars_setting":match.group(5).strip(),
    }

def _ranked_summary(documents:list, graph_novel_ids:set[str]) -> list[dict]:
    return [{
        "rank":idx,
        "chunk_id":_doc_id(doc),
        "source":_meta(doc,"source"),
        "section_path":_meta(doc,"section_path"),
        "relevance_score":float(getattr(doc,"metadata",{}).get("rerank_score",0)),
        "v214_new_graph_chunk":_doc_id(doc) in graph_novel_ids,
    } for idx,doc in enumerate(documents,1)]

def _same_text_and_source(doc_a, doc_b) -> bool:
    from graph_tools.graph_chunk_linker_v211 import canonical_source
    return (getattr(doc_a,"page_content",None)==getattr(doc_b,"page_content",None)
            and canonical_source(_meta(doc_a,"source"))==
                canonical_source(_meta(doc_b,"source")))

def _new_for_top10(base_docs:list, graph_docs:list) -> tuple[list,list]:
    """Exact ID and same-content+source dedupe without touching retrieval."""
    seen={_doc_id(d) for d in base_docs}
    _require(len(seen)==len(base_docs) and "" not in seen,
             "Original Hybrid Top10 has missing or duplicate Chunk ID")
    added=[]
    skipped=[]
    for candidate in graph_docs:
        cid=_doc_id(candidate)
        _require(bool(cid),"Graph candidate is missing its original Chunk ID")
        if cid in seen or any(_same_text_and_source(d,candidate)
                              for d in base_docs+added):
            skipped.append(cid)
            continue
        seen.add(cid)
        added.append(candidate)
    return added,skipped

def _run_actual_production_reranker(
    base_class, query:str, pool:list, top_n:int=3
) -> tuple[list[dict],dict,str]:
    """One paid DashScope call. Fresh frozen test instance per condition."""
    _require(1<=top_n<=len(pool),"Invalid rerank top_n")
    frozen=_prepare_frozen_service(base_class,pool)
    buf=io.StringIO()
    with redirect_stdout(buf):
        ranked=frozen.search_with_rerank(query,recall_k=len(pool),top_n=top_n)
    output=buf.getvalue()
    print(output, end="")
    _require(0<len(ranked)<=top_n,
             "Unexpected empty / excessive result from production reranker")
    _require(all(_doc_id(d) for d in ranked),
             "Production reranker returned a Chunk without its original ID")
    _require(len(set(_doc_id(d) for d in ranked))==len(ranked),
             "Production reranker returned duplicate Chunk IDs")
    return ranked,_parse_rerank_cost(output),output

def _check_input_reports(offline_path:Path, overlap_path:Path):
    from graph_tools.graph_hybrid_overlap_v214 import load_and_validate_report
    offline_obj=load_and_validate_report(offline_path)
    overlap_obj=json.loads(overlap_path.read_text(encoding="utf-8"))
    _require(overlap_obj.get("phase")=="read_only_live_hybrid_candidates_only",
             "Input must be a live unaugmented Hybrid overlap report")
    _require(overlap_obj.get("original_offline_report_sha256")==_sha_file(offline_path),
             "Live overlap report was based on a different offline result")
    _require(overlap_obj.get("chroma_collection_count")==1953 and
             overlap_obj.get("bm25_indexed_chunks")==1953,
             "Unexpected previously audited Chroma/BM25 counts")
    _require(overlap_obj.get("query_decomposition_performed") is False and
             overlap_obj.get("raw_hybrid_top_k")==20 and
             overlap_obj.get("reranker_calls")==0 and
             overlap_obj.get("answer_generation_calls")==0,
             "Unexpected experimental setup in the overlap input")
    offline_by_query={row["query"]:row for row in offline_obj["queries"]}
    for row in overlap_obj.get("results",[]):
        query=row.get("query")
        _require(query in offline_by_query,"Hybrid report includes unknown query")
        src=offline_by_query[query]
        for key_a,key_b in [
            ("v213_graph_candidate_ids","v213_candidate_ids"),
            ("v214_budgeted_graph_candidate_ids","v214_budgeted_candidate_ids"),
            ("v214_new_vs_v213_graph_ids","new_ids_in_budgeted_v214"),
        ]:
            _require(row.get(key_a)==src.get(key_b),
                     "Local report candidate discrepancy: "+key_a)
    return offline_obj,overlap_obj

def _live_verify_origin(overlap_obj:dict, bundle, paths) -> None:
    observed=overlap_obj.get("source_artifact_sha256") or {}
    source_paths={
        "v212_graph":paths.graph, "v212_links":paths.links,
        "v212_aliases":paths.aliases, "v212_snapshot":paths.chunks,
        "v212_gate":paths.gate, "textbook":paths.source,
    }
    for key,path in source_paths.items():
        _require(observed.get(key)==_sha_file(Path(path)),
                 "V2.12 formal source changed since Hybrid audit: "+key)

def _recheck_live_hybrid(kb, input_row:dict, graph_by_id:dict):
    from graph_tools.graph_hybrid_overlap_v214 import analyze_hybrid_overlap
    all_hybrid=kb.search_hybrid(input_row["query"],k=20)
    ids=[_doc_id(x) for x in all_hybrid]
    _require(ids==input_row["hybrid_top20_chunk_ids"],
             "Raw Hybrid Top20 has drifted since prior audit; stop before reranker calls")
    new_audit=analyze_hybrid_overlap(
        all_hybrid,graph_by_id,
        input_row["v213_graph_candidate_ids"],
        input_row["v214_budgeted_graph_candidate_ids"],
        input_row["v214_new_vs_v213_graph_ids"]
    )
    _require(
        new_audit["v214_new_candidate_audit"]==
        input_row["v214_new_candidate_audit"],
        "Graph/Hybrid overlap changed; stop before reranker calls")
    return all_hybrid,new_audit

def run_probe(offline_path:Path, overlap_path:Path)->dict:
    from graph_tools.graph_retriever_v213 import (
        load_verified_bundle, _live_graph_documents, GraphFiles
    )
    from knowledge_base import KnowledgeBaseService
    import config_data as config

    offline, overlap = _check_input_reports(offline_path,overlap_path)
    verified=load_verified_bundle()
    paths=GraphFiles()
    _live_verify_origin(overlap,verified,paths)

    chroma_dir=Path(config.persist_directory).expanduser()
    _require(chroma_dir.is_dir(),
             "Local Chroma directory does not exist; refusing initialization")
    _require(os.getenv("DASHSCOPE_API_KEY"),"DASHSCOPE_API_KEY missing")
    service=KnowledgeBaseService()
    count=service.chroma._collection.count()
    _require(count==overlap["chroma_collection_count"],
             "Chroma collection count changed since prior audit")
    total=service.build_bm25_index()
    _require(total==overlap["bm25_indexed_chunks"],
             "BM25 document count differs from previous audit")
    service.build_hybrid_retriever()

    experiments=[]
    for row in overlap["results"]:
        novelty_ids={
            entry["chunk_id"] for entry in row["v214_new_candidate_audit"]
            if entry["candidate_novel_vs_hybrid_top20"] is True
        }
        if not novelty_ids:
            print("[V2.14 RERANK AB] skip (no Hybrid Top20 candidate novelty):",
                  row["query"])
            continue
        ids=row["v214_budgeted_graph_candidate_ids"]
        _require(all(cid in verified.chunk_rows for cid in ids),
                 "Graph candidate missing from the verified snapshot")
        # V2.13's safety gate intentionally returns (body, metadata)
        # tuples, not LangChain Documents. Preserve the verified tuples
        # for overlap replay; convert to Documents only for reranking.
        verified_rows=_live_graph_documents(service,ids,verified)
        from langchain_core.documents import Document
        live_tuples=dict(zip(ids,verified_rows))
        _require(len(live_tuples)==len(ids) and
                 all(meta.get("chunk_id")==cid
                     for cid,(body,meta) in live_tuples.items()),
                 "V2.14 hydrated Chroma IDs were not preserved")
        live_map={
            cid:Document(page_content=body,metadata=dict(meta))
            for cid,(body,meta) in live_tuples.items()
        }
        all_hybrid,rechecked=_recheck_live_hybrid(
            service,row,live_tuples)
        raw10=all_hybrid[:10]
        eligible=[live_map[cid] for cid in ids if cid in novelty_ids]
        additions,skipped=_new_for_top10(raw10,eligible)
        _require({ _doc_id(doc) for doc in additions }==novelty_ids,
                 "Expected novel graph documents became duplicate/absent")
        _require(1<=len(additions)<=2,
                 "Unexpected count of verified relation-expanded documents")
        _require(row["budget"]==2,
                 "This focused A/B study assumes the original 2-graph candidate budget")

        print("\n[V2.14 RERANK AB] query:",row["query"])
        print("[V2.14 RERANK AB] baseline=HybridTop10; graph_condition=HybridTop10+"+
              str(len(additions))+" verified Chunk(s)")
        start=time.perf_counter()
        base_ranked, base_cost, _=_run_actual_production_reranker(
            KnowledgeBaseService,row["query"],raw10,top_n=3)
        base_sec=round(time.perf_counter()-start,3)
        start=time.perf_counter()
        plus_ranked, plus_cost, _=_run_actual_production_reranker(
            KnowledgeBaseService,row["query"],raw10+additions,top_n=3)
        plus_sec=round(time.perf_counter()-start,3)
        _require(base_cost["candidate_count"]==len(raw10) and
                 plus_cost["candidate_count"]==len(raw10)+len(additions),
                 "Rerank backend did not use the expected exact candidate pools")

        b=[_doc_id(d) for d in base_ranked]
        a=[_doc_id(d) for d in plus_ranked]
        selected=[cid for cid in a if cid in novelty_ids]
        experiments.append({
            "query":row["query"],
            "novel_verified_graph_chunk_ids":sorted(novelty_ids),
            "two_hop_claim_paths":row.get("matched_two_hop_claim_paths",[]),
            "baseline_hybrid_top10_ids":[_doc_id(d) for d in raw10],
            "baseline_rerank_top3_ids":b,
            "graph_augmented_rerank_top3_ids":a,
            "new_graph_chunk_selected_top3_ids":selected,
            "novel_graph_survived_rerank_top3":bool(selected),
            "top3_changed":b!=a,
            "top3_overlap_size":len(set(a)&set(b)),
            "baseline_rerank_top3":_ranked_summary(base_ranked,novelty_ids),
            "graph_augmented_rerank_top3":_ranked_summary(plus_ranked,novelty_ids),
            "baseline_rerank_cost":base_cost,
            "graph_augmented_rerank_cost":plus_cost,
            "extra_rerank_tokens_if_reported":(
                plus_cost["official_total_tokens"]-base_cost["official_total_tokens"]
                if plus_cost["official_total_tokens"] is not None and
                   base_cost["official_total_tokens"] is not None else None
            ),
            "baseline_rerank_call_seconds":base_sec,
            "graph_augmented_rerank_call_seconds":plus_sec,
            "no_answer_generated":True,
            "no_final_rag_citation_ascertained":True,
        })
    _require(experiments,"No novel-vs-Hybrid Top20 graph candidate left to test")
    return {
        "experiment":"DouXian_V2.14_ReadOnly_Production_Reranker_AB",
        "phase":"controlled_single_query_reranker_top3_only",
        "run_utc":datetime.now(timezone.utc).isoformat(),
        "offline_sha256":_sha_file(offline_path),
        "hybrid_overlap_sha256":_sha_file(overlap_path),
        "source_sha256":overlap["source_artifact_sha256"],
        "local_chroma_count":count,
        "fixed_subqueries":True,
        "query_decomposition_performed":False,
        "reranker_backend":"KnowledgeBaseService.search_with_rerank / qwen3-vl-rerank",
        "top_n":3,
        "baseline_pool":"original_unaugmented_hybrid_top10",
        "graph_pool":"same_baseline_top10_plus_new_verified_v214_chunks",
        "force_inclusion_in_rerank_top3":False,
        "reranker_model_calls":len(experiments)*2,
        "answer_generation_calls":0,
        "results":experiments,
        "limitations":[
            "Candidate budget grows from 10 to 11 (or 12) when graph adds docs; "
            "reranker cost may increase.",
            "This is one call per condition; reranking may show model randomness.",
            "Top3 after reranking is not the final cited evidence or answer.",
            "Only fixed single-question Top10 pools are tested, not query decomposition.",
            "No answer generation, final citation comparison or quality judgments."
        ],
    }

def main(argv=None):
    from graph_tools.graph_paths import REPORTS_DIR
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline",type=Path,default=REPORTS_DIR/"graph_v214_local_run01.json")
    p.add_argument("--overlap",type=Path,default=REPORTS_DIR/"graph_v214_hybrid_overlap_run01.json")
    p.add_argument("--out",type=Path,default=REPORTS_DIR/"graph_v214_rerank_ab_run01.json")
    args=p.parse_args(argv)
    offline=args.offline.expanduser().resolve()
    overlap=args.overlap.expanduser().resolve()
    dest=args.out.expanduser().resolve()
    for src in (offline,overlap):
        _require(src.is_file(),"Missing input report: "+str(src))
    _require(dest.parent==REPORTS_DIR.resolve() and dest.suffix.lower()==".json",
             "Output JSON must be directly inside graph_reports/")
    _require(not dest.exists(),
             "Output already exists; give a fresh report name, never overwrite")
    report=run_probe(offline,overlap)
    with dest.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
        f.write("\n")
    print("[Graph V2.14 RERANK AB] report:",dest)
    for x in report["results"]:
        print("QUESTION:",x["query"])
        print("  hybrid_top3:",x["baseline_rerank_top3_ids"])
        print("  hybrid_plus_graph_top3:",x["graph_augmented_rerank_top3_ids"])
        print("  NEW graph Chunk selected:",x["new_graph_chunk_selected_top3_ids"])
        print("  official rerank token delta:",x["extra_rerank_tokens_if_reported"])

if __name__=="__main__":
    main()
