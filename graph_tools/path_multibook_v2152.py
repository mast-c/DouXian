# -*- coding: utf-8 -*-
"""DouXian V2.15.2 controlled cross-textbook two-hop candidate experiment.

OFFLINE (default): validates original 29 verified edges, 54 pending-new
edge bindings, frozen source SHA references and 3 fixed cross-book paths.
LIVE (--live): compares both exact evidence Chunk IDs (especially the
second hop) against UNMODIFIED Hybrid Top10/Top20 on exactly 3 questions.
No answer generation, Qwen reranking, Streamlit patches, relation promotion,
Chroma writes, or inferred direct A-to-C knowledge edge.

Use as a module inside project's `data` folder.
"""
from __future__ import annotations
from datetime import datetime,timezone
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import time

from .verify_machine_graph_v215 import (
    HERE, OLD_GRAPH, OLD_LINKS, OLD_ALIASES, SNAPSHOT, SRC, LINK_REL,
    GRAPH_REL, INDEX_REL, verify_only
)

CASEFILE=HERE/"graph_data/expansions/v2152_fixed_paths.json"
EXPECTED_CASES=("gibbs_mcmc_markov","mh_mcmc_markov","undirected_mrf_potential")
EXPECTED_PATTERNS=(
    ("IS_A","BASED_ON"),
    ("IS_A","BASED_ON"),
    ("ALSO_CALLED","USES"),
)
EXPECTED_STARTS=("吉布斯采样","Metropolis-Hastings算法","无向图模型")
EXPECTED_ENDS=("马尔可夫链","马尔可夫链","势函数")

def fail_unless(flag,description):
    if not flag:
        raise RuntimeError(description)

def sha(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):
            h.update(b)
    return h.hexdigest()

def load_and_verify_cases():
    """Re-read all frozen source material; never trust a JSON 'verified' flag alone."""
    gate=verify_only()
    spec=json.loads(CASEFILE.read_text(encoding="utf-8"))
    fail_unless(spec.get("phase")=="offline_pending_semantic_review",
                "This experiment must remain OFFLINE before semantic promotion")
    cases=spec.get("cases")
    fail_unless(isinstance(cases,list) and len(cases)==3
                and tuple(x.get("case_id") for x in cases)==EXPECTED_CASES,
                "V2.15.2 frozen 3-case sequence differs")
    fail_unless(len(spec.get("negative_controls",[]))==2
                and all(x.get("expected_path_expansion") is False
                        for x in spec["negative_controls"]),
                "Expected two explicit no-expansion control questions")

    old=json.loads(OLD_LINKS.read_text(encoding="utf-8"))
    new=json.loads((HERE/LINK_REL).read_text(encoding="utf-8"))
    fail_unless(len(old["claims"])==29 and len(new["claims"])==54,
                "Frozen claim count has drifted")
    edges={}
    for c in old["claims"]:
        fail_unless(c.get("status")=="verified",
                    "Previously promoted V2.12 edge status is no longer verified")
        edges[c["claim_id"]]={**c,"book_role":"old_verified"}
    for c in new["claims"]:
        fail_unless(c.get("status")=="evidence_bound_pending_independent_semantic_review"
                    and c.get("online_eligible") is False,
                    "V2.15 new edge unexpectedly promoted/modified")
        fail_unless(c["claim_id"] not in edges,
                    "Reused edge ID between books")
        edges[c["claim_id"]]={**c,"book_role":"new_pending_review"}
    # Reported live collection is indexed by THESE native IDs, not by textual paragraphs.
    all_chunks={}
    for lineno,line in enumerate(SNAPSHOT.read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip():
            continue
        item=json.loads(line)
        cid=item["chunk_id"]
        fail_unless(cid not in all_chunks,
                    f"Duplicate frozen Chroma ID at snapshot line {lineno}")
        all_chunks[cid]=item
    fail_unless(len(all_chunks)==1953,
                "Unexpected Chroma snapshot length in path report")

    compiled=[]
    for idx,case in enumerate(cases):
        paths=case.get("path_claim_ids")
        fail_unless(isinstance(paths,list) and len(paths)==2 and
                    all(c in edges for c in paths),
                    "Unrecognized/missing relation ID in "+case.get("case_id",""))
        e1,e2=[edges[c] for c in paths]
        fail_unless(e1["book_role"]!=e2["book_role"],
                    "Both path edges come from the same textbook")
        fail_unless((e1["predicate"],e2["predicate"])==EXPECTED_PATTERNS[idx],
                    "Path directed predicate pattern unexpectedly changed")
        fail_unless(e1["object"]==e2["subject"]
                    and e1["subject"]==EXPECTED_STARTS[idx]
                    and e2["object"]==EXPECTED_ENDS[idx],
                    "Path labels/direction do not form A->B->C")
        fail_unless(e1["subject"] in case["query"] and
                    e1["object"] not in case["query"] and
                    e2["object"] not in case["query"],
                    "Question leaks intermediate/destination terms or misses A")
        fail_unless(len(e1["chunk_ids"])==len(e2["chunk_ids"])==1,
                    "Evidence edge cannot resolve to a unique frozen native Chunk ID")
        for claim in (e1,e2):
            row=all_chunks.get(claim["chunk_ids"][0])
            fail_unless(bool(row),
                        "Edge's original Chunk ID absent from frozen snapshot")
            fail_unless(row["source"]==claim["source_file"]
                        and row["section_path"].endswith(claim["section_path"])
                        and claim["evidence_text"] in row["content"],
                        "Path edge lacks exact source/section/evidence quotation")
        fail_unless(e1["chunk_ids"][0]!=e2["chunk_ids"][0],
                    "Two-hop path has no distinct second evidence Chunk")
        compiled.append({
            "case_id":case["case_id"],
            "query":case["query"],
            "intent":case["intent"],
            "sequence":[e1["subject"],e1["object"],e2["object"]],
            "edge_predicates":[e1["predicate"],e2["predicate"]],
            "first_hop_chunk_id":e1["chunk_ids"][0],
            "second_hop_chunk_id":e2["chunk_ids"][0],
            "cross_book":True,
            "new_book_edge_semantic_review_pending":True,
            "online_eligible":False,
            "edge_evidence":[
                {
                    "claim_id":c["claim_id"],
                    "book_role":c["book_role"],
                    "source_file":c["source_file"],
                    "section_path":c["section_path"],
                    "source_line":c["source_line"],
                    "subject":c["subject"],
                    "predicate":c["predicate"],
                    "object":c["object"],
                    "source_quote":c["evidence_text"],
                    "chunk_id":c["chunk_ids"][0],
                    "qualifier":c.get("qualifier","")
                } for c in (e1,e2)
            ],
            "risk_note":case["risk_note"],
            "scope":"Two verified-by-quote relation edges; not an inferred A-to-C semantic claim."
        })
    return {
        "experiment":"DouXian_V2.15.2_CrossBook_TwoHop_ReadOnly",
        "phase":"offline_path_candidates_semantic_review_pending",
        "source_sha256":{
            "old_graph":sha(OLD_GRAPH),
            "old_claims":sha(OLD_LINKS),
            "old_aliases":sha(OLD_ALIASES),
            "frozen_snapshot":sha(SNAPSHOT),
            "new_textbook":sha(HERE/"graph_data/sources/机器学习.md"),
            "new_graph":sha(HERE/GRAPH_REL),
            "new_claims":sha(HERE/LINK_REL),
            "new_index":sha(HERE/INDEX_REL),
            "cases":sha(CASEFILE),
        },
        "source_rows":gate["snapshot_total_chunk_count"],
        "old_verified_relations":29,
        "new_pending_relations":54,
        "two_hop_cross_book_cases":compiled,
        "negative_controls":spec["negative_controls"],
        "no_online_promotion":True,
        "not_measured":["actual_live_chroma_content","hybrid_overlap",
                         "relevance_manual_judgement","reranker_top3",
                         "final_answer_or_citation"]
    }

def run_live(offline_obj:dict)->dict:
    from graph_tools.graph_retriever_v213 import (
        load_verified_bundle,_live_graph_documents,GraphFiles
    )
    from graph_tools.audit_multisource_hybrid_v2151 import analyze_query
    from knowledge_base import KnowledgeBaseService
    import config_data as config

    # Independently re-run the old 29-relations promotion gate and textbook SHA.
    old_bundle=load_verified_bundle()
    fail_unless(old_bundle.total_verified==29,
                "Legacy V2.12 no longer validates its 29 promoted claims")
    sources=GraphFiles()
    fail_unless(sha(sources.chunks)==offline_obj["source_sha256"]["frozen_snapshot"],
                "Old gate/Chroma snapshot and new path cases disagree")
    database=Path(config.persist_directory).expanduser()
    fail_unless(database.is_dir(),
                "Configured Chroma directory does not exist. No DB initialized.")
    start=time.perf_counter()
    kb=KnowledgeBaseService()
    count=kb.chroma._collection.count()
    fail_unless(count==1953,
                "Live Chroma collection is not 1953 chunks")
    bm25_count=kb.build_bm25_index()
    fail_unless(bm25_count==1953,
                "Read-only in-memory BM25 index count drifted")
    kb.build_hybrid_retriever()
    build_time=round(time.perf_counter()-start,3)

    all_ids=list(dict.fromkeys(
        cid for case in offline_obj["two_hop_cross_book_cases"]
        for cid in (case["first_hop_chunk_id"],case["second_hop_chunk_id"])
    ))
    fail_unless(len(all_ids)<=6,"Unexpected number of path evidence IDs")
    hydrated=_live_graph_documents(kb,all_ids,old_bundle)
    ids_to_rows=dict(zip(all_ids,hydrated))
    fail_unless(len(ids_to_rows)==len(all_ids),
                "Live Chroma evidence Chunk ID hydration failed")
    comparisons=[]
    for case in offline_obj["two_hop_cross_book_cases"]:
        q=case["query"]
        start=time.perf_counter()
        hybrid=kb.search_hybrid(q,k=20)
        seconds=round(time.perf_counter()-start,3)
        fake_lookup={
            "query":q,
            "matched_labels":[case["sequence"][0]],
            "matched_claim_count":2,
            "chunks":[
                {"chunk_id":c["chunk_id"],"source_file":c["source_file"],
                 "chapter":c["section_path"],
                 "claim_ids":[c["claim_id"]],
                 "status":"verified" if c["book_role"]=="old_verified"
                 else "evidence_bound_pending_independent_semantic_review"}
                for c in case["edge_evidence"]
            ]
        }
        audit=analyze_query(fake_lookup,hybrid,ids_to_rows)
        second_id=case["second_hop_chunk_id"]
        second=next(x for x in audit["candidate_audit"]
                    if x["chunk_id"]==second_id)
        comparisons.append({
            "case_id":case["case_id"],
            "query":q,
            "first_hop_id":case["first_hop_chunk_id"],
            "second_hop_id":second_id,
            "first_hop_source":case["edge_evidence"][0]["source_file"],
            "second_hop_source":case["edge_evidence"][1]["source_file"],
            "two_hop_second_edge_evidence":case["edge_evidence"][1]["source_quote"],
            "first_hop_hybrid_rank":next(x["hybrid_rank_top20"]
                    for x in audit["candidate_audit"]
                    if x["chunk_id"]==case["first_hop_chunk_id"]),
            "second_hop_hybrid_rank":second["hybrid_rank_top20"],
            "second_hop_novel_vs_hybrid_top10":second["novel_candidate_vs_hybrid_top10"],
            "second_hop_novel_vs_hybrid_top20":second["novel_candidate_vs_hybrid_top20"],
            "second_hop_content_duplicates_top20":second["same_body_same_source_other_id_top20"],
            "raw_hybrid_query_seconds":seconds,
            "raw_hybrid_top10_ids":audit["hybrid_top10_ids"],
            "raw_hybrid_top20_ids":audit["hybrid_top20_ids"],
            "all_evidence_checks":"same_native_Chroma_ID_content_source_section_as_frozen_snapshot",
            "second_hop_semantic_relevance_to_query":"not_yet_human_labeled",
        })
        print("Q:",q)
        print("  second-hop",second_id[:15]+"...",
              "Top20 rank:",second["hybrid_rank_top20"],
              "NEW vs Top10:",second["novel_candidate_vs_hybrid_top10"],
              "NEW vs Top20:",second["novel_candidate_vs_hybrid_top20"])

    return {
        "experiment":offline_obj["experiment"],
        "phase":"read_only_cross_book_path_candidate_vs_unaugmented_hybrid",
        "run_utc":datetime.now(timezone.utc).isoformat(),
        "offline_config_sha256":offline_obj["source_sha256"],
        "chroma_chunks":count,
        "bm25_in_memory_chunks":bm25_count,
        "bm25_build_seconds":build_time,
        "fixed_queries":True,
        "unaugmented_hybrid_top_k":20,
        "query_decomposition_performed":False,
        "reranker_calls":0,
        "answer_generation_calls":0,
        "candidate_budget":"one first-hop and one second-hop ID per case, not injected online",
        "new_semantic_relations_promoted":False,
        "path_cases":comparisons,
        "second_hop_new_vs_top10_count":sum(x["second_hop_novel_vs_hybrid_top10"]
                                          for x in comparisons),
        "second_hop_new_vs_top20_count":sum(x["second_hop_novel_vs_hybrid_top20"]
                                          for x in comparisons),
        "limitations":[
            "These are three purpose-built grounded relation questions, not a statistically representative query set.",
            "New-book relation edges are still pending independent semantic review.",
            "Even absent-from-Hybrid evidence may not be relevant or useful.",
            "No online Graph ON injection, reranking or final citation/answer comparison.",
            "Full question embedding may call a paid API; no inference LLM is used."
        ],
    }

def main(argv=None):
    from graph_tools.graph_paths import REPORTS_DIR
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--offline",action="store_true",
                       help="Zero API calls. Verify the 3 real two-book paths.")
    group.add_argument("--live",action="store_true",
                       help="3 queries to original live Hybrid; Embedding API may charge.")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    target=args.out.expanduser().resolve()
    reports=(REPORTS_DIR/"expansions").resolve()
    fail_unless(target.parent==reports and target.suffix.lower()==".json",
                "Output must be new JSON under graph_reports/expansions/")
    fail_unless(not target.exists(),
                "Existing report cannot be overwritten; select a new filename.")
    offline=load_and_verify_cases()
    result=run_live(offline) if args.live else offline
    with target.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write("\n")
    print("[V2.15.2] report:",target)
    print("[V2.15.2] no Graph edges promoted; production RAG untouched")

if __name__=="__main__":
    main()
