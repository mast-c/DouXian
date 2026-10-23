# -*- coding: utf-8 -*-
"""V2.15 multi-book exact-label evidence lookup demo, OFFLINE ONLY.

This is NOT Graph ON and does NOT use embeddings, BM25, reranking or answer
generation. New V2.15 rows remain pending independent semantic review.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from .verify_machine_graph_v215 import HERE,INDEX_REL,verify_only,normalization

DEFAULT_QUERIES=(
    "吉布斯采样和MCMC有什么关系？",
    "马尔可夫随机场的证据来自哪本教材？",
    "变分推断属于哪类推断方法？",
    "随机森林使用哪些方法训练？",
    "隐马尔可夫模型是什么？",
    "条件随机场与马尔可夫随机场有什么关联？",
)
def match_term(query,label):
    if not label or not isinstance(query,str):
        return False
    if any(ord(ch)>127 for ch in label):
        return normalization(label) in normalization(query)
    return re.search(
        r"(?<![A-Za-z0-9])"+re.escape(label)+r"(?![A-Za-z0-9])",
        query,re.IGNORECASE) is not None

def run_query(index_obj,query,limit=20):
    if not isinstance(query,str) or len(query)>500 or not query.strip():
        return {"query":query,"matched_labels":[],"candidate_count":0,
                "sources":{},"chunks":[]}
    by_label=index_obj["by_exact_label"]
    # Prefer the longest matching term if a shorter exact label occurs ONLY
    # inside the longer label. E.g. "吉布斯采样" must not trigger the generic
    # concept "采样" unless 采样 is mentioned separately in the query.
    hits=[]
    normalized_query=normalization(query)
    for item in by_label.values():
        for word in item["labels"]:
            if not match_term(query,word):
                continue
            needle=normalization(word)
            for hit in re.finditer(re.escape(needle),normalized_query):
                hits.append((hit.start(),hit.end(),word))
    claimed_spans=[]
    labels=set()
    for start,end,word in sorted(hits,
                                 key=lambda x:(-(x[1]-x[0]),x[0],x[2])):
        if any(a<=start and end<=b for a,b in claimed_spans):
            continue
        claimed_spans.append((start,end))
        labels.add(word)
    refs=set()
    for item in by_label.values():
        if any(x in labels for x in item["labels"]):
            refs.update(item["claim_ids"])
    selected=[c for c in index_obj["claims"] if c["claim_id"] in refs]
    count=Counter(c["source_file"] for c in selected)
    chunks=[]
    for c in selected:
        if c["chunk_id"] not in [x["chunk_id"] for x in chunks]:
            chunks.append({
                "chunk_id":c["chunk_id"],
                "source_file":c["source_file"],
                "chapter":c["section_path"],
                "claim_ids":[c["claim_id"]],
                "status":c["status"],
                "online_eligible_in_new_v215":False,
            })
        else:
            for x in chunks:
                if x["chunk_id"]==c["chunk_id"]:
                    x["claim_ids"].append(c["claim_id"])
                    break
    return {
        "query":query,
        "matched_labels":sorted(labels),
        "matched_claim_count":len(selected),
        "candidate_count":len(chunks),
        "sources":dict(sorted(count.items())),
        "chunks":chunks[:limit],
        "truncated":len(chunks)>limit,
        "no_online_promotion":True,
    }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query",action="append",default=[])
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    verify_only()
    index=json.loads((HERE/INDEX_REL).read_text(encoding="utf-8"))
    queries=args.query or DEFAULT_QUERIES
    results=[run_query(index,query) for query in queries]
    for row in results:
        print("Q:",row["query"])
        print("  matched labels:",row["matched_labels"],
              "distinct candidate Chunks:",row["candidate_count"])
        print("  claims by source:",row["sources"])
    if args.out:
        output=args.out.expanduser().resolve()
        reports=(HERE/"graph_reports/expansions").resolve()
        if output.parent!=reports or output.suffix.lower()!=".json":
            raise ValueError("Report must be a JSON inside graph_reports/expansions/")
        with output.open("x",encoding="utf-8") as f:
            json.dump({
                "phase":"v215_offline_exact_label_lookup_only",
                "warning":"New relations are pending semantic review; not a Hybrid or Reranker evaluation.",
                "results":results,
            },f,ensure_ascii=False,indent=2)
            f.write("\n")
        print("Saved read-only lookup demo:",output)

if __name__=="__main__":
    main()
