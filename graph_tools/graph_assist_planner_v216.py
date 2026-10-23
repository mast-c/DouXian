# -*- coding: utf-8 -*-
"""DouXian V2.16 - deterministic, source-grounded Graph Assist query planner.

Design contract:
- NO mutation of original user question or existing RAG subqueries.
- NO Qwen/DashScope/Chroma calls in plan() or load_graph().
- NO generation of direct A->C claims from two verified A->B, B->C edges.
- NO new-book relation receives verified status; pending relations optional,
  explicitly exploratory OFFLINE ONLY.
- Only old V2.12 promoted edges qualify for verified-only planning.
- Distinct query branches are returned separately with directional edge
  paths and ORIGINAL Chroma Chunk IDs.
- Branches containing identical evidence are deduplicated.
- Definition, comparison and unmatched queries are deliberately left alone.

Experiment-only; production rag.py and original graph_assist_v28.py untouched.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any

# Strict, deliberately small pattern grammar.
# Each path pattern represents individually sourced DIRECTED edge relations.
PATTERNS = {
    "theoretical_basis": (
        (("IS_A", "BASED_ON"), 150),
        (("SPECIAL_CASE_OF", "BASED_ON"), 95),
        (("BASED_ON",), 103),
    ),
    "mechanism": (
        (("HAS_PART", "USES"), 154),
        (("IS_A", "BASED_ON"), 142),
        (("ALSO_CALLED", "USES"), 118),
        (("BASED_ON",), 99),
        (("USES",), 83),
        (("VARIANT_OF",), 73),
    ),
    "composition": (
        (("HAS_PART",), 124),
        (("HAS_PART", "USES"), 66),
    ),
    "hierarchy": (
        (("IS_A",), 120),
        (("SPECIAL_CASE_OF",), 92),
        (("IS_A", "IS_A"), 70),
    ),
    "relation": (
        (("IS_A",), 104),
        (("SPECIAL_CASE_OF",), 91),
        (("VARIANT_OF",), 75),
        (("ALSO_CALLED",), 65),
    ),
}
PENDING_ALLOWED = frozenset({
    "IS_A","HAS_PART","USES","BASED_ON",
    "SPECIAL_CASE_OF","VARIANT_OF",
})
# New unreviewed equality and conditional-form claims are NEVER treated as
# verified aliases or traversable semantics in the first exploratory release.
PENDING_BLOCKED = frozenset({"ALSO_CALLED","CONDITIONAL_FORM_OF"})
MAX_QUESTION_LENGTH = 220
# Exact V2.15.1 high-risk semantic screening file shipped in the previous
# installed offline-audit package; a changed/missing file FAILS CLOSED.
FROZEN_V2151_RISK_SHA256 = (
    "93c348945e644b0e69e43bdc2148f613c81702548560694214ad5fe7f62b81d8"
)

RE_THEORY = re.compile(
    r"理论依据|理论基础|基本原理|依赖哪|依赖什么|"
    r"随机过程|采样机制|基于什么|为什么|推导依据"
)
RE_MECHANISM = re.compile(
    r"怎么|怎样|如何|工作机制|训练机制|原理|机制|"
    r"工作过程|训练|计算|实现|产生|采用什么|"
    r"通过什么|使用什么|如何使用|表达概率|怎样进行"
)
RE_COMPOSITION = re.compile(
    r"由什么组成|由哪些.*组成|包含什么|包含哪些|包括什么|"
    r"包括哪些|有哪些层|哪些模块|结构|组成"
)
RE_HIERARCHY = re.compile(
    r"属于|哪类|分类|类别|类型|哪种|特例|子方法"
)
RE_RELATION = re.compile(r"什么关系|有何关系|之间的关系|关联|从属关系")
RE_COMPARISON = re.compile(r"区别|差异|异同|比较|对比|优劣|不同之处")
RE_DEFINITION = re.compile(r"是什么|什么是|定义|概念|指什么|含义")

def _require(ok:bool, why:str)->None:
    if not ok: raise RuntimeError(why)

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with Path(p).open("rb") as fd:
        for block in iter(lambda:fd.read(1048576),b""):
            h.update(block)
    return h.hexdigest()

def key(label:str)->str:
    return re.sub(r"\s+","",str(label or "")).casefold()

def _has_term_at(s:str,term:str)->list[tuple[int,int]]:
    """ASCII word boundaries, Chinese literal substrings."""
    if not term:
        return []
    if all(ord(c)<128 for c in term):
        pat=r"(?<![A-Za-z0-9])"+re.escape(term)+r"(?![A-Za-z0-9])"
        return [(m.start(),m.end()) for m in re.finditer(pat,s,re.IGNORECASE)]
    return [(m.start(),m.end()) for m in re.finditer(re.escape(term),s)]

def term_matches(s:str,term:str)->bool:
    return bool(_has_term_at(s,term))

def detect_intent(question:str)->str:
    """Specific tasks beat a superficial '是什么' substring."""
    if not isinstance(question,str) or not question.strip():
        return "invalid"
    if len(question)>MAX_QUESTION_LENGTH:
        return "too_long"
    if RE_COMPARISON.search(question):
        return "comparison_unsupported"
    if RE_THEORY.search(question):
        return "theoretical_basis"
    if RE_MECHANISM.search(question):
        return "mechanism"
    if RE_COMPOSITION.search(question):
        return "composition"
    if RE_HIERARCHY.search(question):
        return "hierarchy"
    if RE_RELATION.search(question):
        return "relation"
    if RE_DEFINITION.search(question):
        return "definition"
    return "no_supported_intent"

def _read_json(path:Path)->dict:
    o=json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(o,dict),"Expected JSON object: "+str(path))
    return o

def _clone_edge(claim:dict,role:str)->dict:
    return {
        "claim_id":claim["claim_id"],
        "subject":claim["subject"],
        "predicate":claim["predicate"],
        "object":claim["object"],
        "source_file":claim["source_file"],
        "section_path":claim["section_path"],
        "source_line":claim["source_line"],
        "evidence_quote":claim["evidence_text"],
        "chunk_id":claim["chunk_ids"][0],
        "qualifier":claim.get("qualifier") or "",
        "role":role,
        "status":claim["status"],
    }

def load_graph(include_pending:bool=False, project_root:Path|None=None)->dict:
    """Offline frozen graph loading with V2.15 re-generation & evidence gate.

    `include_pending=True` is ALLOWED ONLY for offline exploratory planning;
    it never promotes pending edges, verified aliases or online eligibility.
    """
    from .verify_machine_graph_v215 import (
        HERE,verify_only,OLD_GRAPH,OLD_LINKS,OLD_ALIASES,SNAPSHOT,
        LINK_REL,GRAPH_REL,INDEX_REL
    )
    project=Path(project_root).resolve() if project_root else HERE.resolve()
    # In production the verifier exists in exactly this project's graph_tools.
    _require(project==HERE.resolve(),"V2.16 requires the installed V2.15 data root")
    verify_only()
    graph=_read_json(OLD_GRAPH)
    baseline=_read_json(OLD_LINKS)
    aliases=_read_json(OLD_ALIASES)
    new_links=_read_json(HERE/LINK_REL)
    _require(graph["metadata"]["v212"]["phase"]=="promoted",
             "V2.12 graph is not officially promoted")
    old_edge_map={e["id"]:e for e in graph["edges"]
                  if e.get("relation") not in {"PART_OF","MENTIONED_IN"}}
    old_claims=baseline.get("claims",[])
    _require(len(old_claims)==len(old_edge_map)==29,
             "Old verified graph edge count is not the expected 29")
    new_claims=new_links.get("claims",[])
    _require(len(new_claims)==54,"V2.15 new source claims != 54")

    chunk_map={}
    for line in SNAPSHOT.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():continue
        row=json.loads(line)
        cid=row["chunk_id"]
        _require(cid not in chunk_map and bool(cid),
                 "Duplicate / missing original Chroma Chunk ID")
        chunk_map[cid]=row
    _require(len(chunk_map)==1953,"Unexpected frozen Chunk snapshot size")
    all_claims=[]
    for claim in old_claims:
        original=old_edge_map.get(claim["claim_id"])
        _require(original is not None,"Verified claim has no original graph edge")
        _require(claim["status"]=="verified" and
                 claim["predicate"]==original["relation"] and
                 claim["subject_id"]==original["source"] and
                 claim["object_id"]==original["target"],
                 "V2.12 graph direction/predicate/source validation failure")
        all_claims.append((claim,"verified_v212"))
    suppressed_pending=[]
    flagged_sha=None
    if include_pending:
        # Previous V2.15.1 model-assisted semantic screening is a SAFETY FILTER
        # for exploratory planning. It is NOT an approval or an alias registry.
        risk_file=HERE/"graph_reports/expansions/v2151_preliminary_semantic_flags.json"
        _require(risk_file.is_file(),
                 "Offline pending-mode requires V2.15.1 semantic risk flags; "
                 "no pending edges are permitted without screening")
        _require(sha256_file(risk_file)==FROZEN_V2151_RISK_SHA256,
                 "V2.15.1 semantic screening SHA256 changed; "
                 "pending relation paths are disabled until re-reviewed")
        flags=_read_json(risk_file)
        _require(flags.get("schema")==
                 "douxian-v2151-preliminary-semantic-review-1"
                 and flags.get("new_candidate_relations")==54
                 and flags.get("new_claims_human_approved")==0
                 and flags.get("new_claims_online_eligible")==0,
                 "Unexpected/mutated V2.15.1 semantic screening source")
        blocked=set()
        qualifiers=set()
        for flag in flags.get("flagged_assertions",[]):
            cid=flag["claim_id"]
            if flag["issue_type"] in {"hold","scope","taxonomy","qualify_new_type"}:
                blocked.add(cid)
            if flag["issue_type"]=="qualify":
                qualifiers.add(cid)
        _require(blocked.issubset({x["claim_id"] for x in new_claims}) and
                 qualifiers.issubset({x["claim_id"] for x in new_claims}),
                 "Risk flags contain unknown V2.15 relations")
        flagged_sha=sha256_file(risk_file)
        for claim in new_claims:
            _require(claim["status"]=="evidence_bound_pending_independent_semantic_review"
                     and claim["online_eligible"] is False,
                     "V2.15 new source edge was unexpectedly promoted/altered")
            cid=claim["claim_id"]
            if (claim["predicate"] not in PENDING_ALLOWED or
                cid in blocked or
                (cid in qualifiers and not claim.get("qualifier"))):
                suppressed_pending.append(cid)
                continue
            all_claims.append((claim,"pending_v215"))
    final=[]
    for claim,role in all_claims:
        cid_list=claim.get("chunk_ids")
        _require(isinstance(cid_list,list) and len(cid_list)==1,
                 "Claim does not bind uniquely to a Chunk")
        cid=cid_list[0]
        row=chunk_map.get(cid)
        _require(row and row["source"]==claim["source_file"] and
                 row["section_path"].endswith(claim["section_path"]) and
                 claim["evidence_text"] in row["content"],
                 "Bound graph evidence cannot be recovered from frozen Chroma snapshot")
        final.append(_clone_edge(claim,role))
    old_nodes={n["id"]:n["label"] for n in graph["nodes"]
               if n.get("type")=="Concept" and isinstance(n.get("label"),str)}
    verified_alias_to_root={}
    for grp in aliases["verified_synonym_groups"]:
        canon=grp["canonical_id"]
        _require(canon in old_nodes,"Alias root not in formally verified old graph")
        root=key(old_nodes[canon])
        for nid in grp["member_ids"]:
            _require(nid in old_nodes,"Verified synonym node absent")
            verified_alias_to_root[key(old_nodes[nid])]=root
        for label in grp["verified_labels"]:
            verified_alias_to_root[key(label)]=root
    # Only old VERIFIED synonym groups are equivalence, all other labels are
    # equal-text-only retrieval navigation; search_only_hints are not synonyms.
    terms=set()
    for e in final:
        terms.update((e["subject"],e["object"]))
    terms.update(old_nodes.values())
    for grp in aliases["verified_synonym_groups"]:
        terms.update(grp["verified_labels"])
    return {
        "edges":final,
        "terms":sorted(terms),
        "verified_alias_to_root":verified_alias_to_root,
        "include_pending":include_pending,
        "new_pending_count":sum(e["role"]=="pending_v215" for e in final),
        "suppressed_pending_claim_ids":sorted(suppressed_pending),
        "semantic_flags_sha256":flagged_sha,
        "verified_count":sum(e["role"]=="verified_v212" for e in final),
        "source_checksums":{
            "v212_graph_sha256":sha256_file(OLD_GRAPH),
            "v212_links_sha256":sha256_file(OLD_LINKS),
            "v212_alias_sha256":sha256_file(OLD_ALIASES),
            "snapshot_sha256":sha256_file(SNAPSHOT),
            "v215_claims_sha256":sha256_file(HERE/LINK_REL),
            "v215_graph_sha256":sha256_file(HERE/GRAPH_REL),
            "v215_index_sha256":sha256_file(HERE/INDEX_REL),
            "v2151_semantic_screening_sha256":flagged_sha,
        },
        "new_edges_online_eligible":False,
        "no_prod_mutation":True,
    }

def _concept_key(label:str,bundle:dict)->str:
    return bundle["verified_alias_to_root"].get(key(label),key(label))

def match_anchors(question:str,bundle:dict)->list[dict]:
    """Longest-nonoverlapping term matching + VERIFIED alias collapsing."""
    candidates=[]
    for term in bundle["terms"]:
        for start,end in _has_term_at(question,term):
            candidates.append((start,end,term))
    # Prevent "采样" from firing inside "吉布斯采样"; ties deterministic.
    kept=[]
    for a,b,term in sorted(candidates,key=lambda x:(-(x[1]-x[0]),x[0],x[2])):
        if any(x<=a and b<=y for x,y,_ in kept):
            continue
        kept.append((a,b,term))
    matched=[]
    roots=set()
    for a,b,term in sorted(kept,key=lambda x:(x[0],x[1])):
        root=_concept_key(term,bundle)
        if root not in roots:
            matched.append({
                "label":term,"concept_lookup_key":root,
                "span":[a,b],
            })
            roots.add(root)
    return matched

def _directed_path_candidates(anchor:dict,intent:str,bundle:dict):
    """Only outgoing directed semantic edges; no reverse IS_A/USES expansion."""
    edges=bundle["edges"]
    root=anchor["concept_lookup_key"]
    for pattern,base_score in PATTERNS.get(intent,()):
        if len(pattern)==1:
            for e in edges:
                if e["predicate"]!=pattern[0] or _concept_key(e["subject"],bundle)!=root:
                    continue
                yield [e],base_score
        else:
            first,second=pattern
            for e in edges:
                if e["predicate"]!=first or _concept_key(e["subject"],bundle)!=root:
                    continue
                intermediate=_concept_key(e["object"],bundle)
                for other in edges:
                    if other["claim_id"]==e["claim_id"] or other["predicate"]!=second:
                        continue
                    if _concept_key(other["subject"],bundle)!=intermediate:
                        continue
                    if _concept_key(other["object"],bundle)==root:
                        continue
                    yield [e,other],base_score

def _path_kind(path:list[dict],bundle:dict)->str:
    if len(path)==2 and path[0]["predicate"]=="ALSO_CALLED":
        return "verified_alias_bridge_then_direct_relation"
    if len(path)==2 and path[0]["source_file"]!=path[1]["source_file"]:
        return "cross_textbook_directed_two_edge_path"
    if len(path)==2:
        return "same_textbook_directed_two_edge_path"
    return "single_directed_relation"

def _source_risk(path:list[dict])->str:
    return "pending_new_book_offline_only" if any(
        e["role"]=="pending_v215" for e in path) else "formal_v212_verified"

def _make_branch(path:list[dict],score:float,anchor_label:str,bundle:dict)->dict:
    tail=path[-1]
    if len(path)==2:
        # Target the evidence on the SECOND hop, not the question's root
        query=path[1]["subject"]+" "+path[1]["object"]
    else:
        query=tail["subject"]+" "+tail["object"]
    chunks=[x["chunk_id"] for x in path]
    return {
        "query":query,
        "intent_source_anchor":anchor_label,
        "target_chunk_id":tail["chunk_id"],
        "path_chunk_ids":chunks,
        "path_claim_ids":[x["claim_id"] for x in path],
        "path_labels":[path[0]["subject"]]+[x["object"] for x in path],
        "path_predicates":[x["predicate"] for x in path],
        "path_kind":_path_kind(path,bundle),
        "intermediate_join_type":(
            "exact_label" if key(path[0]["object"])==key(path[1]["subject"])
            else "formally_verified_synonym_lookup"
        ) if len(path)==2 else "none",
        "score":round(float(score),2),
        "source_risk":_source_risk(path),
        "online_eligible":False,
        "asserts_new_direct_fact":False,
        "edges":[
            {
                "claim_id":x["claim_id"],"subject":x["subject"],
                "predicate":x["predicate"],"object":x["object"],
                "source_file":x["source_file"],
                "section_path":x["section_path"],
                "source_line":x["source_line"],
                "evidence_quote":x["evidence_quote"],
                "chunk_id":x["chunk_id"],
                "qualifier":x["qualifier"],
                "status":x["status"],
                "verified_status":"formal_v212" if x["role"]=="verified_v212"
                                  else "pending_independent_semantic_review",
            } for x in path
        ],
    }

def plan(question:str, original_subqueries:list[str],bundle:dict,
         max_branches:int=2)->dict:
    """Return *separate* branches: never append terms to original subqueries.

    The caller may run only original Hybrid and treat branch work as a
    read-only experiment; a branch does not authorize online injection.
    """
    _require(isinstance(bundle,dict) and
             "verified_alias_to_root" in bundle and "edges" in bundle,
             "Missing validated graph bundle")
    _require(type(max_branches) is int and 0<=max_branches<=3,
             "V2.16 Graph Assist permits at most 3 independent branches")
    _require(isinstance(original_subqueries,(list,tuple)) and
             all(isinstance(q,str) for q in original_subqueries),
             "Original subqueries must be a list of strings")
    untouched=list(original_subqueries)
    intent=detect_intent(question)
    result={
        "question":question,
        "intent":intent,
        "original_subqueries":untouched,
        "original_subqueries_unchanged":True,
        "mode":"offline_exploration" if bundle["include_pending"] else "verified_only",
        "offline_only":True,
        "branches":[],
        "matched_anchors":[],
        "status":"no_expansion",
        "reason":"",
        "diagnostics":{
            "old_verified_edges":bundle["verified_count"],
            "pending_edges_explorable":bundle["new_pending_count"],
            "pending_edges_suppressed":len(
                bundle.get("suppressed_pending_claim_ids",[])),
            "candidate_paths_considered":0,
            "skipped_in_question":0,
            "skipped_same_evidence":0,
            "skipped_duplicate_query":0,
            "query_calls_executed":0,
            "reranker_calls_executed":0,
            "online_eligible":False,
        },
    }
    if intent in ("definition","comparison_unsupported","invalid","too_long",
                  "no_supported_intent"):
        result["reason"]="intent_gate_"+intent
        return result
    if max_branches==0 or not untouched:
        result["reason"]="budget_zero_or_no_subqueries"
        return result
    matched=match_anchors(question,bundle)
    result["matched_anchors"]=matched
    if not matched:
        result["reason"]="no_formal_graph_term_in_question"
        return result
    candidates=[]
    for anchor in matched:
        for path,base in _directed_path_candidates(anchor,intent,bundle):
            result["diagnostics"]["candidate_paths_considered"]+=1
            # Avoid returning an already fully explicit relation as an
            # additional semantic query. A path must reveal at least one
            # term not literally given in the user's question.
            target=path[-1]["object"]
            if term_matches(question,target):
                result["diagnostics"]["skipped_in_question"]+=1
                continue
            risk=_source_risk(path)
            score=base
            if len(path)==2: score+=22
            if len(path)==2 and path[0]["source_file"]!=path[1]["source_file"]:
                score+=9
            if risk=="pending_new_book_offline_only": score-=17
            if len(path)==2:
                # An EXACT shared concept label is a more directly auditable
                # join than a different label linked by an old synonym group.
                score+=(6 if key(path[0]["object"])==key(path[1]["subject"]) else -5)
            if path[0]["predicate"]=="ALSO_CALLED" and len(path)==2:
                score-=6 # alias bridge is NOT two independent semantic steps
            if path[-1]["chunk_id"]==path[0]["chunk_id"] and len(path)>1:
                # A two-hop path bound to the same Chunk gives no second
                # piece of evidence; not an incremental target candidate.
                result["diagnostics"]["skipped_same_evidence"]+=1
                continue
            candidate=_make_branch(path,score,anchor["label"],bundle)
            candidates.append(candidate)
    candidates.sort(key=lambda x:(
        -x["score"],
        x["source_risk"]!="formal_v212_verified",
        x["query"],
        tuple(x["path_claim_ids"])
    ))
    seen_query=set()
    seen_last_chunk=set()
    chosen=[]
    for candidate in candidates:
        q=key(candidate["query"])
        last=candidate["target_chunk_id"]
        if q in seen_query:
            result["diagnostics"]["skipped_duplicate_query"]+=1
            continue
        if last in seen_last_chunk:
            result["diagnostics"]["skipped_same_evidence"]+=1
            continue
        # Never count a query which is already one of original subqueries.
        if any(key(x)==q for x in untouched):
            result["diagnostics"]["skipped_duplicate_query"]+=1
            continue
        chosen.append(candidate)
        seen_query.add(q)
        seen_last_chunk.add(last)
        if len(chosen)>=max_branches:break
    result["branches"]=chosen
    result["status"]="planned" if chosen else "no_expansion"
    result["reason"]="verified_path_plan" if chosen and not bundle["include_pending"] \
        else "exploratory_path_plan" if chosen else "no_new_directional_relation"
    return result

def main(argv:list[str]|None=None):
    from .verify_machine_graph_v215 import HERE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--question",required=True,
                   help="Full original Chinese question, unchanged")
    p.add_argument("--subquery",action="append",default=[],
                   help="Optional original RAG subquery, repeatable")
    p.add_argument("--max-branches",type=int,default=2)
    p.add_argument("--explore-pending",action="store_true",
                   help="ONLY for offline research with V2.15 unreviewed relations")
    p.add_argument("--out",type=Path,required=True,
                   help="Fresh JSON under graph_reports/assist_v216")
    args=p.parse_args(argv)
    out=args.out.expanduser().resolve()
    _require(out.parent==(HERE/"graph_reports/assist_v216").resolve()
             and out.suffix.lower()==".json",
             "Output must be a new JSON under graph_reports/assist_v216")
    _require(not out.exists(),"Never overwrite a previous V2.16 audit report")
    bundle=load_graph(include_pending=args.explore_pending)
    queries=args.subquery or [args.question]
    obj=plan(args.question,queries,bundle,max_branches=args.max_branches)
    with out.open("x",encoding="utf-8") as fd:
        json.dump(obj,fd,ensure_ascii=False,indent=2)
        fd.write("\n")
    print("[V2.16] status:",obj["status"],"intent:",obj["intent"],
          "branches:",len(obj["branches"]))
    for branch in obj["branches"]:
        print(" branch:",branch["query"],"| path:",branch["path_labels"],
              "| evidence Chunk:",branch["target_chunk_id"][:16],
              "| risk:",branch["source_risk"])
    print("[V2.16] No graph or production RAG files modified. No API calls.")

if __name__=="__main__":
    main()
