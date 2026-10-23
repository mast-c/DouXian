# -*- coding: utf-8 -*-
"""V2.15 deterministic MULTI-SOURCE graph expansion builder (OFFLINE ONLY).

Rebuilds from:
  graph_data/sources/机器学习.md (frozen SHA)
  graph_bindings/chroma_chunks_v211.json (existing 1953-row JSONL)
  graph_data/knowledge_graph_v212.json (untouched baseline)
  graph_bindings/graph_chunk_links_v212.json (untouched baseline)
  graph_data/graph_alias_registry_v212.json (untouched baseline)
  graph_data/expansions/machine_learning_curated_spec_v215.json

No network, Chroma DB connection, LLM, embeddings, source file mutation,
or automatic relation promotion. Call --verify for read-only integrity.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parents[1]
SRC=HERE/"graph_data/sources/机器学习.md"
SNAPSHOT=HERE/"graph_bindings/chroma_chunks_v211.json"
OLD_GRAPH=HERE/"graph_data/knowledge_graph_v212.json"
OLD_LINKS=HERE/"graph_bindings/graph_chunk_links_v212.json"
OLD_ALIASES=HERE/"graph_data/graph_alias_registry_v212.json"
SPECS=HERE/"graph_data/expansions/machine_learning_curated_spec_v215.json"
GRAPH_REL="graph_data/expansions/machine_learning_relations_v215.json"
LINK_REL="graph_bindings/expansions/machine_learning_claim_links_v215.json"
INDEX_REL="graph_data/expansions/multi_source_claim_index_v215.json"
REPORT_REL="graph_reports/expansions/machine_learning_gate_v215.json"
ARTIFACT_REL=(GRAPH_REL,LINK_REL,INDEX_REL,REPORT_REL)
PREDICATES={
    "IS_A","SPECIAL_CASE_OF","HAS_PART","USES","BASED_ON",
    "SUBFIELD_OF","COMPUTES","ALSO_CALLED",
    "VARIANT_OF","CONDITIONAL_FORM_OF",
}
STATUS_NEW="evidence_bound_pending_independent_semantic_review"

def require(condition,msg):
    if not condition: raise ValueError(msg)

def json_bytes(obj):
    return (json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+"\n").encode("utf-8")

def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()

def sha_file(path):
    return sha_bytes(Path(path).read_bytes())

def read_json(path):
    result=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(result,dict),"Expected JSON object: "+str(path))
    return result

def tag(prefix,token):
    return prefix+hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]

def normalization(s):
    return re.sub(r"\s+","",str(s or "")).casefold()

def section_map(source_lines):
    stack={}
    sections={}
    for lineno,line in enumerate(source_lines,1):
        match=re.match(r"^(#{2,6})\s+(.+?)\s*$",line)
        if match:
            level=len(match.group(1))
            stack[level]=match.group(2).strip()
            for old_level in list(stack):
                if old_level>level: del stack[old_level]
        sections[lineno]="机器学习 / "+" / ".join(
            stack[k] for k in sorted(stack) if k>=2)
    return sections

def create_artifacts(source_path=SRC,snapshot_path=SNAPSHOT,
                     old_graph_path=OLD_GRAPH,old_links_path=OLD_LINKS,
                     old_aliases_path=OLD_ALIASES,specs_path=SPECS):
    paths=(source_path,snapshot_path,old_graph_path,old_links_path,
           old_aliases_path,specs_path)
    for path in paths: require(Path(path).is_file(),"Missing required file: "+str(path))
    hashes={
        "source_sha256":sha_file(source_path),
        "snapshot_sha256":sha_file(snapshot_path),
        "baseline_graph_sha256":sha_file(old_graph_path),
        "baseline_links_sha256":sha_file(old_links_path),
        "baseline_aliases_sha256":sha_file(old_aliases_path),
        "spec_sha256":sha_file(specs_path),
    }
    spec=read_json(specs_path)
    graph=read_json(old_graph_path)
    links=read_json(old_links_path)
    aliases=read_json(old_aliases_path)
    require(spec.get("source_file")=="机器学习.md" and
            spec.get("source_sha256")==hashes["source_sha256"] and
            spec.get("frozen_chroma_snapshot_sha256")==hashes["snapshot_sha256"] and
            spec.get("baseline_verified_links_sha256")==hashes["baseline_links_sha256"],
            "V2.15 curated spec provenance mismatch")
    require(graph.get("metadata",{}).get("v212",{}).get("phase")=="promoted",
            "V2.12 graph baseline is not promoted")
    require(links.get("metadata",{}).get("graph_sha256")==hashes["baseline_graph_sha256"]
            and links.get("metadata",{}).get("snapshot_sha256")==hashes["snapshot_sha256"]
            and links.get("metadata",{}).get("semantic_claims")==29,
            "Frozen V2.12 baseline/snapshot changed")
    require(aliases.get("graph_sha256")==hashes["baseline_graph_sha256"]
            and aliases.get("links_sha256")==hashes["baseline_links_sha256"],
            "V2.12 alias registry hashes are not linked to the formal graph")
    original_claims=links.get("claims",[])
    require(len(original_claims)==29 and
            all(x.get("status")=="verified" and len(x.get("chunk_ids",[]))==1
                for x in original_claims),
            "Baseline must contain 29 verified singly bound relations")

    raw=Path(source_path).read_text(encoding="utf-8-sig").splitlines()
    sections=section_map(raw)
    rows=[json.loads(x) for x in Path(snapshot_path).read_text(
        encoding="utf-8-sig").splitlines() if x.strip()]
    require(len(rows)==1953, "Frozen 1953-row Chroma snapshot is required")
    by_source=defaultdict(list)
    all_ids=set()
    for row in rows:
        require(isinstance(row,dict)
                and all(k in row for k in ("chunk_id","source","section_path","content")),
                "Snapshot missing required Chroma native ID/source/section/content")
        cid=row["chunk_id"]
        require(isinstance(cid,str) and cid not in all_ids and cid,
                "Missing or duplicate native Chroma Chunk ID")
        all_ids.add(cid)
        by_source[row["source"]].append(row)
    require(len(by_source["机器学习.md"])==895 and
            len(by_source["神经网络与深度学习.md"])==1058,
            "Chroma source distribution mismatch")

    candidates=spec.get("candidate_definitions",[])
    require(isinstance(candidates,list) and len(candidates)==55,
            "Unexpected curated specs; this V2.15 release expects 55")
    accepted=[]
    rejected=[]
    uniqueness=set()
    for idx,entry in enumerate(candidates,1):
        num=entry.get("source_line")
        name,relation,target=entry.get("subject"),entry.get("predicate"),entry.get("object")
        start,end=entry.get("begin"),entry.get("end")
        require(type(num) is int and 1<=num<=len(raw) and
                all(isinstance(v,str) and v for v in (name,relation,target,start,end))
                and relation in PREDICATES,
                "Invalid candidate spec row #"+str(idx))
        key=(name,relation,target)
        require(key not in uniqueness,"Duplicate proposed semantic assertion: "+str(key))
        uniqueness.add(key)
        text=raw[num-1]
        begin=text.find(start)
        finish=text.find(end,begin) if begin>=0 else -1
        quote=text[begin:finish+len(end)] if finish>=begin>=0 else ""
        sec=sections[num]
        matches=[row for row in by_source["机器学习.md"]
                 if row["section_path"]==sec
                 and quote and quote in row["content"]]
        reasons=[]
        if not quote: reasons.append("exact_original_span_missing")
        if not re.search(r"机器学习 / 第\d+章",sec):
            reasons.append("not_body_chapter_section")
        if len(matches)!=1:
            reasons.append("no_unique_chunk_match" if not matches
                           else "ambiguous_overlapping_chunk_match")
        if len(quote)<8: reasons.append("insufficient_quote_length")
        if reasons:
            rejected.append({
                "candidate_index":idx,"source_line":num,
                "subject":name,"predicate":relation,"object":target,
                "section_path":sec,"reason_codes":reasons,
                "matching_native_chunk_ids":[m["chunk_id"] for m in matches],
                "evidence_text":quote,
            })
            continue
        edge_id=tag("edge:ml215:", "|".join([hashes["source_sha256"],
            str(num),name,relation,target,quote]))
        accepted.append({
            "claim_id":edge_id,
            "subject":name,"predicate":relation,"object":target,
            "source_file":"机器学习.md",
            "source_line":num,"section_path":sec,
            "evidence_text":quote,
            "chunk_ids":[matches[0]["chunk_id"]],
            "qualifier":entry.get("qualifier",""),
            "status":STATUS_NEW,
            "online_eligible":False,
            "binding_checks":{
                "exact_source_line_span":True,
                "exact_source_file":"机器学习.md",
                "exact_chapter_section":True,
                "unique_snapshot_chunk_id":True,
                "exact_contiguous_quote_in_snapshot_chunk":True,
            },
        })
    require(len(accepted)==54 and len(rejected)==1,
            "Unexpected new evidence gate result; fail closed")
    require(rejected[0]["reason_codes"]==
            ["ambiguous_overlapping_chunk_match"],
            "Expected overlapping evidence candidate no longer ambiguous: review needed")
    require(len({x["claim_id"] for x in accepted})==len(accepted),
            "New edge IDs collide")
    old_ids={c["chunk_ids"][0] for c in original_claims}
    new_chunk_ids={c["chunk_ids"][0] for c in accepted}
    require(not old_ids.intersection(new_chunk_ids),
            "Cross-book Chunk ID collision")

    concepts=sorted({x[label] for x in accepted for label in ("subject","object")})
    concept_to_id={
        label:tag("concept:ml215:",label) for label in concepts
    }
    new_nodes=[
        {"id":tag("doc:ml215:","机器学习.md"),"type":"Document",
         "label":"机器学习.md","source_sha256":hashes["source_sha256"],
         "source_lines":len(raw)}
    ]
    new_nodes += [
        {"id":concept_to_id[label],"type":"Concept","label":label}
        for label in concepts
    ]
    needed_section_paths=set()
    for c in accepted:
        parts=c["section_path"].split(" / ")
        needed_section_paths.update(" / ".join(parts[:idx])
                                    for idx in range(2,len(parts)+1))
    section_to_id={sec:tag("sec:ml215:",sec) for sec in needed_section_paths}
    new_nodes += [
        {"id":section_to_id[sec],"type":"Section",
         "label":sec.rsplit(" / ",1)[-1],
         "section_path":sec,"source_file":"机器学习.md"}
        for sec in sorted(needed_section_paths)
    ]
    semantic_edges=[]
    for c in accepted:
        semantic_edges.append({
            "id":c["claim_id"],
            "source":concept_to_id[c["subject"]],
            "relation":c["predicate"],
            "target":concept_to_id[c["object"]],
            "source_section":section_to_id[c["section_path"]],
            "qualifier":c["qualifier"],
            "status":STATUS_NEW,
            "evidence":{
                "source_file":"机器学习.md","section_path":c["section_path"],
                "line":c["source_line"],"text":c["evidence_text"],
                "chunk_id":c["chunk_ids"][0],
                "extraction":"curated_literal_line_span",
            },
        })
    structural=[]
    doc_id=new_nodes[0]["id"]
    for sec in sorted(section_to_id):
        parent=sec.rsplit(" / ",1)[0]
        parent_id=section_to_id.get(parent,doc_id)
        structural.append({
            "id":tag("edge:ml215:section:",sec),
            "source":section_to_id[sec],
            "relation":"PART_OF",
            "target":parent_id,
            "evidence":{"source_file":"机器学习.md","section_path":sec},
        })

    graph_obj={
        "metadata":{
            "schema":"douxian-kg-multisource-sidecar-v215-1",
            "phase":"offline_evidence_bound_pending_review",
            "source_file":"机器学习.md",
            **hashes,
            "semantic_edges":len(accepted),
            "structural_edges":len(structural),
            "new_concepts":len(concepts),
            "new_sections":len(needed_section_paths),
            "online_eligible":False,
            "old_graph_untouched":True,
            "note":"New statements have exact original line/section/unique Chroma snapshot evidence, but require independent semantic review and live Chroma revalidation before promotion.",
        },
        "nodes":sorted(new_nodes,key=lambda x:x["id"]),
        "edges":sorted(semantic_edges+structural,key=lambda x:x["id"]),
    }
    graph_hash=sha_bytes(json_bytes(graph_obj))
    link_obj={
        "metadata":{
            "schema":"douxian-claim-link-multisource-v215-1",
            "phase":"offline_evidence_bound_pending_review",
            "graph_sha256":graph_hash,
            **hashes,
            "source_file":"机器学习.md",
            "snapshot_rows":len(rows),
            "new_source_snapshot_rows":len(by_source["机器学习.md"]),
            "semantic_claims":len(accepted),
            "status_counts":{STATUS_NEW:len(accepted)},
            "online_eligible_status":None,
            "binding_rule":"same-original-source-line+same-source+exact-full-section+contiguous-quote+unique-existing-Chunk",
        },
        "claims":sorted(accepted,key=lambda x:x["claim_id"]),
    }
    links_hash=sha_bytes(json_bytes(link_obj))

    # Query lookup, not an asserted cross-source semantic identity.
    all_claims=[]
    for c in original_claims:
        all_claims.append({
            "claim_id":c["claim_id"],"subject":c["subject"],
            "predicate":c["predicate"],"object":c["object"],
            "source_file":c["source_file"],"section_path":c["section_path"],
            "source_line":c["source_line"],"chunk_id":c["chunk_ids"][0],
            "origin":"formal_v212_baseline","status":"verified",
            "online_eligible_in_existing_v213":True,
            "online_eligible_in_new_v215":False,
        })
    for c in accepted:
        all_claims.append({
            "claim_id":c["claim_id"],"subject":c["subject"],
            "predicate":c["predicate"],"object":c["object"],
            "source_file":c["source_file"],"section_path":c["section_path"],
            "source_line":c["source_line"],"chunk_id":c["chunk_ids"][0],
            "qualifier":c["qualifier"],"origin":"machine_v215_sidecar",
            "status":STATUS_NEW,
            "online_eligible_in_existing_v213":False,
            "online_eligible_in_new_v215":False,
        })
    by_label=defaultdict(lambda:{"labels":set(),"claim_ids":set(),"source_files":set()})
    for c in all_claims:
        for label in (c["subject"],c["object"]):
            entry=by_label[normalization(label)]
            entry["labels"].add(label)
            entry["claim_ids"].add(c["claim_id"])
            entry["source_files"].add(c["source_file"])
    label_index={
        name:{
            "labels":sorted(x["labels"]),
            "claim_ids":sorted(x["claim_ids"]),
            "source_files":sorted(x["source_files"]),
            "is_cross_source_exact_label_candidate":len(x["source_files"])>1
        }
        for name,x in sorted(by_label.items())
    }
    shared=[
        {"canonical_lexical_key":term, "labels":item["labels"],
         "source_files":item["source_files"],
         "status":"exact_label_lookup_only_not_claimed_equivalent"}
        for term,item in label_index.items()
        if item["is_cross_source_exact_label_candidate"]
    ]
    index_obj={
        "metadata":{
            "schema":"douxian-multi-source-lookup-v215-1",
            "phase":"offline_search_lookup_only",
            "source_sha256":hashes["source_sha256"],
            "snapshot_sha256":hashes["snapshot_sha256"],
            "baseline_graph_sha256":hashes["baseline_graph_sha256"],
            "baseline_links_sha256":hashes["baseline_links_sha256"],
            "new_graph_sha256":graph_hash,
            "new_links_sha256":links_hash,
            "baseline_verified_claim_count":29,
            "machine_pending_review_claim_count":len(accepted),
            "online_eligible":False,
            "cross_source_alias_promotion":False,
            "warning":"Exact same label across books is only a search lookup bridge; it does not prove synonymous identities or create an inferred relationship.",
        },
        "claims":sorted(all_claims,key=lambda x:x["claim_id"]),
        "by_exact_label":label_index,
        "cross_source_exact_label_candidates":shared,
    }
    report_obj={
        "schema":"douxian-machine-expansion-evidence-gate-v215-1",
        "phase":"offline_gate_no_online_promotion",
        "source":"机器学习.md",
        "source_lines":len(raw),
        **hashes,
        "new_graph_sha256":graph_hash,
        "new_links_sha256":links_hash,
        "baseline_verified_claims":len(original_claims),
        "candidate_count":len(candidates),
        "passed_exact_quote_section_unique_chunk":len(accepted),
        "rejected_count":len(rejected),
        "rejected_candidates":rejected,
        "new_evidence_distinct_chunk_ids":len(new_chunk_ids),
        "new_source_snapshot_chunk_count":len(by_source["机器学习.md"]),
        "snapshot_total_chunk_count":len(rows),
        "baseline_distinct_chunk_ids":len(old_ids),
        "merged_claim_count_for_offline_analysis":len(all_claims),
        "merged_distinct_bound_chunk_ids":len(old_ids|new_chunk_ids),
        "new_relation_type_counts":dict(sorted(Counter(x["predicate"]
                                                 for x in accepted).items())),
        "new_chapter_counts":dict(sorted(Counter(
            re.search(r"第(\d+)章",x["section_path"]).group(1)
            for x in accepted).items(), key=lambda kv:int(kv[0]))),
        "cross_source_exact_label_candidate_count":len(shared),
        "cross_source_exact_labels":[x["labels"][0] for x in shared],
        "semantic_review_gate":{
            "literal_quote_and_binding":"passed for accepted candidates",
            "independent_human_semantic_review":"pending",
            "new_relation_types_require_explicit_retriever_support":[
                "VARIANT_OF","CONDITIONAL_FORM_OF"],
            "qualified_assertions_must_keep_modality":True,
            "new_edges_online_eligible":False,
            "old_v212_graph_modified":False,
            "chroma_database_modified":False,
            "live_chroma_verified":False,
            "qa_rerank_impact_verified":False,
        },
        "note":"This is a reproducible evidence-bound candidate graph, NOT a verified online GraphRAG upgrade.",
    }
    return {
        GRAPH_REL:graph_obj,LINK_REL:link_obj,
        INDEX_REL:index_obj,REPORT_REL:report_obj,
    }

def verify_only(project_dir=HERE):
    base=Path(project_dir)
    generated=create_artifacts(
        source_path=base/"graph_data/sources/机器学习.md",
        snapshot_path=base/"graph_bindings/chroma_chunks_v211.json",
        old_graph_path=base/"graph_data/knowledge_graph_v212.json",
        old_links_path=base/"graph_bindings/graph_chunk_links_v212.json",
        old_aliases_path=base/"graph_data/graph_alias_registry_v212.json",
        specs_path=base/"graph_data/expansions/machine_learning_curated_spec_v215.json",
    )
    for relative,obj in generated.items():
        file=base/relative
        require(file.is_file(),"Missing V2.15 expansion artifact: "+relative)
        require(file.read_bytes()==json_bytes(obj),
                "V2.15 artifact has changed or is not reproducible: "+relative)
    report=generated[REPORT_REL]
    print("[V2.15] VERIFY PASS: 54 evidence-bound / 1 rejected ambiguous")
    print("[V2.15] Source chunks: 895 / 1953; new distinct evidence chunks:",
          report["new_evidence_distinct_chunk_ids"])
    print("[V2.15] Baseline relations: 29 (unchanged); offline total:",
          report["merged_claim_count_for_offline_analysis"])
    print("[V2.15] Cross-source exact-label lookup candidates:",
          report["cross_source_exact_label_candidate_count"])
    print("[V2.15] New graph remains OFFLINE / NOT ONLINE ELIGIBLE.")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true",
                        help="Recompute all artifacts in memory and compare exact bytes")
    args=parser.parse_args()
    if not args.verify:
        parser.error("Only read-only --verify is exposed for the released frozen artifacts")
    verify_only()

if __name__=="__main__":
    main()
