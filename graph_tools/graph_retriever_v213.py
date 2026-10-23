# -*- coding: utf-8 -*-
"""DouXian V2.13: verified Graph Chunk injection *before* the existing Reranker.

Use create_graph_augmented_service(KnowledgeBaseService) only in Graph ON mode.
The subclass overrides search_hybrid, NOT search_with_rerank, so the already
installed reranker gets the combined Hybrid+Graph pool once.

Fail closed: stale/missing/ambiguous graph artifacts, mismatched live Chroma
documents, or bad metadata -> keep original Hybrid candidates unchanged.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import time

try:
    from .graph_paths import CONTENT_DIR, BINDINGS_DIR, REPORTS_DIR, SOURCE_FILE
    from .graph_chunk_linker_v211 import (
        check_graph_source, verify_links, canonical_section, canonical_source, norm
    )
except ImportError:
    from graph_paths import CONTENT_DIR, BINDINGS_DIR, REPORTS_DIR, SOURCE_FILE
    from graph_chunk_linker_v211 import (
        check_graph_source, verify_links, canonical_section, canonical_source, norm
    )


GRAPH_FILE = CONTENT_DIR / "knowledge_graph_v212.json"
LINKS_FILE = BINDINGS_DIR / "graph_chunk_links_v212.json"
ALIASES_FILE = CONTENT_DIR / "graph_alias_registry_v212.json"
SNAPSHOT_FILE = BINDINGS_DIR / "chroma_chunks_v211.json"
GATE_FILE = REPORTS_DIR / "graph_v212_upgrade_report.json"
SEMANTIC = frozenset({
    "IS_A", "SPECIAL_CASE_OF", "HAS_PART", "USES", "BASED_ON",
    "SUBFIELD_OF", "COMPUTES", "ALSO_CALLED",
})
MAX_GRAPH_CHUNKS = 2   # capped independently of Hybrid recall_k
MAX_QUERY_LENGTH = 500


class GraphV213Error(ValueError):
    """An invalid, stale, or unavailable graph source."""


def _require(condition, message):
    if not condition:
        raise GraphV213Error(message)


def _sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _lexical(text):
    return re.sub(r"\s+","",str(text or "")).casefold()


def _has_term(text,term):
    if not isinstance(text,str) or not isinstance(term,str) or not term.strip():
        return False
    if any(ord(ch)>=128 for ch in term):
        return term in text
    return re.search(r"(?<![A-Za-z])"+re.escape(term)+r"(?![A-Za-z])",
                     text,re.IGNORECASE) is not None


def _chunk_id(doc):
    meta=getattr(doc,"metadata",{})
    return str(meta.get("chunk_id") or "").strip() if isinstance(meta,dict) else ""


@dataclass(frozen=True)
class GraphFiles:
    graph: Path = GRAPH_FILE
    links: Path = LINKS_FILE
    aliases: Path = ALIASES_FILE
    chunks: Path = SNAPSHOT_FILE
    source: Path = SOURCE_FILE
    gate: Path = GATE_FILE

    def required_paths(self):
        return tuple(Path(getattr(self,key)).resolve() for key in
                     ("graph","links","aliases","chunks","source","gate"))


@dataclass
class GraphBundle:
    concepts: dict
    edges: list
    linked_ids: dict
    chunk_rows: dict
    term_to_nodes: dict
    canonical_by_node: dict
    total_verified: int


def _build_bundle(files):
    """All authoritative checks are run before a graph edge can be considered."""
    graph_path,links_path,aliases_path,chunks_path,source_path,gate_path=files
    g=_read_json(graph_path)
    links=_read_json(links_path)
    alias=_read_json(aliases_path)
    gate=_read_json(gate_path)

    _require(g.get("metadata",{}).get("v212",{}).get("phase")=="promoted",
             "知识图谱不是 V2.12 正式 promoted 版本")
    _require(links.get("metadata",{}).get("graph_sha256")==_sha_file(graph_path),
             "图谱 SHA 与正式绑定文件不一致")
    _require(links["metadata"].get("source_sha256")==_sha_file(source_path),
             "教材 SHA 与绑定文件不一致")
    _require(links["metadata"].get("snapshot_sha256")==_sha_file(chunks_path),
             "Chunk 快照与正式绑定不一致")
    _require(alias.get("graph_sha256")==_sha_file(graph_path) and
             alias.get("links_sha256")==_sha_file(links_path),
             "实体别名登记表与图谱绑定不一致")
    _require(gate.get("final_graph_sha256")==_sha_file(graph_path) and
             gate.get("final_links_sha256")==_sha_file(links_path),
             "V2.12 升级门禁不属于当前正式图谱或绑定")
    _require(gate.get("alias_registry_sha256")==_sha_file(aliases_path),
             "V2.12 升级门禁与别名登记表不一致")
    _require(gate.get("snapshot_sha256")==_sha_file(chunks_path),
             "V2.12 升级门禁与当前 Chunk 快照不一致")
    check_graph_source(g,source_path)
    verify_links(g,source_path,chunks_path,links_path)

    # Check all chunks independently; do not trust verified flags from JSON.
    rows={}
    for ln,line in enumerate(chunks_path.read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip():continue
        row=json.loads(line)
        cid=row.get("chunk_id")
        _require(isinstance(cid,str) and cid and cid not in rows,
                 f"快照第 {ln} 行 ID 无效或重复")
        _require(isinstance(row.get("content"),str) and
                 isinstance(row.get("source"),str) and
                 isinstance(row.get("section_path"),str),
                 f"快照第 {ln} 行缺少正文/来源/章节")
        rows[cid]=row

    concepts={n["id"]:n for n in g.get("nodes",[])
              if n.get("type")=="Concept"}
    semantic={e["id"]:e for e in g.get("edges",[])
              if e.get("relation") in SEMANTIC}
    byid={r["claim_id"]:r for r in links.get("claims",[])}
    _require(len(byid)==len(links.get("claims",[])),
             "正式绑定中有重复 claim_id")
    _require(set(semantic)==set(byid),
             "正式绑定与图谱语义边集合不一致")
    linked={}
    for cid,edge in semantic.items():
        claim=byid[cid]
        chunk_ids=claim.get("chunk_ids")
        _require(claim.get("status")=="verified" and
                 isinstance(chunk_ids,list) and len(chunk_ids)==1,
                 "正式图谱存在非唯一 verified 语义关系："+cid)
        chunk_id=chunk_ids[0]
        row=rows.get(chunk_id)
        _require(row is not None,"正式绑定指向快照外的 Chunk: "+cid)
        ev=edge["evidence"]
        _require(canonical_source(row["source"])==canonical_source(ev["source_file"]) and
                 canonical_section(row["section_path"],ev["source_file"])==
                 canonical_section(ev["section_path"],ev["source_file"]) and
                 norm(ev["text"].rstrip("…")) in norm(row["content"]),
                 "图谱关系证据与快照 Chunk 不一致: "+cid)
        _require(edge["source"] in concepts and edge["target"] in concepts,
                 "语义关系指向缺失的概念")
        linked[cid]=chunk_id

    _require(len(linked)>0,"正式图谱没有经过验证的语义关系")
    _require(len(linked)==int(gate.get("final_verified",-1)),
             "正式图谱数量与升级报告不一致")

    canonical={nid:nid for nid in concepts}
    for group in alias.get("verified_synonym_groups",[]):
        leader=group.get("canonical_id")
        members=group.get("member_ids",[])
        _require(leader in concepts and isinstance(members,list) and leader in members,
                 "verified 同义词组缺少有效 canonical ID")
        for nid in members:
            _require(nid in concepts,"同义词组成员不在图谱")
            _require(canonical[nid]==nid or canonical[nid]==leader,
                     "同义词组出现重叠冲突")
            canonical[nid]=leader
    # All legacy search hints are just *query matching signals*.
    # They never become relation evidence or establish concept identity.
    term_to_nodes=defaultdict(set)
    ambiguous={entry["lexical_key"] for entry in alias.get("ambiguous_terms",[])}
    def remember(term,nid):
        # Preserve real spaces/word boundaries in English aliases when matching
        # questions. Normalize only for collision checking.
        if (isinstance(term,str) and term.strip()
                and _lexical(term) not in ambiguous):
            term_to_nodes[term.strip()].add(nid)
    for nid,node in concepts.items():
        remember(node["label"],nid)
        # Original V2.7 aliases are query hints, not verified equivalence.
        for search_hint in node.get("aliases_for_search",[]):
            remember(search_hint,nid)
    for hint in alias.get("search_only_hints",[]):
        nid=hint.get("concept_id")
        if nid in concepts:
            remember(hint.get("term"),nid)
    for group in alias.get("verified_synonym_groups",[]):
        for word in group.get("verified_labels",[]):
            for nid in group["member_ids"]:
                remember(word,nid)
    return GraphBundle(
        concepts=concepts,
        edges=list(semantic.values()),
        linked_ids=linked,
        chunk_rows=rows,
        term_to_nodes=dict(term_to_nodes),
        canonical_by_node=canonical,
        total_verified=len(linked),
    )


@lru_cache(maxsize=2)
def _cached_bundle(filepaths, stats):
    del stats  # used only as cache key
    return _build_bundle(filepaths)


def load_verified_bundle(paths=None):
    paths=paths or GraphFiles()
    files=paths.required_paths()
    missing=[p for p in files if not p.is_file()]
    if missing:
        raise GraphV213Error("缺少图谱正式文件: "+", ".join(p.name for p in missing))
    stats=tuple((p.stat().st_mtime_ns,p.stat().st_size) for p in files)
    return _cached_bundle(files,stats)


def candidate_chunk_ids(query,bundle,max_candidates=MAX_GRAPH_CHUNKS):
    """Retrieve only real Chunk IDs linked to uniquely verified claims."""
    if not isinstance(query,str) or not query.strip() or len(query)>MAX_QUERY_LENGTH:
        return [],{"matched_labels":[],"matched_edges":0}
    detected=set()
    for term,nids in bundle.term_to_nodes.items():
        if _has_term(query,term):
            detected.update(nids)
    if not detected:
        return [],{"matched_labels":[],"matched_edges":0}

    detected_roots={bundle.canonical_by_node.get(n,n) for n in detected}
    labels=sorted({bundle.concepts[n]["label"] for n in detected})
    scored=[]
    for e in bundle.edges:
        subject=e["source"]
        target=e["target"]
        src_root=bundle.canonical_by_node.get(subject,subject)
        dst_root=bundle.canonical_by_node.get(target,target)
        if src_root not in detected_roots and dst_root not in detected_roots:
            continue
        candidate=bundle.linked_ids[e["id"]]
        # Direct lexical concept mention beats indirect one-hop synonyms.
        direct=int(subject in detected)+int(target in detected)
        double=int(src_root in detected_roots and dst_root in detected_roots)
        evidence=e["evidence"]["text"]
        evidence_score=sum(_has_term(query,word) for word in
                           (bundle.concepts[subject]["label"],
                            bundle.concepts[target]["label"]))
        priority=double*20+direct*5+evidence_score*2 + int(e["relation"]=="ALSO_CALLED")
        scored.append((-priority,e["id"],candidate))
    scored.sort()
    unique=[]
    seen=set()
    for _,eid,chunk_id in scored:
        if chunk_id in seen:continue
        unique.append(chunk_id)
        seen.add(chunk_id)
        if len(unique)>=max(1,min(3,int(max_candidates))):break
    return unique,{
        "matched_labels":labels[:12],
        "matched_edges":len(scored),
    }


def _live_graph_documents(service, ids, bundle):
    """Hydrate by original Chroma IDs, verify against the exported snapshot."""
    data=service.chroma.get(ids=ids,include=["documents","metadatas"])
    found_ids=data.get("ids") or []
    content=data.get("documents") or []
    meta=data.get("metadatas") or []
    _require(len(found_ids)==len(content)==len(meta)==len(ids) and
             set(found_ids)==set(ids), "Chroma 的真实 Chunk 与已验证快照 ID 不一致")
    byid={cid:(body,metadata) for cid,body,metadata in
          zip(found_ids,content,meta)}
    result=[]
    for cid in ids:
        body,metadata=byid[cid]
        metadata=dict(metadata or {})
        saved=bundle.chunk_rows[cid]
        _require(body==saved["content"] and
                 canonical_source(metadata.get("source"))==
                 canonical_source(saved["source"]) and
                 canonical_section(metadata.get("section_path"),
                                   saved["source"])==
                 canonical_section(saved["section_path"],saved["source"]),
                 "真实 Chroma Chunk 与导出的快照内容或 Metadata 不一致")
        if metadata.get("chunk_id") not in (None,cid):
            raise GraphV213Error("真实 Chroma 原生 ID 与 metadata.chunk_id 不一致")
        metadata["chunk_id"]=cid
        result.append((body,metadata))
    return result


def create_graph_augmented_service(base_class, graph_paths=None):
    """Wrap current KnowledgeBaseService without editing its source file.

    Uses inherited search_with_rerank, which must call self.search_hybrid.
    Reject changed service interfaces instead of silently misreporting Graph ON.
    """
    method=getattr(base_class,"search_with_rerank",None)
    if not callable(method):
        raise GraphV213Error("KnowledgeBaseService 缺少 search_with_rerank")
    try:
        method_src=inspect.getsource(method)
    except (OSError,TypeError) as e:
        raise GraphV213Error("无法确认当前 Reranker 候选来源；拒绝接入") from e
    if "self.search_hybrid(" not in method_src:
        raise GraphV213Error("当前 Reranker 未调用 self.search_hybrid；拒绝无效接入")

    paths=graph_paths or GraphFiles()
    class GraphAugmentedKnowledgeBaseService(base_class):
        def search_hybrid(self,query,k=5):
            original=super().search_hybrid(query,k)
            # Measure only incremental Graph work; exclude baseline Hybrid.
            started=time.perf_counter()
            info={
                "status":"no_match", "mode":"v213",
                "matched_edges":0,"matched_labels":[],
                "graph_candidates":0,"added_chunks":0,
                "injected_chunk_ids":[],"fallback_reason":"",
                "graph_candidate_ids":[],"already_in_hybrid_ids":[],
                "matched_by_content_ids":[],"matched_by_content_map":{},
            }
            try:
                bundle=load_verified_bundle(paths)
                limit=max(1,min(3,int(os.getenv("DOUXIAN_GRAPH_V213_MAX_EXTRA","2"))))
                ids,hints=candidate_chunk_ids(query,bundle,max_candidates=limit)
                info["matched_edges"]=hints["matched_edges"]
                info["matched_labels"]=hints["matched_labels"]
                info["verified_relations_in_graph"]=bundle.total_verified
                info["graph_candidates"]=len(ids)
                info["graph_candidate_ids"]=list(ids)
                if ids:
                    # Validate ALL selected Chunk IDs against live Chroma *before*
                    # injecting any one of them. Partial drift -> all Hybrid.
                    live=_live_graph_documents(self,ids,bundle)
                    seen={_chunk_id(doc) for doc in original if _chunk_id(doc)}
                    # Audit mapping is read-only; same content is not same Chunk ID.
                    original_content_ids=defaultdict(list)
                    for _doc in original:
                        _meta=getattr(_doc,"metadata",{})
                        _meta=_meta if isinstance(_meta,dict) else {}
                        _key=(str(getattr(_doc,"page_content","")),
                              canonical_source(_meta.get("source")))
                        original_content_ids[_key].append(_chunk_id(_doc))
                    seen_content=set(original_content_ids)
                    added=[]
                    for cid,(body,metadata) in zip(ids,live):
                        key=(body,canonical_source(metadata.get("source")))
                        if cid in seen:
                            info["already_in_hybrid_ids"].append(cid)
                            continue
                        if key in seen_content:
                            info["matched_by_content_ids"].append(cid)
                            info["matched_by_content_map"][cid]=[
                                value for value in original_content_ids[key] if value
                            ]
                            continue
                        if original:
                            document_type=type(original[0])
                        else:
                            from langchain_core.documents import Document as document_type
                        try:
                            new_doc=document_type(page_content=body,metadata=metadata)
                        except (TypeError,ValueError):
                            from langchain_core.documents import Document
                            new_doc=Document(page_content=body,metadata=metadata)
                        added.append(new_doc)
                        seen.add(cid)
                        seen_content.add(key)
                    info["added_chunks"]=len(added)
                    info["injected_chunk_ids"]=[
                        _chunk_id(doc) for doc in added if _chunk_id(doc)
                    ]
                    info["status"]="applied" if added else "already_in_hybrid"
                    original=list(original)+added
            except Exception as exc:
                info["status"]="fallback_error"
                info["fallback_reason"]=type(exc).__name__
                # Graph failure must never alter the Hybrid answer or crash RAG.
                print("[Graph V2.13] fallback: "+type(exc).__name__)
            info["elapsed_ms"]=round((time.perf_counter()-started)*1000,2)
            self.graph_v213_last_diagnostics=info
            print("[Graph V2.13] status="+info["status"]+
                  " matches="+str(info["matched_edges"])+
                  " added="+str(info["added_chunks"]))
            return original

    GraphAugmentedKnowledgeBaseService.__name__="GraphAugmentedKnowledgeBaseService"
    return GraphAugmentedKnowledgeBaseService
