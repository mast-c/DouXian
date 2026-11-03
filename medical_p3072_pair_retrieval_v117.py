# -*- coding: utf-8 -*-
"""P3-07.2 pair-agnostic retrieval from an all-Drug/all-safety-Section snapshot.

* No hard-coded pair, no drug-drug edges, no automatic clinical approval.
* All sections derived from Neo4j Drug->SourceRecord->Section only.
* Drug aliases derived from name morphology are *search candidates*, not
  verified ingredient/salt equivalence, brand substitution or approved labels.
* A class mention without separately curated taxonomy is NOT pair evidence.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any
from urllib.parse import urlparse

from medical_p3072_corpus_index_v117 import (
    CorpusBlocked,normalized,open_snapshot,
)

TRACE_KEY="p3072_full_corpus_interactions"
ENABLE_FLAG="MEDICAL_P3072_FULL_INTERACTION_ENABLED"
DEFAULT_INDEX="stage4_reports/p3072_full_drug_interaction_index.sqlite"
RISK_TOKENS=(
    "出血","胃肠道出血","凝血","血小板","呼吸抑制","心律失常",
    "低血压","低血糖","肝损伤","肾损伤","心脏毒性","致死",
    "严重不良反应","高钾血症","肌无力","惊厥",
    "bleeding","hemorrhage","haemorrhage","seizure",
    "arrhythmia","renal injury","hepatotoxicity",
)
CAUTION_TOKENS=(
    "禁用","禁忌","慎用","不宜","不应","避免合用","不推荐",
    "禁止","不得","相互作用","监测","风险增加",
    "contraindicated","avoid","do not combine",
)
# These are generic vocabulary search leads, *not* a taxonomy assigning
# drugs to the class. A class mentioned in local Section is NOT enough
# to prove the other named drug belongs to the class.
CLASS_TEXT_TOKENS=(
    "抗凝药","抗凝剂","非甾体抗炎药","非甾体抗炎",
    "NSAID","双香豆素","香豆素","抗血小板药","抗心律失常药",
    "阿片类","大环内酯类","抗真菌药","喹诺酮类",
    "CYP3A4","细胞色素P450","口服抗凝药",
)
MAX_EXCERPT=420


def index_path(root:Path|str,index:Path|str|None=None)->Path:
    root=Path(root).resolve()
    if index is None:
        raw=os.getenv("MEDICAL_P3072_INDEX_PATH",DEFAULT_INDEX)
    else:
        raw=str(index)
    path=Path(raw)
    return (path if path.is_absolute() else root/path).resolve()


@lru_cache(maxsize=4)
def _load_aliases(path:str,mtime_ns:int)->tuple:
    # Catalog is reused for repeated Streamlit questions while indexed
    # source snapshot remains the same. Rebuild invalidates by mtime.
    conn,meta=open_snapshot(Path(path))
    try:
        rows=conn.execute(
            """SELECT DISTINCT surface,canonical_key,basis
               FROM aliases ORDER BY LENGTH(surface) DESC,surface,canonical_key"""
        ).fetchall()
        return tuple((r[0],r[1],r[2]) for r in rows)
    finally:
        conn.close()


def _has_english_boundary(text:str,start:int,end:int)->bool:
    before=text[start-1] if start>0 else ""
    after=text[end] if end<len(text) else ""
    return not (
        before.isascii() and before.isalpha()
        or after.isascii() and after.isalpha()
    )


def _token_positions(text:str,term:str,limit:int=30)->list[int]:
    """Literal lookups only, preserving source offsets."""
    hay=text.casefold()
    low=term.casefold()
    if not low:return []
    found=[]
    pos=0
    for _ in range(limit):
        at=hay.find(low,pos)
        if at<0:break
        stop=at+len(low)
        if not all(ord(c)<128 and c.isalpha() for c in low) or (
            _has_english_boundary(hay,at,stop)
        ):
            found.append(at)
        pos=at+1
    return found


def recognize_drugs(question:str,*,alias_rows:tuple,limit:int=6)->dict:
    """Longest non-overlapping name tokens; refuse ambiguous and 3+ ingredients."""
    text=normalized(str(question or ""))[:500]
    if not text:
        return {"status":"EMPTY_QUESTION","matches":[]}
    hits=[]
    for surface,key,basis in alias_rows:
        if len(surface)<2:
            continue
        for at in _token_positions(text,surface,limit=4):
            hits.append({
                "start":at,"end":at+len(surface),"surface":surface,
                "canonical_key":key,"basis":basis
            })
    if not hits:
        return {"status":"NO_CATALOG_DRUG_DETECTED","matches":[]}
    basis_priority={
        "exact_formulation":0,
        "dosage_suffix_removed":1,
        "heuristic_salt_morphology":2,
    }
    hits.sort(key=lambda item:(
        -(item["end"]-item["start"]),
        basis_priority.get(item["basis"],9),
        item["start"],item["canonical_key"]
    ))
    occupied=[]
    accepted=[]
    for item in hits:
        if any(item["start"]<b and a<item["end"] for a,b in occupied):
            continue
        same_span=[
            h for h in hits
            if h["start"]==item["start"]
            and h["end"]==item["end"]
        ]
        canonical={h["canonical_key"] for h in same_span}
        if len(canonical)>1:
            return {
                "status":"AMBIGUOUS_CATALOG_ALIAS",
                "matches":[],
                "ambiguous_surface":item["surface"],
                "group_candidates":sorted(canonical)[:10],
            }
        occupied.append((item["start"],item["end"]))
        accepted.append(item)
    groups={}
    for item in sorted(accepted,key=lambda x:x["start"]):
        key=item["canonical_key"]
        if key not in groups:
            groups[key]={
                "canonical_key":key,
                "surface":item["surface"],
                "basis":item["basis"],
                "match_start":item["start"],
                "name_resolution_is_clinically_verified":False,
            }
    matches=list(groups.values())[:limit]
    if len(groups)>2:
        status="THREE_OR_MORE_DRUGS_NOT_A_PAIR"
    elif len(groups)<2:
        status="ONLY_ONE_DRUG_DETECTED"
    else:
        status="EXACTLY_TWO_CATALOG_GROUPS_DETECTED"
    return {
        "status":status,"matches":matches,
        "detected_group_count":len(groups),
        "catalog_name_morphology_only":True,
    }


def _source_link(url)->str:
    if not isinstance(url,str) or len(url)>2048:return ""
    parsed=urlparse(url)
    return url if parsed.scheme in ("https","http") and parsed.netloc else ""


def _evidence_literal(text:str,anchors:list[str])->tuple[str,int,str]:
    """Only source quote; choose anchor with longest length for context."""
    possibilities=[]
    for term in anchors:
        for pos in _token_positions(text,term,limit=4):
            possibilities.append((pos,-len(term),term))
    if not possibilities:
        return text[:MAX_EXCERPT],0,""
    pos,_,term=min(possibilities)
    start=max(0,pos-145)
    stop=min(len(text),start+MAX_EXCERPT)
    return text[start:stop],start,term


def _neighborhood_terms(text:str,anchor:str,at:int)->dict:
    context=text[max(0,at-150):min(len(text),at+len(anchor)+160)]
    risk=[word for word in RISK_TOKENS if word.casefold() in context.casefold()]
    caution=[word for word in CAUTION_TOKENS if word.casefold() in context.casefold()]
    return {"risk_terms":list(dict.fromkeys(risk))[:10],
            "caution_terms":list(dict.fromkeys(caution))[:10]}


def _class_map(path:Path|None)->dict:
    """Optional MANUALLY CURATED source-supported research mapping.

    Never interprets a class relation as a real Neo4j edge. A source URL
    is a reference, not an automated approval or a latest-label guarantee.
    """
    if path is None or not Path(path).is_file():
        return {}
    raw=Path(path).read_bytes()
    if len(raw)>512*1024:
        raise CorpusBlocked("人工分类映射不能超过 512 KiB")
    data=json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data,dict) or data.get("schema_version")!=1:
        raise CorpusBlocked("分类映射格式不匹配")
    entries=data.get("entries",[])
    if not isinstance(entries,list) or len(entries)>30000:
        raise CorpusBlocked("人工分类条目超过预算")
    result={}
    for row in entries:
        if not isinstance(row,dict) or row.get("status")!="reviewed":
            continue
        key=normalized(row.get("canonical_key"))
        items=row.get("class_terms",[])
        source=row.get("source_url")
        reviewer=row.get("reviewed_by")
        if not (
            2<=len(key)<=80 and isinstance(items,list)
            and 1<=len(items)<=20 and
            isinstance(source,str) and source.startswith("https://")
            and isinstance(reviewer,str) and reviewer.strip()
        ):
            continue
        terms=[normalized(x) for x in items
               if isinstance(x,str) and 2<=len(normalized(x))<=60]
        if terms:
            result[key]={
                "terms":list(dict.fromkeys(terms)),
                "source_url":source[:2048],
                "reviewed_by":reviewer[:100],
                "clinical_verified":False,
                "neo4j_class_edge_present":False,
            }
    return result


def _analyze_section(row,opposite_aliases:list[str],
                     reviewed_classes:list[str],own_key:str)->dict:
    """Text mention + nearby risk words, no cross-source clinical claim."""
    text=row["section_text"]
    field=row["section_field"]
    direct=[]
    for term in opposite_aliases:
        for at in _token_positions(text,term,limit=8):
            direct.append((at,-len(term),term))
    class_hits=[]
    for term in reviewed_classes:
        for at in _token_positions(text,term,limit=8):
            class_hits.append((at,-len(term),term))
    unreviewed=[]
    if not direct and not class_hits:
        for term in CLASS_TEXT_TOKENS:
            for at in _token_positions(text,term,limit=2):
                unreviewed.append((at,-len(term),term))

    if direct:
        at,_,anchor=min(direct)
        flag="LITERAL_OTHER_DRUG_MENTION"
        bridge="exact_alias_in_source_text"
    elif class_hits:
        at,_,anchor=min(class_hits)
        flag="REVIEWED_CLASS_TERM_IN_SOURCE_TEXT"
        bridge="external_reviewed_mapping_not_neo4j_edge"
    elif unreviewed:
        at,_,anchor=min(unreviewed)
        flag="UNRESOLVED_CLASS_MENTION_ONLY"
        bridge="no_verified_link_to_other_drug"
    else:
        at,anchor=-1,""
        flag="NO_COUNTERPART_MENTION"
        bridge="none"
    markers=(
        _neighborhood_terms(text,anchor,at)
        if at>=0 else {"risk_terms":[],"caution_terms":[]}
    )
    if flag=="LITERAL_OTHER_DRUG_MENTION":
        if markers["risk_terms"] or markers["caution_terms"]:
            status="DIRECT_DRUG_NAME_PLUS_WARNING_WORDS"
            score=100
        else:
            status="DIRECT_DRUG_NAME_MENTION_ONLY"
            score=70
    elif flag=="REVIEWED_CLASS_TERM_IN_SOURCE_TEXT":
        status=(
            "REVIEWED_CLASS_BRIDGE_WITH_WARNING_TERMS"
            if markers["risk_terms"] or markers["caution_terms"]
            else "REVIEWED_CLASS_TERM_WITHOUT_WARNING"
        )
        score=60 if markers["risk_terms"] or markers["caution_terms"] else 40
    elif flag=="UNRESOLVED_CLASS_MENTION_ONLY":
        status=(
            "UNRESOLVED_CLASS_WARNING_NOT_PAIR_EVIDENCE"
            if markers["risk_terms"] or markers["caution_terms"]
            else "UNRESOLVED_CLASS_MENTION_NOT_PAIR_EVIDENCE"
        )
        score=15
    else:
        status="NO_PAIR_RELEVANT_LITERAL_EVIDENCE"
        score=0
    anchors=[anchor] if anchor else []
    quote,offset,matched=_evidence_literal(text,anchors)
    field_priority={"drug_interactions":3,"warnings":2,"contraindications":1}
    return {
        "status":status,
        "score":score+field_priority.get(field,0),
        "matched_literal":matched,
        "bridge_kind":bridge,
        "risk_terms_near_match":markers["risk_terms"],
        "caution_terms_near_match":markers["caution_terms"],
        "quote":quote,
        "quote_start":offset,
        "actual_interaction_graph_edge_verified":False,
        "medical_truth_reviewed":False,
    }


def _per_group_rows(conn:sqlite3.Connection,key:str,
                    limit:int)->tuple[list[sqlite3.Row],bool,int]:
    count=conn.execute(
        """SELECT COUNT(*) FROM sections s
           JOIN drugs d ON d.drug_id=s.drug_id
           WHERE d.canonical_key=?""",(key,)
    ).fetchone()[0]
    rows=conn.execute("""
SELECT s.drug_id,d.drug_name,s.record_id,s.section_id,
       s.section_field,s.section_text,s.section_sha256,
       s.source_url,s.authority_status
FROM sections s JOIN drugs d ON d.drug_id=s.drug_id
WHERE d.canonical_key=?
ORDER BY CASE s.section_field
         WHEN 'drug_interactions' THEN 0
         WHEN 'warnings' THEN 1 ELSE 2 END,
         d.drug_name,s.record_id,s.section_id
LIMIT ?
""",(key,limit)).fetchall()
    return rows,count>limit,count


def inspect_pair(
    question:str,*,root:Path|str=".",index:Path|str|None=None,
    curated_map:Path|str|None=None,
    max_sections_per_group:int=400,
    max_evidence:int=16,
)->dict:
    """Works for arbitrary TWO detected drugs present in the full graph catalog.

    A source signal is never automatically a drug interaction or a safe
    combination; no Neo4j/Chroma writes and no fabricated clinical classes.
    """
    if not 1<=max_sections_per_group<=2000 or not 1<=max_evidence<=24:
        raise CorpusBlocked("检索预算超出限定范围")
    snapshot=index_path(root,index)
    conn,meta=open_snapshot(snapshot)
    try:
        alias_rows=_load_aliases(str(snapshot),snapshot.stat().st_mtime_ns)
        detected=recognize_drugs(question,alias_rows=alias_rows)
        result={
            "stage":"P3-07.2",
            "question":str(question or "")[:500],
            "status":"OUT_OF_SCOPE",
            "name_resolution":detected,
            "index_snapshot":{
                "name":snapshot.name,
                "created_utc":meta.get("completed_utc"),
                "drugs_indexed":meta.get("drugs_indexed"),
                "drugs_with_safety_sections":meta.get("drugs_with_safety_sections"),
                "sections_indexed":meta.get("sections_indexed"),
                "sections_overlong_skipped":meta.get("sections_overlong_skipped"),
                "orphan_sections_skipped":meta.get("orphan_sections_skipped"),
                "coverage_pct":meta.get("coverage_pct"),
                "snapshot_may_be_stale":True,
            },
            "clinical_verified":False,
            "research_only":True,
            "medical_decision":None,
            "pair_safe_to_combine":None,
            "interaction_exists_in_graph":False,
            "read_source_graph_edges_only":True,
            "neo4j_writes":0,
            "chroma_writes":0,
            "local_index_writes":0,
            "source_rows_scanned":0,
            "source_rows_total":0,
            "search_budget_truncated":False,
            "source_evidence":[],
            "evidence_summary":{
                "direct_name_warning_count":0,
                "reviewed_class_bridge_count":0,
                "unresolved_class_warning_count":0,
            },
            "notes":[
                "名称词形归一化不是经医学核验的成分等价声明。",
                "仅可读取 Drug->SourceRecord->Section 的真实2跳来源链。",
                "药物相互作用关键词共现、分类桥接都不是自动生成的Neo4j边。",
                "未发现相互作用证据不代表两药可以安全合用。",
            ],
        }
        if detected["status"]!="EXACTLY_TWO_CATALOG_GROUPS_DETECTED":
            result["status"]=detected["status"]
            return result
        groups=detected["matches"]
        keys=[item["canonical_key"] for item in groups]
        curated=_class_map(Path(curated_map) if curated_map else None)
        aliases={}
        for key in keys:
            aliases[key]=[
                row[0] for row in conn.execute(
                    "SELECT DISTINCT surface FROM aliases WHERE canonical_key=? "
                    "ORDER BY LENGTH(surface) DESC LIMIT 150",(key,)
                ).fetchall()
                if 2<=len(row[0])<=100
            ]
            # canonical itself is a search token, if not in full drug aliases.
            if key not in aliases[key]:aliases[key].append(key)
        found=[]
        for own,other in ((keys[0],keys[1]),(keys[1],keys[0])):
            rows,truncated,total=_per_group_rows(
                conn,own,max_sections_per_group
            )
            result["source_rows_scanned"]+=len(rows)
            result["source_rows_total"]+=total
            result["search_budget_truncated"]|=truncated
            reviewed=curated.get(other,{}).get("terms",[])
            for row in rows:
                text=row["section_text"]
                expected=hashlib.sha256(text.encode("utf-8")).hexdigest()
                if expected!=row["section_sha256"]:
                    raise CorpusBlocked(
                        "本地派生索引 Section 哈希不一致，拒绝展示受损证据"
                    )
                analysis=_analyze_section(
                    row,aliases[other],reviewed,own
                )
                item={
                    "owner_canonical":own,
                    "other_canonical":other,
                    "owner_drug_name":row["drug_name"],
                    "node_ids":[row["drug_id"],row["record_id"],row["section_id"]],
                    "node_labels":["Drug","SourceRecord","Section"],
                    "edge_types":["HAS_RECORD","HAS_SECTION"],
                    "actual_graph_hops":2,
                    "section_field":row["section_field"],
                    "section_sha256":row["section_sha256"],
                    "source_url":_source_link(row["source_url"]),
                    "authority_status":row["authority_status"],
                    "source_status":"third_party_unverified",
                    "source_snapshot_not_live_reverified":True,
                    "clinical_verified":False,
                    "reviewed_class_link":curated.get(other) if reviewed else None,
                    "analysis":analysis,
                }
                if analysis["score"]>0:
                    found.append(item)
        found.sort(
            key=lambda row:(-row["analysis"]["score"],
                            row["owner_canonical"],row["node_ids"])
        )
        dedup=set()
        evidence=[]
        for row in found:
            key=(row["owner_canonical"],tuple(row["node_ids"]))
            if key in dedup:continue
            dedup.add(key)
            evidence.append(row)
            if len(evidence)>=max_evidence:break
        result["source_evidence"]=evidence
        for item in found:
            status=item["analysis"]["status"]
            if status=="DIRECT_DRUG_NAME_PLUS_WARNING_WORDS":
                result["evidence_summary"]["direct_name_warning_count"]+=1
            elif status=="REVIEWED_CLASS_BRIDGE_WITH_WARNING_TERMS":
                result["evidence_summary"]["reviewed_class_bridge_count"]+=1
            elif status in (
                "UNRESOLVED_CLASS_WARNING_NOT_PAIR_EVIDENCE",
                "UNRESOLVED_CLASS_MENTION_NOT_PAIR_EVIDENCE"
            ):
                result["evidence_summary"]["unresolved_class_warning_count"]+=1
        if result["search_budget_truncated"]:
            result["status"]="PARTIAL_COVERAGE_QUERY_BUDGET"
        elif result["evidence_summary"]["direct_name_warning_count"]>0:
            result["status"]="LOCAL_DIRECT_MENTION_WITH_WARNING_WORDS"
        elif result["evidence_summary"]["reviewed_class_bridge_count"]>0:
            result["status"]="LOCAL_REVIEWED_CLASS_TEXT_BRIDGE"
        elif result["evidence_summary"]["unresolved_class_warning_count"]>0:
            result["status"]="LOCAL_UNRESOLVED_CLASS_WARNING_ONLY"
        elif evidence:
            result["status"]="LOCAL_OTHER_DRUG_MENTION_WITHOUT_WARNING"
        else:
            result["status"]="NO_PAIR_SPECIFIC_EVIDENCE_FOUND"
        return result
    finally:
        conn.close()
