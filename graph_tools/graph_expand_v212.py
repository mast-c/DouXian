# -*- coding: utf-8 -*-
"""DouXian V2.12: strictly evidence-based OFFLINE graph expansion & alias registry.

CLI: propose -> [V2.11.1 link] -> promote -> verify
- Original V2.7 graph, textbooks, Chroma database, and RAG files are never modified.
- Proposed rules are curated, deterministic, exact textbook phrases.
- Proposals alone are NOT verified knowledge. Only real unique Chroma Chunk matches
  may be promoted; previous V2.11.1 verified Chunk IDs must remain unchanged.
- One original textbook per graph. Multi-book merge is deliberately out of scope.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import unicodedata

if __package__:
    from .graph_paths import BINDINGS_DIR, CONTENT_DIR, REPORTS_DIR, SOURCE_FILE
else:
    from graph_paths import BINDINGS_DIR, CONTENT_DIR, REPORTS_DIR, SOURCE_FILE


SEMANTIC = frozenset({
    "IS_A", "SPECIAL_CASE_OF", "HAS_PART", "USES",
    "BASED_ON", "SUBFIELD_OF", "COMPUTES", "ALSO_CALLED",
})
GRAPH_SCHEMA = "douxian-kg-2.7-offline-1"  # kept compatible with existing V2.11.1
EXT_SCHEMA = "douxian-kg-expansion-2.12-1"
HEADER_CHAPTER = re.compile(r"^第[0-9一二三四五六七八九十百]+章")


# Manually inspected against "神经网络与深度学习.md" from the user's graph snapshot.
# Only these specific, explicitly stated facts may become new candidates.
# Every rule also pins an exact heading and original line fragment.
# "full_line" yields a stronger evidence quote when a local sentence is required.
RULES = (
    ("最大汇聚", "IS_A", "汇聚函数", "5.2.3 汇聚层",
     "（1） 最大汇聚（Maximum Pooling或Max Pooling）", "needle"),
    ("平均汇聚", "IS_A", "汇聚函数", "5.2.3 汇聚层",
     "（2） 平均汇聚（Mean Pooling）", "needle"),
    ("卷积层", "USES", "卷积核", "5.2.2 卷积层",
     "为了计算输出特征映射", "full_line"),
    ("卷积层", "USES", "非线性激活函数", "5.2.2 卷积层",
     "再经过非线性激活函数后得到输出特征映射", "full_line"),
    ("汇聚层", "USES", "最大汇聚", "5.2.3 汇聚层",
     "典型的汇聚层是将每个特征映射划分为", "full_line"),
    ("自动微分", "USES", "链式法则", "4.5.3 自动微分",
     "然后利用链式法则来自动计算一个复合函数的梯度", "needle"),
    ("无向图模型", "ALSO_CALLED", "马尔可夫随机场", "11.1.3 无向图模型",
     "无向图模型，也称为马尔可夫随机场", "needle"),
    ("无向图模型", "ALSO_CALLED", "马尔可夫网络", "11.1.3 无向图模型",
     "无向图模型，也称为马尔可夫随机场（Markov Random Field，MRF）或马尔可夫网络", "needle"),
    ("变分推断", "IS_A", "近似推断", "11.3.2 近似推断",
     "变分推断（Variational Inference）是引入一个变分分布", "needle"),
    ("变分推断", "ALSO_CALLED", "变分贝叶斯", "11.4 变分推断",
     "也称为 变分贝叶斯（Varia-tionalBayesian）", "needle"),
    ("拒绝采样", "IS_A", "采样法", "11.5.2 拒绝采样",
     "拒绝采样（Rejection Sampling）是一种间接采样方法", "needle"),
    ("拒绝采样", "ALSO_CALLED", "接受-拒绝采样", "11.5.2 拒绝采样",
     "也称为接受-拒绝采样（Acceptance-Rejection Sampling）", "needle"),
    ("残差网络", "USES", "残差连接", "5.4.4 残差网络",
     "残差网络（ResidualNetwork，ResNet）通过给非线性的卷积层增加直连边", "full_line"),
    ("残差网络", "HAS_PART", "残差单元", "5.4.4 残差网络",
     "残差网络就是将很多个残差单元串联起来构成的一个非常深的网络", "needle"),
    ("残差连接", "ALSO_CALLED", "直连边", "5.4.4 残差网络",
     "直连边（Shortcut Connection）（也称为残差连接（Residual Connection））", "needle"),
    ("卷积核", "ALSO_CALLED", "滤波器", "5.1.1.1 一维卷积",
     "称为滤波器（Filter）或卷积核（Convolution Kernel）", "needle"),
    ("互相关", "ALSO_CALLED", "不翻转卷积", "5.1.2 互相关",
     "因此互相关也可以称为不翻转卷积", "needle"),
    ("MCMC", "IS_A", "采样法", "11.5.4 马尔可夫链蒙特卡罗方法",
     "（Markov Chain Monte Carlo，MCMC）方法是一种更好的采样方法", "needle"),
)


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def stable_id(prefix, value):
    return prefix + ":" + sha_bytes(value.encode("utf-8"))[:16]


def norm_text(text):
    return re.sub(r"\s+", "", str(text or "")).replace("\u200b", "")


def lexical_key(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or ""))).casefold()


def source_name(path):
    return Path(str(path).replace("\\", "/")).name.casefold()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def serialized(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def semantic_edges(graph):
    return [e for e in graph["edges"] if e.get("relation") in SEMANTIC]


def validate_source_graph(graph, source_path):
    """Validate original immutable provenance and evidence line references."""
    raw = Path(source_path).read_bytes()
    meta = graph.get("metadata") or {}
    require(meta.get("source_sha256") == sha_bytes(raw),
            "源教材 SHA256 与图谱不一致：禁止继续")
    require(source_name(meta.get("source_file")) == source_name(source_path),
            "教材文件名与图谱 source_file 不匹配")
    require(meta.get("schema") == GRAPH_SCHEMA, "仅支持 V2.7 图谱 schema")
    lines = raw.decode("utf-8-sig").splitlines()

    nodes=graph.get("nodes") or []
    edges=graph.get("edges") or []
    node_ids={n["id"] for n in nodes}
    edge_ids={e["id"] for e in edges}
    require(len(node_ids)==len(nodes), "节点 ID 重复")
    require(len(edge_ids)==len(edges), "边 ID 重复")
    for e in edges:
        require(e["source"] in node_ids and e["target"] in node_ids,
                "图谱存在悬空关系: "+str(e.get("id")))
        if e.get("relation") not in SEMANTIC:
            continue
        ev=e["evidence"]
        num=ev.get("line")
        require(isinstance(num,int) and 1<=num<=len(lines),
                "证据行号无效: "+e["id"])
        quote=ev.get("text", "").rstrip("…").strip()
        require(ev.get("extraction") == "exact_source_phrase" and quote,
                "语义边缺少严格原文证据: "+e["id"])
        require(norm_text(quote) in norm_text(lines[num-1]),
                "语义边证据与原文不符: "+e["id"])
        require(source_name(ev.get("source_file"))==source_name(source_path),
                "语义边来源不属于该教材: "+e["id"])
    return lines


def find_rule_evidence(rule, nodes_by_section, lines, filename):
    subject,relation,obj,title,needle,mode=rule
    matches=[]
    for section in nodes_by_section.get(title, []):
        start=section["body_start_line"]
        end=section["body_end_line"]
        for index in range(start-1,end):
            if needle in lines[index]:
                matches.append((section,index+1,lines[index]))
    require(len(matches)==1,
            ("策划规则未唯一命中原文：" + subject+" "+relation+" "+obj+
             "；标题="+title+"；命中数="+str(len(matches))))
    section,line_number,line=matches[0]
    quote=line.strip() if mode=="full_line" else needle
    require(quote and norm_text(quote) in norm_text(line),
            "证据片段不在原始行："+subject)
    # V2.11 must also be able to match the complete literal quote inside one Chunk.
    require(len(quote)<=420,"候选原文证据超过420字符，请调整规则")
    return {
        "source_file":filename,
        "section_path":section["path"],
        "line":line_number,
        "text":quote,
        "extraction":"exact_source_phrase",
    }, section["id"]


def refresh_graph_stats(graph):
    """Keep materialized graph metadata honest after additions or filtering."""
    graph["metadata"]["node_counts"]=dict(Counter(n["type"] for n in graph["nodes"]))
    graph["metadata"]["edge_counts"]=dict(Counter(e["relation"] for e in graph["edges"]))
    graph["metadata"]["semantic_rules_matched"]=len(semantic_edges(graph))
    return graph


def propose_graph(base_graph, source_path, baseline_path, rules=RULES):
    source=Path(source_path)
    lines=validate_source_graph(base_graph,source)
    require(source_name(source.name)=="神经网络与深度学习.md" or rules is not RULES,
            "内置精选规则仅适用于《神经网络与深度学习》；其他教材请使用独立规则与图谱")
    sections=defaultdict(list)
    for node in base_graph["nodes"]:
        if node.get("type")=="Section":
            sections[node["title"]].append(node)
    original_node_ids={n["id"] for n in base_graph["nodes"]}
    concept_ids={n["label"]:n["id"] for n in base_graph["nodes"] if n.get("type")=="Concept"}
    seen={(e["source"],e["relation"],e["target"],e["evidence"]["section_path"])
          for e in semantic_edges(base_graph)}
    new_nodes=[]
    new_edges=[]
    for rule in rules:
        subject,relation,obj,title,needle,mode=rule
        require(relation in SEMANTIC, "非法关系类型")
        require(subject != obj,"候选关系不能自指")
        evidence,section_id=find_rule_evidence(rule,sections,lines,source.name)
        for label in (subject,obj):
            if label not in concept_ids:
                cid=stable_id("concept",label)
                require(cid not in original_node_ids and
                        all(n["id"]!=cid for n in new_nodes),
                        "新概念 ID 冲突: "+label)
                concept_ids[label]=cid
                new_nodes.append({
                    "id":cid,"type":"Concept",
                    "label":label,"aliases_for_search":[],
                    "v212_origin":"curated_source_phrase",
                })
        key=(concept_ids[subject],relation,concept_ids[obj],evidence["section_path"])
        require(key not in seen,"候选关系与基线或其他规则重复: "+str(rule[:3]))
        seen.add(key)
        edge_key=(concept_ids[subject],relation,concept_ids[obj],section_id)
        edge_id=stable_id("edge","\u241f".join(map(str,edge_key)))
        require(edge_id not in {e["id"] for e in base_graph["edges"]+new_edges},
                "新增 edge ID 冲突："+edge_id)
        new_edges.append({
            "id":edge_id,"source":concept_ids[subject],
            "relation":relation,"target":concept_ids[obj],
            "evidence":evidence,
            "v212_origin":"curated_candidate",
        })
    require(new_edges,"没有新增语义关系")
    candidate=deepcopy(base_graph)
    candidate["nodes"].extend(new_nodes)
    candidate["edges"].extend(new_edges)
    refresh_graph_stats(candidate)
    candidate["metadata"]["v212"]={
        "schema":EXT_SCHEMA,
        "phase":"candidate",
        "baseline_graph_sha256":sha_file(baseline_path),
        "original_semantic_count":len(semantic_edges(base_graph)),
        "proposed_count":len(new_edges),
        "new_concept_count":len(new_nodes),
        "extraction":"curated_source_phrase_no_model",
        "created_utc":timestamp(),
        "warning":"CANDIDATE ONLY. Must bind against real Chroma and promote; do not use online.",
    }
    names={node["id"]:node["label"] for node in candidate["nodes"]}
    catalog={
        "schema":EXT_SCHEMA,
        "baseline_graph_sha256":sha_file(baseline_path),
        "source_sha256":sha_file(source),
        "baseline_semantic_count":len(semantic_edges(base_graph)),
        "proposed_count":len(new_edges),
        "new_concept_count":len(new_nodes),
        "warning":"All proposed facts require exact real Chroma Chunk verification.",
        "candidates":[{
            "claim_id":e["id"],
            "subject":names[e["source"]],
            "predicate":e["relation"],
            "object":names[e["target"]],
            "evidence":e["evidence"],
            "status":"pending_real_chunk_binding",
        } for e in new_edges],
    }
    return candidate,catalog


def require_patched_linker():
    try:
        binder=(importlib.import_module(".graph_chunk_linker_v211", __package__)
                if __package__ else importlib.import_module("graph_chunk_linker_v211"))
    except (ImportError,ModuleNotFoundError) as exc:
        raise RuntimeError(
            "请将 V2.11.1 验收通过的 graph_chunk_linker_v211.py 放在 graph_tools 目录") from exc
    try:
        params=inspect.signature(binder.canonical_section).parameters
        require("source_file" in params, "检测到旧版 V2.11，请使用已验收的 V2.11.1")
    except AttributeError as exc:
        raise RuntimeError("graph_chunk_linker_v211.py 缺少已验收的章节归一化接口") from exc
    return binder


def check_real_binding(base_graph, base_graph_path, source, chunks, base_links_path,
                       candidate_graph=None, candidate_graph_path=None,
                       candidate_links_path=None):
    binder=require_patched_linker()
    base_links=load_json(base_links_path)
    require(base_links["metadata"].get("graph_sha256")==sha_file(base_graph_path),
            "旧版 V2.11.1 绑定文件与当前基线图谱不一致")
    binder.verify_links(base_graph,Path(source),Path(chunks),Path(base_links_path))
    baseline_claims={c["claim_id"]:c for c in base_links["claims"]}
    base_edges=semantic_edges(base_graph)
    require(set(baseline_claims)=={e["id"] for e in base_edges},
            "旧版绑定未覆盖全部原始语义边")
    require(all(c["status"]=="verified" and len(c["chunk_ids"])==1
                for c in baseline_claims.values()),
            "基线存在未 verified 的语义边：拒绝升级")

    candidate_links=None
    if candidate_graph is not None:
        candidate_links=load_json(candidate_links_path)
        require(candidate_links["metadata"].get("graph_sha256")==sha_file(candidate_graph_path),
                "候选绑定文件与候选图谱 SHA 不一致")
        binder.verify_links(candidate_graph,Path(source),Path(chunks),
                            Path(candidate_links_path))
        proposed_claims={c["claim_id"]:c for c in candidate_links["claims"]}
        for cid,original in baseline_claims.items():
            current=proposed_claims.get(cid)
            require(current is not None and
                    current["status"]=="verified" and
                    current["chunk_ids"]==original["chunk_ids"],
                    "候选图谱改变了原有 12 条可信关系的 Chunk："+cid)
        require(set(proposed_claims)=={e["id"] for e in semantic_edges(candidate_graph)},
                "候选绑定报告与候选图谱语义边集合不一致")
    return binder,baseline_claims,candidate_links


def derive_alias_registry(graph,link_report):
    """Strict synonyms only from verified ALSO_CALLED edges.

    Legacy aliases_for_search remain non-authoritative hints, never used to
    union nodes or declare identity. Contradictory lexical hints are flagged.
    """
    nodes={n["id"]:n for n in graph["nodes"] if n.get("type")=="Concept"}
    parents={cid:cid for cid in nodes}
    def root(x):
        while parents[x]!=x:
            parents[x]=parents[parents[x]]
            x=parents[x]
        return x
    def union(x,y):
        a,b=root(x),root(y)
        if a!=b:
            parents[b]=a

    claims={c["claim_id"]:c for c in link_report["claims"]}
    synonym_edges=[]
    for edge in graph["edges"]:
        if edge.get("relation")!="ALSO_CALLED":
            continue
        row=claims.get(edge["id"])
        if row is None or row["status"]!="verified" or len(row["chunk_ids"])!=1:
            continue
        require(edge["source"] in nodes and edge["target"] in nodes,
                "ALSO_CALLED 边非概念节点，不能生成同义词组")
        union(edge["source"],edge["target"])
        synonym_edges.append(edge)
    groups=defaultdict(list)
    for cid in nodes:
        groups[root(cid)].append(cid)
    # Prefer the subject of explicitly verified ALSO_CALLED relations as root.
    subject_count=Counter(e["source"] for e in synonym_edges)
    concept_order={cid:i for i,cid in enumerate(nodes)}
    group_map={}
    synonym_groups=[]
    for members in groups.values():
        anchor=sorted(members,key=lambda cid:(
            -subject_count[cid],concept_order[cid],cid))[0]
        for cid in members:
            group_map[cid]=anchor
        if len(members)>1:
            evidence_ids=[
                e["id"] for e in synonym_edges
                if e["source"] in members and e["target"] in members
            ]
            synonym_groups.append({
                "canonical_id":anchor,
                "canonical_label":nodes[anchor]["label"],
                "member_ids":members,
                "verified_labels":[nodes[cid]["label"] for cid in members],
                "verified_claim_ids":evidence_ids,
            })
    synonym_groups.sort(key=lambda x:x["canonical_label"])
    term_to_groups=defaultdict(set)
    search_only=[]
    verified_term_groups=defaultdict(set)
    for cid,node in nodes.items():
        anchor=group_map[cid]
        term_to_groups[lexical_key(node["label"])].add(anchor)
        verified_term_groups[lexical_key(node["label"])].add(anchor)
        for alias in node.get("aliases_for_search",[]):
            term_to_groups[lexical_key(alias)].add(anchor)
            search_only.append({
                "concept_id":cid,
                "term":alias,
                "status":"search_hint_only_not_proven_equivalent",
            })
    ambiguous_terms=[
        {
            "lexical_key":key,
            "canonical_ids":sorted(roots),
            "reason":"name_or_search_hint_used_in_multiple_distinct_verified_groups"
        }
        for key,roots in sorted(term_to_groups.items()) if len(roots)>1
    ]
    verified_map={
        key: next(iter(roots))
        for key,roots in sorted(verified_term_groups.items())
        if len(roots)==1
    }
    return {
        "schema":"douxian-verified-aliases-2.12-1",
        "created_utc":timestamp(),
        "verified_synonym_groups":synonym_groups,
        "verified_label_to_canonical_id":verified_map,
        "search_only_hints":search_only,
        "ambiguous_terms":ambiguous_terms,
        "warning":"Only verified ALSO_CALLED claims establish equivalence. Search hints never do.",
    }


def create_promoted_graph(base, candidate, candidate_links, base_path):
    """Promote only verifiably unique new claims, preserve all baseline edges."""
    base_ids={e["id"] for e in base["edges"]}
    new=[e for e in candidate["edges"] if e["id"] not in base_ids]
    status={c["claim_id"]:c for c in candidate_links["claims"]}
    decisions=[]
    selected=[]
    for e in new:
        claim=status.get(e["id"])
        require(claim is not None,"新增关系缺少绑定信息："+e["id"])
        eligible=(claim["status"]=="verified" and len(claim["chunk_ids"])==1)
        decisions.append({
            "claim_id":e["id"],"relation":e["relation"],
            "status":claim["status"],"promoted":eligible,
            "chunk_ids":claim["chunk_ids"] if eligible else [],
            "reason":claim["reason"],
        })
        if eligible:
            selected.append(deepcopy(e))
    # Never silently call a baseline-only graph a successful expansion.
    require(selected,"新增关系没有任何 verified Chunk；未生成正式 V2.12 图谱")
    node_ids={n["id"] for n in base["nodes"]}
    byid={n["id"]:n for n in candidate["nodes"]}
    selected_ids={e["source"] for e in selected}|{e["target"] for e in selected}
    extra_nodes=[deepcopy(byid[cid]) for cid in byid
                 if cid in selected_ids and cid not in node_ids]
    result=deepcopy(base)
    result["nodes"].extend(extra_nodes)
    for edge in selected:
        edge["v212_origin"]="promoted_unique_real_chunk"
        result["edges"].append(edge)
    require(all(e["source"] in {n["id"] for n in result["nodes"]}
                and e["target"] in {n["id"] for n in result["nodes"]}
                for e in selected),"升级产生了悬空概念节点")
    refresh_graph_stats(result)
    result["metadata"]["v212"]={
        "schema":EXT_SCHEMA,"phase":"promoted",
        "baseline_graph_sha256":sha_file(base_path),
        "original_semantic_count":len(semantic_edges(base)),
        "candidate_count":len(new),
        "new_verified_count":len(selected),
        "new_rejected_count":len(new)-len(selected),
        "new_concept_count":len(extra_nodes),
        "created_utc":timestamp(),
        "note":"Only unique, exact-source/section/quote-matched Chroma Chunk bindings promoted.",
    }
    return result,decisions


def recompute_for_graph(graph, graph_bytes, source, chunks, binder):
    result=binder.build_link_report(graph,Path(source),None,Path(chunks))
    result["metadata"]["graph_sha256"]=sha_bytes(graph_bytes)
    return result


def final_verification(base_path, source, chunks_path, base_links_path,
                       final_graph_path, final_links_path, gate_path=None,
                       alias_path=None, rules=RULES):
    base=load_json(base_path)
    final=load_json(final_graph_path)
    validate_source_graph(base,source)
    validate_source_graph(final,source)
    binder,baseline_claims,_=check_real_binding(
        base,base_path,source,chunks_path,base_links_path)
    require(final.get("metadata",{}).get("v212",{}).get("phase")=="promoted",
            "最终图谱缺少 V2.12 promoted 标识")
    require(final["metadata"]["v212"]["baseline_graph_sha256"]==sha_file(base_path),
            "最终图谱源基线哈希不一致")
    require(final["nodes"][:len(base["nodes"])]==base["nodes"] and
            final["edges"][:len(base["edges"])]==base["edges"],
            "原始图谱节点或关系被修改")
    # Defense-in-depth: a graph with recomputed SHA and valid source quote still
    # cannot introduce a relation absent from the actual vetted V2.12 rules.
    curated,_=propose_graph(base,source,base_path,rules=rules)
    expected_edges={e["id"]:e for e in curated["edges"][len(base["edges"]):]}
    expected_nodes={n["id"]:n for n in curated["nodes"][len(base["nodes"]):]}
    for edge in final["edges"][len(base["edges"]):]:
        baseline_expected=expected_edges.get(edge["id"])
        require(baseline_expected is not None,
                "最终图谱出现未经人工规则审核的语义关系："+edge["id"])
        got=deepcopy(edge)
        got["v212_origin"]="curated_candidate"
        require(got==baseline_expected,
                "最终图谱出现超出证据规则的关系字段："+edge["id"])
    for node in final["nodes"][len(base["nodes"]):]:
        require(node.get("id") in expected_nodes and
                node==expected_nodes[node["id"]],
                "最终图谱出现未经规则生成的概念节点")
    used={e["source"] for e in final["edges"][len(base["edges"]):]} | {
        e["target"] for e in final["edges"][len(base["edges"]):]
    }
    require({n["id"] for n in final["nodes"][len(base["nodes"]):]} ==
            (used-{n["id"] for n in base["nodes"]}),
            "正式图谱包含未被验证关系引用的新增概念")
    require(all(e.get("relation") in SEMANTIC for e in final["edges"][len(base["edges"]):]),
            "V2.12 新增了无法由 V2.11 绑定的关系类型")
    binder.verify_links(final,Path(source),Path(chunks_path),Path(final_links_path))
    final_links=load_json(final_links_path)
    require(final_links["metadata"].get("graph_sha256")==sha_file(final_graph_path),
            "最终图谱与绑定的 graph_sha256 不一致")
    final_claims={c["claim_id"]:c for c in final_links["claims"]}
    for cid,original in baseline_claims.items():
        require(cid in final_claims and
                final_claims[cid]["status"]=="verified" and
                final_claims[cid]["chunk_ids"]==original["chunk_ids"],
                "升级更改了原始已验证的 Chunk："+cid)
    new_edges=semantic_edges(final)[len(semantic_edges(base)):]
    require(new_edges,"最终图谱没有新增关系")
    require(all(final_claims[e["id"]]["status"]=="verified" and
                len(final_claims[e["id"]]["chunk_ids"])==1 for e in new_edges),
            "最终图谱中出现未经验证的新增 Claim")
    require(len(new_edges)==final["metadata"]["v212"]["new_verified_count"],
            "V2.12 metadata 数量不一致")
    require(set(final_claims)=={e["id"] for e in semantic_edges(final)},
            "最终绑定集合与语义边集合不一致")
    if gate_path:
        gate=load_json(gate_path)
        require(gate["final_graph_sha256"]==sha_file(final_graph_path) and
                gate["final_links_sha256"]==sha_file(final_links_path) and
                gate["baseline_links_sha256"]==sha_file(base_links_path),
                "V2.12 门禁报告哈希不匹配")
    if alias_path:
        reg=load_json(alias_path)
        if gate_path:
            require(gate.get("alias_registry_sha256")==sha_file(alias_path),
                    "门禁报告的别名登记表 SHA256 不匹配")
        require(reg["graph_sha256"]==sha_file(final_graph_path) and
                reg["links_sha256"]==sha_file(final_links_path),
                "别名登记表不属于当前图谱/绑定")
        recomputed=derive_alias_registry(final,final_links)
        for field in ("verified_synonym_groups","verified_label_to_canonical_id",
                      "search_only_hints","ambiguous_terms"):
            require(reg[field]==recomputed[field],
                    "别名登记表与已验证关系不一致："+field)
    return {
        "baseline_verified":len(baseline_claims),
        "new_verified":len(new_edges),
        "total_verified":len(final_claims),
        "snapshot_sha256":sha_file(chunks_path),
    }


def safe_write_json_many(outputs):
    """Never overwrite any existing file; on error revert newly written outputs."""
    prepared=[(Path(path),serialized(payload).encode("utf-8"))
              for path,payload in outputs]
    dests=[str(path.resolve()) for path,_ in prepared]
    require(len(set(dests))==len(dests),"多个输出参数指向同一文件")
    for path,_ in prepared:
        require(not path.exists(),"输出文件已存在（拒绝覆盖）: "+str(path))
    created=[]
    try:
        for path,data in prepared:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("xb") as f:
                created.append(path)
                f.write(data)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def run_propose(args):
    graph=load_json(args.baseline_graph)
    candidate,catalog=propose_graph(graph,args.source,args.baseline_graph)
    safe_write_json_many([
        (args.candidate_graph,candidate),
        (args.catalog,catalog),
    ])
    print("[OK] Baseline verified relations (must be checked later):",
          catalog["baseline_semantic_count"])
    print("[OK] Exact-source new candidates:",catalog["proposed_count"])
    print("[OK] Candidate concepts:",catalog["new_concept_count"])
    print("[INFO] Candidates are not verified! Bind with V2.11.1 first.")


def run_promote(args, rules=RULES):
    base=load_json(args.baseline_graph)
    candidate=load_json(args.candidate_graph)
    validate_source_graph(base,args.source)
    validate_source_graph(candidate,args.source)
    base_sha=sha_file(args.baseline_graph)
    require(candidate.get("metadata",{}).get("v212",{}).get("phase")=="candidate" and
            candidate["metadata"]["v212"]["baseline_graph_sha256"]==base_sha,
            "候选图谱的 V2.12 基线标识不一致")

    # Reject edited candidates. Re-mine exactly from textbook + built-in rules.
    regenerated,_=propose_graph(base,args.source,args.baseline_graph,rules=rules)
    require(candidate["nodes"]==regenerated["nodes"] and
            candidate["edges"]==regenerated["edges"],
            "候选图谱节点或关系与经核对的教材提取规则不一致")
    binder,baseline_claims,candidate_links=check_real_binding(
        base,args.baseline_graph,args.source,args.chunks,args.baseline_links,
        candidate,args.candidate_graph,args.candidate_links)
    final,decisions=create_promoted_graph(base,candidate,candidate_links,args.baseline_graph)
    final_bytes=serialized(final).encode("utf-8")
    final_links=recompute_for_graph(final,final_bytes,args.source,args.chunks,binder)
    final_status={c["claim_id"]:c for c in final_links["claims"]}
    require(all(c["status"]=="verified" and len(c["chunk_ids"])==1
                for c in final_links["claims"]),
            "最终重算出现未验证的 Claim，拒绝写出")
    require(all(final_status[cid]["chunk_ids"]==c["chunk_ids"]
                for cid,c in baseline_claims.items()),
            "最终重算改变了基线证据")
    verify_new={e["id"] for e in final["edges"] if e.get("v212_origin")=="promoted_unique_real_chunk"}
    require(len(verify_new)==final["metadata"]["v212"]["new_verified_count"],
            "新增关系元数据统计失败")
    alias=derive_alias_registry(final,final_links)
    # Serialize once so embedded file digests are exact.
    final_links_bytes=serialized(final_links).encode("utf-8")
    alias["graph_sha256"]=sha_bytes(final_bytes)
    alias["links_sha256"]=sha_bytes(final_links_bytes)
    gate={
        "schema":"douxian-kg-expansion-gate-2.12-1",
        "created_utc":timestamp(),
        "source_sha256":sha_file(args.source),
        "snapshot_sha256":sha_file(args.chunks),
        "baseline_graph_sha256":sha_file(args.baseline_graph),
        "baseline_links_sha256":sha_file(args.baseline_links),
        "candidate_graph_sha256":sha_file(args.candidate_graph),
        "candidate_links_sha256":sha_file(args.candidate_links),
        "final_graph_sha256":sha_bytes(final_bytes),
        "final_links_sha256":sha_bytes(final_links_bytes),
        "alias_registry_sha256":sha_bytes(serialized(alias).encode("utf-8")),
        "baseline_verified":len(baseline_claims),
        "candidate_new_count":len(decisions),
        "new_verified":len(verify_new),
        "new_not_promoted":sum(not d["promoted"] for d in decisions),
        "new_status_counts":dict(Counter(d["status"] for d in decisions)),
        "final_verified":len(final_links["claims"]),
        "alias_verified_groups":len(alias["verified_synonym_groups"]),
        "alias_ambiguous_terms":len(alias["ambiguous_terms"]),
        "new_claim_decisions":decisions,
        "note":"Promoted only claims with real, unique, source+section+literal Chunk evidence. "
               "Excluded claims remain candidates; original 12 Chunk IDs unchanged.",
    }
    # For precision, verify precomputed final dicts are byte-identical to disk
    # without touching any core inputs or user's existing V2.11.1 outputs.
    safe_write_json_many([
        (args.graph_out,final),
        (args.links_out,final_links),
        (args.aliases_out,alias),
        (args.report_out,gate),
    ])
    try:
        result=final_verification(
            args.baseline_graph,args.source,args.chunks,args.baseline_links,
            args.graph_out,args.links_out,args.report_out,args.aliases_out,
            rules=rules)
    except Exception:
        # These four destinations did not exist before safe_write_json_many.
        for name in (args.graph_out,args.links_out,args.aliases_out,args.report_out):
            Path(name).unlink(missing_ok=True)
        raise
    print("[OK] Preserved baseline verified:",result["baseline_verified"])
    print("[OK] New verified promoted:",result["new_verified"])
    print("[OK] Final verified claims:",result["total_verified"])
    print("[OK] Synonym evidence groups:",len(alias["verified_synonym_groups"]))
    print("[OK] V2.12 graph+link+report validation passed")


def run_verify(args):
    result=final_verification(
        args.baseline_graph,args.source,args.chunks,args.baseline_links,
        args.graph,args.links,args.report,args.aliases)
    print("[OK] V2.12 baseline verified:",result["baseline_verified"])
    print("[OK] V2.12 newly verified:",result["new_verified"])
    print("[OK] V2.12 total verified:",result["total_verified"])
    print("[OK] No baseline Chunk ID changed")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    cmds=parser.add_subparsers(dest="cmd",required=True)
    propose=cmds.add_parser("propose",help="Curated evidence candidates, no Chroma/LLM")
    propose.add_argument("--baseline-graph",default=str(CONTENT_DIR / "knowledge_graph_v27.json"))
    propose.add_argument("--source",default=str(SOURCE_FILE))
    propose.add_argument("--candidate-graph",default=str(CONTENT_DIR / "knowledge_graph_v212_candidates.json"))
    propose.add_argument("--catalog",default=str(CONTENT_DIR / "graph_v212_candidate_catalog.json"))

    promote=cmds.add_parser("promote",help="Verify real Chunk links and promote only supported new edges")
    promote.add_argument("--baseline-graph",default=str(CONTENT_DIR / "knowledge_graph_v27.json"))
    promote.add_argument("--source",default=str(SOURCE_FILE))
    promote.add_argument("--chunks",default=str(BINDINGS_DIR / "chroma_chunks_v211.json"))
    promote.add_argument("--baseline-links",default=str(BINDINGS_DIR / "graph_chunk_links_v2111.json"))
    promote.add_argument("--candidate-graph",default=str(CONTENT_DIR / "knowledge_graph_v212_candidates.json"))
    promote.add_argument("--candidate-links",default=str(BINDINGS_DIR / "graph_chunk_links_v212_candidates.json"))
    promote.add_argument("--graph-out",default=str(CONTENT_DIR / "knowledge_graph_v212.json"))
    promote.add_argument("--links-out",default=str(BINDINGS_DIR / "graph_chunk_links_v212.json"))
    promote.add_argument("--aliases-out",default=str(CONTENT_DIR / "graph_alias_registry_v212.json"))
    promote.add_argument("--report-out",default=str(REPORTS_DIR / "graph_v212_upgrade_report.json"))

    verify=cmds.add_parser("verify",help="Revalidate source/snapshot/graph/base and new Chunk IDs")
    verify.add_argument("--baseline-graph",default=str(CONTENT_DIR / "knowledge_graph_v27.json"))
    verify.add_argument("--source",default=str(SOURCE_FILE))
    verify.add_argument("--chunks",default=str(BINDINGS_DIR / "chroma_chunks_v211.json"))
    verify.add_argument("--baseline-links",default=str(BINDINGS_DIR / "graph_chunk_links_v2111.json"))
    verify.add_argument("--graph",default=str(CONTENT_DIR / "knowledge_graph_v212.json"))
    verify.add_argument("--links",default=str(BINDINGS_DIR / "graph_chunk_links_v212.json"))
    verify.add_argument("--aliases",default=str(CONTENT_DIR / "graph_alias_registry_v212.json"))
    verify.add_argument("--report",default=str(REPORTS_DIR / "graph_v212_upgrade_report.json"))
    args=parser.parse_args(argv)
    try:
        if args.cmd=="propose":
            run_propose(args)
        elif args.cmd=="promote":
            run_promote(args)
        elif args.cmd=="verify":
            run_verify(args)
    except (OSError,ValueError,KeyError,RuntimeError,TypeError) as exc:
        parser.exit(2,"[ERROR] "+str(exc)+"\n")


if __name__=="__main__":
    main()
