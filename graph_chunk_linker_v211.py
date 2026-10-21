# -*- coding: utf-8 -*-
"""豆馅 V2.11｜离线图谱 Claim 与真实 Chroma Chunk 绑定。

运行流程：export -> link -> verify，严格 fail-closed；无模型、无 embedding 请求。
原始教材、JSON 图谱、Chroma 记录不被此工具主动修改。
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, timezone

SCHEMA="douxian-claim-link-2.11-1"
SEMANTIC=frozenset({
    "IS_A","SPECIAL_CASE_OF","HAS_PART","USES","BASED_ON",
    "SUBFIELD_OF","COMPUTES","ALSO_CALLED",
})

def sha_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for part in iter(lambda:f.read(1048576),b""):
            h.update(part)
    return h.hexdigest()

def norm(s):
    # Chroma Chunk 和教材行有空白、表格换行差异时只删除空白；不改实体与事实。
    return re.sub(r"\s+","",str(s or "")).replace("\u200b","")

def canonical_source(s):
    return Path(str(s or "").replace("\\","/")).name.casefold()

def canonical_section(s):
    return norm(s).replace("《","").replace("》","").casefold()

def quote_for_matching(edge):
    ev=edge.get("evidence") or {}
    if ev.get("extraction")!="exact_source_phrase":
        return None
    quote=ev.get("text")
    if not isinstance(quote,str) or not quote.strip():
        return None
    # 校验保存的 excerpt 是教材中确切子串；省略号是之前摘录器自动添加的。
    return quote.rstrip("…").strip()

def check_graph_source(graph,source):
    meta=graph.get("metadata",{})
    if not meta.get("source_sha256"):
        raise ValueError("图谱缺少 source_sha256")
    if canonical_source(meta.get("source_file")) != canonical_source(source.name):
        raise ValueError("图谱文件名与指定的原始教材不一致")
    if sha_file(source)!=meta["source_sha256"]:
        raise ValueError("原始教材 SHA256 和图谱不一致，请先重建图谱")
    lines=source.read_text(encoding="utf-8-sig").splitlines()
    for edge in graph.get("edges",[]):
        if edge.get("relation") not in SEMANTIC:
            continue
        q=quote_for_matching(edge)
        ev=edge.get("evidence") or {}
        line_no=ev.get("line")
        if not q or not isinstance(line_no,int) or not 1<=line_no<=len(lines):
            raise ValueError(f"语义关系缺少有效原文证据: {edge.get('id')}")
        if norm(q) not in norm(lines[line_no-1]):
            raise ValueError(f"语义关系与原始教材行不一致: {edge.get('id')}")
    return True

def read_snapshot(path):
    if not path.is_file():raise FileNotFoundError(f"找不到 Chroma 快照：{path}")
    records=[]
    ids=set()
    for no,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip(): continue
        try:obj=json.loads(line)
        except Exception as e:raise ValueError(f"快照 {no} 行不是 JSON") from e
        if not isinstance(obj,dict):raise ValueError(f"快照 {no} 行不是对象")
        chunk_id=obj.get("chunk_id")
        if not isinstance(chunk_id,str) or not chunk_id:
            raise ValueError(f"快照 {no} 行缺少 chunk_id")
        if chunk_id in ids:raise ValueError(f"快照中 chunk_id 重复: {chunk_id}")
        ids.add(chunk_id)
        if not isinstance(obj.get("content"),str):
            raise ValueError(f"快照 {no} 行缺少 content 正文")
        for field in ("source","section_path"):
            if not isinstance(obj.get(field),str):
                raise ValueError(f"快照 {no} 行缺少 {field}，请检查 Chroma Metadata")
        records.append(obj)
    if not records:raise ValueError("快照为空，不能建立绑定")
    return records

def build_link_report(graph,source,snapshot,snapshot_path):
    check_graph_source(graph,source)
    records=read_snapshot(snapshot_path)
    by_source=defaultdict(list)
    for rec in records:by_source[canonical_source(rec["source"])].append(rec)
    concepts={n["id"]:n["label"] for n in graph["nodes"] if n.get("type")=="Concept"}
    claims=[]
    status_counter=Counter()
    for edge in graph["edges"]:
        if edge.get("relation") not in SEMANTIC: continue
        quote=quote_for_matching(edge)
        ev=edge["evidence"]
        source_name=canonical_source(ev["source_file"])
        eligible=by_source.get(source_name,[])
        qnorm=norm(quote)
        section=canonical_section(ev["section_path"])
        # 把父章节完整路径绑定给真正 Chunk。必须同时来源相同 + 原文命中；
        # 只有 section 匹配但无原文，是不够的。
        candidates=[]
        section_mismatches=[]
        for rec in eligible:
            content=norm(rec["content"])
            if not qnorm or qnorm not in content:
                continue
            chunk_section=canonical_section(rec["section_path"])
            if chunk_section==section:
                candidates.append(rec["chunk_id"])
            else:
                section_mismatches.append(rec["chunk_id"])
        candidates=sorted(set(candidates))
        if len(candidates)==1:
            status="verified"
            reason="exact_evidence_and_section"
        elif len(candidates)>1:
            status="ambiguous"
            reason="exact_evidence_in_multiple_chunks"
        elif section_mismatches:
            status="section_conflict"
            reason="evidence_found_but_section_path_differs"
        else:
            status="unbound"
            reason="no_exact_contiguous_quote_in_same_source_and_section"
        # 不管有没有命中，都保留原 edge 的原始证据；不猜测关系。
        row={
            "claim_id":edge["id"],
            "subject_id":edge["source"],
            "subject":concepts.get(edge["source"],""),
            "predicate":edge["relation"],
            "object_id":edge["target"],
            "object":concepts.get(edge["target"],""),
            "source_file":ev["source_file"],
            "section_path":ev["section_path"],
            "source_line":ev["line"],
            "evidence_text":ev["text"],
            "chunk_ids":candidates if status=="verified" else [],
            "candidate_chunk_ids":candidates if status=="ambiguous" else [],
            "status":status,
            "reason":reason,
            "extraction_method":ev["extraction"],
        }
        claims.append(row)
        status_counter[status]+=1
    result={
        "metadata":{
            "schema":SCHEMA,
            "created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "graph_schema":graph["metadata"].get("schema"),
            "graph_sha256":None,
            "source_sha256":sha_file(source),
            "snapshot_sha256":sha_file(snapshot_path),
            "snapshot_rows":len(records),
            "semantic_claims":len(claims),
            "status_counts":dict(status_counter),
            "online_eligible_status":"verified",
            "binding_rule":"same-source+same-section+exact-contiguous-evidence-quote;unique-chunk",
            "note":"仅 verified 可供未来在线图谱关系检索；其它状态禁止自动当作已支持的知识",
        },
        "claims":claims,
    }
    return result

def verify_links(graph,source,snapshot_path,links_path):
    report=json.loads(links_path.read_text(encoding="utf-8"))
    if report.get("metadata",{}).get("schema")!=SCHEMA:
        raise ValueError("绑定文件 schema 不兼容")
    if report["metadata"].get("source_sha256") != sha_file(source):
        raise ValueError("原教材内容变化")
    if report["metadata"].get("snapshot_sha256") != sha_file(snapshot_path):
        raise ValueError("Chroma 快照变化，请重新绑定")
    # 通过重新运行判定逻辑验证，不盲信 JSON 中的 verified 状态。
    current=build_link_report(graph,source,None,snapshot_path)
    originals={x["claim_id"]:{
        key:x[key] for key in ("chunk_ids","status","reason","source_file","section_path","evidence_text")
    } for x in report.get("claims",[])}
    recalculated={x["claim_id"]:{
        key:x[key] for key in ("chunk_ids","status","reason","source_file","section_path","evidence_text")
    } for x in current["claims"]}
    if originals!=recalculated:
        raise ValueError("绑定结果与当前图谱/快照不一致或已被修改")
    return len(recalculated)

def export_chroma(persist_dir,collection,out_path,batch_size=200):
    """只请求元数据和文档，不要求 embedding；此函数不执行新增/删除/更新。

    注意：Chroma 的 PersistentClient 打开数据库时可能执行其内部迁移。
    先备份数据库，关闭 Streamlit 后执行。
    """
    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError("缺少 chromadb；请在原 rag 环境安装/运行") from e
    if not Path(persist_dir).is_dir():
        raise FileNotFoundError(f"Chroma 目录不存在：{persist_dir}")
    client=chromadb.PersistentClient(path=str(Path(persist_dir).resolve()))
    col=client.get_collection(name=collection) # 不自动新建集合
    count=col.count()
    if count<1:raise ValueError("当前 collection 没有记录")
    # 避免导出到图谱/项目源码目录下的旧文件，覆盖需要人工明确允许。
    if out_path.exists():raise FileExistsError(f"输出已存在，请换文件名：{out_path}")
    out_path.parent.mkdir(parents=True,exist_ok=True)
    written=0
    try:
        with out_path.open("w",encoding="utf-8") as f:
            for offset in range(0,count,batch_size):
                page=col.get(include=["documents","metadatas"],limit=batch_size,offset=offset)
                ids=page.get("ids") or []
                texts=page.get("documents") or []
                meta=page.get("metadatas") or []
                if len(ids)!=len(texts) or len(ids)!=len(meta):
                    raise ValueError(f"Chroma 分页 {offset} 字段数量不同，导出终止")
                for cid,body,m in zip(ids,texts,meta):
                    m=m or {}
                    if m.get("chunk_id") not in (None,cid):
                        raise ValueError(f"元数据 chunk_id 与 Chroma 原生 ID 不一致: {cid}")
                    if not isinstance(body,str):
                        raise ValueError(f"文档正文不可用: {cid}")
                    f.write(json.dumps({
                        "chunk_id":cid,
                        "source":str(m.get("source") or ""),
                        "section_path":str(m.get("section_path") or ""),
                        "content":body,
                    },ensure_ascii=False)+"\n")
                    written+=1
        if written != count:raise ValueError(f"Chroma 数量变化：初始 {count}，实际导出 {written}")
    except Exception:
        out_path.unlink(missing_ok=True)
        raise
    return written

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="cmd",required=True)
    exp=sub.add_parser("export",help="从本地 Chroma 导出真实 Chunk JSONL")
    exp.add_argument("--persist-directory",required=True)
    exp.add_argument("--collection",required=True)
    exp.add_argument("--output",default="graph_chroma_chunks_v211.jsonl")
    exp.add_argument("--batch-size",type=int,default=200)
    for command in ("link","verify"):
        sp=sub.add_parser(command)
        sp.add_argument("--graph",default="knowledge_graph_v27.json")
        sp.add_argument("--source",default="graph_sources/神经网络与深度学习.md")
        sp.add_argument("--chunks",default="graph_chroma_chunks_v211.jsonl")
        sp.add_argument("--links",default="graph_claim_links_v211.json")
        if command=="link":
            sp.add_argument("--report",default="graph_integrity_report_v211.json")
    args=ap.parse_args(argv)
    if args.cmd=="export":
        if args.batch_size<=0:ap.error("batch-size 必须为正整数")
        total=export_chroma(args.persist_directory,args.collection,Path(args.output),args.batch_size)
        print(f"✅ 导出 {total} 个真实 Chunk 至 {args.output}（包含原文和元数据，不包含向量）")
        return
    graph_p,source,chunks,links=map(Path,(args.graph,args.source,args.chunks,args.links))
    for p in (graph_p,source,chunks):
        if not p.is_file():ap.error(f"缺少所需文件：{p}")
    graph=json.loads(graph_p.read_text(encoding="utf-8"))
    if args.cmd=="link":
        report=build_link_report(graph,source,None,chunks)
        report["metadata"]["graph_sha256"]=sha_file(graph_p)
        if links.exists():ap.error(f"已有输出文件，不会覆盖：{links}")
        report_path=Path(args.report)
        if report_path.exists():ap.error(f"已有报告文件，不会覆盖：{report_path}")
        links.parent.mkdir(parents=True,exist_ok=True)
        report_path.parent.mkdir(parents=True,exist_ok=True)
        links.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        overview={"schema":SCHEMA, **report["metadata"],
                  "failed_claims":[{
                      "claim_id":x["claim_id"],
                      "subject":x["subject"],"predicate":x["predicate"],
                      "object":x["object"],"status":x["status"],
                      "reason":x["reason"],"section_path":x["section_path"]
                  } for x in report["claims"] if x["status"]!="verified"]}
        report_path.write_text(json.dumps(overview,ensure_ascii=False,indent=2),encoding="utf-8")
        print("✅ 图谱语义关系总数",report["metadata"]["semantic_claims"])
        print("✅ 状态统计",report["metadata"]["status_counts"])
        print("✅ 绑定文件",links)
        print("✅ 完整性报告",report_path)
    else:
        if not links.is_file():ap.error(f"缺少绑定文件：{links}")
        check=json.loads(links.read_text(encoding="utf-8"))
        if check.get("metadata",{}).get("graph_sha256")!=sha_file(graph_p):
            raise ValueError("图谱 JSON 内容已变化，请重新绑定")
        n=verify_links(graph,source,chunks,links)
        print(f"✅ 原文、快照、绑定关系全部校验通过，覆盖 {n} 条语义关系")
if __name__=="__main__":
    main()
