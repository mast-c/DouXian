"""A source-separated graph for local dataset categories, never pharmacology.

The two JSON files contain mixed therapeutic, department and pharmacological
labels. Their structural membership is searchable but must not be used as a
clinical class-membership bridge in interaction retrieval.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import tempfile
import os
import threading

from medical_semantic_graph import fingerprint, GraphBlocked, digest

FILES = ("drug_categories_western.json", "drug_categories_chinese.json")
RAW = "data/raw/drug_db/MedGraphRAG-Drug-DB"
VERSION = "dataset-category-1"
_LOCK = threading.Lock()
RELATIONS = {"DATASET_CATEGORY_MEMBERSHIP":"本地数据集分类记载",
             "DATASET_CATEGORY_PARENT":"本地数据集上级分类"}


def node_id(kind, *parts):
    return kind.lower()+"_"+digest(json.dumps(parts,ensure_ascii=False))[:24]


def pointer(*parts):
    return "/"+"/".join(str(p).replace("~","~0").replace("/","~1") for p in parts)


@lru_cache(maxsize=4)
def _load(path, size, sha):
    raw=Path(path).read_bytes()
    if digest(raw)!=sha:
        raise GraphBlocked("Category source changed")
    return json.loads(raw.decode("utf-8-sig"))


def source_catalog(root):
    catalog=[]
    for name in FILES:
        path=root/RAW/name
        if not path.exists():
            continue
        fp=fingerprint(path)
        data=_load(str(path),fp["size"],fp["sha256"])
        if not isinstance(data.get("categories"),dict):
            raise GraphBlocked("Invalid category file schema")
        catalog.append((name,fp,data))
    return catalog


def ensure_category_graph(root, nodes):
    root=Path(root).resolve()
    catalog=source_catalog(root)
    if not catalog:
        return None, {"status":"category_sources_missing"}
    products={n["name"]:n["id"] for n in nodes.values() if n["kind"]=="Drug"}
    signature=digest(json.dumps({"version":VERSION,"builder":fingerprint(Path(__file__)),
                                "files":{n:f for n,f,_ in catalog},"products":products},sort_keys=True,ensure_ascii=False))
    target=root/"semantic_graph/dataset_categories.sqlite"
    manifest_path=target.with_suffix(".manifest.json")
    with _LOCK:
        if target.exists() and manifest_path.exists():
            manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature")==signature and manifest.get("index")==fingerprint(target):
                return target,manifest
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,temporary=tempfile.mkstemp(prefix="category_graph_",suffix=".sqlite",dir=target.parent)
        os.close(fd)
        conn=sqlite3.connect(temporary)
        matched,unmatched=0,0
        try:
            conn.executescript("""CREATE TABLE entities(id TEXT PRIMARY KEY,kind TEXT,name TEXT);
            CREATE TABLE edges(id TEXT PRIMARY KEY,src TEXT,dst TEXT,relation TEXT,file TEXT,
                               json_pointer TEXT,source_sha256 TEXT);
            CREATE INDEX category_src ON edges(src,relation);
            CREATE INDEX category_dst ON edges(dst,relation);""")
            def node(identity,kind,name):
                conn.execute("INSERT OR IGNORE INTO entities VALUES(?,?,?)",(identity,kind,name))
            def edge(src,dst,relation,file,ptr,sha):
                identity=node_id("edge",src,dst,relation,file,ptr)
                conn.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?,?,?,?,?)",(identity,src,dst,relation,file,ptr,sha))
            for file,fp,data in catalog:
                for top,children in data["categories"].items():
                    if not isinstance(children,dict):
                        raise GraphBlocked("Invalid category children")
                    top_id=node_id("datasetcategory",file,top)
                    node(top_id,"DatasetCategory",top)
                    for leaf,names in children.items():
                        if not isinstance(names,list) or not all(isinstance(n,str) for n in names):
                            raise GraphBlocked("Invalid category members")
                        leaf_id=node_id("datasetcategory",file,top,leaf)
                        node(leaf_id,"DatasetCategory",leaf)
                        edge(leaf_id,top_id,"DATASET_CATEGORY_PARENT",file,pointer("categories",top,leaf),fp["sha256"])
                        for index,name in enumerate(names):
                            if name not in products:
                                # Keep unlinked category entries as catalog entities,
                                # never fuzzy-match them to another formulation.
                                identity=node_id("catalogdrug",file,name)
                                node(identity,"CatalogDrug",name)
                                unmatched+=1
                            else:
                                identity=products[name]
                                node(identity,"Drug",name)
                                matched+=1
                            edge(identity,leaf_id,"DATASET_CATEGORY_MEMBERSHIP",file,pointer("categories",top,leaf,index),fp["sha256"])
            conn.commit()
            manifest={"version":VERSION,"signature":signature,"files":{n:f for n,f,_ in catalog},
                      "matched_memberships":matched,"unmatched_memberships":unmatched,
                      "nodes":conn.execute("SELECT count(*) FROM entities").fetchone()[0],
                      "edges":conn.execute("SELECT count(*) FROM edges").fetchone()[0],
                      "clinical_validation":False,"pharmacological_class_equivalence":False}
            conn.close()
            for name,fp,_ in catalog:
                if fingerprint(root/RAW/name)!=fp:
                    raise GraphBlocked("Category source changed during build")
            manifest["index"]=fingerprint(Path(temporary))
            os.replace(temporary,target)
            tmp_manifest=manifest_path.with_suffix(".tmp")
            tmp_manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
            os.replace(tmp_manifest,manifest_path)
            return target,manifest
        finally:
            conn.close()
            if Path(temporary).exists():
                Path(temporary).unlink()


def category_only(question, understanding):
    intent=understanding.get("intent",{})
    if not intent.get("classification_question") or any(intent.get(k) for k in (
        "interaction_question","risk_question","adverse_event_question","efficacy_question")):
        return False
    # Clinical and other source-field questions stay with the full retrieval lane.
    import re
    text=question
    for m in understanding.get("medicines",[]):
        text=text.replace(m["surface"],"")
    return not bool(re.search(r"成分|组成|用法|用量|剂量|疗效|作用|机制|治疗|禁忌|副作用|不良反应|注意|安全|能否|能不能|停药|停用|过敏.{0,12}(?:能|可|吃|用)|怀孕|孕妇|小孩|宝宝|肾功能|肝功能|适合|建议",text))


def extend_category_entities(question, root, nodes, understanding):
    """Exact category-only names may exist outside the medical product export.

    Keep them as catalog entries, never patient product identification or
    candidates for therapeutic/interaction retrieval.
    """
    if not category_only(question,understanding):
        return nodes,understanding
    from medical_semantic_graph import Lexicon
    from medical_query_understanding import patient_boundary
    catalog=source_catalog(Path(root))
    names={}
    for file,_,data in catalog:
        for children in data['categories'].values():
            if not isinstance(children,dict):
                raise GraphBlocked('Invalid category children')
            for values in children.values():
                if not isinstance(values,list) or not all(isinstance(n,str) for n in values):
                    raise GraphBlocked('Invalid category members')
                for name in values:
                    names.setdefault(name,set()).add(file)
    existing={m['surface'] for m in understanding['medicines']}
    additions=[]
    extra_nodes={}
    normalized=understanding.get('normalized_question',question)
    for name,start,end in Lexicon(names).find(normalized[:600]):
        if name in existing or not patient_boundary(normalized,start,end,exact_product=True):
            continue
        if any(m['start']<end and start<m['end'] for m in understanding['medicines']):
            continue
        ids=[]
        for file in sorted(names[name]):
            identity=node_id('catalogdrug',file,name)
            extra_nodes[identity]={'id':identity,'kind':'CatalogDrug','name':name}
            ids.append(identity)
        additions.append({'surface':name,'start':start,'end':end,'candidate_drug_ids':ids,
            'candidate_drug_names':[name]*len(ids),'ingredient_ids':[],
            'candidate_count':len(ids),'candidates_truncated':False,'actual_patient_product_known':False,
            'clinical_equivalence_asserted':False,'catalog_entry_only':True,'basis':'exact_category_catalog_entry',
            'role':'mentioned','assertion':'unspecified','route_hint':'','mentions':[]})
    if not additions:
        return nodes,understanding
    return {**nodes,**extra_nodes},{**understanding,'medicines':sorted(understanding['medicines']+additions,key=lambda m:m['start'])}


def _parts(ptr):
    return [p.replace("~1","/").replace("~0","~") for p in ptr.lstrip("/").split("/")]


def verify_category_edge(root, row, src, dst):
    fp=fingerprint(root/RAW/row["file"])
    if fp["sha256"]!=row["source_sha256"]:
        raise GraphBlocked("Category edge source changed")
    data=_load(str(root/RAW/row["file"]),fp["size"],fp["sha256"])
    parts=_parts(row["json_pointer"])
    if row["relation"]=="DATASET_CATEGORY_MEMBERSHIP":
        if len(parts)!=4 or parts[0]!="categories":
            raise GraphBlocked("Invalid category membership pointer")
        _,top,leaf,index=parts
        if (data["categories"][top][leaf][int(index)]!=src["name"] or dst["name"]!=leaf
            or dst["id"]!=node_id("datasetcategory",row["file"],top,leaf)):
            raise GraphBlocked("Category membership mismatch")
        # This is a structural JSON source check, not a fabricated medical quote.
        quote=json.dumps(src["name"],ensure_ascii=False)
    else:
        if len(parts)!=3 or parts[0]!="categories":
            raise GraphBlocked("Invalid category parent pointer")
        _,top,leaf=parts
        if (leaf not in data["categories"][top] or src["name"]!=leaf or dst["name"]!=top
            or src["id"]!=node_id("datasetcategory",row["file"],top,leaf)
            or dst["id"]!=node_id("datasetcategory",row["file"],top)):
            raise GraphBlocked("Category parent mismatch")
        quote=json.dumps(leaf,ensure_ascii=False)
    return {**dict(row),"src":src,"dst":dst,"relation_label":RELATIONS[row["relation"]],
            "source_id":row["file"]+":"+row["json_pointer"],"source_file":RAW+"/"+row["file"],
            "source_url":"","record_id":row["json_pointer"],"section_field":"dataset_category",
            "quote":quote,"quote_start":None,"quote_end":None,
            "source_status":"local_dataset_unverified","status":"dataset_classification_not_pharmacological_assertion",
            "verification_method":"json_pointer_structure_and_file_sha256"}


def search_category_graph(question,root,nodes,understanding,max_paths=8):
    root=Path(root).resolve()
    target,manifest=ensure_category_graph(root,nodes)
    report={"storage":"independent_local_dataset_category_graph","clinical_validation":False,
            "category_index":manifest,"paths":[],"returned_paths":0,"generation_evidence":True}
    if target is None:
        return [],{**report,"status":"category_sources_missing"}
    from urllib.parse import quote
    conn=sqlite3.connect("file:"+quote(target.as_posix(),safe="/:")+"?mode=ro",uri=True)
    conn.row_factory=sqlite3.Row
    try:
        entities={r["id"]:dict(r) for r in conn.execute("SELECT * FROM entities")}
        owners=list(dict.fromkeys(i for m in understanding["medicines"] for i in m["candidate_drug_ids"]))
        rows=list(conn.execute("""SELECT e.id FROM edges e JOIN entities d ON d.id=e.dst
            WHERE e.src IN (SELECT value FROM json_each(?)) AND e.relation='DATASET_CATEGORY_MEMBERSHIP'
            ORDER BY e.file,d.name,e.id LIMIT ?""",(json.dumps(owners),max_paths+1)))
        paths,excerpts=[],[]
        for hit in rows[:max_paths]:
            member=dict(conn.execute("SELECT * FROM edges WHERE id=?",(hit["id"],)).fetchone())
            parent=dict(conn.execute("SELECT * FROM edges WHERE src=? AND relation='DATASET_CATEGORY_PARENT'",(member["dst"],)).fetchone())
            drug,leaf,top=(entities[i] for i in (member["src"],member["dst"],parent["dst"]))
            if drug["id"] not in nodes or nodes[drug["id"]]["name"]!=drug["name"]:
                raise GraphBlocked("Category product is not the recognized source formulation")
            edges=[verify_category_edge(root,member,drug,leaf),verify_category_edge(root,parent,leaf,top)]
            proof={"id":node_id("categorypath",*[e["id"] for e in edges]),"nodes":[drug,leaf,top],"edges":edges,
                   "hops":2,"relations":list(RELATIONS),"verified_against_original":True,
                   "clinical_validation":False,"evidence_scope":"dataset_category_structure_only"}
            content=("本地数据集分类记载："+drug["name"]+" → "+leaf["name"]+" → "+top["name"]
                     +"。\n来源："+edges[0]["source_file"]+"\nJSON路径："+member["json_pointer"]
                     +"\n逐项核对：JSON数组元素为 "+edges[0]["quote"]+"；分类键为 "+edges[1]["quote"]
                     +"。\n已核对JSON结构与文件哈希；这是数据集分类，未验证为权威药理类别，不能用于推断适应症或合用安全。")
            paths.append(proof)
            excerpts.append({"id":len(excerpts)+1,"source":"本地分类文件 · "+member["file"],"section_path":top["name"]+" / "+leaf["name"],
                "content":content,"drug_name":drug["name"],"evidence_type":"dataset_category_graph_path",
                "source_url":"","source_status":"local_dataset_unverified","graph_verified":True,"graph_path":proof,
                "_retrieval_tool":"search_category_graph"})
        report.update(status="ok" if paths else "category_not_in_dataset",paths=paths,returned_paths=len(paths),
                      truncated=len(rows)>max_paths,multi_relation_paths=len(paths))
        return excerpts,report
    finally:
        conn.close()


def category_source_answer(excerpts):
    lines=[]
    for item in excerpts:
        path=item["graph_path"]
        drug,leaf,top=path["nodes"]
        source_kind="西药" if path['edges'][0]['file']=='drug_categories_western.json' else "中药"
        lines.append(f"本地{source_kind}分类文件将 **{drug['name']}** 列在 **{top['name']} → {leaf['name']}** 下[{item['id']}]。")
    lines.append("这是本地数据集的分类记载，尚未核验为权威药理分类。")
    lines.append("分类归属及来源位置可在证据栏的**多关系图路径**中展开查看。")
    return "\n\n".join(lines)
