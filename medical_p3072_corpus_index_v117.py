# -*- coding: utf-8 -*-
"""P3-07.2: complete Neo4j Drug + safety Section inventory to a local SQLite snapshot.

Graph is ALWAYS read-only. Writes only a separate derived SQLite file named
p3072_*.sqlite (never Neo4j/Chroma). The generated snapshot is not medical
ground truth and may be stale after the graph changes.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from typing import Any

VERSION="P3-07.2"
SCHEMA_VERSION=1
SAFETY_FIELDS=("drug_interactions","warnings","contraindications")
MAX_SECTION_CHARS=200_000
MAX_DRUGS=300_000
MAX_SECTIONS=750_000
BATCH_MIN,BATCH_MAX=10,1000
SUFFIXES=tuple(sorted((
    "缓释胶囊","肠溶胶囊","软胶囊","分散片","缓释片","肠溶片","咀嚼片",
    "口崩片","口含片","泡腾片","滴丸","注射液","注射用","鼻喷雾剂",
    "喷雾剂","鼻气雾剂","气雾剂","口服溶液","口服液","口服混悬液",
    "混悬液","颗粒剂","颗粒","胶囊剂","胶囊","片剂","片",
    "滴眼液","滴耳液","眼膏","软膏","乳膏","凝胶","栓剂","栓",
    "贴剂","贴膏","糖浆","干混悬剂","粉针剂","丸剂","丸","散剂","散",
    "洗剂","乳剂","合剂","冲剂","粉剂","溶液","针剂",
),key=len,reverse=True))
SALT_PREFIXES=tuple(sorted((
    "盐酸","氢溴酸","乳酸","醋酸","硫酸","枸橼酸","磷酸","马来酸",
    "酒石酸","富马酸","苯磺酸","甲磺酸","葡萄糖酸","琥珀酸",
),key=len,reverse=True))
SALT_SUFFIXES=("钠","钾")
STOP_ALIAS={"复方","缓释","胶囊","注射","口服","溶液","颗粒","干混悬"}
Q_DRUG_COUNTS="""
MATCH (d:Drug)
RETURN count(d) AS total_nodes,
       sum(CASE WHEN d.id IS NOT NULL AND d.name IS NOT NULL
                    AND trim(d.name) <> '' THEN 1 ELSE 0 END) AS eligible_nodes
""".strip()
Q_DRUGS="""
MATCH (d:Drug)
WHERE d.id IS NOT NULL AND d.name IS NOT NULL
  AND trim(d.name) <> '' AND ($after_id = '' OR d.id > $after_id)
RETURN d.id AS drug_id, d.name AS drug_name
ORDER BY d.id
LIMIT $batch
""".strip()
Q_SECTIONS="""
MATCH (d:Drug)-[:HAS_RECORD]->(r:SourceRecord)-[:HAS_SECTION]->(s:Section)
WHERE d.id IS NOT NULL AND d.name IS NOT NULL
  AND r.id IS NOT NULL AND s.id IS NOT NULL
  AND toLower(coalesce(s.field,'')) IN $fields
  AND s.text IS NOT NULL AND trim(s.text) <> ''
  AND (
    $last_section = '' OR s.id > $last_section
    OR (s.id = $last_section AND (
       d.id > $last_drug OR (d.id = $last_drug AND r.id > $last_record)
    ))
  )
WITH DISTINCT d,r,s
RETURN d.id AS drug_id, d.name AS drug_name,
       r.id AS record_id, r.source_url AS source_url,
       r.authority_status AS authority_status,
       s.id AS section_id, s.field AS section_field, s.text AS section_text
ORDER BY s.id, d.id, r.id
LIMIT $batch
""".strip()


class CorpusBlocked(RuntimeError):
    pass


def normalized(value:Any)->str:
    """Case-fold ASCII, preserve Chinese characters, collapse spaces."""
    if not isinstance(value,str):
        return ""
    text=unicodedata.normalize("NFKC",value).strip().casefold()
    return "".join(text.split())


def alias_lexicon(drug_name:str)->list[tuple[str,str,str]]:
    """Surface, grouping token, basis. Naming morphology IS NOT clinical equivalence."""
    full=normalized(drug_name)
    if not full or len(full)<2:
        return []
    stem=full
    for form in SUFFIXES:
        if stem.endswith(form) and len(stem)>=len(form)+2:
            stem=stem[:-len(form)]
            break
    # Only as a candidate group key; prefix/salt stripping isn't clinical proof.
    core=stem
    for prefix in SALT_PREFIXES:
        if core.startswith(prefix) and len(core)>=len(prefix)+2:
            core=core[len(prefix):]
            break
    for salt in SALT_SUFFIXES:
        if core.endswith(salt) and len(core)>=len(salt)+3:
            core=core[:-len(salt)]
            break
    # Avoid assigning exact entire formulation name to the core if no known
    # dosage form can be removed: group by stable stem and expose morphology.
    group=core if len(core)>=2 else stem
    candidates=[(full,"exact_formulation"),(stem,"dosage_suffix_removed"),
                (group,"heuristic_salt_morphology")]
    unique=[]
    seen=set()
    for token,basis in candidates:
        if len(token)<2 or token in STOP_ALIAS or token in seen:
            continue
        # Two-character aliases are accepted only when both are Han
        # characters. Ambiguity will still be blocked at query time.
        if len(token)==2 and not all("\u4e00"<=c<="\u9fff" for c in token):
            continue
        if token.isdigit() or any(x in token for x in ("\n","\r","\t",";")):
            continue
        seen.add(token)
        unique.append((token,group,basis))
    return unique


def _safe_string(x:Any,limit:int=300)->str:
    return str(x or "").strip()[:limit]


def _schema(conn:sqlite3.Connection)->None:
    conn.executescript("""
PRAGMA foreign_keys=ON;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE drugs (
    drug_id TEXT PRIMARY KEY, drug_name TEXT NOT NULL,
    canonical_key TEXT NOT NULL
);
CREATE TABLE aliases (
    surface TEXT NOT NULL, canonical_key TEXT NOT NULL,
    drug_id TEXT NOT NULL, basis TEXT NOT NULL,
    PRIMARY KEY(surface,canonical_key,drug_id),
    FOREIGN KEY(drug_id) REFERENCES drugs(drug_id)
);
CREATE TABLE sections (
    drug_id TEXT NOT NULL, record_id TEXT NOT NULL,
    section_id TEXT NOT NULL, section_field TEXT NOT NULL,
    source_url TEXT NOT NULL, authority_status TEXT NOT NULL,
    section_text TEXT NOT NULL, section_sha256 TEXT NOT NULL,
    PRIMARY KEY(drug_id,record_id,section_id),
    FOREIGN KEY(drug_id) REFERENCES drugs(drug_id)
);
CREATE INDEX idx_drugs_canonical ON drugs(canonical_key);
CREATE INDEX idx_alias_surface ON aliases(surface);
CREATE INDEX idx_sections_field ON sections(section_field);
""")
    conn.commit()


def _mark(conn,key:str,value:Any)->None:
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
        (key,json.dumps(value,ensure_ascii=False))
    )


def read_metadata(conn:sqlite3.Connection)->dict:
    rows=conn.execute("SELECT key,value FROM metadata").fetchall()
    return {key:json.loads(value) for key,value in rows}


def require_complete(conn:sqlite3.Connection)->dict:
    try:
        meta=read_metadata(conn)
    except (sqlite3.DatabaseError,ValueError) as exc:
        raise CorpusBlocked("索引不是有效的 P3-07.2 SQLite 文件") from exc
    if meta.get("schema_version")!=SCHEMA_VERSION or meta.get("status")!="complete":
        raise CorpusBlocked("索引尚未完整建成，请先执行 --build")
    return meta


def _target_checked(path:Path)->Path:
    p=Path(path).resolve()
    if not (p.name.startswith("p3072_") and p.suffix==".sqlite"):
        raise CorpusBlocked(
            "派生文件名必须为 p3072_*.sqlite，防止覆盖 Neo4j/Chroma/原项目数据库"
        )
    return p


class Neo4jReadPager:
    """All graph Cypher is MATCH/RETURN. Session.run(Query) avoids old tx.run bug."""

    def __init__(self,root:Path):
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(root)/".env",override=False)
        except ImportError:
            pass
        from neo4j import GraphDatabase,READ_ACCESS,Query
        uri=os.getenv("NEO4J_URI","").strip()
        password=os.getenv("NEO4J_PASSWORD","")
        if not uri or not password:
            raise CorpusBlocked("缺少 NEO4J_URI / NEO4J_PASSWORD")
        user=os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
        self.db=os.getenv("NEO4J_DATABASE","neo4j")
        self.driver=GraphDatabase.driver(
            uri,auth=(user,password),connection_timeout=5.0
        )
        self.READ_ACCESS=READ_ACCESS
        self.Query=Query
        self.query_count=0

    def _run(self,cypher:str,**params)->list[dict]:
        with self.driver.session(
            database=self.db,default_access_mode=self.READ_ACCESS
        ) as session:
            rows=[
                item.data() for item in session.run(
                    self.Query(cypher,timeout=25.0),**params
                )
            ]
        self.query_count+=1
        return rows

    def graph_counts(self)->dict:
        rows=self._run(Q_DRUG_COUNTS)
        if len(rows)!=1:
            raise CorpusBlocked("无法读取 Neo4j Drug 库存总数")
        return rows[0]

    def drug_page(self,cursor:str,batch:int)->list[dict]:
        return self._run(Q_DRUGS,after_id=cursor,batch=batch)

    def section_page(self,cursor:tuple[str,str,str],batch:int)->list[dict]:
        return self._run(
            Q_SECTIONS,last_section=cursor[0],
            last_drug=cursor[1],last_record=cursor[2],
            batch=batch,fields=list(SAFETY_FIELDS)
        )

    def close(self):
        self.driver.close()


def _check_cursor(previous,current,kind)->None:
    if current<=previous:
        raise CorpusBlocked(f"{kind} 分页游标没有前进，拒绝生成不完整索引")


def build_snapshot(
    output:Path,*,pager,replace:bool=False,batch:int=256,
    progress=None,
)->dict:
    """Read ENTIRE graph inventory and all 3 safety fields.

    Never silently truncate when maximum limits are reached. Incomplete
    temp snapshots are deleted and never published.
    """
    output=_target_checked(output)
    if not (BATCH_MIN<=batch<=BATCH_MAX):
        raise CorpusBlocked("batch 必须为 10～1000")
    if output.exists() and not replace:
        raise CorpusBlocked(
            f"{output.name} 已存在；默认禁止覆盖，先使用 --replace 并留存备份"
        )
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,temp_path=tempfile.mkstemp(
        prefix="p3072_build_",suffix=".sqlite.tmp",dir=output.parent
    )
    os.close(fd)
    temp=Path(temp_path)
    stats={
        "schema_version":SCHEMA_VERSION,
        "stage":VERSION,
        "status":"building",
        "started_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_source":"Neo4j Drug / SourceRecord / Section",
        "safety_fields":list(SAFETY_FIELDS),
        "drugs_scanned":0,"drugs_indexed":0,"sections_scanned":0,
        "sections_indexed":0,"sections_overlong_skipped":0,
        "invalid_rows_skipped":0,"orphan_sections_skipped":0,
        "duplicate_rows":0,"aliases_indexed":0,"query_count":0,
        "source_verified_by_live_neo4j_scan":True,
        "clinical_review_performed":False,
        "source_rights_verified":False,
        "graph_writes":0,"chroma_writes":0,
    }
    connection=None
    try:
        connection=sqlite3.connect(temp)
        _schema(connection)
        counts=pager.graph_counts() if hasattr(pager,"graph_counts") else None
        if isinstance(counts,dict):
            total=int(counts.get("total_nodes") or 0)
            eligible=int(counts.get("eligible_nodes") or 0)
            if total<eligible or eligible<0:
                raise CorpusBlocked("Neo4j 药品节点计数不一致")
            stats["all_graph_drug_nodes"]=total
            stats["eligible_graph_drug_nodes"]=eligible
            stats["drug_nodes_missing_id_or_name"]=total-eligible
        else:
            stats["all_graph_drug_nodes"]=None
            stats["eligible_graph_drug_nodes"]=None
            stats["drug_nodes_missing_id_or_name"]=None
        cursor=""
        while True:
            rows=pager.drug_page(cursor,batch)
            if not isinstance(rows,list) or len(rows)>batch:
                raise CorpusBlocked("药品扫描超出分页配额")
            if not rows:
                break
            ids=[str(x.get("drug_id") or "") for x in rows]
            if not all(ids) or ids!=sorted(ids) or len(set(ids))!=len(ids):
                raise CorpusBlocked("Drug.id 分页存在空 ID、重复或排序异常")
            _check_cursor(cursor,ids[0],"Drug")
            _check_cursor(cursor,ids[-1],"Drug")
            cursor=ids[-1]
            stats["drugs_scanned"]+=len(rows)
            if stats["drugs_scanned"]>MAX_DRUGS:
                raise CorpusBlocked(
                    "Drug 总量超过本阶段安全上限，禁止生成截断的全库快照"
                )
            for row in rows:
                name=_safe_string(row.get("drug_name"),250)
                drugid=_safe_string(row.get("drug_id"),300)
                if not name or not drugid:
                    stats["invalid_rows_skipped"]+=1
                    continue
                lex=alias_lexicon(name)
                canonical=lex[0][1] if lex else normalized(name)
                connection.execute(
                    "INSERT INTO drugs(drug_id,drug_name,canonical_key) VALUES (?,?,?)",
                    (drugid,name,canonical)
                )
                stats["drugs_indexed"]+=1
                for surface,group,basis in lex:
                    connection.execute("""
INSERT OR IGNORE INTO aliases(surface,canonical_key,drug_id,basis)
VALUES (?,?,?,?)""",(surface,group,drugid,basis))
            connection.commit()
            if progress and stats["drugs_scanned"]%max(batch*20,batch)==0:
                progress(dict(stats))
            if len(rows)<batch:break

        cursor3=("","","")
        while True:
            rows=pager.section_page(cursor3,batch)
            if not isinstance(rows,list) or len(rows)>batch:
                raise CorpusBlocked("药品安全章节扫描超出分页配额")
            if not rows:break
            cursors=[
                (str(x.get("section_id") or ""),
                 str(x.get("drug_id") or ""),
                 str(x.get("record_id") or ""))
                for x in rows
            ]
            if not all(all(t) for t in cursors):
                raise CorpusBlocked("Section 分页中出现空 ID，拒绝丢失游标")
            if cursors!=sorted(cursors) or len(set(cursors))!=len(cursors):
                raise CorpusBlocked("Section 分页出现未排序或重复的复合键")
            _check_cursor(cursor3,cursors[0],"Section")
            _check_cursor(cursor3,cursors[-1],"Section")
            cursor3=cursors[-1]
            stats["sections_scanned"]+=len(rows)
            if stats["sections_scanned"]>MAX_SECTIONS:
                raise CorpusBlocked(
                    "Section 数量超过安全上限；不能谎报全库覆盖"
                )
            for x in rows:
                text=x.get("section_text")
                field=_safe_string(x.get("section_field"),60).lower()
                if not isinstance(text,str) or not text.strip() or field not in SAFETY_FIELDS:
                    stats["invalid_rows_skipped"]+=1
                    continue
                if len(text)>MAX_SECTION_CHARS:
                    stats["sections_overlong_skipped"]+=1
                    continue
                did=_safe_string(x["drug_id"],300)
                record=_safe_string(x["record_id"],300)
                section=_safe_string(x["section_id"],300)
                name=_safe_string(x.get("drug_name"),250)
                exists=connection.execute(
                    "SELECT drug_name FROM drugs WHERE drug_id=?",(did,)
                ).fetchone()
                if not exists or exists[0]!=name:
                    stats["orphan_sections_skipped"]+=1
                    continue
                before=connection.total_changes
                connection.execute("""
INSERT OR IGNORE INTO sections(
drug_id,record_id,section_id,section_field,source_url,
authority_status,section_text,section_sha256
) VALUES (?,?,?,?,?,?,?,?)""",
                    (did,record,section,field,
                     _safe_string(x.get("source_url"),2048),
                     _safe_string(x.get("authority_status"),90),
                     text,hashlib.sha256(text.encode("utf-8")).hexdigest())
                )
                if connection.total_changes>before:
                    stats["sections_indexed"]+=1
                else:
                    stats["duplicate_rows"]+=1
            connection.commit()
            if progress and stats["sections_scanned"]%max(batch*20,batch)==0:
                progress(dict(stats))
            if len(rows)<batch:break
        if not stats["drugs_indexed"]:
            raise CorpusBlocked("图谱里没有找到有效 Drug 节点，禁止发布空全库索引")
        if stats["eligible_graph_drug_nodes"] is not None and (
            stats["drugs_scanned"]!=stats["eligible_graph_drug_nodes"]
        ):
            raise CorpusBlocked(
                "扫描 Drug 数量与扫描前图谱库存计数不一致；"
                "可能是图谱在扫描时变化，拒绝发布不一致的全库快照"
            )
        stats["aliases_indexed"]=connection.execute(
            "SELECT COUNT(*) FROM aliases"
        ).fetchone()[0]
        stats["drugs_with_safety_sections"]=connection.execute(
            "SELECT COUNT(DISTINCT drug_id) FROM sections"
        ).fetchone()[0]
        stats["drugs_without_safety_sections"]=(
            stats["drugs_indexed"]-stats["drugs_with_safety_sections"]
        )
        stats["coverage_pct"]=round(
            100*stats["drugs_with_safety_sections"]/stats["drugs_indexed"],2
        )
        stats["status"]="complete"
        stats["completed_utc"]=datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        stats["query_count"]=getattr(pager,"query_count",None)
        for k,v in stats.items():_mark(connection,k,v)
        connection.commit()
        require_complete(connection)
        connection.close()
        connection=None
        if output.exists():
            backup=output.with_name(
                output.stem+"_backup_"+datetime.now(timezone.utc).strftime(
                    "%Y%m%d_%H%M%S_%f"
                )+".sqlite.bak"
            )
            shutil.copy2(output,backup)
            stats["previous_snapshot_backup"]=str(backup)
        os.replace(temp,output)
        stats["index_path"]=str(output)
        stats["index_bytes"]=output.stat().st_size
        return stats
    finally:
        if connection is not None:connection.close()
        if temp.exists():temp.unlink()


def open_snapshot(path:Path)->tuple[sqlite3.Connection,dict]:
    path=_target_checked(path)
    if not path.is_file():
        raise CorpusBlocked(f"缺少全库派生索引：{path}")
    # SQLite URI read-only: never create a fresh DB accidentally.
    uri=path.as_uri()+"?mode=ro"
    conn=sqlite3.connect(uri,uri=True,timeout=5.0)
    conn.row_factory=sqlite3.Row
    try:
        meta=require_complete(conn)
    except Exception:
        conn.close()
        raise
    return conn,meta
