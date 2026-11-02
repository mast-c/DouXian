# -*- coding: utf-8 -*-
"""P3-04.3 read-only exact disease name alignment: medical.json -> Neo4j DiseaseTerm.

No MERGE/CREATE/upsert and no TREAT/INDICATED_FOR relationship inference.
Outputs only a new audit JSON and a mapping JSONL for exact, unique ID matches.
All unmatched or ambiguous names remain unresolved.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

from medical_p3041_medical_json_prepare_v117 import (
    get_oid,iterate_source,sha256_file
)

READ_ONLY_MATCH = """
MATCH (t:DiseaseTerm)
WHERE t.name IN $names
RETURN t.id AS term_id, t.name AS term_name
ORDER BY term_name, term_id
"""
COUNT_TERMS = "MATCH (t:DiseaseTerm) RETURN count(t) AS n"


def read_original_names(path: Path) -> dict[str,list[str]]:
    names={}
    seen_oids=set()
    for line_num,row,_ in iterate_source(path):
        oid=get_oid(row)
        name=row.get("name")
        if not oid or not isinstance(name,str) or not name.strip():
            raise ValueError(f"源文件第 {line_num} 行缺少原始 ID 或名称")
        if oid in seen_oids:
            raise ValueError(f"重复疾病记录 OID：{oid}")
        seen_oids.add(oid)
        names.setdefault(name.strip(),[]).append(oid)
    if not names:
        raise ValueError("没有疾病术语可对齐")
    return names


def align_names(
    names: dict[str,list[str]],
    fetch: Callable[[list[str]],list[dict]],
    *,
    batch_size: int=400,
    graph_total_terms: int | None=None
) -> tuple[dict,list[dict]]:
    if not 1<=batch_size<=1000:
        raise ValueError("batch_size out of range")
    matches={}
    unique_names=sorted(names)
    for offset in range(0,len(unique_names),batch_size):
        batch=unique_names[offset:offset+batch_size]
        records=fetch(batch)
        for record in records:
            name=record.get("term_name")
            term_id=record.get("term_id")
            if name not in batch:
                raise ValueError("Neo4j 返回非本批请求的 DiseaseTerm 名称")
            matches.setdefault(name,[]).append(term_id)
    summary=Counter()
    mapping=[]
    unmapped=[]
    ambiguous=[]
    for name in unique_names:
        ids=matches.get(name,[])
        valid={item for item in ids if isinstance(item,str) and item.strip()}
        if not ids:
            summary["unmatched_names"]+=1
            if len(unmapped)<30:unmapped.append(name)
        elif len(ids)!=1 or len(valid)!=1:
            summary["ambiguous_or_missing_term_id"]+=1
            if len(ambiguous)<30:
                ambiguous.append({"disease_name":name,"term_ids":ids})
        else:
            term_id=next(iter(valid))
            summary["matched_exact_unique_names"]+=1
            for oid in names[name]:
                mapping.append({
                    "disease_name":name,
                    "medical_json_record_oid":oid,
                    "disease_term_id":term_id,
                    "match_type":"exact_name_only",
                    "medical_validation":"not_performed",
                    "is_treatment_relation":False
                })
    summary["source_records"]=sum(len(oids) for oids in names.values())
    summary["unique_source_names"]=len(names)
    summary["matched_source_records"]=len(mapping)
    summary["duplicate_name_source_records"]=sum(
        len(items)-1 for items in names.values()
    )
    summary["neo4j_total_disease_term_nodes"]=graph_total_terms
    report={
        "stage":"P3-04.3",
        "mode":"read_only_exact_name_neo4j_alignment",
        "status":"EXACT_NAME_ALIGNMENT_COMPLETE_WITH_GAPS"
                 if summary["unmatched_names"] or
                    summary["ambiguous_or_missing_term_id"]
                 else "EXACT_NAMES_ALL_MATCHED",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts":dict(summary),
        "unmatched_examples":unmapped,
        "ambiguous_examples":ambiguous,
        "medical_conclusion":"NOT_EVALUATED",
        "new_neo4j_relationships":0,
        "note":"仅疾病名称相同，不代表疾病规范化或同义词审核，不生成任何药物治疗关系"
    }
    return report,mapping


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(Path.cwd()/".env",override=False)


def neo4j_fetchers() -> tuple[Callable[[list[str]],list[dict]], Callable[[],int],Any]:
    from neo4j import GraphDatabase
    uri=os.getenv("NEO4J_URI","").strip()
    username=os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
    password=os.getenv("NEO4J_PASSWORD","")
    db=os.getenv("NEO4J_DATABASE","neo4j")
    if not uri or not password:
        raise ValueError("需要 NEO4J_URI/NEO4J_PASSWORD，且只能使用只读 MATCH")
    driver=GraphDatabase.driver(uri,auth=(username,password))
    driver.verify_connectivity()

    def fetch(batch: list[str]) -> list[dict]:
        with driver.session(database=db) as session:
            return session.execute_read(
                lambda tx:[x.data() for x in tx.run(READ_ONLY_MATCH,names=batch)]
            )

    def count() -> int:
        with driver.session(database=db) as session:
            return int(session.execute_read(
                lambda tx:tx.run(COUNT_TERMS).single().get("n",0)
            ))
    return fetch,count,driver


def run(
    source: Path, *, report: Path, links: Path,
    fetch: Callable[[list[str]],list[dict]],
    count_terms: Callable[[],int], batch_size: int=400
) -> dict:
    source=source.resolve()
    for path in (report,links):
        if path.exists():
            raise FileExistsError(f"输出文件已存在，拒绝覆盖：{path}")
    if report.resolve()==links.resolve():
        raise ValueError("两个输出路径不能相同")
    names=read_original_names(source)
    summary,linked=align_names(
        names,fetch,batch_size=batch_size,
        graph_total_terms=count_terms()
    )
    summary["source_file_sha256"]=sha256_file(source)
    summary["links_count"]=len(linked)
    summary["links_sha256"]=hashlib.sha256(
        "".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n"
                for x in linked).encode("utf-8")
    ).hexdigest()
    for path in (report,links):
        path.parent.mkdir(parents=True,exist_ok=True)
    with links.open("x",encoding="utf-8") as output:
        for item in linked:
            output.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+"\n")
    with report.open("x",encoding="utf-8") as output:
        json.dump(summary,output,ensure_ascii=False,indent=2)
        output.write("\n")
    return summary


def main(argv=None) -> int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=400)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--links",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        load_env()
        fetch,count,driver=neo4j_fetchers()
        try:
            summary=run(args.source,report=args.report,links=args.links,
                        fetch=fetch,count_terms=count,
                        batch_size=args.batch_size)
        finally:
            driver.close()
        print("[P3-04.3]",summary["status"])
        print("统计：",json.dumps(summary["counts"],ensure_ascii=False))
        print("报告：",args.report.resolve())
        print("映射：",args.links.resolve())
        return 0
    except (OSError,ValueError,RuntimeError) as exc:
        print("[P3-04.3] BLOCKED:",exc,file=sys.stderr)
        return 2


if __name__=="__main__":
    sys.exit(main())
