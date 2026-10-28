"""Stage 3: conservative disease candidates -> human-reviewed claims -> Neo4j.

Add this module to the existing RAG_project/kg/ package.
Uses local kg_output_clean_v2 JSONL. Does not touch your ChromaDB.

Clinical safety: text matches are candidate MENTIONS, never a confirmed
regulatory-approved treatment indication. Only explicitly reviewed rows
can become IndicationClaim nodes, marked clinical_status=not_validated.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re

CSV_FIELDS = [
    "candidate_id", "drug_name", "disease_name", "source_record_id",
    "section_id", "source_url", "evidence_quote", "quote_start", "quote_end",
    "disease_start", "disease_end", "context_flag", "review_status",
    "relation_type", "reviewer", "review_note",
]
# A positive relation type must be chosen by a reviewer, not inferred by
# a substring matcher. Even after review it is a third-party TEXT claim.
RELATION_TYPES = {"treatment", "prevention", "symptom_relief"}
NEGATIVE_CONTEXT = re.compile(
    r"不适用于|不用于|禁用于|不得用于|不能用于|不推荐|"
    r"不宜用于|无治疗作用|无效|禁忌|慎用|避免使用|"
    r"不作为|仅作为诊断|用于鉴别诊断"
)
SENTENCE_ENDINGS = "。；;！？\n"

def hash_id(kind, *parts):
    encoded = json.dumps([kind, *parts], ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return kind.lower() + "_" + hashlib.sha256(encoded).hexdigest()[:24]

def each_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open("r",encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from exc
            if not isinstance(row,dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象")
            yield row

def iter_diseases(path):
    """Supports medical.json JSONL from QASystemOnMedicalKG or JSON array."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open("r",encoding="utf-8-sig") as f:
        prefix=f.read(2048).lstrip()
        f.seek(0)
        if prefix.startswith("["):
            obj=json.load(f)
            if not isinstance(obj,list):
                raise ValueError("疾病词表的 JSON 根目录必须是列表")
            yield from obj
        elif prefix.startswith("{"):
            # JSONL if first nonblank line is a complete JSON object.
            first_line=next((x for x in f if x.strip()), "")
            try:
                json.loads(first_line)
            except json.JSONDecodeError:
                f.seek(0)
                obj=json.load(f)
                if isinstance(obj,dict):
                    for key in ("diseases","records","data","items"):
                        if isinstance(obj.get(key),list):
                            yield from obj[key]
                            return
                    yield obj
                    return
                raise ValueError("未知疾病数据结构")
            f.seek(0)
            for number,line in enumerate(f,1):
                if line.strip():
                    try:
                        item=json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"medical.json 第 {number} 行格式错误") from exc
                    if isinstance(item,dict):
                        yield item
        else:
            raise ValueError("medical.json 必须为 JSONL 或 JSON")

def inspect_vocabulary(path):
    """Inspect schema before treating a local JSON file as a disease lexicon."""
    fields=Counter()
    examples=[]
    entries=0
    disease_schema_records=0
    drug_schema_records=0
    for row in iter_diseases(path):
        if not isinstance(row,dict):
            continue
        entries+=1
        fields.update(row.keys())
        if any(x in row for x in ("symptom","common_drug","recommand_drug","cure_way")):
            disease_schema_records+=1
        if any(x in row for x in ("indication","contraindications","approval_numbers")):
            drug_schema_records+=1
        name=row.get("name",row.get("disease_name"))
        if isinstance(name,str) and len(examples)<8:
            examples.append(name[:60])
    return {
        "record_count":entries,
        "sample_names":examples,
        "top_fields":dict(fields.most_common(14)),
        "records_with_disease_schema":disease_schema_records,
        "records_with_drug_schema":drug_schema_records,
        "warning":"请人工确认 sample_names 为疾病名，而不是药品名。"
    }

def disease_terms(path, min_chars=3, max_chars=50):
    terms=set()
    for row in iter_diseases(path):
        if not isinstance(row,dict):
            continue
        name=row.get("name") or row.get("disease_name")
        if isinstance(name,str):
            name=name.strip()
            if min_chars<=len(name)<=max_chars and not re.search(r"[\t\n\r]",name):
                terms.add(name)
    if not terms:
        raise ValueError("没有提取到疾病名称；请检查 medical.json 是否为疾病数据")
    return terms

class AhoMatcher:
    """Efficient multi-pattern exact substring matcher, pure stdlib."""
    def __init__(self, terms):
        self.trans=[{}]
        self.fail=[0]
        self.out=[[]]
        for term in sorted(set(terms)):
            state=0
            for ch in term:
                if ch not in self.trans[state]:
                    self.trans[state][ch]=len(self.trans)
                    self.trans.append({})
                    self.fail.append(0)
                    self.out.append([])
                state=self.trans[state][ch]
            self.out[state].append(term)

        queue=deque(self.trans[0].values())
        while queue:
            state=queue.popleft()
            for ch,nxt in self.trans[state].items():
                queue.append(nxt)
                parent=self.fail[state]
                while parent and ch not in self.trans[parent]:
                    parent=self.fail[parent]
                self.fail[nxt]=self.trans[parent].get(ch,0)
                self.out[nxt].extend(self.out[self.fail[nxt]])

    def candidates(self, content):
        matches=[]
        state=0
        for i,ch in enumerate(content):
            while state and ch not in self.trans[state]:
                state=self.fail[state]
            state=self.trans[state].get(ch,0)
            for term in self.out[state]:
                matches.append((i+1-len(term),i+1,term))
        # Keep longer overlapping disease names, but treat all as unverified.
        matches.sort(key=lambda m:(m[0],-(m[1]-m[0]),m[2]))
        chosen=[]
        end=-1
        for start,stop,term in matches:
            if start>=end:
                chosen.append((start,stop,term))
                end=stop
        return chosen

def evidence_span(text, start, end, max_size=300):
    left=max((text.rfind(ch,0,start) for ch in SENTENCE_ENDINGS),default=-1)+1
    right_candidates=[text.find(ch,end) for ch in SENTENCE_ENDINGS]
    right_candidates=[x for x in right_candidates if x!=-1]
    right=min(right_candidates)+1 if right_candidates else len(text)
    if right-left>max_size:
        left=max(left,start-100)
        right=min(right,max(left+max_size,end+80))
        if right-left>max_size:
            left=max(left,right-max_size)
    if not (left<=start<end<=right):
        raise ValueError("Evidence quote incorrectly clipped")
    return left,right,text[left:right]

def context_flag(quote, disease):
    if NEGATIVE_CONTEXT.search(quote):
        return "negation_or_restriction_check"
    if len(disease)<=3:
        return "short_term_manual_check"
    return "manual_semantic_review"

def file_sha256(path):
    checksum=hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024*1024), b""):
            checksum.update(block)
    return checksum.hexdigest()

def graph_snapshot_sha256(in_dir):
    report_file=Path(in_dir)/"build_report.json"
    if not report_file.exists():
        return ""
    report=json.loads(report_file.read_text(encoding="utf-8"))
    return report.get("input_sha256","")

def write_jsonl(path, rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")

def extract(medical, in_dir, output_csv, *, limit_drugs=300,
            seed=42, min_chars=3, max_per_drug=60):
    in_dir,output_csv=Path(in_dir),Path(output_csv)
    vocabulary_info=inspect_vocabulary(medical)
    if (vocabulary_info["records_with_drug_schema"] > 0
            and vocabulary_info["records_with_disease_schema"] == 0):
        raise ValueError("medical.json 疑似药品信息而非疾病词表，请先运行 inspect 核对")
    terms=disease_terms(medical,min_chars=min_chars)
    matcher=AhoMatcher(terms)
    drugs={x["id"]:x["name"] for x in each_jsonl(in_dir/"drugs.jsonl")}
    source_records={r["id"]:r for r in each_jsonl(in_dir/"source_records.jsonl")}
    if not drugs or not source_records:
        raise ValueError("缺少有效的图谱 Drug / SourceRecord 文件")
    if limit_drugs<0 or min_chars<2 or max_per_drug<1:
        raise ValueError("limit_drugs >= 0, min_chars >= 2, max_per_drug >= 1")

    eligible=set(drugs)
    if limit_drugs and len(eligible)>limit_drugs:
        eligible=set(random.Random(seed).sample(sorted(eligible),limit_drugs))

    output_csv.parent.mkdir(parents=True,exist_ok=True)
    counts=Counter()
    candidate_ids=set()
    per_drug=Counter()
    drug_hits=set()
    with output_csv.open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=CSV_FIELDS)
        writer.writeheader()
        for section in each_jsonl(in_dir/"sections.jsonl"):
            if section.get("field")!="indication" or section.get("placeholder"):
                continue
            record=source_records.get(section.get("record_id"))
            if not record:
                raise ValueError("Section 对应的 SourceRecord 缺失")
            drug_id=record["drug_id"]
            if drug_id not in drugs:
                raise ValueError("SourceRecord 对应的 Drug 缺失")
            if drug_id not in eligible:
                continue
            content=section.get("text", "")
            if not isinstance(content,str) or not content.strip():
                continue
            counts["scanned_indication_sections"]+=1
            for start,end,term in matcher.candidates(content):
                if per_drug[drug_id]>=max_per_drug:
                    counts["over_limit"]+=1
                    continue
                quote_start,quote_end,quote=evidence_span(content,start,end)
                candidate_id=hash_id("DiseaseCandidate",section["id"],term,start,end)
                if candidate_id in candidate_ids:
                    continue
                candidate_ids.add(candidate_id)
                flag=context_flag(quote,term)
                writer.writerow({
                    "candidate_id":candidate_id,
                    "drug_name":drugs[drug_id],
                    "disease_name":term,
                    "source_record_id":record["id"],
                    "section_id":section["id"],
                    "source_url":record.get("source_url",""),
                    "evidence_quote":quote,
                    "quote_start":quote_start,"quote_end":quote_end,
                    "disease_start":start,"disease_end":end,
                    "context_flag":flag,
                    "review_status":"pending",
                    "relation_type":"",
                    "reviewer":"",
                    "review_note":"",
                })
                counts["candidate_mentions"]+=1
                counts["flag_"+flag]+=1
                per_drug[drug_id]+=1
                drug_hits.add(drug_id)
    report={
        "status":"all_candidates_pending_manual_review",
        "disease_lexicon_size":len(terms),
        "vocabulary_sha256":file_sha256(medical),
        "graph_source_sha256":graph_snapshot_sha256(in_dir),
        "vocabulary_schema_check":vocabulary_info,
        "random_seed":seed,
        "sample_limit":limit_drugs,
        "unique_drugs_with_candidates":len(drug_hits),
        **dict(counts),
        "output_csv":str(output_csv),
        "note":"Clinical validity and relation semantics have not been established."
    }
    report_path=output_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),
                           encoding="utf-8")
    return report

def review(input_csv, in_dir, output_dir):
    """Require explicit positive human review and EXACT source alignment."""
    input_csv,in_dir,output_dir=Path(input_csv),Path(in_dir),Path(output_dir)
    drugs={r["id"]:r for r in each_jsonl(in_dir/"drugs.jsonl")}
    source_records={r["id"]:r for r in each_jsonl(in_dir/"source_records.jsonl")}
    sections={s["id"]:s for s in each_jsonl(in_dir/"sections.jsonl")}
    diseases={}
    claims={}
    reasons=Counter()
    total=0
    output_dir.mkdir(parents=True,exist_ok=True)
    with input_csv.open("r",encoding="utf-8-sig",newline="") as file:
        reader=csv.DictReader(file)
        if not set(CSV_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("CSV 缺失必要字段，请不要删除表头列")
        for row in reader:
            total+=1
            if (row.get("review_status") or "").strip()!="approved":
                reasons["unapproved"]+=1
                continue
            relation=(row.get("relation_type") or "").strip().lower()
            reviewer=(row.get("reviewer") or "").strip()
            term=(row.get("disease_name") or "").strip()
            quote=row.get("evidence_quote","")
            if relation not in RELATION_TYPES or not reviewer or not term:
                reasons["missing_human_relation_type_or_reviewer"]+=1
                continue

            section=sections.get(row.get("section_id"))
            record=source_records.get(row.get("source_record_id"))
            if (not section or not record or
                    section.get("record_id")!=record["id"] or
                    section.get("field")!="indication"):
                reasons["wrong_source_or_section"]+=1
                continue
            drug_id=record["drug_id"]
            if drug_id not in drugs or drugs[drug_id]["name"]!=row.get("drug_name"):
                reasons["wrong_drug"]+=1
                continue
            try:
                qs,qe=int(row["quote_start"]),int(row["quote_end"])
                ds,de=int(row["disease_start"]),int(row["disease_end"])
            except (ValueError,KeyError,TypeError):
                reasons["invalid_offsets"]+=1
                continue
            text=section["text"]
            if (not (0<=qs<=ds<de<=qe<=len(text))
                    or text[qs:qe]!=quote
                    or text[ds:de]!=term):
                reasons["evidence_mismatch"]+=1
                continue
            if context_flag(quote,term)=="negation_or_restriction_check":
                reasons["negative_or_restricted_context"]+=1
                continue
            expected_id=hash_id("DiseaseCandidate",section["id"],term,ds,de)
            if row["candidate_id"]!=expected_id:
                reasons["candidate_id_mismatch"]+=1
                continue
            disease_id=hash_id("DiseaseTerm",term)
            diseases[disease_id]={
                "id":disease_id,
                "name":term,
                "normalization_status":"dataset_exact_match_not_ontology_mapped"
            }
            claim_id=hash_id("IndicationClaim",section["id"],disease_id,ds,de,relation)
            claims[claim_id]={
                "id":claim_id,
                "drug_id":drug_id,
                "disease_id":disease_id,
                "section_id":section["id"],
                "source_record_id":record["id"],
                "predicate":"INDICATION_TEXT_MENTIONS",
                "relation_type":relation,
                "evidence_quote":quote,
                "char_start":qs,"char_end":qe,
                "disease_char_start":ds,"disease_char_end":de,
                "review_status":"approved_text_claim",
                "reviewer":reviewer,
                "clinical_status":"not_validated",
                "regulatory_status":"source_version_unverified",
            }
    write_jsonl(output_dir/"disease_terms.jsonl",diseases.values())
    write_jsonl(output_dir/"indication_claims.jsonl",claims.values())
    report={
        "csv_rows":total,
        "candidate_csv_sha256":file_sha256(input_csv),
        "graph_source_sha256":graph_snapshot_sha256(in_dir),
        "candidate_diseases_in_reviewed_claims":len(diseases),
        "accepted_claims":len(claims),
        "rejected_or_unreviewed":total-len(claims),
        "reject_reasons":dict(reasons),
        "output_dir":str(output_dir),
        "note":"Human review is not verification of clinical efficacy or approval status."
    }
    (output_dir/"disease_review_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report

def connect_driver(uri=None,user=None,password=None):
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise RuntimeError("请先安装 Neo4j 驱动: pip install neo4j") from e
    password=password or os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("必须配置 NEO4J_PASSWORD")
    return GraphDatabase.driver(
        uri or os.environ.get("NEO4J_URI","bolt://localhost:7687"),
        auth=(user or os.environ.get("NEO4J_USER","neo4j"),password)
    )

def in_batches(rows, size=200):
    group=[]
    for row in rows:
        group.append(row)
        if len(group)>=size:
            yield group
            group=[]
    if group:
        yield group

DISEASE_CYPHER = """
UNWIND $rows AS r
MERGE (d:DiseaseTerm {id:r.id})
SET d.name=r.name, d.normalization_status=r.normalization_status
"""
PREFLIGHT_CYPHER = """
UNWIND $rows AS r
MATCH (drug:Drug {id:r.drug_id})-[:HAS_RECORD]->
      (src:SourceRecord {id:r.source_record_id})-[:HAS_SECTION]->
      (sec:Section {id:r.section_id})
WHERE substring(sec.text, r.char_start, r.char_end - r.char_start) = r.evidence_quote
  AND substring(sec.text, r.disease_char_start, r.disease_char_end - r.disease_char_start)
      = r.disease_name
RETURN count(*) AS n
"""

CLAIM_CYPHER = """
UNWIND $rows AS r
MATCH (drug:Drug {id:r.drug_id})
MATCH (sec:Section {id:r.section_id})
MATCH (disease:DiseaseTerm {id:r.disease_id})
MATCH (src:SourceRecord {id:r.source_record_id})-[:HAS_SECTION]->(sec)
MATCH (drug)-[:HAS_RECORD]->(src)
WHERE substring(sec.text, r.char_start, r.char_end - r.char_start) = r.evidence_quote
  AND substring(sec.text, r.disease_char_start, r.disease_char_end - r.disease_char_start)
      = disease.name
MERGE (claim:IndicationClaim {id:r.id})
SET claim.predicate=r.predicate,
    claim.relation_type=r.relation_type,
    claim.evidence_quote=r.evidence_quote,
    claim.char_start=r.char_start,
    claim.char_end=r.char_end,
    claim.disease_char_start=r.disease_char_start,
    claim.disease_char_end=r.disease_char_end,
    claim.review_status=r.review_status,
    claim.reviewer=r.reviewer,
    claim.clinical_status=r.clinical_status,
    claim.regulatory_status=r.regulatory_status
MERGE (claim)-[:SUBJECT]->(drug)
MERGE (claim)-[:OBJECT]->(disease)
MERGE (claim)-[:SUPPORTED_BY]->(sec)
"""

def import_claims(out_dir, *, apply=False, database="neo4j"):
    """Imports approved claims only. Does NOT re-import Section/Chunk corpus."""
    out_dir=Path(out_dir)
    disease_rows=list(each_jsonl(out_dir/"disease_terms.jsonl"))
    claim_rows=list(each_jsonl(out_dir/"indication_claims.jsonl"))
    report={
        "mode":"apply" if apply else "dry-run",
        "disease_terms":len(disease_rows),
        "indication_claims":len(claim_rows),
        "database":database,
    }
    if not apply:
        return report
    if not claim_rows:
        raise RuntimeError("0 条人工审核 Claim；不需要写入数据库")
    review_report=out_dir/"disease_review_report.json"
    if review_report.is_file():
        recorded=json.loads(review_report.read_text(encoding="utf-8"))
        original_graph=recorded.get("graph_source_sha256","")
        current_graph=graph_snapshot_sha256(out_dir)
        if original_graph and current_graph and original_graph!=current_graph:
            raise RuntimeError("审核证据使用的图谱快照已变化，必须重新审核")
    driver=connect_driver()
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            # Fail before writing anything if data no longer matches Neo4j.
            disease_names={d["id"]:d["name"] for d in disease_rows}
            preflight_count=0
            for rows in in_batches(claim_rows):
                checked=[]
                for row in rows:
                    if row["disease_id"] not in disease_names:
                        raise RuntimeError("Claim contains an unknown DiseaseTerm ID")
                    checked.append({**row,"disease_name":disease_names[row["disease_id"]]})
                preflight_count+=session.run(
                    PREFLIGHT_CYPHER,rows=checked
                ).single()["n"]
            if preflight_count!=len(claim_rows):
                raise RuntimeError(
                    f"Preflight failed: {preflight_count}/{len(claim_rows)} "
                    "claims have matching Drug/SourceRecord/Section/evidence. "
                    "No new nodes were written."
                )
            session.run(
                "CREATE CONSTRAINT disease_term_p3_id IF NOT EXISTS "
                "FOR (n:DiseaseTerm) REQUIRE n.id IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT indication_claim_p3_id IF NOT EXISTS "
                "FOR (n:IndicationClaim) REQUIRE n.id IS UNIQUE"
            ).consume()
            for rows in in_batches(disease_rows):
                def tx_diseases(tx, batch):
                    return tx.run(DISEASE_CYPHER,rows=batch).consume()
                session.execute_write(tx_diseases, rows)
            matched=0
            for rows in in_batches(claim_rows):
                def tx_claims(tx, batch):
                    return tx.run(CLAIM_CYPHER,rows=batch).consume()
                session.execute_write(tx_claims,rows)
                # Verify requested claim IDs were created (and not dropped by MATCH).
                result=session.run(
                    "UNWIND $ids AS id MATCH (c:IndicationClaim {id:id}) "
                    "RETURN count(c) AS n",
                    ids=[x["id"] for x in rows]
                ).single()
                matched+=result["n"]
            report["actually_present_claims"]=matched
            if matched != len(claim_rows):
                raise RuntimeError(
                    "部分 Claim 未导入：Drug/Section/SourceRecord 未在 Neo4j 中匹配，"
                    "请检查数据是否来自同一 cleaned_v2 快照"
                )
        return report
    finally:
        driver.close()

def main():
    parser=argparse.ArgumentParser(
        description="药品—疾病候选抽取、审核与 Neo4j 入库 V3"
    )
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("inspect")
    p.add_argument("--medical",required=True)
    p=sub.add_parser("extract")
    p.add_argument("--medical",required=True)
    p.add_argument("--in-dir",default="data/kg_output_clean_v2")
    p.add_argument("--out",default="data/kg_output_clean_v2/disease_candidates.csv")
    p.add_argument("--limit-drugs",type=int,default=300)
    p.add_argument("--min-chars",type=int,default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-per-drug",type=int,default=60)
    p=sub.add_parser("review")
    p.add_argument("--csv",required=True)
    p.add_argument("--in-dir",default="data/kg_output_clean_v2")
    p.add_argument("--out-dir",default="data/kg_output_clean_v2")
    p=sub.add_parser("import")
    p.add_argument("--out-dir",default="data/kg_output_clean_v2")
    p.add_argument("--database",default="neo4j")
    p.add_argument("--apply",action="store_true")
    args=parser.parse_args()
    if args.command=="inspect":
        result=inspect_vocabulary(args.medical)
    elif args.command=="extract":
        result=extract(args.medical,args.in_dir,args.out,
                       limit_drugs=args.limit_drugs,min_chars=args.min_chars,
                       max_per_drug=args.max_per_drug,seed=args.seed)
    elif args.command=="review":
        result=review(args.csv,args.in_dir,args.out_dir)
    else:
        result=import_claims(args.out_dir,apply=args.apply,database=args.database)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
