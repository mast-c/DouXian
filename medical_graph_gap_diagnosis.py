"""Read-only, source-backed diagnosis before expanding graph relations.

This audits only named-product interaction/composition questions. It does not
measure clinical correctness, live Chroma recall or general graph reachability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from medical_semantic_graph import (
    SOURCE_DIR, GraphBlocked, clause_span, fingerprint,
    ingredient_spans, json_rows, literal_entity_boundary, open_graph,
)
from medical_semantic_retrieval import search_semantic_graph

FIELDS = ("composition", "indication", "contraindications", "warnings", "drug_interactions", "adverse_reactions", "dosage")
UNINFORMATIVE = re.compile(r"^(?:尚不明确|尚不清楚|不详|暂无|无相关资料|详见说明书)[。；;\s]*$")


def load_source_sections(root, drug_ids, manifest):
    """Scan original exports once per batch, including sections omitted by graph."""
    folder = root / SOURCE_DIR
    records = {}
    for row, *_ in json_rows(folder / "source_records.jsonl"):
        if row.get("drug_id") in drug_ids:
            records[row["id"]] = row
    sections = defaultdict(list)
    for row, at, length, sha in json_rows(folder / "sections.jsonl"):
        record = records.get(row.get("record_id"))
        if record:
            sections[record["drug_id"]].append({**row, "source_url": record.get("source_url", ""),
                "source_file": f"{SOURCE_DIR}/sections.jsonl", "byte_start": at, "byte_length": length,
                "row_sha256": sha})
    # A report must not combine a changed export with the opened graph snapshot.
    for name in ("source_records.jsonl", "sections.jsonl"):
        relative = f"{SOURCE_DIR}/{name}"
        if fingerprint(root / relative) != manifest["inputs"][relative]:
            raise GraphBlocked("Source changed during diagnosis: " + relative)
    return sections


def source_reference(section, start=0, end=None):
    text = section.get("text") or ""
    end = len(text) if end is None else end
    # Preview truncation is explicit; source coordinates refer to the full quote.
    quote = text[start:end]
    return {"section_id": section["id"], "record_id": section["record_id"], "field": section["field"],
            "source_url": section["source_url"], "source_file": section["source_file"],
            "byte_start": section["byte_start"], "byte_length": section["byte_length"],
            "row_sha256": section["row_sha256"], "quote_start": start, "quote_end": end,
            "quote_preview": quote[:700], "quote_truncated": len(quote) > 700,
            "scope": "third_party_source_text_not_clinical_validation"}


def audit_products(medicines, sections, edges):
    products = []
    for medicine in medicines:
        for identity, name in zip(medicine["candidate_drug_ids"], medicine["candidate_drug_names"]):
            own = edges.get(identity, [])
            columns = []
            for field in FIELDS:
                rows = [s for s in sections.get(identity, []) if s.get("field") == field
                        and isinstance(s.get("text"), str) and s["text"].strip() and not s.get("placeholder")]
                count = sum(e["field"] == field for e in own)
                status = "source_missing_in_local_snapshot" if not rows else "source_present"
                if rows and field == "dosage":
                    status = "text_only_by_design"
                elif rows and all(UNINFORMATIVE.fullmatch(s["text"].strip()) for s in rows):
                    status = "source_explicitly_unspecified"
                elif rows and field == "composition" and not any(e["relation"] == "HAS_INGREDIENT" for e in own):
                    status = "composition_extraction_gap_candidate" if any(ingredient_spans(s["text"]) for s in rows) else "composition_requires_parser_review"
                columns.append({"field": field, "source_sections": len(rows), "graph_edges": count,
                                "status": status, "sources": [source_reference(s) for s in rows[:3]],
                                "sources_omitted_from_report": max(0, len(rows)-3)})
            products.append({"drug_id": identity, "drug_name": name,
                             "actual_patient_product_known": medicine["actual_patient_product_known"], "columns": columns})
    return products


def audit_pair(medicines, sections, edges, nodes):
    if len(medicines) != 2:
        return {"status": "not_two_resolved_medicines"}
    sides = [set(m["candidate_drug_ids"]) for m in medicines]
    # Identity membership is per product and record. Never infer classes from
    # a medication's name, another formulation or ordinary class mentions.
    identity_edges = [defaultdict(list), defaultdict(list)]
    interactions = [[], []]
    for side, owners in enumerate(sides):
        for owner in owners:
            for edge in edges.get(owner, []):
                if edge["relation"] in ("HAS_INGREDIENT", "CLASS_MEMBERSHIP_ASSERTION"):
                    identity_edges[side][edge["dst"]].append(edge)
                elif edge["relation"] == "INTERACTION_MENTION":
                    interactions[side].append(edge)
    joins, leads, classes, class_source_leads = [], [], [], []
    for side in (0,1):
        for edge in interactions[side]:
            if edge["dst"] in sides[1-side]:
                joins.append({"edge_ids":[edge["id"]], "kind":"direct_product_mention"})
            for identity in identity_edges[1-side].get(edge["dst"], []):
                joins.append({"edge_ids":[edge["id"],identity["id"]], "kind":"source_identity_join"})
            if nodes[edge["dst"]]["kind"] == "DrugClass" and edge["dst"] not in identity_edges[1-side]:
                class_name = nodes[edge["dst"]]["name"]
                classes.append({"class_name":class_name, "interaction_edge_id":edge["id"],
                    "reason":"类别仅为相互作用对象，不能假定另一药属于此类；该项本身不证明缺边"})
                for other in sides[1-side]:
                    for source in sections.get(other, []):
                        if source.get("field") not in ("indication", "composition", "warnings") or source.get("placeholder"):
                            continue
                        for match in re.finditer(re.escape(class_name), source.get("text") or ""):
                            qs, qe = clause_span(source["text"],match.start(),match.end())
                            class_source_leads.append({"drug_id":other,"class_name":class_name,
                                "source":source_reference(source,qs,qe), "automatic_edge_creation_allowed":False,
                                "reason":"该药原文出现同一类别名；仍须核对是自我分类、比较、否定还是背景提及"})
        other_names = {name for name in medicines[1-side]["candidate_drug_names"]}
        other_names.update(nodes[i]["name"] for i in identity_edges[1-side] if nodes[i]["kind"] == "Ingredient")
        for owner in sorted(sides[side]):
            for section in sections.get(owner, []):
                if section.get("field") != "drug_interactions" or section.get("placeholder"):
                    continue
                text = section.get("text") or ""
                for name in sorted(other_names):
                    for match in re.finditer(re.escape(name), text):
                        # A literal hit is a review lead, not an interaction assertion.
                        edge_hits = [e["id"] for e in edges.get(owner, []) if e["source_id"] == section["id"]
                                     and e["relation"] == "INTERACTION_MENTION" and nodes[e["dst"]]["name"] == name]
                        if edge_hits:
                            continue
                        own_ingredient = any(e["relation"] == "HAS_INGREDIENT" and e["record_id"] == section["record_id"]
                                             and nodes[e["dst"]]["name"] == name for e in edges.get(owner, []))
                        boundary = literal_entity_boundary(text, match.start(), match.end())
                        qs, qe = clause_span(text, match.start(), match.end())
                        leads.append({"drug_id":owner,"target_literal":name,"source":source_reference(section,qs,qe),
                            "reason":"own_ingredient_mention_not_partner" if own_ingredient else "boundary_review_required" if not boundary else "unextracted_literal_mention_candidate",
                            "automatic_edge_creation_allowed":False})
    # A -> X <- B is not evidence for A interacting with B.
    shared = {e["dst"] for e in interactions[0]} & {e["dst"] for e in interactions[1]}
    shared -= set.union(*sides)
    partners = sorted({(nodes[i]["name"],nodes[i]["kind"]) for i in shared})
    return {"status":"diagnosed", "source_identity_joins":joins,
            "raw_mentions_without_edges":leads, "unlinked_class_mentions":classes,
            "class_source_review_leads":class_source_leads,
            "shared_third_partners":[{"name":name,"kind":kind,
                 "usable_for_pair_conclusion":False} for name,kind in partners],
            "boundary":"仅核对两端制剂的本地相互作用原文及成分/明确类别归属，不代表全库或临床结论"}


def classify_case(graph, products, pair):
    status = graph.get("status", "")
    if status in ("index_missing", "source_verification_blocked", "disabled"):
        return "diagnosis_blocked", "先修复索引或来源校验，再判断缺边"
    if status in ("needs_medication_clarification", "unresolved_named_entity", "no_exact_entity", "ambiguous_or_too_many_entities"):
        return "entity_resolution_or_clarification", "先核对药名和患者实际制剂，避免按错误实体补边"
    if graph.get("returned_paths"):
        return "graph_paths_available", "图路径已经返回；若最终仍拒答，检查 evidence_flow 和审核原因"
    if graph.get("stats", {}).get("time_budget_exhausted") or graph.get("truncated"):
        return "retrieval_budget_or_truncation_review", "结果受预算或截断影响，先检查遍历诊断"
    if not graph.get("returned_paths") and pair.get("source_identity_joins"):
        return "existing_join_not_returned_candidate", "存在结构化连接线索，先检查遍历约束与路径筛选"
    if any(x["reason"] == "unextracted_literal_mention_candidate" for x in pair.get("raw_mentions_without_edges", [])):
        return "interaction_extraction_gap_candidate", "原文有未结构化的对象字面提及，核验完整语义后修复抽取"
    if any(c["status"].startswith("composition_extraction_gap") for p in products for c in p["columns"]):
        return "composition_extraction_gap_candidate", "先补成分抽取；须保留同制剂、同来源记录约束"
    if pair.get("class_source_review_leads"):
        return "class_membership_source_review_candidate", "两端有同类别字面线索，核对是否明确自我分类；不可把普通提及变成类别归属"
    return "no_pair_link_in_checked_sources", "当前检查范围未找到两药连接依据；先核对或补充可靠来源，不能虚构相互作用边"


def diagnose_questions(questions, root, *, index=None):
    root = Path(root).resolve()
    runs, owners = [], set()
    for question in questions:
        _, graph = search_semantic_graph(question, project_root=root, path=index)
        medicines = graph.get("query_understanding", {}).get("medicines", [])
        owners.update(i for m in medicines for i in m["candidate_drug_ids"])
        runs.append((question, graph, medicines))
    conn, manifest = open_graph(root, index)
    try:
        sections = load_source_sections(root, owners, manifest)
        nodes = {r["id"]:dict(r) for r in conn.execute("SELECT id,kind,name FROM entities")}
        edges = defaultdict(list)
        for row in conn.execute("""SELECT e.*,s.record_id,s.field FROM edges e JOIN sources s ON s.id=e.source_id
                                   WHERE e.src IN (SELECT value FROM json_each(?))""", (json.dumps(sorted(owners)),)):
            edges[row["src"]].append(dict(row))
        cases = []
        for question, graph, medicines in runs:
            products = audit_products(medicines, sections, edges)
            pair = (audit_pair(medicines, sections, edges, nodes)
                    if graph.get("query_understanding", {}).get("intent", {}).get("interaction_question")
                    and graph.get("retrieval_plan", {}).get("status") == "planned"
                    else {"status":"skipped_unconfirmed_entities_or_not_interaction"})
            category, next_step = classify_case(graph,products,pair)
            cases.append({"question":question,"category":category,"next_step":next_step,
                          "graph_status":graph["status"],"graph_seconds":graph.get("seconds"),
                          "returned_paths":graph.get("returned_paths",0),"truncated":graph.get("truncated",False),
                          "products":products,"pair_diagnosis":pair})
        return {"generated_at":datetime.now(timezone.utc).isoformat(), "graph_snapshot":manifest.get("created_at"),
                "scope":"read_only_local_exports_and_derived_graph_not_live_chroma_or_clinical_audit",
                "edges_created":0,"summary":dict(Counter(c["category"] for c in cases)),"cases":cases}
    finally:
        conn.close()


def render_markdown(report):
    lines = ["# 图谱缺口诊断", "", "只核对本地来源和派生图，不创建图边，不验证临床结论或云端回答质量。", "",
             "| 问题 | 分类 | 路径数 | 下一步 |", "| --- | --- | ---: | --- |"]
    for case in report["cases"]:
        cells = [case["question"],case["category"],str(case["returned_paths"]),case["next_step"]]
        lines.append("| " + " | ".join(str(x).replace("|","／").replace("\n"," ") for x in cells) + " |")
    for case in report["cases"]:
        lines.extend(["", "## " + case["question"], "", case["next_step"]])
        shared = case["pair_diagnosis"].get("shared_third_partners", [])
        if shared:
            lines.extend(["", "共同提及的第三方对象：" + "、".join(sorted({x["name"] for x in shared})) + "。不能作为这两药相互作用的连接依据。"])
        for product in case["products"]:
            gaps = [c["field"] + "："+c["status"] for c in product["columns"] if c["status"] != "source_present"]
            if gaps:
                lines.extend(["", "- " + product["drug_name"] + "：" + "；".join(gaps)])
    lines.extend(["", "JSON 报告保留来源网址、原文片段、行哈希与位置。未命中不代表安全；分类仅为工程排查候选。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--index",type=Path)
    parser.add_argument("--cases",type=Path)
    parser.add_argument("--question",action="append",default=[])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    questions=list(args.question)
    if args.cases:
        questions.extend(json.loads(args.cases.read_text(encoding="utf-8"))["questions"])
    if not questions or len(questions)>30 or not all(isinstance(q,str) and q.strip() for q in questions):
        parser.error("provide 1 to 30 nonempty questions")
    report=diagnose_questions(questions,args.root,index=args.index)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(report),encoding="utf-8")
    print(json.dumps({"report":str(args.output),"summary":report["summary"],"edges_created":0},ensure_ascii=False))


if __name__ == "__main__":
    main()
