"""Bounded category candidate discovery followed by independent source lookups.

Dataset membership selects products; it never substitutes for a clinical class
assertion, a composition edge, or an interaction statement. No model calls.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import quote

from medical_category_graph import ensure_category_graph, verify_category_edge, node_id
from medical_query_plan import FIELD_RELATIONS

EVIDENCE_TYPE = "category_discovery_source_path"
SAME_CATEGORY = r"(?<!不)同(?:一)?(?:大类|子类|小类|类别|类)|这一类|这类|同属"


def category_discovery_plan(question, understanding):
    """Parse set operations independently of particular medicine names."""
    medicines = understanding.get("medicines", [])
    if not medicines or not re.search(SAME_CATEGORY, question):
        return None
    intent = understanding.get("intent", {})
    interaction = bool(intent.get("interaction_question"))
    if len(medicines) > 1 and not interaction and not re.search(r"哪些|有哪些|其他|其它|排除|除了|成分|禁忌|副作用|不良反应|注意|用途|适应症", question):
        return None
    fields = []
    for pattern, field in ((r"成分|组成", "composition"), (r"禁忌|禁用", "contraindications"),
                           (r"不良反应|副作用", "adverse_reactions"), (r"注意|慎用", "warnings"),
                           (r"适应症|用途|治疗", "indication")):
        if re.search(pattern, question):
            fields.append(field)
    if interaction:
        fields.insert(0, "drug_interactions")
    # Category exploration is not a substitute for patient-specific advice.
    clinical = bool(intent.get("risk_question") or intent.get("efficacy_question") or re.search(
        r"我.{0,12}(?:能吃|能用|能服|可以吃|可以用)|能不能吃|能不能用|该吃|该用|推荐|换药|替代|剂量|用量|怎么吃|怎么用", question))
    same = re.search(SAME_CATEGORY, question).group()
    level = "top" if "大类" in same else "leaf"
    other_level = ("top" if re.search(r"(?:其他|其它|不同).{0,2}大类", question) else
                   "leaf" if re.search(r"(?:其他|其它|不同).{0,2}(?:子类|小类)", question) else level)
    excludes = []
    exclusion_spans = []
    for operator in re.finditer(r"排除|不包括|除了|除去|不要", question):
        stop = re.search(r"[，,。；;？?]|与|跟", question[operator.end():])
        exclusion_spans.append((operator.end(), operator.end()+stop.start() if stop else len(question)))
    for medicine in medicines:
        start = medicine.get("start", question.find(medicine["surface"]))
        if any(begin <= start < end for begin,end in exclusion_spans):
            excludes.extend(medicine.get("candidate_drug_ids", []))
    seeds = [m for m in medicines if not set(m.get("candidate_drug_ids", [])) & set(excludes)]
    if not seeds:
        seeds = medicines[:1]
    seed = seeds[0]
    # An explicitly named second medicine is a target constraint, not another seed.
    partners = [i for m in seeds[1:] for i in m.get("candidate_drug_ids", [])] if interaction else []
    return {"version": 1, "route": "category_candidate_source_join", "level": level,
            "level_label": "一级大类" if level == "top" else "二级子类",
            "other_level": other_level,
            "level_basis": "explicit" if re.search(r"大类|子类|小类", same) else "default_leaf_explained",
            "seed_surface": seed["surface"], "seed_ids": seed.get("candidate_drug_ids", []),
            "original_question": question,
            "seed_candidates_truncated": bool(seed.get("candidates_truncated")),
            "exclude_seed": not bool(re.search(r"包括(?:它|本品|原药)|包含(?:它|本品|原药)", question)),
            "excluded_ids": sorted(set(excludes)), "partner_ids": sorted(set(partners)),
            "other_category": bool(re.search(r"其他.{0,4}(?:类|类别)|其它.{0,4}(?:类|类别)|不同.{0,4}(?:类|类别)|跨类", question)),
            "fields": list(dict.fromkeys(fields)), "operation": "interaction" if interaction else "source_fields" if fields else "members",
            "different_ingredient": bool(re.search(r"不同成分|成分不同|成分不一样|成分有区别", question)),
            "unsupported_negative_constraint": bool(re.search(r"不含|不包含.{0,6}成分|成分相同|相同成分|(?<!不)同成分|(?<!有)没有.{0,6}(?:禁忌|相互作用|副作用)", question)),
            "unsupported_clinical_advice": clinical, "model_calls": 0,
            "limits": {"candidate_products": 48, "medical_edges": 2400, "identity_joins": 96,
                       "seconds": 3.0, "evidence_chars": 32000, "max_hops": 8},
            "policy": "dataset_candidate_selection_then_independent_product_source_evidence"}


def _path(nodes, edges, scope):
    return {"id": node_id("discoverypath", scope, *[e["id"] for e in edges]),
            "nodes": nodes, "edges": edges, "hops": len(edges),
            "relations": list(dict.fromkeys(e["relation"] for e in edges)),
            "verified_against_original": True, "clinical_validation": False, "evidence_scope": scope}


def _excerpt(proof, number, role, **metadata):
    lines = ["分类驱动的来源检索：分类仅筛选候选，具体药物关系由独立原文支持。"]
    for e in proof["edges"]:
        lines.extend([f"{e['src']['name']} —{e['relation_label']}→ {e['dst']['name']}",
                      f"来源：{e['source_file']}；记录：{e['record_id']}；章节：{e['section_field']}。",
                      ("JSON路径："+e["json_pointer"]) if e.get("json_pointer") else f"原文位置：{e['quote_start']}:{e['quote_end']}。",
                      "原文："+e["quote"]])
    return {"id": number, "source": "分类分步检索 · "+proof["id"], "section_path": " / ".join(proof["relations"]),
            "content": "\n".join(lines), "evidence_type": EVIDENCE_TYPE,
            "source_url": "", "source_status": "mixed_local_dataset_and_third_party_unverified",
            "graph_verified": True, "graph_path": proof, "discovery_role": role,
            "_retrieval_tool": "search_category_discovery", **metadata}


def _route_family(name):
    """Only reject clear route conflicts for inferred ingredient/class joins."""
    for pattern, family in ((r"滴眼|眼膏", "ocular"), (r"滴耳|耳丸|耳栓", "otic"),
                            (r"鼻喷|滴鼻|鼻气雾", "nasal"), (r"吸入|雾化", "inhaled"),
                            (r"搽剂|乳膏|软膏|凝胶|贴剂", "cutaneous"), (r"栓", "suppository")):
        if re.search(pattern, name):
            return family
    if re.search(r"注射|片|胶囊|颗粒|口服|糖浆", name):
        return "systemic"
    return "unknown"


def _source_route(conn, edge, product, nodes, route_texts=None):
    """Use same-record route wording before broad formulation morphology."""
    morphology = _route_family(nodes[product]["name"])
    if morphology not in ("systemic", "unknown"):
        return morphology
    source = conn.execute("SELECT record_id FROM sources WHERE id=?", (edge["source_id"],)).fetchone()
    if source:
        text = (route_texts.get((product,source[0]), "") if route_texts is not None else
                "\n".join(r[0] for r in conn.execute(
                    "SELECT text FROM sources WHERE drug_id=? AND record_id=? AND field='dosage'",
                    (product, source[0]))))
        for pattern,family in ((r"阴道给药|置入阴道|放入阴道|阴道内|阴道用", "vaginal"),
                               (r"外用|涂于患处|涂患处|搽患处", "cutaneous"),
                               (r"滴入眼|滴眼|涂入眼", "ocular"), (r"滴入耳|滴耳", "otic"),
                               (r"鼻腔内|滴鼻|喷鼻", "nasal"), (r"口服|静脉滴注|肌内注射|静脉注射", "systemic")):
            if re.search(pattern,text):
                return family
    return morphology


def search_category_discovery(conn, root, nodes, plan, *, max_paths=6, max_seconds=3.0):
    """Execute set selection -> source edges -> exact identity join -> category check.

    Results are source examples, not complete drug lists or validated risks.
    Both directions of literal interaction mentions are searched. Ingredient or
    class mentions require a product's independently proved identity edge.
    """
    from medical_semantic_retrieval import _proof

    started = time.monotonic()
    deadline = started + max(0.001, min(3.0, max_seconds))
    root = Path(root)
    report = {"plan": plan, "status": "not_run", "paths": [], "missing_links": [],
              "candidate_drugs_available": 0, "candidate_drugs_searched": 0,
              "coverage_complete": False, "truncated": bool(plan.get("seed_candidates_truncated")), "model_calls": 0,
              "storage": "local_dataset_category_and_independently_verified_medical_graph"}
    excerpts, source_cache, seen, used_products = [], {}, set(), set()
    cat = None

    def missing(kind):
        if kind not in report["missing_links"]:
            report["missing_links"].append(kind)

    def budget():
        if time.monotonic() >= deadline:
            report["truncated"] = True
            missing("time_budget_exhausted")
            return False
        return True

    def add(proof, role, **metadata):
        key = proof["id"]
        if key in seen:
            return False
        item = _excerpt(proof, len(excerpts)+1, role, **metadata)
        if sum(len(e["content"]) for e in excerpts)+len(item["content"]) > plan["limits"]["evidence_chars"]:
            report["truncated"] = True
            missing("evidence_budget_exhausted")
            return False
        excerpts.append(item)
        seen.add(key)
        return True

    try:
        if plan["unsupported_clinical_advice"]:
            missing("patient_specific_advice_requires_specific_products")
            return excerpts, report
        if plan["unsupported_negative_constraint"]:
            missing("absence_or_complete_identity_not_provable_by_mention_edges")
            return excerpts, report
        target, manifest = ensure_category_graph(root, nodes)
        report["category_index"] = manifest
        if target is None:
            missing("category_sources_missing")
            return excerpts, report
        if not budget():
            return excerpts, report
        cat = sqlite3.connect("file:"+quote(target.as_posix(), safe="/:")+"?mode=ro", uri=True)
        cat.row_factory = sqlite3.Row
        cat.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        entities = {r["id"]: dict(r) for r in cat.execute("SELECT * FROM entities")}
        # A product can have multiple memberships; keep each provenance path.
        memberships = {}
        by_category = {}
        for row in cat.execute("""SELECT m.*,p.id AS parent_edge,p.dst AS top_id FROM edges m
                JOIN edges p ON p.src=m.dst AND p.relation='DATASET_CATEGORY_PARENT'
                WHERE m.relation='DATASET_CATEGORY_MEMBERSHIP' ORDER BY m.file,m.json_pointer,m.id"""):
            row = dict(row)
            memberships.setdefault(row["src"], []).append(row)
            key = row["top_id"] if plan["level"] == "top" else row["dst"]
            by_category.setdefault(key, set()).add(row["src"])
        seeds = [m for i in plan["seed_ids"] for m in memberships.get(i, [])]
        if not seeds:
            missing("seed_category_membership_missing")
            return excerpts, report
        keys = {m["top_id"] if plan["level"] == "top" else m["dst"] for m in seeds}
        other_keys = {m["top_id"] if plan["other_level"] == "top" else m["dst"] for m in seeds}
        report["categories"] = [entities[k]["name"] for k in sorted(keys)]
        report["level_label"] = plan["level_label"]
        report["seed_surface"] = plan["seed_surface"]

        def category_proof(member, level=None):
            level = level or plan["level"]
            drug, leaf, top = [entities[i] for i in (member["src"], member["dst"], member["top_id"])]
            parent = dict(cat.execute("SELECT * FROM edges WHERE id=?", (member["parent_edge"],)).fetchone())
            edges = [verify_category_edge(root, member, drug, leaf)]
            ns = [drug, leaf]
            if level == "top":
                edges.append(verify_category_edge(root, parent, leaf, top))
                ns.append(top)
            return _path(ns, edges, "dataset_category_structure_only")

        # Original anchor classification remains available even when a later join fails.
        for member in seeds[:1]:
            add(category_proof(member, "top"), "seed_category")
        if len(seeds) > 1:
            report["truncated"] = True
        example_limit = max(0, max_paths-len(excerpts))
        candidates = set.union(*(by_category[k] for k in keys))
        if plan["exclude_seed"]:
            candidates.difference_update(plan["seed_ids"])
        candidates.difference_update(plan["excluded_ids"])
        report["candidate_drugs_available"] = len(candidates)
        matched = sorted(i for i in candidates if i in nodes and nodes[i]["kind"] == "Drug")
        report["catalog_only_candidates"] = len(candidates)-len(matched)
        # Catalog-only entries can be enumerated but never become clinical products.
        candidates = sorted(candidates) if plan["operation"] == "members" else matched
        report["truncated"] |= len(candidates) > plan["limits"]["candidate_products"]
        candidates = candidates[:plan["limits"]["candidate_products"]]
        report["candidate_drugs_searched"] = len(candidates)
        report["candidate_names"] = [entities[i]["name"] for i in candidates]
        if not candidates:
            missing("same_category_product_candidates_missing")
            return excerpts, report

        def own_members(product):
            return [m for m in memberships.get(product, [])
                    if (m["top_id"] if plan["level"] == "top" else m["dst"]) in keys]

        def other_members(product):
            rows = memberships.get(product, [])
            # An overlapping membership cannot establish an exclusive other category.
            if plan["other_category"] and any((m["top_id"] if plan["other_level"] == "top" else m["dst"]) in other_keys for m in rows):
                return []
            return rows

        if plan["operation"] == "members":
            for product in candidates:
                if not budget():
                    break
                if len(used_products) >= example_limit:
                    report["truncated"] = True
                    break
                if add(category_proof(own_members(product)[0]), "member", product=entities[product]["name"]):
                    used_products.add(product)
            return excerpts, report

        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        cap = plan["limits"]["medical_edges"]
        wanted = [FIELD_RELATIONS[f] for f in plan["fields"] if f in FIELD_RELATIONS]
        # Interactions are looked up in both directions. No all-pairs expansion.
        relations = list(dict.fromkeys(wanted + (["HAS_INGREDIENT", "CLASS_MEMBERSHIP_ASSERTION"] if plan["operation"] == "interaction" else [])))
        rows = [dict(r) for r in conn.execute("""WITH ranked AS (
            SELECT e.*,ROW_NUMBER() OVER(PARTITION BY src,relation ORDER BY id) AS rnk
            FROM edges e WHERE src IN (SELECT value FROM json_each(?))
              AND relation IN (SELECT value FROM json_each(?)))
            SELECT * FROM ranked ORDER BY rnk,src,relation,id LIMIT ?""",
            (json.dumps(candidates), json.dumps(relations), cap+1))]
        report["truncated"] |= len(rows) > cap
        rows = rows[:cap]
        report["medical_edges_scanned"] = len(rows)
        # An explicit disease/symptom condition is a target filter, not an
        # unrelated section lookup. Preserve the matched original entity names.
        from medical_semantic_graph import Lexicon, literal_entity_boundary
        filter_names = {n["name"]: n["id"] for n in nodes.values()
                        if n["kind"] in ("Disease", "Symptom", "Population", "Condition")}
        context_ids = {filter_names[name] for name, start, end in Lexicon(filter_names).find(plan["original_question"])
                       if literal_entity_boundary(plan["original_question"], start, end, query=True)}
        report["source_target_filters"] = [nodes[i]["name"] for i in sorted(context_ids)]

        def medproof(edges):
            ids = [edges[0]["src"], edges[0]["dst"]]
            if len(edges) == 2:
                ids.append(edges[1]["src"])
            return _proof(conn, root, {"node_ids": ids, "edges": edges,
                          "relations": list(dict.fromkeys(e["relation"] for e in edges))}, nodes, source_cache)

        def combined(product, medical, partner=None):
            a = category_proof(own_members(product)[0])
            # Always orient traversal from selected category to candidate product.
            ns, es = list(reversed(a["nodes"])), list(reversed(a["edges"]))
            mednodes, mededges = medical["nodes"], medical["edges"]
            if mednodes[0]["id"] != product:
                mednodes, mededges = list(reversed(mednodes)), list(reversed(mededges))
            ns.extend(mednodes[1:])
            es.extend(mededges)
            if partner:
                b = category_proof(other_members(partner)[0], plan["other_level"])
                ns.extend(b["nodes"][1:])
                es.extend(b["edges"])
            if len(es) > plan["limits"]["max_hops"]:
                return None
            return _path(ns, es, "dataset_selection_and_independent_medical_source_statements")

        if plan["operation"] != "interaction":
            seed_ingredients = set()
            if plan["different_ingredient"]:
                seed_edges = [dict(r) for r in conn.execute("""SELECT * FROM edges WHERE src IN
                    (SELECT value FROM json_each(?)) AND relation='HAS_INGREDIENT' ORDER BY src,id LIMIT 33""",
                    (json.dumps(plan["seed_ids"]),))]
                if not seed_edges or len(seed_edges) > 32:
                    missing("seed_composition_not_available_within_budget")
                    return excerpts, report
                for edge in seed_edges:
                    if not budget():
                        return excerpts, report
                    seed_ingredients.add(edge["dst"])
                    if len(excerpts) >= max_paths-1:
                        report["truncated"] = True
                        missing("seed_composition_not_available_within_budget")
                        return excerpts, report
                    add(medproof([edge]), "seed_composition", product=nodes[edge["src"]]["name"])
                example_limit = max(0, max_paths-len(excerpts))
            # Round robin fields/products before retaining repeated source passages.
            used = set()
            for edge in rows:
                if not budget():
                    break
                key = (edge["src"], edge["relation"])
                if context_ids and edge["dst"] not in context_ids:
                    continue
                if plan["different_ingredient"] and edge["dst"] in seed_ingredients:
                    continue
                if key in used:
                    continue
                if len(used) >= example_limit:
                    report["truncated"] = True
                    break
                proof = combined(edge["src"], medproof([edge]))
                if proof and add(proof, "source_field", product=nodes[edge["src"]]["name"]):
                    used.add(key)
                    used_products.add(edge["src"])
            if not used:
                missing("requested_product_source_edges_missing")
            return excerpts, report

        accepted, statements = set(), set()
        joins = 0
        route_cache = {}
        # One bounded source scan, rather than a full sources scan for every
        # candidate identity (the primary index intentionally stays untouched).
        route_texts = {(r[0],r[1]):r[2] for r in conn.execute(
            "SELECT drug_id,record_id,text FROM sources WHERE field='dosage'")}

        def source_route(edge, product):
            key = (edge["source_id"], product)
            if key not in route_cache:
                route_cache[key] = _source_route(conn, edge, product, nodes, route_texts)
            return route_cache[key]

        def admit(product, partner, edges):
            nonlocal joins
            joins += 1
            if not budget() or joins > cap:
                report["truncated"] = True
                return
            if product == partner or partner in plan["excluded_ids"]:
                return
            if plan["partner_ids"] and partner not in plan["partner_ids"]:
                return
            if len(edges) > 1:
                product_edge = next(e for e in edges if e["src"] == product)
                partner_edge = next(e for e in edges if e["src"] == partner)
                own_route = source_route(product_edge, product)
                partner_route = source_route(partner_edge, partner)
                if own_route != partner_route or "unknown" in (own_route, partner_route):
                    missing("inferred_identity_join_route_conflict")
                    return
            if not other_members(partner):
                missing("interaction_partner_category_not_proven_or_overlaps")
                return
            key = tuple(sorted((product, partner)))
            statement = (edges[0]["id"], edges[0]["dst"])
            if key in accepted or statement in statements:
                return
            if len(accepted) >= example_limit:
                report["truncated"] = True
                return
            medical = medproof(edges)
            proof = combined(product, medical, partner)
            if proof and add(proof, "interaction_example", product=nodes[product]["name"],
                             partner=nodes[partner]["name"], interaction_edge_id=edges[0]["id"],
                             identity_join=len(edges) > 1):
                accepted.add(key)
                statements.add(statement)
                used_products.add(product)

        # Forward: candidate -> mentioned drug or ingredient <- concrete product.
        for interaction in (e for e in rows if e["relation"] == "INTERACTION_MENTION"):
            if not budget() or len(accepted) >= example_limit:
                if len(accepted) >= example_limit:
                    report["truncated"] = True
                break
            target_id = interaction["dst"]
            if nodes[target_id]["kind"] == "Drug":
                admit(interaction["src"], target_id, [interaction])
                continue
            rel = "CLASS_MEMBERSHIP_ASSERTION" if nodes[target_id]["kind"] == "DrugClass" else "HAS_INGREDIENT"
            identities = list(conn.execute("SELECT * FROM edges WHERE dst=? AND relation=? ORDER BY src,id LIMIT ?",
                                           (target_id, rel, plan["limits"]["identity_joins"]+1)))
            report["truncated"] |= len(identities) > plan["limits"]["identity_joins"]
            for identity in identities[:plan["limits"]["identity_joins"]]:
                if not budget() or len(accepted) >= example_limit:
                    break
                admit(interaction["src"], identity["src"], [interaction, dict(identity)])

        # Reverse: other product -> candidate's name or independently proved identity.
        identities = {i: [None] for i in candidates}
        for e in rows:
            if e["relation"] in ("HAS_INGREDIENT", "CLASS_MEMBERSHIP_ASSERTION"):
                identities.setdefault(e["dst"], []).append(e)
        if len(accepted) < example_limit and budget():
            reverse = [dict(r) for r in conn.execute("""SELECT * FROM edges WHERE dst IN
                (SELECT value FROM json_each(?)) AND relation='INTERACTION_MENTION' ORDER BY src,id LIMIT ?""",
                (json.dumps(list(identities)), cap+1))]
            report["truncated"] |= len(reverse) > cap
            for interaction in reverse[:cap]:
                if not budget() or len(accepted) >= example_limit:
                    if len(accepted) >= example_limit:
                        report["truncated"] = True
                    break
                for identity in identities[interaction["dst"]]:
                    if not budget() or len(accepted) >= example_limit:
                        break
                    product = identity["src"] if identity else interaction["dst"]
                    admit(product, interaction["src"], [interaction]+([identity] if identity else []))
        report["identity_joins_checked"] = joins
        report["matched_drug_pairs"] = len(accepted)
        if not accepted:
            missing("cross_product_interaction_source_path_missing")
        return excerpts, report
    except sqlite3.OperationalError as exc:
        if "interrupted" not in str(exc).lower():
            raise
        report["truncated"] = True
        missing("time_budget_exhausted")
        return excerpts, report
    finally:
        conn.set_progress_handler(None, 0)
        if cat is not None:
            cat.close()
        report["returned_examples"] = sum(e["discovery_role"] not in ("seed_category", "seed_composition") for e in excerpts)
        report["status"] = "ok" if report["returned_examples"] else "partial" if excerpts else "no_discovery_evidence"
        report["paths"] = [e["graph_path"] for e in excerpts]
        report["seconds"] = round(time.monotonic()-started, 4)


MISSING_LABELS = {
    "category_sources_missing": "本地分类文件缺失",
    "seed_category_membership_missing": "未找到起始药物的分类归属",
    "same_category_product_candidates_missing": "未找到可继续核对原文的同类药物候选",
    "requested_product_source_edges_missing": "候选药物缺少所请求栏目的实体关系证据",
    "cross_product_interaction_source_path_missing": "未取得连接具体药物的相互作用原文路径",
    "interaction_partner_category_not_proven_or_overlaps": "部分相互作用对象缺少分类证明或同时属于起始类别",
    "time_budget_exhausted": "达到检索时间预算",
    "evidence_budget_exhausted": "达到证据长度预算",
    "patient_specific_advice_requires_specific_products": "本功能只能查找来源示例；个体用药建议需要明确具体药物及用药条件",
    "inferred_identity_join_route_conflict": "已排除给药途径不同或无法确认、仅凭共享成分或类别连接的候选组合",
    "absence_or_complete_identity_not_provable_by_mention_edges": "当前提及关系不能证明不含某成分、成分完全相同或不存在某项风险；需要完整栏目核对",
    "seed_composition_not_available_within_budget": "未在预算内取得起始药物完整的已提取成分证据，未执行成分比较",
}


def category_discovery_answer(excerpts, report):
    """Keep findings in the answer; provenance and identity proofs live in paths."""
    plan = report["plan"]
    examples = [i for i in excerpts if i["discovery_role"] not in ("seed_category", "seed_composition")]
    lines = [f"按“{plan['seed_surface']}”所在的**{plan['level_label']}**，找到 **{len(examples)} 条来源示例**"
             + ("（已排除该药本身）。" if plan["exclude_seed"] else "。")]
    if plan["level_basis"] == "default_leaf_explained":
        lines.append("问题未指定层级，本次按二级子类理解“同类”。")
    seed_compositions = [i for i in excerpts if i["discovery_role"] == "seed_composition"]
    if seed_compositions:
        lines.append("成分比较仅比较本轮原文明确提取的成分，不证明完整配方不同或制剂可替代。")
    for item in examples:
        cite = f"[{item['id']}]"
        if item["discovery_role"] == "member":
            lines.append(f"- **{item['product']}** {cite}")
            continue
        label = f"**{item['product']} × {item['partner']}**" if item.get("partner") else f"**{item['product']}**"
        lines.append(label+" "+cite)
        medical = [e for e in item["graph_path"]["edges"] if not e.get("json_pointer")]
        # Ingredient/class identity is supporting proof, not another answer item.
        if item.get("partner"):
            medical = [e for e in medical if e["relation"] == "INTERACTION_MENTION"]
        for edge in medical:
            lines.append("资料原文："+edge["quote"].strip())
    if not examples:
        lines.append("本轮没有取得符合全部条件的来源示例，不能据此认定不存在相关药物或相互作用。")
    if report["missing_links"] and not examples:
        lines.append("检索缺口："+"；".join(MISSING_LABELS.get(k,k) for k in report["missing_links"])+"。")
    if report["truncated"]:
        lines.append("以上为本轮有限检索结果，不是完整清单。")
    if any(i.get("identity_join") for i in examples):
        lines.append("以上含成分或类别层面的来源线索，尚未核实该具体制剂组合的临床相互作用；不能据此判断合用安全。")
    else:
        lines.append("分类来自尚未核验的本地数据集，所列内容为资料记载，不代表临床结论。")
    if excerpts:
        supporting = "".join(f"[{i['id']}]" for i in excerpts if i["discovery_role"] in ("seed_category", "seed_composition"))
        lines.append("分类、成分与逐边原文请在证据栏的**多关系图路径**中展开查看。"+supporting)
    return "\n\n".join(lines)
