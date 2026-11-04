"""Bounded joins between two disease-mention drug cohorts, not an all-pairs guess.

Each admitted path contains both original disease anchors, exactly two products,
one interaction mention and any required ingredient/class identity evidence.
Every edge for a product must belong to the same original SourceRecord.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import re


def direct_use_context(edge, disease: str) -> bool:
    """Conservative text scope, never a medically validated TREATS assertion.

    A background risk/complication occurrence cannot admit a product into the
    disease's treatment-query cohort. Ambiguous prose is left out and counted.
    """
    text = edge["text"]
    start, end = edge["mention_start"], edge["mention_end"]
    qs, qe = edge["quote_start"], edge["quote_end"]
    quote = text[qs:qe]
    # Explicitly exclude common confounding contexts even inside indication.
    if re.search(r"排除|继发病因|结核菌素|危险因素|风险因素|高危|并发症|神经痛|神经病变|周围神经|肾病|视网膜病变", quote):
        return False
    if re.match(r"性(?:周围|神经|肾|视网膜)|所致|引起的", text[end:]):
        return False
    before = text[qs:start]
    # Preserve subtypes, populations and original restrictions in the quote.
    if re.search(r"(?:用于|治疗|预防|改善|控制|缓解|适用于)[^。；;\n]{0,140}$", before):
        if re.search(r"(?:不|未|非|不能|不宜|不应|禁止)(?:用于|治疗|适用于)", before):
            return False
        return True
    # Standalone enumerated indication item, e.g. "1.Ⅰ型糖尿病；".
    stripped = re.sub(r"^[\s\d①-⑳一二三四五六七八九十.、．()（）]+", "", quote).strip(" 。；;\n")
    return bool(re.fullmatch(r"(?:Ⅰ型|Ⅱ型|I型|II型|1型|2型|１型|２型|妊娠|成年|成人|儿童|原发性|继发性|轻度|中度|重度|季节性|常年性)*" + re.escape(disease), stripped))


def _rows(conn, sql, params):
    return [dict(row) for row in conn.execute(sql, params)]


def disease_pair_paths(conn, nodes, plan, *, max_paths, max_expansions, max_neighbors):
    groups = plan["anchors"]
    disease_ids = [g["node_ids"][0] for g in groups]
    # Unlike ordinary neighborhood traversal, cohort admission cannot be by a
    # contraindication/side effect/condition shared by an unrelated product.
    cap = min(500, max(16, max_neighbors * 4))
    cohorts = []
    summaries = []
    truncated = False
    for disease in disease_ids:
        counts = conn.execute("""SELECT count(DISTINCT e.src),count(DISTINCT s.record_id)
            FROM edges e JOIN sources s ON s.id=e.source_id
            WHERE e.dst=? AND e.relation='INDICATION_MENTION'""", (disease,)).fetchone()
        raw_cap = min(16000, cap * 32)
        raw_rows = _rows(conn, """SELECT e.*,s.record_id,s.drug_id,s.text
            FROM edges e JOIN sources s ON s.id=e.source_id
            WHERE e.dst=? AND e.relation='INDICATION_MENTION'
            ORDER BY e.src,s.record_id,e.quote_end-e.quote_start,e.id LIMIT ?""", (disease, raw_cap + 1))
        eligible = {}
        rejected = set()
        for edge in raw_rows[:raw_cap]:
            key = (edge["src"], edge["record_id"])
            if direct_use_context(edge, nodes[disease]["name"]):
                def priority(e):
                    quote = e["text"][e["quote_start"]:e["quote_end"]]
                    return (bool(re.search(r"妊娠|孕妇|儿童|新生儿", quote)), e["quote_start"], e["quote_end"]-e["quote_start"])
                if key not in eligible or priority(edge) < priority(eligible[key]):
                    eligible[key] = edge
            else:
                rejected.add(key)
        rows = list(eligible.values())
        if len(rows) > cap or len(raw_rows) > raw_cap:
            truncated = True
        rows = rows[:cap]
        cohorts.append(rows)
        summaries.append({
            "disease": nodes[disease]["name"], "candidate_drugs_available": counts[0],
            "candidate_records_available": counts[1], "candidate_records_searched": len(rows),
            "candidate_drugs_searched": len({r["src"] for r in rows}),
            "excluded_ambiguous_or_background_records": len(rejected - set(eligible)),
            "admission": "same_record_direct_use_wording_or_standalone_indication_item_not_approved_treatment",
            "sample_drugs": [nodes[v]["name"] for v in sorted({r["src"] for r in rows})[:12]],
        })
    report = {"route": "two_disease_drug_cohort_join", "cohorts": summaries,
              "coverage_complete": False, "class_membership_policy": "explicit_same_product_source_assertion_only_no_synonyms",
              "missing_links": [], "truncated": truncated}
    if not all(cohorts):
        report["missing_links"] = [{"type": "disease_drug_candidates_missing", "disease": s["disease"]}
                                   for s in summaries if not s["candidate_records_searched"]]
        return [], {"expanded_states": 0, "candidate_paths": 0}, report
    cohort_map = [defaultdict(list), defaultdict(list)]
    for side, rows in enumerate(cohorts):
        for edge in rows:
            cohort_map[side][(edge["src"], edge["record_id"])].append(edge)
    owners = sorted({e["src"] for rows in cohorts for e in rows})
    placeholders = ",".join("?" for _ in owners)
    edge_cap = min(40000, max(256, max_expansions * 4))
    rows = _rows(conn, f"""WITH ranked AS (
        SELECT e.*,s.record_id,s.drug_id,
            ROW_NUMBER() OVER (PARTITION BY e.src,e.relation ORDER BY e.id) AS owner_rank
        FROM edges e JOIN sources s ON s.id=e.source_id
        WHERE e.src IN ({placeholders})
          AND e.relation IN ('HAS_INGREDIENT','CLASS_MEMBERSHIP_ASSERTION','INTERACTION_MENTION')
        ) SELECT * FROM ranked ORDER BY owner_rank,src,relation,id LIMIT ?""", (*owners, edge_cap + 1))
    if len(rows) > edge_cap:
        report["truncated"] = True
    rows = rows[:edge_cap]
    identities = [defaultdict(list), defaultdict(list)]
    interactions = [[], []]
    for edge in rows:
        for side in (0, 1):
            if (edge["src"], edge["record_id"]) not in cohort_map[side]:
                continue
            if edge["relation"] == "INTERACTION_MENTION":
                interactions[side].append(edge)
            else:
                identities[side][edge["dst"]].append(edge)
    stats = Counter(scanned_bridge_edges=len(rows))
    candidates, seen = [], set()
    missing_classes = []

    def admit(owner_side, interaction, partner, partner_record, identity=None):
        owner, owner_record = interaction["src"], interaction["record_id"]
        if owner == partner:
            return
        for own_ind in cohort_map[owner_side][(owner, owner_record)]:
            for partner_ind in cohort_map[1-owner_side][(partner, partner_record)]:
                if owner_side == 0:
                    ids = [disease_ids[0], owner, interaction["dst"]]
                    edges = [own_ind, interaction]
                    if identity:
                        ids.append(partner)
                        edges.append(identity)
                    ids.append(disease_ids[1])
                    edges.append(partner_ind)
                else:
                    ids = [disease_ids[0], partner]
                    edges = [partner_ind]
                    if identity:
                        ids.append(interaction["dst"])
                        edges.append(identity)
                    ids.extend([owner, disease_ids[1]])
                    edges.extend([interaction, own_ind])
                if len(edges) > plan["max_hops"] or len(set(ids)) != len(ids):
                    continue
                key = tuple(sorted(e["id"] for e in edges))
                if key in seen:
                    continue
                seen.add(key)
                stats["candidate_paths"] += 1
                relations = sorted({e["relation"] for e in edges})
                # Exact product/ingredient links outrank source class assertions.
                score = 100 - len(edges) * 3 - (20 if "CLASS_MEMBERSHIP_ASSERTION" in relations else 0)
                candidates.append({"node_ids": ids, "edges": edges, "relations": relations,
                                   "covered_anchors": [0, 1], "score": score,
                                   "drug_pair": tuple(sorted((owner, partner)))})

    for side in (0, 1):
        for interaction in interactions[side]:
            stats["expanded_states"] += 1
            if stats["expanded_states"] > max_expansions:
                report["truncated"] = True
                break
            target = interaction["dst"]
            target_kind = nodes[target]["kind"]
            found = False
            if target_kind == "Drug":
                for (partner, record) in cohort_map[1-side]:
                    if partner == target:
                        found = True
                        admit(side, interaction, partner, record)
            else:
                expected = "CLASS_MEMBERSHIP_ASSERTION" if target_kind == "DrugClass" else "HAS_INGREDIENT"
                for identity in identities[1-side].get(target, []):
                    if identity["relation"] == expected:
                        found = True
                        admit(side, interaction, identity["src"], identity["record_id"], identity)
            if target_kind == "DrugClass" and not found:
                missing_classes.append((side, target, interaction))
        if stats["expanded_states"] >= max_expansions:
            break
    # A useful report of unsupported taxonomy joins, never invented membership.
    seen_missing = set()
    for side, target, interaction in missing_classes:
        key = (side, target)
        if key in seen_missing:
            continue
        seen_missing.add(key)
        if len(report["missing_links"]) < 10:
            report["missing_links"].append({
                "type": "class_membership_not_proven_for_other_cohort", "class_name": nodes[target]["name"],
                "mentioning_drug": nodes[interaction["src"]]["name"],
                "other_disease": nodes[disease_ids[1-side]]["name"],
                "source_id": interaction["source_id"],
            })
    # Prefer distinct drug pairs over many passages of one pair, while preserving
    # same-record proofs. Diversification is retrieval coverage, not recommendation.
    candidates.sort(key=lambda p: (-p["score"], p["drug_pair"], tuple(e["id"] for e in p["edges"])))
    selected, used_pairs = [], set()
    for path in candidates:
        if path["drug_pair"] not in used_pairs:
            selected.append(path)
            used_pairs.add(path["drug_pair"])
        if len(selected) >= max_paths * 3:
            break
    for path in candidates:
        if len(selected) >= max_paths * 3:
            break
        if path not in selected:
            selected.append(path)
    if len(candidates) > len(selected):
        report["truncated"] = True
    report["matched_drug_pairs"] = len({p["drug_pair"] for p in candidates})
    report["searched_snapshot_exhausted"] = not report["truncated"]
    # Exhausting the eligible local snapshot never establishes clinical coverage.
    report["coverage_complete"] = False
    stats["expanded_states"] = min(stats["expanded_states"], max_expansions)
    stats["expansion_budget_exhausted"] = report["truncated"]
    return selected, dict(stats), report
