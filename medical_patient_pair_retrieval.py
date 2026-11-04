"""Join candidate formulations for colloquial generic-name interaction queries.

Name candidates are query scope only; all reported edges remain literal source
assertions for a concrete product and record, with no invented salt equivalence.
"""
from collections import defaultdict
import json


def patient_pair_paths(conn, nodes, plan, *, max_paths, max_expansions):
    groups = plan["patient_query"]["medicines"]
    maps = [set(g["candidate_drug_ids"]) for g in groups]
    owners = sorted(set.union(*maps))
    if not owners:
        return [], {"expanded_states": 0, "candidate_paths": 0}
    limit = min(20000, max(100, max_expansions * 4))
    rows = [dict(v) for v in conn.execute("""WITH ranked AS (
        SELECT e.*,s.record_id,s.drug_id,substr(s.text,e.quote_start+1,e.quote_end-e.quote_start) AS source_quote,
               ROW_NUMBER() OVER(PARTITION BY e.src,e.relation ORDER BY e.id) AS owner_rank
        FROM edges e JOIN sources s ON s.id=e.source_id
        WHERE e.src IN (SELECT value FROM json_each(?))
          AND e.relation IN ('HAS_INGREDIENT','CLASS_MEMBERSHIP_ASSERTION','INTERACTION_MENTION')
        ) SELECT * FROM ranked ORDER BY owner_rank,src,relation,id LIMIT ?""", (json.dumps(owners), limit+1))]
    truncated = len(rows) > limit or any(g["candidates_truncated"] for g in groups)
    identity_maps = [defaultdict(list), defaultdict(list)]
    compositions = defaultdict(list)
    interactions = [[], []]
    for e in rows[:limit]:
        if e["relation"] == "HAS_INGREDIENT":
            compositions[(e["src"], e["record_id"])].append(e)
        for side in (0, 1):
            if e["src"] not in maps[side]:
                continue
            if e["relation"] == "INTERACTION_MENTION":
                interactions[side].append(e)
            else:
                identity_maps[side][e["dst"]].append(e)
    paths, seen, expanded = [], set(), 0
    for side in (0, 1):
        for interaction in interactions[side]:
            expanded += 1
            if expanded > max_expansions:
                truncated = True
                break
            target = interaction["dst"]
            targets = identity_maps[1-side].get(target, [])
            joins = [(e["src"], e) for e in targets]
            if target in maps[1-side]:
                joins.append((target, None))
            for product, identity in joins:
                if product == interaction["src"]:
                    continue
                if identity and identity["relation"] != ("CLASS_MEMBERSHIP_ASSERTION" if nodes[target]["kind"] == "DrugClass" else "HAS_INGREDIENT"):
                    continue
                path_ids = [interaction["src"], target] + ([product] if identity else [])
                edges = [interaction] + ([identity] if identity else [])
                if side:
                    path_ids.reverse()
                    edges.reverse()
                if len(edges) > plan["max_hops"] or len(set(path_ids)) != len(path_ids):
                    continue
                key = tuple(sorted(e["id"] for e in edges))
                if key in seen:
                    continue
                seen.add(key)
                # Penalize formulation guesses and long source quotes, never
                # rank a combo as a proven match to a single generic ingredient.
                score = 100 - len(edges)*3 - sum(e["quote_end"]-e["quote_start"] for e in edges)/1000
                score += 5 * sum(g["surface"] in interaction["source_quote"] for g in groups)
                if "复方" in nodes[product]["name"] or "复方" in nodes[interaction["src"]]["name"]:
                    score -= 20
                paths.append({"node_ids": path_ids, "edges": edges, "relations": sorted({e["relation"] for e in edges}),
                              "score": score, "covered_anchors": [0, 1]})
        if expanded >= max_expansions:
            break
    if "HAS_INGREDIENT" in plan["requested_relations"]:
        extended = []
        for path in paths:
            interaction = next(e for e in path["edges"] if e["relation"] == "INTERACTION_MENTION")
            owner = interaction["src"]
            for composition in compositions[(owner, interaction["record_id"])]:
                if (composition["dst"] in path["node_ids"] or len(path["edges"]) >= plan["max_hops"]):
                    continue
                if path["node_ids"][0] == owner:
                    ids, edges = [composition["dst"]] + path["node_ids"], [composition] + path["edges"]
                elif path["node_ids"][-1] == owner:
                    ids, edges = path["node_ids"] + [composition["dst"]], path["edges"] + [composition]
                else:
                    continue
                extended.append({**path, "node_ids": ids, "edges": edges,
                                 "relations": sorted({e["relation"] for e in edges}), "score": path["score"] + 10})
        paths = extended + paths
    paths.sort(key=lambda p: (-p["score"], tuple(p["node_ids"])))
    # Different formulations joined to the same literal interaction statement
    # are not independent findings. Keep one complete proof per statement.
    distinct, statements = [], set()
    for path in paths:
        statement = (next(e["id"] for e in path["edges"] if e["relation"] == "INTERACTION_MENTION"),
                     tuple(sorted(e["dst"] for e in path["edges"] if e["relation"] == "HAS_INGREDIENT")))
        if statement not in statements:
            statements.add(statement)
            distinct.append(path)
    plan["patient_pair_search"] = {"candidate_paths": len(paths), "distinct_interaction_statements": len(distinct), "query_scope_only": True,
                                   "actual_patient_formulation_confirmed": all(g["actual_patient_product_known"] for g in groups),
                                   "truncated": truncated}
    return distinct[:max_paths*3], {"expanded_states": min(expanded,max_expansions),
        "candidate_paths": len(paths), "fanout_truncations": int(truncated),
        "expansion_budget_exhausted": expanded > max_expansions}
