# -*- coding: utf-8 -*-
"""GraphRAG v5.4 category citations, provenance, retry and pair-scope tests.

Run from the release directory:
    python -m unittest -v test_v52_regressions test_v53_regressions test_v54_regressions
No model, network, local graph DB or credentials required.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

# Existing v5.2 regression tests install stubs for unavailable graph modules.
import test_v52_regressions  # noqa: F401
from evidence_guard import validate_judge_output
from medical_evidence_contract import (
    retain_provenance, split_retry_targets, verify_retry_integrity,
)
from medical_evidence_coverage import (
    build_answer_targets, evidence_bindings, evidence_key,
    _verified_seed_pair_interaction,
)
from medical_category_discovery import category_discovery_plan

SEED = "复方氨酚烷胺片"
LEAF = "缓解感冒复方OTC"
TOP = "镇痛解热"
CANDIDATE = "示例复方药甲"
OTHER_DRUG = "无关药物乙"
S = "f" * 64


def category_edge(typ, src, dst, quote):
    return {
        "id": "edge_" + typ, "relation": typ, "src": src, "dst": dst,
        "quote": quote, "json_pointer": "/categories/镇痛解热/缓解感冒复方OTC",
        "source_sha256": S, "source_file": "data/raw/drug_db/test.json",
        "section_field": "dataset_category",
        "verification_method": "json_pointer_structure_and_file_sha256",
    }


def drug(name, node_id):
    return {"id": node_id, "name": name, "kind": "Drug"}


def cat(name, node_id):
    return {"id": node_id, "name": name, "kind": "DatasetCategory"}


def seed_excerpt(*, role="seed_category"):
    d, leaf, top = drug(SEED, "seed"), cat(LEAF, "leaf"), cat(TOP, "top")
    edges = [
        category_edge("DATASET_CATEGORY_MEMBERSHIP", d, leaf, json.dumps(SEED, ensure_ascii=False)),
        category_edge("DATASET_CATEGORY_PARENT", leaf, top, json.dumps(LEAF, ensure_ascii=False)),
    ]
    graph = {
        "id": "categorypath_seed", "hops": 2, "nodes": [d, leaf, top],
        "edges": edges, "relations": [e["relation"] for e in edges],
        "verified_against_original": True,
    }
    content = (f"本地数据集分类记载：{SEED} → {LEAF} → {TOP}。\n"
               f"JSON路径：/categories/{TOP}/{LEAF}\n"
               f"原文：{json.dumps(LEAF, ensure_ascii=False)}")
    return {"id": 1, "evidence_type": "category_discovery_source_path",
            "graph_verified": True, "graph_path": graph, "discovery_role": role,
            "source": "test_category.json", "section_path": "本地分类数据集",
            "content": content, "source_status": "local_dataset_unverified"}


def raw_aspect(question, status, evidence):
    return {"question": question, "status": status, "evidence": evidence,
            "reason": "仅测试逐项来源审核，不作临床结论"}


def interaction_excerpt(*, named_pair=False):
    d = drug(CANDIDATE,"candidate")
    seed = drug(SEED,"seed")
    unrelated = drug(OTHER_DRUG,"other")
    leaf = cat(LEAF,"leaf")
    top = cat(TOP,"top")
    member = category_edge("DATASET_CATEGORY_MEMBERSHIP",d,leaf,json.dumps(CANDIDATE,ensure_ascii=False))
    parent = category_edge("DATASET_CATEGORY_PARENT",leaf,top,json.dumps(LEAF,ensure_ascii=False))
    pair = {"id":"pair1", "relation":"INTERACTION_MENTION", "src": d,
            "dst":seed if named_pair else unrelated,
            "quote": "原文：具体药品关系待来源核对。", "section_field": "drug_interactions"}
    nodes = [top,leaf,d,pair["dst"]]
    return {"id":2,"evidence_type":"category_discovery_source_path",
            "discovery_role":"interaction_example", "product":CANDIDATE,
            "partner":pair["dst"]["name"],"graph_verified":True,
            "graph_path":{"id":"path_candidate","verified_against_original":True,
                          "nodes":nodes, "edges":[parent,member,pair],"hops":3,
                          "relations":["DATASET_CATEGORY_PARENT","DATASET_CATEGORY_MEMBERSHIP","INTERACTION_MENTION"]},
            "source":"test.jsonl","content": "来源原文："+pair["quote"]}


def discovery_targets():
    question = f"{SEED}所属的大类下哪些药与{SEED}存在相互作用？"
    understanding={"medicines":[{"surface":SEED,"candidate_drug_names":[SEED]}],
                   "patient_context":[],"intent":{"interaction_question":True}}
    plan={"status":"planned", "category_discovery":{
        "seed_surface":SEED, "level_label":"一级大类",
        "candidate_names":[CANDIDATE],"operation":"interaction",
        "interaction_scope":"seed_pair",}}
    return build_answer_targets(question,understanding,plan)


class CategoryQuoteTests(unittest.TestCase):
    def test_verified_full_top_category_short_quote_is_accepted(self):
        item=seed_excerpt()
        res=validate_judge_output({"aspects":[raw_aspect(
            f"{SEED}的分类归属是什么？","supported",[{"id":1,"quote":TOP}])]},[item])
        self.assertEqual(res["decision"],"answer",res)
        self.assertEqual(res["evidence_ids"],[1])

    def test_verified_category_restored_when_model_failed_to_quote_node(self):
        res=validate_judge_output({"aspects":[
            raw_aspect(f"{SEED}的分类归属是什么？","insufficient",[]),
            raw_aspect(f"{SEED}与其他药物存在相互作用吗？","insufficient",[]),
        ]},[seed_excerpt()])
        self.assertEqual(res["decision"],"partial",res)
        self.assertEqual(res["aspects"][0]["validation_method"],
                         "verified_dataset_json_structure_only")
        self.assertEqual(res["aspects"][1]["status"],"insufficient")

    def test_category_not_restored_from_interaction_example_role(self):
        item=seed_excerpt(role="interaction_example")
        res=validate_judge_output({"aspects":[raw_aspect(
            f"{SEED}的分类归属是什么？","supported",[{"id":1,"quote":TOP}])]},[item])
        self.assertEqual(res["decision"],"refuse")

    def test_category_rejected_without_file_sha256_provenance(self):
        item=seed_excerpt()
        item["graph_path"]["edges"][1].pop("source_sha256")
        res=validate_judge_output({"aspects":[raw_aspect(
            f"{SEED}的分类归属是什么？","supported",[{"id":1,"quote":TOP}])]},[item])
        self.assertEqual(res["decision"],"refuse")

    def test_category_rejected_if_source_not_verified(self):
        item=seed_excerpt()
        item["graph_path"]["verified_against_original"]=False
        res=validate_judge_output({"aspects":[raw_aspect(
            f"{SEED}的分类归属是什么？","supported",[{"id":1,"quote":TOP}])]},[item])
        self.assertEqual(res["decision"],"refuse")

    def test_partial_category_string_not_valid_node_quote(self):
        item=seed_excerpt()
        res=validate_judge_output({"aspects":[raw_aspect(
            f"{SEED}的分类归属是什么？","supported",[{"id":1,"quote":"镇痛"}])]},[item])
        # Original question is a provable literal classification, so the
        # deterministic *full* source fact may still be restored. Never accept
        # the partial quote itself.
        self.assertEqual(res["decision"],"answer")
        self.assertEqual(res["aspects"][0]["quotes"],[{"id":1,"quote":TOP}])
        self.assertEqual(res["aspects"][0]["validation_method"],"verified_dataset_json_structure_only")


class EvidenceTransportTests(unittest.TestCase):
    def test_guard_projection_retains_discovery_role_and_stable_identity(self):
        original=seed_excerpt()
        converted={"id":1,"content":original["content"],"source":original["source"],
                   "evidence_type":original["evidence_type"]}
        output=retain_provenance(converted,original)
        self.assertEqual(output["discovery_role"],"seed_category")
        self.assertEqual(output["graph_path"]["id"],"categorypath_seed")
        self.assertTrue(output["graph_verified"])
        self.assertEqual(output["stable_evidence_key"],evidence_key(original))

    def test_retry_preserves_existing_evidence_then_appends(self):
        prior=[seed_excerpt()]
        after=[deepcopy(prior[0]),{"id":2,"content":"新增药品资料原文", "source":"new",
                                 "evidence_type":"source_section"}]
        old_coverage={"decision":"partial","evidence_ids":[1],"aspects":[{"status":"supported","evidence_ids":[1]}]}
        result,meta=verify_retry_integrity(prior,old_coverage,{
            "excerpts":after,"coverage":old_coverage,"status":"retry_success",
            "new_chunks":1,"retry_queries":[]})
        self.assertEqual(result["status"],"retry_success")
        self.assertEqual(meta["status"],"verified")

    def test_retry_dropped_evidence_falls_back_without_promoting_claim(self):
        prior=[seed_excerpt()]
        old_coverage={"decision":"refuse","evidence_ids":[],"aspects":[]}
        result,meta=verify_retry_integrity(prior,old_coverage,{
            "excerpts":[],"coverage":{"decision":"answer","evidence_ids":[],"aspects":[]},
            "status":"retry_success","new_chunks":0,"retry_queries":[]})
        self.assertEqual(result["status"],"retry_integrity_fallback")
        self.assertEqual(result["coverage"]["decision"],"refuse")
        self.assertEqual(meta["reason"],"prior_evidence_dropped_or_renumbered")

    def test_retry_renumbered_initial_citation_is_rejected(self):
        prior=[seed_excerpt()]
        after=deepcopy(prior);after[0]["id"]=77
        old_coverage={"decision":"refuse","evidence_ids":[],"aspects":[]}
        result,_=verify_retry_integrity(prior,old_coverage,{
            "excerpts":after,"coverage":old_coverage,"status":"ok"})
        self.assertEqual(result["status"],"retry_integrity_fallback")

    def test_split_retry_targets_excludes_json_classification_and_pair_tasks(self):
        supported,skipped=split_retry_targets([
            {"id":"C","field":"dataset_category"},
            {"id":"P","field":"seed_pair_interaction"},
            {"id":"S","field":"warnings"},
        ])
        self.assertEqual([x["id"] for x in supported],["S"])
        self.assertEqual(skipped,["dataset_category","seed_pair_interaction"])


class InteractionEndpointTests(unittest.TestCase):
    def test_separate_candidate_membership_from_seed_interaction(self):
        targets=discovery_targets()
        self.assertEqual([x["id"] for x in targets],
                         ["target_category","target_discovered_sources","target_seed_interaction"])
        general=interaction_excerpt(named_pair=False)
        bindings=evidence_bindings(general,targets)
        self.assertEqual({x["target_id"] for x in bindings}, {"target_discovered_sources"})
        self.assertFalse(_verified_seed_pair_interaction(general,SEED,[CANDIDATE]))
        specific=interaction_excerpt(named_pair=True)
        new_bindings=evidence_bindings(specific,targets)
        self.assertEqual({x["target_id"] for x in new_bindings},
                         {"target_discovered_sources","target_seed_interaction"})

    def test_weak_section_cannot_link_interaction_by_category_only(self):
        targets=discovery_targets()
        item={"evidence_type":"json_neo4j_text_verified",
              "drug_name":CANDIDATE,"section_field":"drug_interactions",
              "parent_verified":True,"content": "药品：示例复方药甲\n原文：本品与无关药物乙的相互作用尚不明确。"}
        self.assertFalse(any(x["target_id"]=="target_seed_interaction"
                             for x in evidence_bindings(item,targets)))
        item["content"]="药品：示例复方药甲\n原文：本品与复方氨酚烷胺片可能存在相互作用，需核对正式说明书。"
        self.assertTrue(any(x["target_id"]=="target_seed_interaction"
                            for x in evidence_bindings(item,targets)))

    def test_model_subquery_constrains_category_pairs_to_seed(self):
        understanding={
            "intent":{"interaction_question":True},
            "medicines":[{"surface":SEED,"candidate_drug_ids":["seed"],
                          "candidate_drug_names":[SEED]}],
            "semantic_sub_queries":[f"{SEED}与哪些同类药物存在相互作用？"],
        }
        question=f"{SEED}所属的大类与该大类下的什么药物存在相互作用？"
        plan=category_discovery_plan(question,understanding)
        self.assertEqual(plan["interaction_scope"],"seed_pair")
        self.assertEqual(plan["partner_ids"],["seed"])
        understanding["semantic_sub_queries"]=["同类药品有哪些常见相互作用示例？"]
        general=category_discovery_plan(question,understanding)
        self.assertEqual(general["interaction_scope"],"general_category_examples")
        self.assertEqual(general["partner_ids"],[])


if __name__ == "__main__":
    unittest.main()
