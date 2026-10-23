# -*- coding: utf-8 -*-
"""V2.16 tests run ONLY offline: no Chroma, APIs, Reranker or answer model."""
from __future__ import annotations
import unittest
from .graph_assist_planner_v216 import (
    load_graph,plan,match_anchors,term_matches,detect_intent,
    _directed_path_candidates,MAX_QUESTION_LENGTH
)

class GraphAssistPlannerV216Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verified=load_graph(include_pending=False)
        cls.research=load_graph(include_pending=True)

    @staticmethod
    def run_plan(q,bundle,max_branches=2,subs=None):
        return plan(q,subs if subs is not None else [q],bundle,max_branches)

    def test_semantic_screening_sha_is_frozen(self):
        from .graph_assist_planner_v216 import (
            FROZEN_V2151_RISK_SHA256,sha256_file
        )
        from .verify_machine_graph_v215 import HERE
        risk=HERE/"graph_reports/expansions/v2151_preliminary_semantic_flags.json"
        self.assertEqual(sha256_file(risk),FROZEN_V2151_RISK_SHA256)
        self.assertEqual(self.research["source_checksums"][
            "v2151_semantic_screening_sha256"],FROZEN_V2151_RISK_SHA256)

    def test_official_verified_count(self):
        self.assertEqual(self.verified["verified_count"],29)
        self.assertEqual(self.verified["new_pending_count"],0)
        self.assertEqual(len(self.verified["edges"]),29)

    def test_pending_count_no_unreviewed_alias_promotion(self):
        self.assertEqual(self.research["verified_count"],29)
        self.assertTrue(self.research["new_pending_count"]>0)
        pending=[e for e in self.research["edges"] if e["role"]=="pending_v215"]
        self.assertTrue(all(e["predicate"] not in
                            {"ALSO_CALLED","CONDITIONAL_FORM_OF"} for e in pending))
        self.assertEqual(self.research["new_edges_online_eligible"],False)

    def test_intent_coverage(self):
        cases={
            "吉布斯采样的理论依据是什么？":"theoretical_basis",
            "吉布斯采样的采样机制依赖哪类随机过程？":"theoretical_basis",
            "无向图模型使用什么函数表达概率分布？":"mechanism",
            "卷积神经网络包含哪些层，这些层怎样完成特征计算？":"mechanism",
            "随机森林如何训练？":"mechanism",
            "卷积神经网络有哪些层？":"composition",
            "吉布斯采样属于哪类算法？":"hierarchy",
            "吉布斯采样和MCMC有什么关系？":"relation",
            "吉布斯采样是什么？":"definition",
            "随机森林与Bagging的区别是什么？":"comparison_unsupported",
        }
        for text,expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(detect_intent(text),expected)

    def test_definition_no_expansion(self):
        for q in ("残差单元是什么？","马尔可夫随机场是什么？"):
            r=self.run_plan(q,self.research)
            self.assertEqual(r["branches"],[])
            self.assertEqual(r["reason"],"intent_gate_definition")

    def test_comparison_no_unproven_difference(self):
        r=self.run_plan("随机森林与Bagging有什么区别？",self.research)
        self.assertEqual(r["branches"],[])
        self.assertEqual(r["intent"],"comparison_unsupported")

    def test_gibbs_verified_two_hop_theory(self):
        r=self.run_plan("吉布斯采样的采样机制依赖哪类随机过程？",
                        self.verified)
        self.assertEqual(r["status"],"planned")
        self.assertTrue(any(
            b["path_labels"]==["吉布斯采样","MCMC","马尔可夫链"]
            and b["path_predicates"]==["IS_A","BASED_ON"]
            and b["target_chunk_id"].startswith("597a2dc0")
            for b in r["branches"]))

    def test_cnn_directed_composition_mechanism(self):
        q="卷积神经网络包含哪些层，这些层怎样完成特征计算？"
        r=self.run_plan(q,self.verified)
        self.assertTrue(any(
            b["path_predicates"]==["HAS_PART","USES"]
            and b["path_labels"][:2]==["卷积神经网络","卷积层"]
            for b in r["branches"]))
        ids=[b["target_chunk_id"] for b in r["branches"]]
        self.assertEqual(len(ids),len(set(ids)))

    def test_mh_crossbook_requires_offline_explicit_opt_in(self):
        q="Metropolis-Hastings算法通过什么随机过程开展采样？"
        baseline=self.run_plan(q,self.verified)
        self.assertEqual(baseline["branches"],[])
        explore=self.run_plan(q,self.research)
        self.assertTrue(any(
            b["path_labels"]==["Metropolis-Hastings算法","MCMC","马尔可夫链"]
            and b["path_predicates"]==["IS_A","BASED_ON"]
            and b["path_kind"]=="cross_textbook_directed_two_edge_path"
            and b["source_risk"]=="pending_new_book_offline_only"
            for b in explore["branches"]))

    def test_undirected_model_alias_bridge_not_called_two_semantic_hops(self):
        q="无向图模型使用什么函数表达概率分布？"
        result=self.run_plan(q,self.research)
        self.assertTrue(any(
            b["path_labels"]==["无向图模型","马尔可夫随机场","势函数"]
            and b["path_kind"]=="verified_alias_bridge_then_direct_relation"
            for b in result["branches"]))

    def test_pending_ineligible_for_online_in_all_output(self):
        r=self.run_plan("随机森林如何训练？",self.research)
        self.assertTrue(r["offline_only"])
        self.assertFalse(r["diagnostics"]["online_eligible"])
        for b in r["branches"]:
            self.assertFalse(b["online_eligible"])
            self.assertFalse(b["asserts_new_direct_fact"])

    def test_verification_gate_keeps_original_queries_exact(self):
        original=["吉布斯采样的理论依据是什么？","学习率调整的作用是什么？"]
        copy=list(original)
        result=plan(original[0],original,self.verified)
        self.assertEqual(result["original_subqueries"],copy)
        self.assertEqual(original,copy)
        self.assertTrue(result["original_subqueries_unchanged"])

    def test_longest_chinese_term_overrides_inside_substring(self):
        q="吉布斯采样的理论依据是什么？"
        labels={x["label"] for x in match_anchors(q,self.verified)}
        self.assertIn("吉布斯采样",labels)
        self.assertNotIn("采样",labels)
        self.assertNotIn("采样法",labels)

    def test_english_concept_boundary(self):
        self.assertTrue(term_matches("MCMC是采样技术","MCMC"))
        self.assertFalse(term_matches("X_MCMCB","MCMC"))
        self.assertFalse(term_matches("SomeXMCX","MC"))

    def test_missing_graph_concept_no_expansion(self):
        q="天文学中的黑洞能否采用量子计算？"
        r=self.run_plan(q,self.research)
        self.assertFalse(r["branches"])

    def test_no_reversed_IS_A_paths(self):
        question="MCMC属于什么分类？"
        r=self.run_plan(question,self.verified)
        self.assertFalse(any(
            b["path_labels"]==["MCMC","吉布斯采样"] for b in r["branches"]))

    def test_no_new_claims_and_grounded_native_chunk_ids(self):
        id_pool={x["chunk_id"] for x in self.research["edges"]}
        for question in (
            "吉布斯采样的采样机制依赖哪类随机过程？",
            "无向图模型使用什么函数表达概率分布？",
            "随机森林如何训练？"
        ):
            r=self.run_plan(question,self.research)
            for b in r["branches"]:
                self.assertIn(b["target_chunk_id"],id_pool)
                self.assertEqual(b["path_chunk_ids"][-1],b["target_chunk_id"])
                self.assertEqual(len(b["path_claim_ids"]),len(b["edges"]))
                self.assertFalse(b["asserts_new_direct_fact"])

    def test_tight_branch_budget_and_unique_target_evidence(self):
        q="卷积神经网络包含哪些层，这些层怎样完成特征计算？"
        for limit in (0,1,2,3):
            r=self.run_plan(q,self.research,max_branches=limit)
            self.assertLessEqual(len(r["branches"]),limit)
            ids=[b["target_chunk_id"] for b in r["branches"]]
            self.assertEqual(len(ids),len(set(ids)))

    def test_invalid_max_budget_fails(self):
        q="吉布斯采样的理论依据是什么？"
        with self.assertRaises(RuntimeError):
            plan(q,[q],self.verified,max_branches=9)

    def test_too_long_fails_closed(self):
        q="吉布斯采样为什么" + "a"*(MAX_QUESTION_LENGTH+1)
        r=self.run_plan(q,self.research)
        self.assertEqual(r["branches"],[])
        self.assertEqual(r["intent"],"too_long")

    def test_no_legacy_query_modification_or_model_calls(self):
        q="吉布斯采样的采样机制依赖哪类随机过程？"
        r=self.run_plan(q,self.verified)
        self.assertEqual(r["diagnostics"]["query_calls_executed"],0)
        self.assertEqual(r["diagnostics"]["reranker_calls_executed"],0)

if __name__=="__main__":
    unittest.main()
