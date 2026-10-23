# -*- coding: utf-8 -*-
"""Pure V2.16 evaluation and safety tests. No live Chroma or API use."""
from __future__ import annotations
import sys
import types
import unittest
from unittest.mock import patch

from .eval_graph_assist_v216 import (
    read_cases,legacy_relation_gate,evaluate_offline,
    rank_and_duplicates
)
from .graph_assist_planner_v216 import load_graph

class FakeDoc:
    def __init__(self,cid,body,source="机器学习.md"):
        self.page_content=body
        self.metadata={"chunk_id":cid,"source":source}

class PlannerEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verified=evaluate_offline(False)
        cls.exploratory=evaluate_offline(True)

    def setUp(self):
        # Pure output comparison, no import of DashScope or production service.
        graph=types.ModuleType("graph_tools.graph_retriever_v213")
        graph.canonical_source=lambda s:str(s or "").strip().casefold()
        self.patch=patch.dict(sys.modules,{"graph_tools.graph_retriever_v213":graph})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_fixture_count_and_uniqueness(self):
        data=read_cases()
        self.assertEqual(len(data),11)
        self.assertEqual(len({r["id"] for r in data}),11)

    def test_legacy_regex_extracted_from_actual_v28_source(self):
        regex, sha=legacy_relation_gate()
        self.assertIsNotNone(sha)
        self.assertTrue(regex.search("月球的环形山属于哪类天文地貌？"))
        self.assertFalse(regex.search("吉布斯采样的理论依据是什么？"))

    def test_verified_only_results_expected(self):
        x=self.verified
        self.assertEqual(x["verified_old_relations"],29)
        self.assertEqual(x["pending_relations_explorable"],0)
        self.assertEqual(x["questions_with_plan"],3)
        self.assertEqual(x["v216_planned_query_count"],4)
        self.assertEqual(x["embedding_calls"],0)
        self.assertEqual(x["reranker_calls"],0)

    def test_exploratory_pending_never_promoted(self):
        x=self.exploratory
        self.assertEqual(x["new_relations_promoted"],0)
        self.assertEqual(x["embedding_calls"],0)
        self.assertEqual(x["questions_with_plan"],6)
        self.assertGreater(x["pending_relations_explorable"],0)
        for row in x["results"]:
            for branch in row["branches"]:
                self.assertFalse(branch["online_eligible"])

    def test_old_intent_false_positive_not_graph_evidence(self):
        x=self.verified
        row=next(r for r in x["results"] if r["case_id"]=="unrelated")
        self.assertTrue(row["old_v28_relation_intent_regex_matched"])
        self.assertFalse(row["branches"])

    def test_negative_controls_do_not_expand(self):
        for row in self.exploratory["results"]:
            if row["group"].startswith("negative_"):
                self.assertEqual(row["branches"],[])

    def test_risk_flags_block_mh_rejection_sampling_edge(self):
        edges=load_graph(include_pending=True)["edges"]
        triples={(r["subject"],r["predicate"],r["object"]) for r in edges}
        self.assertNotIn(("Metropolis-Hastings算法","BASED_ON","拒绝采样"),triples)
        self.assertNotIn(("马尔可夫随机场","ALSO_CALLED","马尔可夫网"),triples)

    def test_original_and_plant_queries_no_external_model_in_offline(self):
        for x in (self.verified,self.exploratory):
            self.assertEqual(x["answer_generation_calls"],0)
            self.assertEqual(x["reranker_calls"],0)
            self.assertEqual(x["old_original_queries_modified"],0)

    def test_overlap_exact_id_already_present(self):
        r=rank_and_duplicates(
            [FakeDoc("a","A"),FakeDoc("graph","body")],
            "graph","body","机器学习.md")
        self.assertEqual(r["rank"],2)
        self.assertFalse(r["novel_in_top20"])

    def test_overlap_missing_verified_target(self):
        r=rank_and_duplicates(
            [FakeDoc("a","A"),FakeDoc("b","B")],
            "graph","body","机器学习.md")
        self.assertIsNone(r["rank"])
        self.assertTrue(r["novel_in_top20"])

    def test_overlap_same_body_source_different_native_id(self):
        r=rank_and_duplicates(
            [FakeDoc("different_id","identical")],
            "graph","identical","机器学习.md")
        self.assertEqual(len(r["same_content_same_source_other_ids"]),1)
        self.assertFalse(r["novel_in_top10"])
        self.assertFalse(r["novel_in_top20"])

    def test_overlap_same_body_different_book_not_duplicate(self):
        r=rank_and_duplicates(
            [FakeDoc("different_id","identical","神经网络与深度学习.md")],
            "graph","identical","机器学习.md")
        self.assertTrue(r["novel_in_top20"])

    def test_hybrid_duplicated_ids_fail(self):
        with self.assertRaises(RuntimeError):
            rank_and_duplicates(
                [FakeDoc("a","A"),FakeDoc("a","B")],
                "target","content","机器学习.md")

if __name__=="__main__":
    unittest.main()
