# -*- coding: utf-8 -*-
"""V2.15.2 offline path verification tests; zero external services."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .path_multibook_v2152 import (
    CASEFILE,EXPECTED_CASES,EXPECTED_PATTERNS,load_and_verify_cases,
)
from .verify_machine_graph_v215 import (
    SRC,SNAPSHOT,OLD_GRAPH,OLD_LINKS,OLD_ALIASES,sha_file
)

class CrossBookPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obj=load_and_verify_cases()
        cls.cases=cls.obj["two_hop_cross_book_cases"]

    def test_three_real_two_hop_paths(self):
        self.assertEqual(len(self.cases),3)
        self.assertEqual(tuple(c["case_id"] for c in self.cases),EXPECTED_CASES)
        self.assertEqual(tuple(tuple(c["edge_predicates"]) for c in self.cases),
                         EXPECTED_PATTERNS)

    def test_two_distinct_evidence_chunk_ids(self):
        for case in self.cases:
            self.assertNotEqual(case["first_hop_chunk_id"],
                                case["second_hop_chunk_id"])
            self.assertEqual(case["edge_evidence"][0]["object"],
                             case["edge_evidence"][1]["subject"])
            self.assertEqual(case["sequence"],[
                case["edge_evidence"][0]["subject"],
                case["edge_evidence"][0]["object"],
                case["edge_evidence"][1]["object"]
            ])

    def test_each_path_joins_two_books(self):
        for case in self.cases:
            first,second=case["edge_evidence"]
            self.assertNotEqual(first["source_file"],second["source_file"])
            self.assertEqual({first["book_role"],second["book_role"]},
                             {"old_verified","new_pending_review"})

    def test_query_does_not_leak_middle_or_target_concept(self):
        for case in self.cases:
            question=case["query"]
            self.assertIn(case["sequence"][0],question)
            self.assertNotIn(case["sequence"][1],question)
            self.assertNotIn(case["sequence"][2],question)

    def test_all_new_graph_edges_offline_only(self):
        self.assertTrue(self.obj["no_online_promotion"])
        for case in self.cases:
            self.assertFalse(case["online_eligible"])
            self.assertTrue(case["new_book_edge_semantic_review_pending"])
            self.assertIn("not an inferred",case["scope"])

    def test_negative_controls_declared(self):
        controls=self.obj["negative_controls"]
        self.assertEqual(len(controls),2)
        self.assertTrue(all(c["expected_path_expansion"] is False
                            for c in controls))

    def test_frozen_baseline_unchanged_when_verifying(self):
        check=[SRC,SNAPSHOT,OLD_GRAPH,OLD_LINKS,OLD_ALIASES]
        before=[sha_file(p) for p in check]
        load_and_verify_cases()
        self.assertEqual(before,[sha_file(p) for p in check])

    def test_reject_swapped_edge_ids(self):
        obj=json.loads(CASEFILE.read_text(encoding="utf-8"))
        obj["cases"][0]["path_claim_ids"].reverse()
        with tempfile.TemporaryDirectory() as t:
            edited=Path(t)/"incorrect.json"
            edited.write_text(json.dumps(obj,ensure_ascii=False),encoding="utf-8")
            with patch("graph_tools.path_multibook_v2152.CASEFILE",edited):
                with self.assertRaises(RuntimeError):
                    load_and_verify_cases()

    def test_reject_query_with_answer_leak(self):
        obj=json.loads(CASEFILE.read_text(encoding="utf-8"))
        obj["cases"][0]["query"]+=" MCMC马尔可夫链"
        with tempfile.TemporaryDirectory() as t:
            edited=Path(t)/"leaky.json"
            edited.write_text(json.dumps(obj,ensure_ascii=False),encoding="utf-8")
            with patch("graph_tools.path_multibook_v2152.CASEFILE",edited):
                with self.assertRaises(RuntimeError):
                    load_and_verify_cases()

if __name__=="__main__":
    unittest.main()
