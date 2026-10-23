# -*- coding: utf-8 -*-
"""Pure offline tests; do not initialize Chroma or call Qwen."""
import sys
import types
import unittest
from unittest.mock import patch

from .graph_hybrid_overlap_v214 import analyze_hybrid_overlap, _rank_map

class FakeDoc:
    def __init__(self, cid, text, source="神经网络与深度学习.md"):
        self.metadata={"chunk_id":cid, "source":source}
        self.page_content=text

def _fake_canonical_source(source):
    return str(source or "").strip().casefold()

class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.module=types.ModuleType("graph_tools.graph_chunk_linker_v211")
        self.module.canonical_source=_fake_canonical_source
        self.patch=patch.dict(sys.modules,{
            "graph_tools.graph_chunk_linker_v211":self.module,
        })
        self.patch.start()
    def tearDown(self):
        self.patch.stop()

    def test_exact_id_overlap(self):
        docs=[FakeDoc("base","baseline"),FakeDoc("new","actual graph body")]
        result=analyze_hybrid_overlap(
            docs, {"base":("baseline",{"source":"book.md"}),
                   "new":("actual graph body",{"source":"book.md"})},
            ["base"],["base","new"],["new"])
        row=result["v214_new_candidate_audit"][0]
        self.assertEqual(row["hybrid_rank_top20"],2)
        self.assertFalse(row["candidate_novel_vs_hybrid_top10"])
        self.assertFalse(row["candidate_novel_vs_hybrid_top20"])

    def test_same_body_source_but_different_chunk_id(self):
        docs=[FakeDoc("base","baseline","book.md"),
              FakeDoc("other","same text","book.md")]
        result=analyze_hybrid_overlap(
            docs,{"base":("baseline",{"source":"book.md"}),
                  "new":("same text",{"source":"book.md"})},
            ["base"],["base","new"],["new"])
        row=result["v214_new_candidate_audit"][0]
        self.assertIsNone(row["hybrid_rank_top20"])
        self.assertEqual(row["same_content_other_id_top20"],["other"])
        self.assertEqual(result["v214_new_vs_hybrid_top20"],0)

    def test_novel_candidate_not_in_hybrid(self):
        docs=[FakeDoc("base","baseline","book.md")]
        result=analyze_hybrid_overlap(
            docs,{"base":("baseline",{"source":"book.md"}),
                  "new":("different text",{"source":"book.md"})},
            ["base"],["base","new"],["new"])
        self.assertEqual(result["v214_new_vs_hybrid_top10"],1)
        self.assertEqual(result["v214_new_vs_hybrid_top20"],1)

    def test_top10_vs_top20(self):
        docs=[FakeDoc(str(i),"text"+str(i),"book.md") for i in range(1,13)]
        result=analyze_hybrid_overlap(
            docs,{"1":("text1",{"source":"book.md"}),
                  "11":("text11",{"source":"book.md"})},
            ["1"],["1","11"],["11"])
        row=result["v214_new_candidate_audit"][0]
        self.assertEqual(row["hybrid_rank_top20"],11)
        self.assertTrue(row["candidate_novel_vs_hybrid_top10"])
        self.assertFalse(row["candidate_novel_vs_hybrid_top20"])

    def test_same_text_different_source_is_not_same_evidence(self):
        docs=[FakeDoc("base","base","book.md"),
              FakeDoc("other","identical","different-book.md")]
        result=analyze_hybrid_overlap(
            docs,{"base":("base",{"source":"book.md"}),
                  "new":("identical",{"source":"book.md"})},
            ["base"],["base","new"],["new"])
        self.assertEqual(result["v214_new_vs_hybrid_top20"],1)

    def test_reject_missing_or_duplicate_ids(self):
        with self.assertRaises(RuntimeError):
            _rank_map([FakeDoc("a","x"),FakeDoc("a","y")])
        with self.assertRaises(RuntimeError):
            _rank_map([FakeDoc("","x")])

    def test_reject_mismatched_new_ids(self):
        with self.assertRaises(RuntimeError):
            analyze_hybrid_overlap(
                [FakeDoc("base","base")],
                {"base":("base",{"source":"book.md"}),
                 "new":("new",{"source":"book.md"})},
                ["base"],["base","new"],[])  # must be ['new']

if __name__ == "__main__":
    unittest.main()
