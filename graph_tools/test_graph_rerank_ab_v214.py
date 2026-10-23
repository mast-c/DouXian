# -*- coding: utf-8 -*-
"""Self-contained unit tests. No Chroma, DashScope API or file modification."""
import sys
import types
import unittest
from unittest.mock import patch

from .graph_rerank_ab_v214 import (
    _prepare_frozen_service, _new_for_top10,
    _parse_rerank_cost, _ranked_summary,
    _run_actual_production_reranker,
)

class FakeDoc:
    def __init__(self, cid, body, source="book.md", section="chapter1",
                 score=0.5):
        self.page_content=body
        self.metadata={
            "chunk_id":cid,
            "source":source,
            "section_path":section,
            "rerank_score":score,
        }

class FakeRerankService:
    def search_with_rerank(self, query, recall_k=10, top_n=3):
        docs=self.search_hybrid(query, k=recall_k)
        print(
            "[Rerank成本] 候选="+str(len(docs))+
            " 原文字符="+str(sum(len(d.page_content) for d in docs))+
            " 发送字符="+str(sum(len(d.page_content) for d in docs))+
            " 官方total_tokens=40 压缩上限=全文"
        )
        return docs[-top_n:]

def fake_canonical(x):
    return str(x or "").strip().casefold()

class RerankABTests(unittest.TestCase):
    def setUp(self):
        module=types.ModuleType("graph_tools.graph_chunk_linker_v211")
        module.canonical_source=fake_canonical
        self.patch=patch.dict(sys.modules,{"graph_tools.graph_chunk_linker_v211":module})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_distinct_graph_candidate_added(self):
        base=[FakeDoc("base","abc","book.md")]
        graph=[FakeDoc("new","xyz","book.md")]
        added,skipped=_new_for_top10(base,graph)
        self.assertEqual([d.metadata["chunk_id"] for d in added],["new"])
        self.assertEqual(skipped,[])

    def test_exact_id_duplicate_discarded(self):
        base=[FakeDoc("base","abc","book.md")]
        graph=[FakeDoc("base","abc","book.md")]
        added,skipped=_new_for_top10(base,graph)
        self.assertEqual(added,[])
        self.assertEqual(skipped,["base"])

    def test_same_content_source_other_id_is_duplicate(self):
        base=[FakeDoc("base","abc","book.md")]
        graph=[FakeDoc("new","abc","BOOK.MD")]
        added,skipped=_new_for_top10(base,graph)
        self.assertEqual(added,[])
        self.assertEqual(skipped,["new"])

    def test_same_content_different_source_remains_distinct(self):
        base=[FakeDoc("base","abc","book.md")]
        graph=[FakeDoc("new","abc","book2.md")]
        added,_=_new_for_top10(base,graph)
        self.assertEqual(len(added),1)

    def test_second_graph_duplicate_discarded(self):
        base=[FakeDoc("base","abc","book.md")]
        graph=[FakeDoc("one","xyz","book.md"),FakeDoc("two","xyz","book.md")]
        added,skipped=_new_for_top10(base,graph)
        self.assertEqual([d.metadata["chunk_id"] for d in added],["one"])
        self.assertEqual(skipped,["two"])

    def test_original_method_is_inherited_only_candidate_pool_substituted(self):
        frozen=_prepare_frozen_service(FakeRerankService,[
            FakeDoc("a","aa"),FakeDoc("b","bb"),FakeDoc("graph","cc")
        ])
        self.assertIsInstance(frozen,FakeRerankService)
        self.assertEqual(len(frozen.search_hybrid("question",k=2)),2)
        self.assertEqual(
            [d.metadata["chunk_id"] for d in frozen.search_with_rerank(
                "question",recall_k=3,top_n=2
            )],["b","graph"]
        )

    def test_call_adapter_cost_and_scores(self):
        pool=[FakeDoc("a","aa"), FakeDoc("graph","cc")]
        ranked,cost,_=_run_actual_production_reranker(
            FakeRerankService,"question",pool,top_n=2)
        self.assertEqual(cost["candidate_count"],2)
        self.assertEqual(cost["official_total_tokens"],40)
        result=_ranked_summary(ranked,{"graph"})
        self.assertFalse(result[0]["v214_new_graph_chunk"])
        self.assertTrue(result[1]["v214_new_graph_chunk"])

    def test_parse_absent_tokens(self):
        trace="[Rerank成本] 候选=10 原文字符=1000 发送字符=500 官方total_tokens=未返回 压缩上限=600"
        got=_parse_rerank_cost(trace)
        self.assertIsNone(got["official_total_tokens"])
        self.assertEqual(got["candidate_count"],10)
        self.assertEqual(got["rerank_doc_chars_setting"],"600")

    def test_malformed_cost_rejected(self):
        with self.assertRaises(RuntimeError):
            _parse_rerank_cost("some strange log")

if __name__=="__main__":
    unittest.main()
