# -*- coding: utf-8 -*-
"""No network, Chroma, reranking or generation in these unit tests."""
from pathlib import Path
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from .audit_multisource_hybrid_v2151 import (
    check_lookup_report, analyze_query
)
from .verify_machine_graph_v215 import HERE, INDEX_REL

class FakeDoc:
    def __init__(self,cid,body,source="机器学习.md"):
        self.page_content=body
        self.metadata={"chunk_id":cid,"source":source}

def fake_canonical(x):
    return str(x or "").strip().casefold()

class OfflineAuditUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index=json.loads((HERE/INDEX_REL).read_text(encoding="utf-8"))
        cls.lookup_path=HERE/"graph_reports/expansions/multi_source_lookup_run01.json"

    def setUp(self):
        # No original Graph dependency needed by a pure comparison helper.
        fake=types.ModuleType("graph_tools.graph_retriever_v213")
        fake.canonical_source=fake_canonical
        self.patch=patch.dict(sys.modules,{"graph_tools.graph_retriever_v213":fake})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_user_pilot_replays_exactly(self):
        r=check_lookup_report(self.lookup_path,self.index)
        self.assertEqual(len(r),6)
        self.assertEqual(sum(x["candidate_count"] for x in r),20)
        self.assertEqual(len({c["chunk_id"] for row in r for c in row["chunks"]}),16)

    def test_falsified_chunk_id_fails_before_api_call(self):
        r=json.loads(self.lookup_path.read_text(encoding="utf-8"))
        r["results"][0]["chunks"][0]["chunk_id"]="tampered-id"
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/"edited.json"
            file.write_text(json.dumps(r,ensure_ascii=False),encoding="utf-8")
            with self.assertRaises(RuntimeError):
                check_lookup_report(file,self.index)

    def test_exact_hybrid_hit(self):
        row={"query":"q","matched_labels":["X"],"matched_claim_count":1,
             "chunks":[{"chunk_id":"new","source_file":"机器学习.md",
                        "chapter":"C","status":"review_pending","claim_ids":["e"]}]}
        data={"new":("abcdef",{"source":"机器学习.md"})}
        outcome=analyze_query(row,[FakeDoc("new","abcdef")],data)
        rec=outcome["candidate_audit"][0]
        self.assertEqual(rec["hybrid_rank_top20"],1)
        self.assertFalse(rec["novel_candidate_vs_hybrid_top20"])
        self.assertTrue(rec["semantic_review_pending"])

    def test_new_graph_chunk_not_in_hybrid(self):
        row={"query":"q","matched_labels":["X"],"matched_claim_count":1,
             "chunks":[{"chunk_id":"new","source_file":"机器学习.md",
                        "chapter":"C","status":"verified","claim_ids":["e"]}]}
        data={"new":("new body",{"source":"机器学习.md"})}
        outcome=analyze_query(row,[FakeDoc("old","old body")],data)
        rec=outcome["candidate_audit"][0]
        self.assertIsNone(rec["hybrid_rank_top20"])
        self.assertTrue(rec["novel_candidate_vs_hybrid_top20"])
        self.assertFalse(rec["semantic_review_pending"])

    def test_content_duplicate_different_id_does_not_count_as_increment(self):
        row={"query":"q","matched_labels":["X"],"matched_claim_count":1,
             "chunks":[{"chunk_id":"new","source_file":"机器学习.md",
                        "chapter":"C","status":"review_pending","claim_ids":["e"]}]}
        data={"new":("identical",{"source":"机器学习.md"})}
        result=analyze_query(row,[FakeDoc("old","identical")],data)
        rec=result["candidate_audit"][0]
        self.assertEqual(rec["same_body_same_source_other_id_top20"],
                         [{"chunk_id":"old","hybrid_rank":1}])
        self.assertFalse(rec["novel_candidate_vs_hybrid_top20"])

    def test_same_content_different_book_is_not_a_duplicate(self):
        row={"query":"q","matched_labels":["X"],"matched_claim_count":1,
             "chunks":[{"chunk_id":"new","source_file":"机器学习.md",
                        "chapter":"C","status":"review_pending","claim_ids":["e"]}]}
        data={"new":("identical",{"source":"机器学习.md"})}
        result=analyze_query(row,[FakeDoc("old","identical","神经网络与深度学习.md")],data)
        rec=result["candidate_audit"][0]
        self.assertEqual(rec["same_body_same_source_other_id_top20"],[])
        self.assertTrue(rec["novel_candidate_vs_hybrid_top20"])

    def test_rank_11_novel_top10_but_not_top20(self):
        row={"query":"q","matched_labels":["X"],"matched_claim_count":1,
             "chunks":[{"chunk_id":"11","source_file":"机器学习.md",
                        "chapter":"C","status":"review_pending","claim_ids":["e"]}]}
        docs=[FakeDoc(str(i),f"body{i}") for i in range(1,13)]
        data={"11":("body11",{"source":"机器学习.md"})}
        outcome=analyze_query(row,docs,data)
        rec=outcome["candidate_audit"][0]
        self.assertEqual(rec["hybrid_rank_top20"],11)
        self.assertTrue(rec["novel_candidate_vs_hybrid_top10"])
        self.assertFalse(rec["novel_candidate_vs_hybrid_top20"])

    def test_reject_duplicate_hybrid_chunk_ids(self):
        row={"query":"q","matched_labels":[],"matched_claim_count":0,
             "chunks":[]}
        with self.assertRaises(RuntimeError):
            analyze_query(row,[FakeDoc("same","x"),FakeDoc("same","y")],{})

    def test_reject_graph_source_drift(self):
        row={"query":"q","matched_labels":["X"],"matched_claim_count":1,
             "chunks":[{"chunk_id":"new","source_file":"机器学习.md",
                        "chapter":"C","status":"review_pending","claim_ids":["e"]}]}
        data={"new":("x",{"source":"wrong.md"})}
        with self.assertRaises(RuntimeError):
            analyze_query(row,[FakeDoc("base","y")],data)

if __name__=="__main__":
    unittest.main()
