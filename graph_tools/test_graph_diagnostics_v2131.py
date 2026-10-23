import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graph_tools.graph_diagnostics_v2131 import finish_run, aggregate

class Doc:
    def __init__(self, cid):
        self.metadata = {"chunk_id":cid}

class TestGraphMetrics(unittest.TestCase):
    def test_exact_hybrid_overlap_without_injection(self):
        run={"graph_candidate_ids":["a"],"already_in_hybrid_ids":["a"],
             "injected_chunk_ids":[]}
        done=finish_run(run,[Doc("a")],[Doc("a")])
        self.assertEqual(done["graph_evidence_in_final"],1)
        self.assertEqual(done["graph_incremental_final"],0)
        self.assertEqual(done["survived_final"],0)
    def test_content_overlap_is_not_exact(self):
        run={"graph_candidate_ids":["a"],"matched_by_content_ids":["a"],
             "matched_by_content_map":{"a":["b"]},"injected_chunk_ids":[]}
        done=finish_run(run,[Doc("b")],[Doc("b")])
        self.assertEqual(done["graph_evidence_in_final"],0)
        self.assertEqual(done["content_equivalent_in_final"],1)
    def test_incremental_final_and_rerank(self):
        run={"graph_candidate_ids":["a","b"],"injected_chunk_ids":["a","b"]}
        done=finish_run(run,[Doc("a")],[Doc("a")])
        self.assertEqual(done["survived_rerank"],1)
        self.assertEqual(done["graph_incremental_final"],1)
        self.assertEqual(done["survived_final"],1)
    def test_multisubquery_deduplicates_ids(self):
        runs=[finish_run({"graph_candidate_ids":["a"],"injected_chunk_ids":["a"]},
                         [Doc("a")],[Doc("a")]),
              finish_run({"graph_candidate_ids":["a"],"injected_chunk_ids":["a"]},
                         [Doc("a")],[Doc("a")])]
        total=aggregate(runs,[Doc("a")])
        self.assertEqual(total["graph_incremental_final"],1)

if __name__=="__main__":
    unittest.main()
