# -*- coding: utf-8 -*-
"""V2.14 offline probe tests: no Chroma, DashScope, Streamlit or file edits."""
import ast
import json
from pathlib import Path
import re
import tempfile
import unittest

from .graph_path_probe_v214 import (
    OfflineGraphError, load_offline_bundle, graph_inventory,
    probe_query, v213_candidates, find_two_hop_paths, make_offline_report,
)

ROOT = Path(__file__).resolve().parents[1]

class GraphPathProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_offline_bundle()

    def test_formal_inventory(self):
        inv = graph_inventory(self.bundle)
        self.assertEqual(inv["semantic_relation_count"], 29)
        self.assertEqual(inv["distinct_verified_bound_chunk_ids"], 18)
        self.assertEqual(inv["directed_two_hop_path_count_excluding_synonyms"], 6)
        self.assertNotIn("MENTIONED_IN", inv["relation_type_counts"])
        self.assertNotIn("PART_OF", inv["relation_type_counts"])

    def test_original_v213_source_parity(self):
        """Execute only the original pure baseline functions extracted from AST."""
        old_file = ROOT / "graph_tools" / "graph_retriever_v213.py"
        self.assertTrue(old_file.is_file(), "Original V2.13 is required for parity")
        source = ast.parse(old_file.read_text(encoding="utf-8"))
        nodes = [n for n in source.body
                 if isinstance(n, ast.FunctionDef)
                 and n.name in ("_has_term", "candidate_chunk_ids")]
        self.assertEqual(len(nodes), 2)
        isolated = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        namespace = {"re": re, "MAX_QUERY_LENGTH": 500, "MAX_GRAPH_CHUNKS": 2}
        exec(compile(isolated, str(old_file), "exec"), namespace)
        native = namespace["candidate_chunk_ids"]
        for query in (
            "残差单元是什么？",
            "马尔可夫随机场是什么？",
            "吉布斯采样的理论依据是什么？",
            "卷积神经网络包含哪些层，这些层怎样进行特征计算？",
            "变分贝叶斯是什么？",
            "汇聚层如何使用最大汇聚进行下采样？",
        ):
            expected, native_info = native(query, self.bundle, max_candidates=2)
            actual, probe_info = v213_candidates(query, self.bundle, limit=2)
            self.assertEqual(actual, expected, query)
            self.assertEqual(probe_info, native_info, query)

    def test_two_hop_gibbs(self):
        query = "吉布斯采样的理论依据是什么？"
        result = probe_query(query, self.bundle)
        self.assertEqual(result["intent_gates"], ["theoretical_basis"])
        self.assertEqual(len(result["two_hop_paths"]), 1)
        p = result["two_hop_paths"][0]
        self.assertEqual([e["predicate"] for e in p["edges"]], ["IS_A", "BASED_ON"])
        self.assertEqual([e["subject"] for e in p["edges"]], ["吉布斯采样", "MCMC"])
        self.assertEqual(len(p["path_chunk_ids"]), 2)
        self.assertEqual(len(result["new_ids_in_budgeted_v214"]), 1)
        self.assertEqual(len(result["v214_budgeted_candidate_ids"]), 2)
        self.assertEqual(len(result["v213_ids_displaced_by_budget"]), 0)

    def test_two_hop_convolution(self):
        result = probe_query(
            "卷积神经网络包含哪些层，这些层怎样进行特征计算？",
            self.bundle
        )
        self.assertEqual(len(result["two_hop_paths"]), 3)
        self.assertEqual(len(result["v213_candidate_ids"]), 1)
        self.assertEqual(len(result["new_ids_in_budgeted_v214"]), 1)
        self.assertEqual(len(result["v214_budgeted_candidate_ids"]), 2)
        for path in result["two_hop_paths"]:
            self.assertEqual([e["predicate"] for e in path["edges"]],
                             ["HAS_PART", "USES"])

    def test_no_definition_expansion(self):
        for query in ("残差单元是什么？", "马尔可夫随机场是什么？"):
            result = probe_query(query, self.bundle)
            self.assertEqual(result["two_hop_paths"], [])
            self.assertEqual(result["v214_budgeted_candidate_ids"],
                             result["v213_candidate_ids"])

    def test_restrict_budget_without_hidden_reservations(self):
        result = probe_query("吉布斯采样的理论依据是什么？",
                             self.bundle, max_graph_candidates=1)
        self.assertEqual(result["v214_budgeted_candidate_ids"],
                         result["v213_candidate_ids"])
        self.assertEqual(result["new_ids_in_budgeted_v214"], [])
        self.assertEqual(len(result["new_two_hop_ids_before_budget"]), 1)

    def test_rejected_average_pooling_edge_not_resurrected(self):
        self.assertFalse(any(
            e["relation"] == "IS_A" and
            self.bundle.concepts[e["source"]]["label"] == "平均汇聚"
            for e in self.bundle.edges
        ))

    def test_no_unverified_path_synthesis(self):
        report = make_offline_report(self.bundle)
        self.assertTrue(report["no_live_qa_claim"])
        for q in report["queries"]:
            self.assertLessEqual(len(q["v214_budgeted_candidate_ids"]), 2)
            for path in q["two_hop_paths"]:
                self.assertIn("not a new direct", path["note"])
                for e in path["edges"]:
                    claim = self.bundle.claim_by_edge[e["claim_id"]]
                    self.assertEqual(claim["status"], "verified")
                    self.assertEqual(e["chunk_id"], claim["chunk_ids"][0])

    def test_corrupted_sha_rejected(self):
        content = ROOT / "graph_data" / "knowledge_graph_v212.json"
        linked = ROOT / "graph_bindings" / "graph_chunk_links_v212.json"
        aliases = ROOT / "graph_data" / "graph_alias_registry_v212.json"
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            # No permanent files modified.
            for name, source in (("graph.json", content), ("links.json", linked),
                                 ("aliases.json", aliases)):
                (base/name).write_bytes(source.read_bytes())
            changed = json.loads((base/"aliases.json").read_text(encoding="utf-8"))
            changed["graph_sha256"] = "0"*64
            (base/"aliases.json").write_text(
                json.dumps(changed, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(OfflineGraphError):
                load_offline_bundle(base/"graph.json", base/"links.json",
                                    base/"aliases.json")

if __name__ == "__main__":
    unittest.main()
