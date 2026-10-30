"""离线测试：不访问真实 Neo4j、Chroma、模型，也不触及用户数据。"""
import ast
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "medical_graph_test_target", ROOT / "drug_graph_retriever.py"
)
graph_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_mod)


class GraphSectionTest(unittest.TestCase):
    def test_medical_field_routing(self):
        self.assertEqual(
            graph_mod.fields_for_question("阿奇霉素葡萄糖注射液的禁忌有哪些？"),
            ["contraindications", "warnings"],
        )
        self.assertEqual(
            graph_mod.fields_for_question("药品有哪些副作用？"),
            ["adverse_reactions", "warnings"],
        )
        self.assertEqual(
            graph_mod.fields_for_question("两种药有什么相互作用？"),
            ["drug_interactions", "warnings"],
        )
        self.assertEqual(
            graph_mod.fields_for_question("阿托伐他汀钙片与动脉粥样硬化有关吗？"),
            ["indication"],
        )

    def test_format_source_section_is_cautious_and_source_linked(self):
        section = graph_mod.format_section({
            "drug_name": "阿托伐他汀钙片",
            "section_field": "indication",
            "section_text": "高胆固醇血症并有动脉粥样硬化危险的病人。",
            "source_url": "https://example.org/drugs/1",
            "authority_status": "third_party_unverified",
            "record_id": "r1",
            "section_id": "s1",
        })
        self.assertEqual(section["evidence_type"], "unverified_drug_source_section")
        self.assertEqual(section["source_url"], "https://example.org/drugs/1")
        self.assertIn("第三方资料", section["content"])
        self.assertIn("动脉粥样硬化危险", section["content"])
        self.assertNotIn("已确认", section["content"])

    def test_section_readonly_and_field_restricted(self):
        retriever = graph_mod.DrugGraphRetriever(driver=object())
        calls = []

        def fake_run(query, **params):
            calls.append((query, params))
            if query == graph_mod._DRUG_IDENTIFY:
                return [{"id": "drug_1", "name": "阿托伐他汀钙片"}]
            if query == graph_mod._SECTION_BY_DRUG:
                self.assertEqual(params["fields"], ["indication"])
                self.assertEqual(params["id"], "drug_1")
                return [{
                    "drug_name": "阿托伐他汀钙片",
                    "section_field": "indication",
                    "section_text": "高胆固醇血症并有动脉粥样硬化危险的病人。",
                    "record_id": "r1",
                    "section_id": "s1",
                    "source_url": "https://example.org/1",
                }]
            raise AssertionError("Unexpected query")
        retriever._run = fake_run
        results = retriever.search_sections("阿托伐他汀钙片与动脉粥样硬化有什么关联？")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["section_field"], "indication")
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("CREATE " not in query and "DELETE " not in query
                            and "MERGE " not in query for query, _ in calls))

    def test_invalid_section_and_unsafe_url(self):
        self.assertIsNone(graph_mod.format_section({
            "drug_name": "药品X", "section_field": "unknown",
            "section_text": "irrelevant"
        }))
        obj = graph_mod.format_section({
            "drug_name": "药品X", "section_field": "warnings",
            "section_text": "注意事项正文",
            "source_url": "javascript:alert(1)",
        })
        self.assertEqual(obj["source_url"], "")

    def test_search_sections_no_drug_match(self):
        retriever = graph_mod.DrugGraphRetriever(driver=object())
        retriever._run = lambda query, **params: []
        self.assertEqual(retriever.search_sections("不知道药名的提问"), [])


class IntegrationStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = (ROOT / "rag.py").read_text(encoding="utf-8")
        cls.ui = (ROOT / "豆馅.py").read_text(encoding="utf-8")
        ast.parse(cls.rag)
        ast.parse(cls.ui)

    def test_knowledge_base_calls_both_backends(self):
        self.assertIn(
            'elif mode in ("知识库", "GraphRAG"):', self.rag
        )
        self.assertIn(
            'names = ["search_drug_graph", "search_knowledge_base"]', self.rag
        )
        self.assertIn('mode not in ("知识库", "GraphRAG")', self.rag)

    def test_empty_chroma_does_not_block_graph(self):
        self.assertIn('"status": "empty_collection"', self.rag)
        self.assertIn('if not ok:', self.rag)
        self.assertIn('tool_reports[tool_name] = {"status": "error", "hits": 0}', self.rag)
        self.assertIn('continue', self.rag)

    def test_ui_has_single_knowledge_entry_and_diagnostics(self):
        self.assertIn('options=["知识库", "自动", "联网"]', self.ui)
        self.assertNotIn('options=["知识库", "GraphRAG"', self.ui)
        self.assertIn('Neo4j {graph_hits} 条 / Chroma {chroma_hits} 条', self.ui)
        self.assertIn('chroma_status == "empty_collection"', self.ui)

    def test_retrieval_provenance_survives_guard(self):
        self.assertIn('"source_url": original_item.get("source_url", "")', self.rag)
        self.assertIn('reference_map[idx] = (', self.rag)
        self.assertIn('_reference_label(item, source)', self.rag)
        self.assertIn('build_candidate_answer(candidate_rows)', self.rag)

    def test_hybrid_and_safety_components_retained(self):
        for token in (
            'search_with_rerank(', 'build_hybrid_retriever()',
            'assess_evidence(', 'retry_insufficient_evidence(',
            'render_kb_references(', 'MEDICAL_KB_ANSWER_PROMPT',
            'pending_human_review',
        ):
            self.assertIn(token, self.rag)

    def test_diagnostic_is_readonly(self):
        script = (ROOT / "diagnose_chroma.py").read_text(encoding="utf-8")
        ast.parse(script)
        self.assertIn('mode=ro', script)
        self.assertIn('PRAGMA query_only = ON', script)
        for suspicious in ("conn.execute(\"DELETE", "conn.execute(\"UPDATE", "conn.execute(\"DROP"):
            self.assertNotIn(suspicious, script)


if __name__ == "__main__":
    unittest.main()
