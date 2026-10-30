# -*- coding: utf-8 -*-
"""独立离线测试：不连接本地 Neo4j / Chroma / Qwen。"""
import ast
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "medical_evidence_bridge", ROOT/"medical_evidence_bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def make_doc(field="indication", quote="高胆固醇血症并有动脉粥样硬化危险的病人。",
             drug="阿托伐他汀钙片", record_id="record_1", section_id="section_1"):
    parent = {
        "drug_name": drug,
        "field": field,
        "source_url": "https://example.org/medicine",
        "full_section": "该药信息：" + quote,
    }
    meta = {
        "data_origin": "json_matched_to_neo4j_section",
        "drug_name": drug,
        "drug_id": "drug_1",
        "record_id": record_id,
        "section_id": section_id,
        "source_url": parent["source_url"],
        "source_status": "third_party_unverified",
        "section_field": field,
        "quote_start": len("该药信息："),
        "quote_end": len(parent["full_section"]),
        "section_sha256": hashlib.sha256(
            parent["full_section"].encode("utf-8")
        ).hexdigest(),
        "section_path": f"{drug} / {field}",
    }
    content = f"药品：{drug}\n栏目：{field}\n资料状态：第三方来源尚未完成医学权威核验\n证据原文：{quote}"
    return FakeDocument(content, meta), parent


class TestMedicalEvidenceBridge(unittest.TestCase):
    def test_relation_question_prefers_indication(self):
        question = "阿托伐他汀钙片与动脉粥样硬化有什么关联？请给出原文证据"
        self.assertEqual(bridge.expected_section_fields(question),
                         frozenset({"indication"}))

    def test_multi_intent_includes_several_relevant_sections(self):
        self.assertEqual(
            bridge.expected_section_fields("药品的禁忌及不良反应分别有哪些？"),
            frozenset({"contraindications", "adverse_reactions"}),
        )
        self.assertIsNone(bridge.expected_section_fields("请介绍一下这款药品"))

    def test_relevant_verified_chunk_only(self):
        indication, parent_indication = make_doc()
        warnings, parent_warnings = make_doc(
            field="warnings", quote="该药的某条注意事项。",
            section_id="section_2",
        )
        contraindication, parent_contraindication = make_doc(
            field="contraindications", quote="某条禁忌。",
            section_id="section_3",
        )
        docs = [indication, contraindication, warnings]
        parents = {d.metadata["section_id"]: parent
                   for d,parent in zip(docs,[parent_indication,
                                              parent_contraindication,parent_warnings])}
        calls=[]
        def fetch(metadata):
            calls.append(metadata["section_id"])
            return parents[metadata["section_id"]]
        selected, report = bridge.prepare_medical_documents(
            docs,
            "阿托伐他汀钙片与动脉粥样硬化有什么关联？",
            fetch, limit=3,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].metadata["section_field"], "indication")
        self.assertEqual(calls, ["section_1"])
        self.assertTrue(selected[0].metadata["provenance_verified"])
        self.assertEqual(report["skipped_section"], 2)
        self.assertEqual(report["verified"], 1)

    def test_verified_does_not_mean_clinically_approved(self):
        doc, parent = make_doc()
        docs, report = bridge.prepare_medical_documents(
            [doc], "治疗什么疾病？", lambda meta: parent,
        )
        self.assertEqual(report["verified"], 1)
        self.assertEqual(
            docs[0].metadata["source_status"], "third_party_unverified"
        )
        self.assertNotIn("medical_approved", docs[0].metadata)

    def test_tamper_parent_sha_fails_closed(self):
        doc, parent = make_doc()
        new_parent = dict(parent, full_section=parent["full_section"]+"变化")
        accepted, report = bridge.prepare_medical_documents(
            [doc], "有什么关联？", lambda meta: new_parent,
        )
        self.assertEqual(accepted, [])
        self.assertEqual(report["failed_provenance"], 1)
        self.assertEqual(report["reasons"].get("parent_changed"), 1)

    def test_tamper_quote_fails_closed(self):
        doc, parent = make_doc()
        doc.page_content=doc.page_content.replace("危险", "疾病")
        accepted, report = bridge.prepare_medical_documents(
            [doc], "有什么关联？", lambda meta: parent,
        )
        self.assertEqual(len(accepted), 0)
        self.assertEqual(report["reasons"].get("exact_quote_mismatch"), 1)

    def test_mismatch_source_rejected(self):
        doc,parent=make_doc()
        parent["source_url"]="https://another.example.org"
        accepted, report = bridge.prepare_medical_documents(
            [doc], "有什么关联？", lambda meta: parent,
        )
        self.assertEqual(accepted, [])
        self.assertEqual(report["reasons"].get("source_mismatch"), 1)

    def test_neo4j_offline_fails_closed(self):
        doc, parent = make_doc()
        def offline(meta):
            raise ConnectionError("no neo4j")
        accepted, report = bridge.prepare_medical_documents(
            [doc], "与动脉粥样硬化有什么关联？", offline,
        )
        self.assertEqual(accepted, [])
        self.assertEqual(report["failed_provenance"], 1)

    def test_legacy_documents_do_not_claim_verified(self):
        doc=FakeDocument("非新版医疗文本",{"source":"doc.md"})
        accepted, report = bridge.prepare_medical_documents(
            [doc], "疾病关系", lambda meta: None,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(report["legacy_unverified"], 1)
        self.assertNotIn("provenance_verified", doc.metadata)

    def test_collapse_duplicate_graph_parent_but_keep_candidate(self):
        items = [
            {"evidence_type":"unverified_drug_disease_candidate",
             "record_id":"r1","section_id":"s1","id":10},
            {"evidence_type":"unverified_drug_source_section",
             "record_id":"r1","section_id":"s1","id":11},
            {"evidence_type":"json_neo4j_text_verified",
             "record_id":"r1","section_id":"s1","id":12},
        ]
        merged=bridge.collapse_duplicate_parent_sections(items)
        self.assertEqual(len(merged),2)
        self.assertEqual([row["id"] for row in merged],[1,2])
        self.assertEqual(merged[0]["evidence_type"],
                         "unverified_drug_disease_candidate")
        self.assertEqual(merged[1]["evidence_type"],
                         "json_neo4j_text_verified")

    def test_rag_primary_and_self_rag_both_verified(self):
        rag=(ROOT/"rag.py").read_text(encoding="utf-8")
        tree=ast.parse(rag)
        self.assertGreaterEqual(rag.count("prepare_medical_documents("),2)
        self.assertIn("fetch_parent=self._get_medical_parent_section",rag)
        self.assertIn("collapse_duplicate_parent_sections(merged_excerpts)",rag)
        self.assertIn("self.knowledge_service.search_with_rerank(",rag)
        self.assertIn("build_bm25_index()",rag)
        self.assertIn("build_hybrid_retriever()",rag)
        self.assertIn("render_kb_references(",rag)

    def test_all_changes_are_read_only_neo4j(self):
        graph=(ROOT/"drug_graph_retriever.py").read_text(encoding="utf-8")
        tree=ast.parse(graph)
        self.assertIn("def fetch_parent_section", graph)
        self.assertIn("session.execute_read(",graph)
        for verb in ("CREATE (","MERGE (","DETACH DELETE","SET d."):
            self.assertNotIn(verb,graph)

    def test_streamlit_ui_keeps_modes(self):
        ui=(ROOT/"豆馅.py").read_text(encoding="utf-8")
        ast.parse(ui)
        self.assertIn('options=["知识库", "自动", "联网"]',ui)
        self.assertIn("原文回溯：Neo4j Section 精确匹配",ui)


if __name__ == "__main__":
    unittest.main()
