"""Offline behavioral checks. All fixture drugs/diseases are fictional."""
from __future__ import annotations
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from medical_semantic_graph import build_graph, clause_span, ingredient_spans, index_path, qualifiers
from medical_semantic_retrieval import (
    EVIDENCE_TYPE, graph_metadata, mark_graph_usage, merge_graph_evidence,
    search_semantic_graph, semantic_graph_html,
)
from evidence_guard import assess_evidence, validate_judge_output
from medical_quality_acceptance import make_export_payload


def fixture(root, *, interaction=True, shared_target=False, variant=False):
    graph = root / "kg_output_clean_v2"
    graph.mkdir()
    drugs = [{"id": "drug_a", "name": "甲药片"}, {"id": "drug_b", "name": "乙药胶囊"},
             {"id": "drug_c", "name": "丙药片"}, {"id": "drug_d", "name": "丁药片"}]
    records = [{"id": "r_" + v["id"], "drug_id": v["id"], "source_url": "https://example.invalid/fixture/" + v["id"],
                "authority_status": "third_party_unverified"} for v in drugs]
    sections = []
    def section(sid, drug, field, text, record=None):
        sections.append({"id": sid, "record_id": record or "r_" + drug, "field": field,
                         "text": text, "placeholder": False})
    section("a_comp", "drug_a", "composition", "本品主要成分为甲试素。")
    section("b_comp", "drug_b", "composition", "本品主要成分为乙试素。")
    section("c_comp", "drug_c", "composition", "共试素。")
    section("d_comp", "drug_d", "composition", "丁试素。")
    section("a_ind", "drug_a", "indication", "虚构测试资料提及试验疾病，仅供软件测试。")
    section("a_contra", "drug_a", "contraindications", "妊娠期患者禁用本品。")
    if interaction:
        section("b_interact", "drug_b", "drug_interactions", "与甲试素同用尚未见不良相互作用，但临床相关性尚不明确。")
    if shared_target:
        section("a_shared", "drug_a", "drug_interactions", "与共试素同用可能出现虚构测试风险。")
        section("b_shared", "drug_b", "drug_interactions", "与共试素同用可能出现虚构测试风险。")
    if variant:
        records.append({"id": "r_a_variant", "drug_id": "drug_a", "source_url": "", "authority_status": "third_party_unverified"})
        sections[:] = [s for s in sections if s["id"] != "a_contra"]
        section("a_contra_other_record", "drug_a", "contraindications", "妊娠期患者禁用本品。", "r_a_variant")
    for name, rows in (("drugs", drugs), ("source_records", records), ("sections", sections)):
        (graph / (name + ".jsonl")).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in rows), encoding="utf-8")
    quote = next(s["text"] for s in sections if s["id"] == "a_ind")
    start = quote.index("试验疾病")
    row = {"drug_name": "甲药片", "disease_name": "试验疾病", "source_record_id": "r_drug_a",
           "section_id": "a_ind", "evidence_quote": quote, "quote_start": 0,
           "quote_end": len(quote), "disease_start": start, "disease_end": start + 4}
    with (graph / "disease_candidates_full.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    disease = root / "data/raw/drug_db/MedGraphRAG-Drug-DB/medical.json"
    disease.parent.mkdir(parents=True)
    disease.write_text(json.dumps({"_id": {"$oid": "fixture-disease"}, "name": "试验疾病", "symptom": ["胸痛", "虚构症状"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    return build_graph(root)


class SemanticGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="semantic_graph_test_")
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def query(self, question, **kwargs):
        return search_semantic_graph(question, project_root=self.root, **kwargs)

    def rebuild(self, **kwargs):
        import shutil
        shutil.rmtree(self.root / "kg_output_clean_v2")
        shutil.rmtree(self.root / "data")
        fixture(self.root, **kwargs)

    def test_product_ingredient_interaction_join(self):
        evidence, report = self.query("甲药片和乙药胶囊的成分与相互作用")
        self.assertEqual(report["status"], "ok")
        self.assertTrue(any(p["hops"] >= 2 and {"HAS_INGREDIENT", "INTERACTION_MENTION"} <= set(p["relations"]) for p in report["paths"]))
        self.assertTrue(all({"甲药片", "乙药胶囊"} <= {n["name"] for n in p["nodes"]} for p in report["paths"]))
        for path in report["paths"]:
            for edge in path["edges"]:
                if edge["relation"] == "INTERACTION_MENTION":
                    self.assertTrue(edge["qualifiers"]["negative_or_absent"])
                    self.assertTrue(edge["qualifiers"]["uncertain"])
                    self.assertIn("尚未见", edge["quote"])
        self.assertEqual(evidence[0]["evidence_type"], EVIDENCE_TYPE)

    def test_four_hop_heterogeneous_path(self):
        _, report = self.query("胸痛相关药品与乙药胶囊的成分和相互作用有哪些？")
        self.assertEqual(report["status"], "ok")
        self.assertTrue(any({"HAS_SYMPTOM", "INDICATION_MENTION", "HAS_INGREDIENT", "INTERACTION_MENTION"} <= set(p["relations"]) and p["hops"] >= 4 for p in report["paths"]))

    def test_shared_third_interaction_target_does_not_prove_pair(self):
        self.rebuild(interaction=False, shared_target=True)
        evidence, report = self.query("甲药片与乙药胶囊的相互作用")
        self.assertEqual(evidence, [])
        self.assertEqual(report["status"], "no_connected_evidence_path")

    def test_disconnected_named_entity_never_dropped(self):
        evidence, report = self.query("甲药片与丁药片的相互作用")
        self.assertFalse(evidence)
        self.assertEqual(len(report["plan"]["anchors"]), 2)

    def test_no_name_morphology_or_unknown_salt_equivalence(self):
        evidence, report = self.query("甲试素钠的相互作用")
        self.assertFalse(evidence)
        self.assertEqual(report["status"], "no_exact_entity")

    def test_unknown_named_product_in_pair_is_not_silently_dropped(self):
        evidence, report = self.query("甲药片和未收录药片的相互作用")
        self.assertFalse(evidence)
        self.assertEqual(report["status"], "unresolved_named_entity")

    def test_class_reference_is_not_promoted_to_exact_ingredient(self):
        path = self.root / "kg_output_clean_v2/sections.jsonl"
        text = path.read_text(encoding="utf-8").replace("与甲试素同用", "与甲试素类同用")
        path.write_text(text, encoding="utf-8")
        build_graph(self.root)
        evidence, report = self.query("甲药片与乙药胶囊的相互作用")
        self.assertFalse(evidence)
        self.assertEqual(report["status"], "no_connected_evidence_path")

    def test_single_drug_does_not_collect_other_drugs_via_common_condition(self):
        evidence, report = self.query("甲药片的成分和禁忌")
        self.assertTrue(evidence)
        self.assertTrue(any({"HAS_INGREDIENT", "CONTRAINDICATION_MENTION"} <= set(p["relations"]) for p in report["paths"]))
        self.assertTrue(all(n["name"] == "甲药片" for p in report["paths"] for n in p["nodes"] if n["kind"] == "Drug"))

    def test_specific_condition_does_not_append_unrequested_branches(self):
        evidence, report = self.query("妊娠期患者使用甲药片有什么禁忌？")
        self.assertTrue(evidence)
        self.assertTrue(all(p["hops"] == 1 and p["relations"] == ["CONTRAINDICATION_MENTION"] for p in report["paths"]))

    def test_records_of_same_drug_must_not_be_stitched(self):
        self.rebuild(variant=True)
        _, report = self.query("甲药片的成分和禁忌")
        self.assertFalse(any({"HAS_INGREDIENT", "CONTRAINDICATION_MENTION"} <= set(p["relations"]) for p in report["paths"]))
        self.assertGreater(report["stats"].get("source_record_conflicts_rejected", 0), 0)

    def test_source_change_blocks_stale_graph(self):
        path = self.root / "kg_output_clean_v2/sections.jsonl"
        path.write_bytes(path.read_bytes().replace("甲试素".encode(), "戊试素".encode()))
        evidence, report = self.query("甲药片的成分")
        self.assertFalse(evidence)
        self.assertEqual(report["status"], "source_verification_blocked")

    def test_index_change_blocks_evidence(self):
        conn = sqlite3.connect(index_path(self.root))
        conn.execute("UPDATE edges SET relation='HAS_SYMPTOM'")
        conn.commit()
        conn.close()
        evidence, report = self.query("甲药片的成分")
        self.assertFalse(evidence)
        self.assertEqual(report["status"], "source_verification_blocked")

    def test_hop_and_expansion_budgets(self):
        evidence, report = self.query("甲药片和乙药胶囊的相互作用", max_hops=1)
        self.assertFalse(evidence)
        _, report = self.query("甲药片的成分与禁忌", max_expansions=1, max_neighbors=1)
        self.assertLessEqual(report["stats"]["expanded_states"], 1)
        self.assertTrue(report["truncated"])

    def test_conservative_composition_parser(self):
        spans = ingredient_spans("本品主要成分为甲试素和乙试素；辅料为丙试素。")
        self.assertEqual([s[0] for s in spans], ["甲试素", "乙试素"])
        self.assertFalse(ingredient_spans("本品为一种药物，不能从名称推断成分。"))
        self.assertEqual(ingredient_spans("盐酸甲试素。"), [("盐酸甲试素", 0, 5)])

    def test_following_uncertainty_sentence_is_not_dropped(self):
        text = "与甲试素同用会影响指标。该相互作用的临床相关性尚不明确。"
        start, end = clause_span(text, 1, 4)
        self.assertEqual(text[start:end], text)
        self.assertTrue(qualifiers(text[start:end])["uncertain"])

    def test_self_ingredient_is_not_an_interacting_partner(self):
        path = self.root / "kg_output_clean_v2/sections.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": "a_self", "record_id": "r_drug_a", "field": "drug_interactions", "text": "本品与丙磺舒同用可使甲试素血浓度增加。"}, ensure_ascii=False) + "\n")
        build_graph(self.root)
        conn = sqlite3.connect(index_path(self.root))
        try:
            self.assertEqual(conn.execute("SELECT count(*) FROM edges WHERE source_id='a_self'").fetchone()[0], 0)
        finally:
            conn.close()

    def test_source_only_citations_and_short_structured_symptom(self):
        evidence, _ = self.query("试验疾病的症状")
        item = next(e for e in evidence if e["graph_path"]["edges"][0]["quote"] == "胸痛")
        raw = {"aspects": [{"question": "试验疾病的症状", "status": "supported",
                            "evidence": [{"id": item["id"], "quote": "胸痛"}]}]}
        self.assertEqual(validate_judge_output(raw, evidence)["decision"], "answer")
        raw["aspects"][0]["evidence"][0]["quote"] = "解释边界：仅连接资料中的陈述"
        self.assertEqual(validate_judge_output(raw, evidence)["decision"], "refuse")
        # Exercise the real assess_evidence serialization boundary too.
        raw["aspects"][0]["evidence"][0]["quote"] = "胸痛"
        result = assess_evidence(model=None, question="试验疾病的症状", excerpts=evidence,
                                 ask_fn=lambda *args: raw)
        self.assertEqual(result["decision"], "answer")

    def test_whole_path_citation_export_and_html(self):
        evidence, report = self.query("甲药片与乙药胶囊的相互作用")
        merged = merge_graph_evidence([{"content": "ordinary"}], evidence, 8)
        self.assertEqual([i["id"] for i in merged], list(range(1, len(merged)+1)))
        self.assertIn("graph_path", graph_metadata(merged[0]))
        trace = {"semantic_graph": report, "knowledge_evidence": merged}
        mark_graph_usage(trace, merged, [1], "这是虚构测试[1]。")
        exported = make_export_payload("fixture", "test[1]", trace)
        self.assertTrue(exported["trace"]["semantic_graph"]["evidence_bindings"][0]["cited_in_answer"])
        report["paths"][0]["edges"][0]["quote"] = "<script>alert(1)</script>"
        rendered = semantic_graph_html(trace)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_complete_ingredient_name_short_quote_and_missed_field(self):
        evidence, _ = self.query("甲药片和乙药胶囊的成分与相互作用")
        item = next(e for e in evidence if any(v["relation"] == "HAS_INGREDIENT" and v["src"]["name"] == "甲药片" for v in e["graph_path"]["edges"]))
        aspect = {"question": "甲药片的成分", "status": "supported", "evidence": [{"id": item["id"], "quote": "甲试素"}]}
        result = validate_judge_output({"aspects": [aspect]}, evidence)
        self.assertEqual(result["decision"], "answer")
        aspect.update(status="insufficient", evidence=[])
        result = assess_evidence(model=None, question="甲药片和乙药胶囊的成分与相互作用", excerpts=evidence,
                                 ask_fn=lambda *args: {"aspects": [aspect]})
        self.assertEqual(result["decision"], "answer")
        self.assertEqual(result["aspects"][0]["validation_method"], "verified_composition_field_lookup")
        self.assertIn("甲试素", result["aspects"][0]["quotes"][0]["quote"])
        for question in ("甲药片与乙药胶囊的相互作用", "甲药片的成分是否可以治疗试验疾病", "甲药片的成分是否安全"):
            aspect["question"] = question
            self.assertEqual(validate_judge_output({"aspects": [aspect]}, evidence)["decision"], "refuse")

    def test_visual_arrows_keep_assertion_direction_and_no_empty_list(self):
        _, report = self.query("甲药片和乙药胶囊的成分与相互作用")
        from html.parser import HTMLParser
        class Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.directions = []
                self.node_count = 0
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                classes = attrs.get("class", "").split()
                if "kb-sg-arrow" in classes:
                    self.directions.append("right" if "kb-sg-arrow-right" in classes else "left")
                if "kb-sg-node" in classes:
                    self.node_count += 1
        rendered = semantic_graph_html({"semantic_graph": report})
        parser = Parser()
        parser.feed(rendered)
        expected = ["right" if edge["src"]["id"] == path["nodes"][i]["id"] else "left"
                    for path in report["paths"] for i, edge in enumerate(path["edges"])]
        self.assertEqual(parser.directions, expected)
        self.assertEqual(parser.node_count, sum(len(p["nodes"]) for p in report["paths"]))
        self.assertNotIn("<li>", rendered)
        self.assertIn("逐边原文与来源", rendered)

    def test_legacy_source_discovery_is_collapsed_without_losing_report(self):
        from medical_p3072_interaction_ui_v117 import interaction_full_corpus_html
        trace = {"p3072_full_corpus_interactions": {
            "stage": "P3-07.2", "status": "NO_PAIR_SPECIFIC_EVIDENCE_FOUND",
            "source_evidence": [], "index_snapshot": {},
        }}
        rendered = interaction_full_corpus_html(trace)
        self.assertTrue(rendered.startswith('<details class="kb-p3072-archive">'))
        self.assertIn("包含其他剂型，不参与本轮回答", rendered)
        self.assertNotIn('<details class="kb-p3072-archive" open', rendered)
        self.assertEqual(trace["p3072_full_corpus_interactions"]["status"], "NO_PAIR_SPECIFIC_EVIDENCE_FOUND")

    def test_backend_supplies_actual_paths_before_generation_and_persists_cites(self):
        # Isolate unused vector/database services; execute the real RagService
        # orchestration, real evidence guard and real LangChain message objects.
        stubs = {}
        for name, symbol in (("knowledge_base", "KnowledgeBaseService"), ("vector_stores", "VectorStoreService")):
            module = ModuleType(name)
            setattr(module, symbol, type(symbol, (), {}))
            stubs[name] = module
        try:
            with patch.dict(sys.modules, stubs):
                import rag
                from langchain_core.messages import AIMessageChunk
        except ModuleNotFoundError as exc:
            self.skipTest("Run backend integration in RAG environment: " + str(exc))
        seen = []
        class Model:
            def bind(self, **kwargs):
                return self
            def invoke(self, messages):
                payload = json.loads(messages[-1].content)
                item = payload["knowledge_evidence"][0]
                question = (payload.get("required_aspects") or [payload["original_question"]])[0]
                return SimpleNamespace(content=json.dumps({"aspects": [{
                    "question": question, "status": "supported",
                    "evidence": [{"id": item["id"], "quote": e["quote"]} for e in item["graph_path"]["edges"]],
                }]}, ensure_ascii=False))
            def stream(self, messages):
                seen.extend(messages)
                yield AIMessageChunk(content="虚构测试资料：合用的临床相关性尚不明确[1]。")
        service = rag.RagService.__new__(rag.RagService)
        service.chat_model = Model()
        service.knowledge_service = None
        service._call_tool = lambda *args: (json.dumps({"found": False, "excerpts": [], "status": "no_match"}), True)
        trace = {}
        def semantic(question, **kwargs):
            return self.query(question)
        for kb_mode in ("fast", "strict"):
            seen.clear()
            trace.clear()
            with patch.object(rag, "search_multihop_enhanced_disease_encyclopedia", return_value=([], {"route": "NONE"})), \
                 patch.object(rag, "_p3063_attach_multihop_path_cards"), \
                 patch.object(rag, "search_semantic_graph", side_effect=semantic), \
                 patch.object(rag, "graph_intent", return_value=True), \
                 patch.object(rag, "get_history"), \
                 patch.dict(os.environ, {"DOUXIAN_KB_MODE": kb_mode}), \
                 contextlib.redirect_stdout(io.StringIO()):
                answer = "".join(service._stream_agent({"input": "甲药片和乙药胶囊的相互作用", "mode": "自动", "trace": trace, "history": []}, "fixture"))
            self.assertIn("[1]", answer)
            self.assertTrue(seen, trace)
            self.assertIn("尚未见不良相互作用", seen[-1].content)
            self.assertIn("HAS_SYMPTOM 不能用于确诊", seen[0].content)
            self.assertTrue(trace["knowledge_evidence"][0]["graph_path"]["verified_against_original"])
            self.assertTrue(trace["semantic_graph"]["evidence_bindings"][0]["cited_in_answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
