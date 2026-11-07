"""Regression tests for lost composition facts, honest coverage and bounded gaps."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.documents import Document
from evidence_guard import assess_evidence, validate_judge_output
from medical_evidence_coverage import (
    build_answer_targets, required_source_aspects, evidence_bindings, evidence_key,
    start_evidence_flow, record_evidence_stage, record_review, record_answer_usage,
)
from medical_query_plan import build_retrieval_plan
from medical_semantic_retrieval import merge_graph_evidence
from medical_targeted_retry import retry_targeted_evidence
from medical_evidence_bridge import prepare_medical_documents
from medical_quality_acceptance import make_export_payload
from self_rag_retry import reassess_fixed_aspects

A, B = "甲试片", "乙试片"
QUESTION = A + "与" + B + "的成分和相互作用"


def patient(interaction=True):
    return {"medicines": [{"surface": name, "candidate_drug_names": [name], "actual_patient_product_known": True}
                           for name in (A, B)],
            "intent": {"interaction_question": interaction, "fields": ["composition"] + (["drug_interactions"] if interaction else [])}}


def targets(interaction=True):
    p = patient(interaction)
    q = QUESTION if interaction else A + "与" + B + "的成分"
    return build_answer_targets(q, p, build_retrieval_plan(q, p))


def graph(drug=A, ingredient="甲试素", index=1, relation="HAS_INGREDIENT", field="composition"):
    src = {"id": "drug_"+drug, "name": drug, "kind": "Drug"}
    dst = {"id": "ingredient_"+ingredient, "name": ingredient, "kind": "Ingredient"}
    quote = "本品主要成分为" + ingredient + "。"
    return {"id": index, "source": "图路径"+str(index), "evidence_type": "evidence_backed_semantic_graph_path",
            "_retrieval_tool": "search_semantic_graph", "content": "原文："+quote, "graph_verified": True,
            "graph_path": {"id": "p"+str(index), "verified_against_original": True, "nodes": [src, dst],
                           "edges": [{"id": "e"+str(index), "src": src, "dst": dst, "relation": relation,
                                      "section_field": field, "record_id": "record_"+drug, "quote": quote}]}}


def section(drug=A, field="composition", text="甲试素。", index=4):
    return {"id": index, "source": "资料"+drug, "evidence_type": "json_neo4j_text_verified",
            "drug_name": drug, "section_field": field, "record_id": "r"+drug, "section_id": "s"+field+drug,
            "content": "药品："+drug+"\n证据原文："+text, "parent_verified": True,
            "_retrieval_tool": "search_knowledge_base"}


def initial():
    qs = [A+"的成分原文如何描述？", B+"的成分原文如何描述？"]
    old = graph()
    coverage = validate_judge_output({"aspects": [
        {"question": qs[0], "status": "supported", "evidence": [{"id": 1, "quote": "甲试素"}]},
        {"question": qs[1], "status": "insufficient", "evidence": []},
    ]}, [old], required_aspects=qs)
    return [old], coverage


def gap_document():
    text = "乙试素。"
    meta = {"chunk_id": "b_composition", "drug_name": B, "drug_id": "b", "section_field": "composition",
            "record_id": "rb", "section_id": "sb", "source_url": "https://example.invalid/b",
            "source": "乙试资料", "data_origin": "json_matched_to_neo4j_section", "quote_start": 0,
            "quote_end": len(text), "section_sha256": hashlib.sha256(text.encode()).hexdigest()}
    doc = Document(page_content="药品："+B+"\n证据原文："+text, metadata=meta)
    parent = {**meta, "field": "composition", "full_section": text}
    return doc, parent


class CoverageTests(unittest.TestCase):
    def test_symptom_drug_sources_bind_as_candidates_not_recommendations(self):
        question = "发烧了应该吃什么药？"
        symptom = {"surface": "发烧", "kind": "symptom_word", "retrieval_terms": ["发烧", "发热"],
                   "assertion": "user_reported"}
        patient_query = {"medicines": [], "patient_context": [symptom],
                         "intent": {"drug_discovery_question": True, "risk_question": False,
                                    "symptom_question": False, "fields": ["indication"]}}
        plan = build_retrieval_plan(question, patient_query)
        targets = build_answer_targets(question, patient_query, plan)
        self.assertEqual([target["field"] for target in targets], ["symptom_drug_candidates"])
        item = {"evidence_type": "json_neo4j_text_verified", "section_field": "indication",
                "parent_verified": True, "retrieval_task_ids": ["symptom_drug_source_candidates"],
                "content": "药品：测试制剂\n证据原文：用于发热相关症状的说明书适应症候选。"}
        binding, = evidence_bindings(item, targets)
        self.assertEqual(binding["scope"], "symptom_related_candidate_leaflet_source_not_patient_recommendation")
        self.assertTrue(binding["source_verified"])

    def test_disease_symptom_path_is_bound_as_a_partial_answer_target(self):
        question = "感冒的症状是什么？应该吃什么药？与哪些药不能同服？"
        disease = {"surface": "感冒", "kind": "disease_mention", "assertion": "user_reported"}
        patient_query = {"medicines": [], "patient_context": [disease],
                         "intent": {"symptom_question": True, "drug_discovery_question": True,
                                    "interaction_question": True, "fields": []}}
        plan = {"status": "planned", "fields": ["indication"]}
        answer_targets = build_answer_targets(question, patient_query, plan)
        self.assertEqual([target["field"] for target in answer_targets],
                         ["symptoms", "disease_drug_candidates", "disease_interactions"])
        quote = "咳嗽、咽痛"
        path = {"verified_against_original": True, "nodes": [
                    {"id": "d", "kind": "Disease", "name": "感冒"},
                    {"id": "s", "kind": "Symptom", "name": quote}],
                "edges": [{"id": "e", "relation": "HAS_SYMPTOM", "section_field": "symptom",
                           "src": {"name": "感冒"}, "quote": quote}]}
        item = {"id": 1, "evidence_type": "evidence_backed_semantic_graph_path",
                "graph_verified": True, "graph_path": path, "content": "原文：" + quote}
        bound = evidence_bindings(item, answer_targets)
        self.assertEqual([entry["target_id"] for entry in bound], ["target_1"])

    def test_disease_classification_is_planned_and_bound_separately_from_symptoms(self):
        question = "荨麻疹属于什么疾病类型？"
        disease = {"surface": "荨麻疹", "kind": "disease_mention",
                   "assertion": "user_reported", "retrieval_terms": ["荨麻疹"]}
        patient_query = {"medicines": [], "patient_context": [disease],
                         "intent": {"classification_question": True,
                                    "symptom_question": True, "fields": []}}
        plan = build_retrieval_plan(question, patient_query)
        self.assertEqual([task["id"] for task in plan["tasks"]],
                         ["disease_classification"])
        targets = build_answer_targets(question, patient_query, plan)
        self.assertEqual([target["field"] for target in targets],
                         ["disease_classification"])
        category = "皮肤性病科"
        path = {"verified_against_original": True, "nodes": [
                    {"id": "d", "kind": "Disease", "name": "荨麻疹"},
                    {"id": "c", "kind": "DiseaseCategory", "name": category}],
                "edges": [{"id": "e", "relation": "DISEASE_CATEGORY_MENTION",
                           "section_field": "category", "src": {"name": "荨麻疹"},
                           "dst": {"name": category}, "quote": category}]}
        item = {"id": 1, "evidence_type": "evidence_backed_semantic_graph_path",
                "graph_verified": True, "graph_path": path,
                "content": "原文：" + category}
        bound = evidence_bindings(item, targets)
        self.assertEqual([entry["target_id"] for entry in bound], ["target_1"])
        self.assertIn("classification", bound[0]["scope"])

    def test_two_compositions_and_pair_form_three_required_aspects(self):
        p = patient()
        required = required_source_aspects(QUESTION, p, targets())
        self.assertEqual(len(required), 3)
        self.assertIn(A+"的成分原文如何描述？", required)
        self.assertIn(B+"的成分原文如何描述？", required)
        self.assertTrue(any(A+"与"+B in q and "相互作用" in q for q in required))
        self.assertFalse(any("注意事项" in q for q in required))

    def test_patient_clinical_question_is_not_reduced_to_fields(self):
        for question in (QUESTION+"，我能一起吃吗", QUESTION+"，为什么鼻子疼", QUESTION+"，作用机制是什么"):
            self.assertEqual(required_source_aspects(question, patient(), targets()), [])
        p = patient()
        p["medicines"][0]["actual_patient_product_known"] = False
        self.assertEqual(required_source_aspects(QUESTION, p, targets()), [])

    def test_separate_field_scopes_are_not_turned_into_a_shared_checklist(self):
        question = A + "的成分和" + B + "的相互作用"
        self.assertEqual(required_source_aspects(question, patient(), targets()), [])

    def test_merge_preserves_unique_composition_over_redundant_paths(self):
        redundant = [graph(index=i) for i in range(1, 7)]
        unique = section(B, text="乙试素。")
        merged = merge_graph_evidence([unique], redundant, 2, targets=targets(False))
        self.assertEqual(len(merged), 2)
        self.assertTrue(any(i.get("drug_name") == B for i in merged))
        bound = {b["target_id"] for item in merged for b in evidence_bindings(item, targets(False))}
        self.assertEqual(len(bound), 2)

    def test_independent_composition_paths_do_not_cover_interaction(self):
        pair = next(t for t in targets() if t["pair"])
        self.assertEqual(evidence_bindings(graph(), [pair]), [])
        item = graph(relation="INTERACTION_MENTION", field="drug_interactions")
        self.assertEqual(evidence_bindings(item, [pair]), [])

    def test_mechanism_target_accepts_verified_mechanism_edges_only(self):
        p = patient()
        p["intent"]["mechanism_question"] = True
        q = QUESTION + "背后的药理机制是什么"
        mechanism_targets = build_answer_targets(q, p, build_retrieval_plan(q, p))
        mechanism = next(t for t in mechanism_targets if t["field"] == "mechanism")
        direct = graph(relation="INTERACTION_MENTION", field="drug_interactions")
        self.assertEqual(evidence_bindings(direct, [mechanism]), [])
        supported = graph(relation="MECHANISM_OUTCOME_MENTION", field="drug_interactions")
        self.assertEqual(evidence_bindings(supported, [mechanism]), [])
        supported["graph_path"]["nodes"].append({"id": "drug_"+B, "name": B, "kind": "Drug"})
        self.assertEqual(evidence_bindings(supported, [mechanism])[0]["scope"],
                         "verified_mechanism_source_mention_not_causal_validation")

    def test_mechanism_target_accepts_verified_original_mechanism_section(self):
        p = patient()
        p["intent"]["mechanism_question"] = True
        q = QUESTION + "为什么"
        mechanism = next(t for t in build_answer_targets(q, p, build_retrieval_plan(q, p))
                         if t["field"] == "mechanism")
        row = section(A, "drug_interactions", A+"与"+B+"合用时抑制CYP3A4，使血药浓度升高。")
        self.assertTrue(evidence_bindings(row, [mechanism]))
        row["parent_verified"] = False
        self.assertFalse(evidence_bindings(row, [mechanism])[0]["source_verified"])

    def test_only_actual_original_text_counts_for_pair_section(self):
        pair = next(t for t in targets() if t["pair"])
        row = section(A, "drug_interactions", "尚不明确。")
        row["content"] = "检索药物："+A+"、"+B+"\n证据原文：尚不明确。"
        self.assertEqual(evidence_bindings(row, [pair]), [])
        row = section(A, "drug_interactions", "与"+B+"合用时需核对条件。")
        self.assertTrue(evidence_bindings(row, [pair]))

    def test_unverified_graph_and_other_formulation_do_not_cover(self):
        row = graph()
        row["graph_verified"] = False
        self.assertEqual(evidence_bindings(row, targets(False)), [])
        self.assertEqual(evidence_bindings(section(A+"缓释片"), targets(False)), [])

    def test_source_tracking_survives_renumbering_and_reports_exclusion(self):
        flow = start_evidence_flow(targets(False))
        a, b = graph(), section(B)
        record_evidence_stage(flow, "recalled", [a,b])
        moved = {**a, "id": 8}
        self.assertEqual(evidence_key(a), evidence_key(moved))
        record_evidence_stage(flow, "merged", [moved], reason="capacity")
        self.assertEqual(len(flow["transitions"][0]["excluded"]), 1)
        self.assertEqual(flow["transitions"][0]["excluded"][0]["reason"], "capacity")
        self.assertEqual(make_export_payload(QUESTION, "", {"evidence_flow":flow})["trace"]["evidence_flow"], flow)

    def test_path_citation_is_not_claimed_as_clinical_support_for_all_targets(self):
        flow = start_evidence_flow(targets(False))
        row = graph()
        record_review(flow, [row], {"evidence_ids":[1], "aspects":[], "decision":"partial"})
        record_answer_usage(flow, "资料记载[1]。\n\n**参考资料**\n- [2] 其他")
        self.assertEqual(flow["answer"][0]["cited_candidate_ids"], [1])
        self.assertEqual(flow["answer"][1]["cited_candidate_ids"], [])
        self.assertNotIn("supported", str(flow["answer"]))

    def test_reviewer_missing_composition_does_not_erase_verified_literal_facts(self):
        p, ts = patient(), targets()
        required = required_source_aspects(QUESTION, p, ts)
        excerpts = [graph(), graph(B, "乙试素", 2)]
        def ask(system, payload):
            data = json.loads(payload)
            self.assertEqual(data["required_aspects"], required)
            return {"aspects": [{"question": q, "status": "insufficient", "evidence": []} for q in required]}
        report = assess_evidence(None, QUESTION, excerpts, ask_fn=ask, required_aspects=required)
        self.assertEqual(report["decision"], "partial")
        self.assertEqual(len(report["answer_questions"]), 2)
        self.assertTrue(all("成分" in q for q in report["answer_questions"]))
        self.assertEqual(report["evidence_ids"], [1,2])
        self.assertTrue(any("相互作用" in q for q in report["refused_questions"]))

    def test_reassessment_cannot_silently_drop_required_aspect(self):
        excerpts, coverage = initial()
        raw = {"aspects": [{"question": coverage["aspects"][0]["question"], "status": "supported",
                            "evidence": [{"id":1,"quote":"甲试素"}]}]}
        model = SimpleNamespace(bind=lambda **k: SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=json.dumps(raw))))
        report = reassess_fixed_aspects(model=model, question=QUESTION, excerpts=excerpts,
                                        retrieval_queries=[], original_coverage=coverage)
        self.assertEqual(len(report["aspects"]), 2)
        self.assertEqual(report["aspects"][1]["status"], "insufficient")


class TargetedRetryTests(unittest.TestCase):
    def run_retry(self, *, search=None, verify=None, judge=None, **budgets):
        excerpts, coverage = initial()
        doc, parent = gap_document()
        def actual_verify(docs, routing):
            return prepare_medical_documents(docs, routing, lambda m: parent)
        def actual_judge(**kwargs):
            return validate_judge_output({"aspects": [
                {"question": coverage["aspects"][0]["question"], "status":"supported", "evidence":[{"id":1,"quote":"甲试素"}]},
                {"question": coverage["aspects"][1]["question"], "status":"supported", "evidence":[{"id":2,"quote":"乙试素。"}]},
            ]}, kwargs["excerpts"], required_aspects=[a["question"] for a in coverage["aspects"]])
        with contextlib.redirect_stdout(io.StringIO()):
            return retry_targeted_evidence(model=None, question=A+"与"+B+"的成分", targets=targets(False),
                excerpts=excerpts, coverage=coverage, original_queries=[], search=search or (lambda *a,**k:[doc]),
                verify=verify or actual_verify, judge=judge or actual_judge, **budgets)

    def test_only_gap_is_searched_once_without_rerank_and_old_fact_survives(self):
        calls = []
        doc, _ = gap_document()
        def search(q, **kwargs):
            calls.append(q)
            self.assertIn(B, q)
            self.assertNotIn(A, q)
            return [doc]
        result = self.run_retry(search=search)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "improved")
        self.assertEqual(result["coverage"]["decision"], "answer")
        self.assertEqual(result["execution"]["rerank_calls"], 0)
        self.assertTrue(result["excerpts"][1]["parent_verified"])
        self.assertEqual(result["excerpts"][0]["id"], 1)

    def test_failed_parent_provenance_cannot_be_added_or_reviewed(self):
        def fail_verify(docs, routing):
            return prepare_medical_documents(docs, routing, lambda m: None)
        result = self.run_retry(verify=fail_verify, judge=lambda **kw: self.fail("unverified source"))
        self.assertEqual(result["status"], "no_new_evidence")
        self.assertEqual(result["new_chunks"], 0)
        self.assertEqual(result["coverage"]["decision"], "partial")
        self.assertEqual(result["execution"]["validation"]["failed_provenance"], 1)
        self.assertEqual(result["execution"]["validation"]["decisions"][0]["reason"], "parent_missing")

    def test_slow_provenance_is_inside_recall_wait_budget(self):
        release = threading.Event()
        def slow_verify(docs, routing):
            release.wait(timeout=1)
            return docs, {}
        started = time.monotonic()
        try:
            result = self.run_retry(verify=slow_verify, recall_seconds=.02)
        finally:
            release.set()
        self.assertLess(time.monotonic()-started, .5)
        self.assertEqual(result["status"], "retrieval_timeout")
        self.assertEqual(result["new_chunks"], 0)

    def test_review_timeout_preserves_original_coverage(self):
        release = threading.Event()
        def slow_judge(**kwargs):
            release.wait(timeout=1)
            raise RuntimeError("late result ignored")
        try:
            result = self.run_retry(judge=slow_judge, review_seconds=.02)
        finally:
            release.set()
        self.assertEqual(result["status"], "judge_error")
        self.assertEqual(result["coverage"]["decision"], "partial")
        self.assertEqual(len(result["excerpts"]), 1)
        self.assertTrue(result["execution"]["review_timed_out"])
        self.assertEqual(result["new_chunks"], 0)
        self.assertEqual(result["execution"]["candidate_new_chunks"], 1)

    def test_reviewer_cannot_gain_new_fact_by_dropping_old_fact(self):
        _, original = initial()
        revised = {**original, "answer_questions": [original["aspects"][1]["question"]],
                   "aspects": [{**original["aspects"][0], "status":"insufficient"},
                               {**original["aspects"][1], "status":"supported"}]}
        result = self.run_retry(judge=lambda **kw: revised)
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(result["coverage"], original)

    def test_already_available_source_does_not_trigger_redundant_retry(self):
        excerpts, coverage = initial()
        excerpts.append(section(B))
        result = retry_targeted_evidence(model=None, question=QUESTION, targets=targets(False),
            excerpts=excerpts, coverage=coverage, original_queries=[],
            search=lambda *a,**kw: self.fail("no source gap"), verify=None, judge=None)
        self.assertEqual(result["status"], "skipped_no_source_gap")


class RuntimeGoalIntegrationTests(unittest.TestCase):
    def test_full_backend_keeps_both_compositions_and_records_final_citations(self):
        stubs = {}
        for name, symbol in (("knowledge_base", "KnowledgeBaseService"), ("vector_stores", "VectorStoreService")):
            module = ModuleType(name)
            setattr(module, symbol, type(symbol, (), {}))
            stubs[name] = module
        with patch.dict(sys.modules, stubs):
            import rag
        from langchain_core.messages import AIMessageChunk
        seen = {}
        p = patient()
        plan = build_retrieval_plan(QUESTION, p)
        semantic = [graph(), graph(B, "乙试素", 2)]
        report = {"status":"ok", "query_understanding":p, "retrieval_plan":plan,
                  "multi_relation_paths":0, "returned_paths":2}
        class Model:
            def bind(self, **kwargs):
                return self
            def invoke(self, messages):
                payload = json.loads(messages[-1].content)
                seen["checklist"] = payload["required_aspects"]
                seen["source_index"] = payload["source_coverage_index_not_sufficiency"]
                return SimpleNamespace(content=json.dumps({"aspects":[
                    {"question":q,"status":"insufficient","evidence":[]} for q in payload["required_aspects"]]}))
            def stream(self, messages):
                seen["generation"] = messages
                yield AIMessageChunk(content="甲试片的资料记载主要成分为甲试素[1]。乙试片的资料记载主要成分为乙试素[2]。")
        service = rag.RagService.__new__(rag.RagService)
        service.chat_model = Model()
        service.knowledge_service = SimpleNamespace(search_hybrid=lambda *a,**k:[])
        service._call_tool = lambda *a,**k:(json.dumps({"found":False,"excerpts":[]}), True)
        trace = {}
        with patch.object(rag,"search_semantic_graph",return_value=(semantic,report)), \
             patch.object(rag,"search_multihop_enhanced_disease_encyclopedia",return_value=([],{"route":"NONE"})), \
             patch.object(rag,"_p3063_attach_multihop_path_cards"), patch.object(rag,"get_history"), \
             patch.dict(os.environ,{"DOUXIAN_KB_MODE":"strict"}), contextlib.redirect_stdout(io.StringIO()):
            answer = ''.join(service._stream_agent({"input":QUESTION,"mode":"知识库","trace":trace,"history":[]},"fixture"))
        self.assertEqual(len(seen["checklist"]),3)
        self.assertEqual(len(trace["evidence_coverage"]["answer_questions"]),2)
        self.assertIn("乙试素[2]",answer)
        self.assertTrue(seen["source_index"])
        self.assertIn("逐项覆盖",seen["generation"][0].content)
        self.assertEqual(len(trace["evidence_flow"]["answer"]), len(targets()))
        self.assertTrue(any(row["cited_candidate_ids"] == [2] for row in trace["evidence_flow"]["answer"]))
        self.assertEqual(trace["query_execution"]["source_gap"]["rerank_calls"],0)


if __name__ == "__main__":
    unittest.main()
