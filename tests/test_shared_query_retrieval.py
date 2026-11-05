"""Shared planning, bounded execution and evidence integrity regressions."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.documents import Document
from medical_query_plan import build_retrieval_plan, section_routing_question
from medical_query_execution import execute_retrieval_plan
from medical_semantic_graph import build_graph
from medical_semantic_retrieval import graph_insufficiency_message, search_semantic_graph
from medical_semantic_graph_view import relation_label
from medical_query_understanding import patient_intent
from medical_quality_acceptance import make_export_payload
from medical_evidence_bridge import expected_section_fields, prepare_medical_documents


QUESTION = "我喷了甲试鼻喷雾剂好几天了，鼻子还是堵，今天鼻子里有点疼，这是药不管用还是其他问题"


def understanding():
    return {"medicines": [{"surface": "甲试鼻喷雾剂", "actual_patient_product_known": True,
                            "candidate_drug_names": ["甲试鼻喷雾剂"]}],
            "intent": {"adverse_event_question": True, "efficacy_question": True,
                       "fields": ["adverse_reactions", "warnings", "indication", "dosage"]},
            "patient_context": [{"retrieval_terms": ["鼻塞", "鼻腔疼痛"], "assertion": "user_reported"},
                                {"retrieval_terms": ["糖尿病"], "assertion": "negated"}],
            "timing": [{"surface": "好几天", "medicine_surface": "甲试鼻喷雾剂"}]}


def doc(key, field="warnings", drug="甲试鼻喷雾剂"):
    return Document(page_content=f"虚构测试原文 {key}", metadata={"chunk_id": key, "section_field": field, "drug_name": drug})


def fixture(root):
    export = root / "kg_output_clean_v2"
    export.mkdir()
    drug = {"id": "drug_a", "name": "甲试鼻喷雾剂", "aliases": []}
    record = {"id": "record_a", "drug_id": "drug_a", "source_url": "https://example.invalid/a"}
    sections = [{"id": "s"+field, "record_id": "record_a", "field": field, "text": text}
                for field, text in [("composition", "甲试素。"),
                                    ("warnings", "儿童应在医师指导下使用，皮疹患者慎用。"),
                                    ("adverse_reactions", "可能出现皮疹，原因尚不清楚。"),
                                    ("dosage", "仅供虚构测试，不作为医疗建议。")]]
    for name, rows in [("drugs", [drug]), ("source_records", [record]), ("sections", sections)]:
        (export / (name + ".jsonl")).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding="utf-8")
    (export / "disease_candidates_full.csv").write_text("section_id\n", encoding="utf-8")
    disease = root / "data/raw/drug_db/MedGraphRAG-Drug-DB/medical.json"
    disease.parent.mkdir(parents=True)
    disease.write_text(json.dumps({"name": "试验病", "symptom": ["皮疹"]}, ensure_ascii=False)+'\n', encoding="utf-8")
    build_graph(root)


class SharedPlanningTests(unittest.TestCase):
    def test_three_complementary_goals_keep_product_symptoms_and_time(self):
        plan = build_retrieval_plan(QUESTION, understanding())
        self.assertEqual([t["id"] for t in plan["tasks"]], ["adverse_event", "efficacy", "administration"])
        self.assertTrue(all("甲试鼻喷雾剂" in t["query"] for t in plan["tasks"]))
        self.assertIn("鼻腔疼痛", plan["tasks"][0]["query"])
        self.assertIn("好几天", plan["tasks"][1]["query"])
        self.assertNotIn("糖尿病", str(plan["tasks"]))
        self.assertNotIn("14天", str(plan["tasks"]))
        self.assertEqual(plan["planning_model_calls"], 0)
        self.assertEqual(plan["limits"]["rerank_calls"], 1)

    def test_overflow_merges_fields_instead_of_dropping_them(self):
        p = understanding()
        p["intent"].update(interaction_question=True, risk_question=True)
        p["intent"]["fields"] += ["composition", "contraindications"]
        plan = build_retrieval_plan(QUESTION + "成分和禁忌", p)
        self.assertLessEqual(len(plan["tasks"]), 3)
        covered = {f for t in plan["tasks"] for f in t["fields"]}
        self.assertTrue({"composition", "contraindications", "drug_interactions", "dosage"} <= covered)

    def test_ambiguity_has_no_executable_tasks(self):
        p = understanding()
        p["unresolved_medication_mentions"] = [{"surface": "错药名"}]
        plan = build_retrieval_plan(QUESTION, p)
        self.assertEqual(plan["status"], "needs_clarification")
        self.assertEqual(plan["tasks"], [])

    def test_simple_field_stays_single_query(self):
        p = understanding()
        p["intent"] = {"fields": []}
        plan = build_retrieval_plan("甲试鼻喷雾剂的成分是什么", p)
        self.assertEqual(len(plan["tasks"]), 1)
        self.assertEqual(plan["tasks"][0]["fields"], ["composition"])

    def test_mechanism_language_creates_a_bounded_source_lookup(self):
        intent = patient_intent("阿托伐他汀和克拉霉素合用的风险及背后药理机制是什么？", medicine_count=2)
        self.assertTrue(intent["interaction_question"])
        self.assertTrue(intent["mechanism_question"])
        p = understanding()
        p["medicines"] = [
            {"surface": "阿托伐他汀", "actual_patient_product_known": False, "candidate_drug_names": []},
            {"surface": "克拉霉素", "actual_patient_product_known": False, "candidate_drug_names": []},
        ]
        p["intent"].update(intent)
        plan = build_retrieval_plan("阿托伐他汀和克拉霉素合用的风险及背后药理机制是什么？", p)
        mechanism = next(task for task in plan["tasks"] if task["id"] == "mechanism")
        self.assertIn("drug_interactions", mechanism["fields"])
        self.assertIn("ENZYME_MODULATION_MENTION", mechanism["graph_relations"])
        self.assertLessEqual(len(plan["tasks"]), 3)

    def test_dosage_has_no_fabricated_graph_relation(self):
        p = understanding()
        p["intent"] = {"fields": ["dosage"]}
        plan = build_retrieval_plan("甲试鼻喷雾剂怎么用", p)
        self.assertEqual(plan["graph_relations"], [])

    def test_column_router_matches_the_shared_plan(self):
        plan = build_retrieval_plan(QUESTION, understanding())
        self.assertTrue(set(plan["fields"]) <= expected_section_fields(section_routing_question(QUESTION, plan)))

    def test_plan_and_cost_diagnostics_survive_export(self):
        trace = {"query_plan": build_retrieval_plan(QUESTION, understanding()), "query_execution": {"rerank_calls": 1}}
        self.assertEqual(make_export_payload(QUESTION, "", trace)["trace"], trace)


class BoundedExecutionTests(unittest.TestCase):
    def plan(self):
        return build_retrieval_plan(QUESTION, understanding())

    def test_parallel_recall_dedup_and_exactly_one_rerank(self):
        plan = self.plan()
        active = peak = 0
        lock = threading.Lock()
        gate = threading.Barrier(2)
        calls = []
        def search(q, k):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                index = len(calls)
                calls.append(q)
            if index < 2:
                gate.wait(timeout=2)
            with lock:
                active -= 1
            return [doc("shared"), doc(str(index), "dosage" if index == 2 else "adverse_reactions" if index == 0 else "indication")]
        reranks = []
        def rerank(q, documents, **kwargs):
            reranks.append((q, documents, kwargs))
            return list(reversed(documents))
        selected, report = execute_retrieval_plan(plan, search=search, rerank=rerank)
        self.assertEqual(len(calls), 3)
        self.assertEqual(peak, 2)
        self.assertEqual(len(reranks), 1)
        self.assertEqual(reranks[0][0], QUESTION)
        self.assertLessEqual(len(reranks[0][1]), 36)
        self.assertEqual(len({d.metadata["chunk_id"] for d in selected}), len(selected))
        self.assertEqual(report["uncovered_tasks"], [])
        shared = next(d for d in selected if d.metadata["chunk_id"] == "shared")
        self.assertEqual(len(shared.metadata["retrieval_task_ids"]), 3)
        self.assertEqual(shared.page_content, "虚构测试原文 shared")

    def test_foreign_drug_and_wrong_field_do_not_enter_rerank(self):
        plan = self.plan()
        sent = []
        def rerank(q, docs, **kwargs):
            sent.extend(docs)
            return docs
        execute_retrieval_plan(plan, search=lambda *a, **k: [doc("wrong", "composition"), doc("foreign", drug="其他药"), doc("valid")], rerank=rerank)
        self.assertEqual([d.metadata["chunk_id"] for d in sent], ["valid"])

    def test_rerank_error_falls_back_to_real_recall_candidates(self):
        def fail(*args, **kwargs):
            raise RuntimeError("service error")
        docs, report = execute_retrieval_plan(self.plan(), search=lambda *a, **k: [doc("real")], rerank=fail)
        self.assertEqual([d.metadata["chunk_id"] for d in docs], ["real"])
        self.assertEqual(report["rerank_status"], "fallback_recall_order")
        self.assertNotIn("evidence_ids", report)

    def test_changed_rerank_text_is_rejected(self):
        docs, report = execute_retrieval_plan(self.plan(), search=lambda *a, **k: [doc("real")],
            rerank=lambda q, docs, **kw: [Document(page_content="fabricated", metadata=docs[0].metadata)])
        self.assertEqual(docs[0].page_content, "虚构测试原文 real")
        self.assertEqual(report["rerank_status"], "fallback_recall_order")

    def test_slow_recall_returns_at_budget_and_reports_missing_tasks(self):
        plan = self.plan()
        plan["limits"].update(recall_seconds=.03, total_seconds=.04)
        release = threading.Event()
        def slow(*a, **kw):
            release.wait(timeout=2)
            return [doc("late")]
        started = time.monotonic()
        try:
            docs, report = execute_retrieval_plan(plan, search=slow, rerank=lambda *a, **kw: self.fail("no candidates"))
        finally:
            release.set()
        self.assertLess(time.monotonic()-started, .5)
        self.assertEqual(docs, [])
        self.assertTrue(report["timed_out"])
        self.assertEqual(len(report["uncovered_tasks"]), 3)

    def test_slow_rerank_returns_existing_candidates_at_deadline(self):
        plan = self.plan()
        plan["limits"].update(rerank_seconds=.03, total_seconds=.15)
        release = threading.Event()
        def slow(*a, **kw):
            release.wait(timeout=2)
            return [doc("late")]
        try:
            docs, report = execute_retrieval_plan(plan, search=lambda *a, **kw: [doc("real")], rerank=slow)
        finally:
            release.set()
        self.assertEqual(docs[0].metadata["chunk_id"], "real")
        self.assertTrue(report["timed_out"])
        self.assertEqual(report["rerank_calls"], 1)

    def test_partial_recall_failure_keeps_successful_queries(self):
        plan = self.plan()
        def search(q, **kw):
            if "用法用量" in q:
                raise ConnectionError()
            return [doc("real")]
        docs, report = execute_retrieval_plan(plan, search=search, rerank=lambda q, ds, **kw: ds)
        self.assertTrue(docs)
        self.assertIn("administration", report["uncovered_tasks"])
        self.assertEqual(sum(t["status"] == "error" for t in report["tasks"]), 1)

    def test_unverified_fallback_still_fails_original_provenance_gate(self):
        source = doc("real")
        source.metadata.update(data_origin="json_matched_to_neo4j_section")
        docs, report = execute_retrieval_plan(self.plan(), search=lambda *a, **k: [source], rerank=lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
        verified, validation = prepare_medical_documents(docs, "注意事项", lambda meta: None)
        self.assertEqual(verified, [])
        self.assertEqual(validation["failed_provenance"], 1)


class GraphIntegrationTests(unittest.TestCase):
    def test_mechanism_pair_path_joins_both_anchors_from_either_source_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            export = root / "kg_output_clean_v2"
            drugs_path = export / "drugs.jsonl"
            drugs = [json.loads(line) for line in drugs_path.read_text(encoding="utf-8").splitlines()]
            drugs.append({"id": "drug_b", "name": "乙试片", "aliases": []})
            drugs_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in drugs), encoding="utf-8")
            records_path = export / "source_records.jsonl"
            records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
            records.append({"id": "record_b", "drug_id": "drug_b", "source_url": "https://example.invalid/b"})
            records_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
            sections_path = export / "sections.jsonl"
            sections = [json.loads(line) for line in sections_path.read_text(encoding="utf-8").splitlines()]
            sections.extend([
                {"id": "scomposition_b", "record_id": "record_b", "field": "composition", "text": "乙试成分。"},
                {"id": "smechanism_b", "record_id": "record_b", "field": "drug_interactions",
                 "text": "本品与甲试鼻喷雾剂合用时经CYP2C19代谢，导致活性代谢物减少，抗血小板作用减弱。"},
            ])
            sections_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sections), encoding="utf-8")
            build_graph(root)
            question = "甲试鼻喷雾剂与乙试片合用为什么可能影响作用？药理机制是什么？"
            excerpts, report = search_semantic_graph(question, project_root=root)
            self.assertTrue(excerpts, report)
            paired = [item for item in excerpts if any(
                edge["relation"] == "MECHANISM_PARTNER_MENTION"
                and edge["dst"]["name"] in {"甲试鼻喷雾剂", "乙试片"}
                for edge in item["graph_path"]["edges"]
            ) and len(item["graph_path"]["nodes"]) >= 3]
            self.assertTrue(paired, report)
            evidence = [edge for item in paired for edge in item["graph_path"]["edges"]]
            self.assertTrue(any(edge["relation"] == "MECHANISM_PARTNER_MENTION" and edge["dst"]["name"] == "甲试鼻喷雾剂" for edge in evidence))
            self.assertTrue(any(edge["relation"] == "HAS_INGREDIENT" and edge["dst"]["name"] == "甲试鼻喷雾剂" for edge in evidence) is False)
            self.assertTrue(all(edge["quote"] == sections[-1]["text"] for edge in evidence))

    def test_mechanism_relation_labels_are_specific_and_future_relations_are_legible(self):
        self.assertEqual(relation_label("MECHANISM_PARTNER_MENTION"), "机制陈述提及相关药物")
        self.assertEqual(relation_label("EXPOSURE_CHANGE_MENTION"), "药物暴露变化原文提及")
        self.assertNotEqual(relation_label("FUTURE_RELATION"), "资料关系")

    def test_mechanism_edges_are_built_and_verified_from_original_clause(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            export = root / "kg_output_clean_v2"
            sections = [json.loads(line) for line in (export / "sections.jsonl").read_text(encoding="utf-8").splitlines()]
            sections.append({"id": "smechanism", "record_id": "record_a", "field": "drug_interactions",
                             "text": "本品经CYP3A4代谢，合用抑制剂后血药浓度升高。"})
            (export / "sections.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sections), encoding="utf-8")
            build_graph(root)
            excerpts, report = search_semantic_graph("甲试鼻喷雾剂的药理机制是什么", project_root=root)
            self.assertTrue(excerpts, report)
            proof_edges = [edge for item in excerpts for edge in item["graph_path"]["edges"]]
            self.assertTrue(any(edge["relation"] == "METABOLISM_MENTION" and edge["dst"]["name"] == "CYP3A4" for edge in proof_edges))
            self.assertTrue(any(edge["relation"] == "EXPOSURE_CHANGE_MENTION" and "血药浓度" in edge["quote"] for edge in proof_edges))
            mechanism_edges = [edge for edge in proof_edges if edge["relation"].endswith("_MENTION") and edge["dst"]["kind"] == "Mechanism"]
            self.assertTrue(mechanism_edges)
            self.assertTrue(all(edge["quote"] == "本品经CYP3A4代谢，合用抑制剂后血药浓度升高。" for edge in mechanism_edges))

    def test_one_traversal_shares_plan_and_verifies_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            with patch("medical_semantic_retrieval._traverse", wraps=__import__('medical_semantic_retrieval')._traverse) as traversal:
                excerpts, report = search_semantic_graph(QUESTION, project_root=root)
            self.assertTrue(excerpts, report)
            self.assertEqual(traversal.call_count, 1)
            self.assertEqual(report["plan"]["retrieval_plan_id"], report["retrieval_plan"]["plan_id"])
            self.assertTrue(all(e["graph_path"]["verified_against_original"] for e in excerpts))
            self.assertIn("dosage", {f for t in report["task_coverage"] for f in t["text_only_fields"]})

    def test_sql_deadline_is_reported_as_timeout_not_absent_relationship(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            with patch("medical_semantic_retrieval._traverse", side_effect=sqlite3.OperationalError("interrupted")):
                excerpts, report = search_semantic_graph(QUESTION, project_root=root)
            self.assertEqual(report["status"], "retrieval_timeout")
            self.assertTrue(report["truncated"])
            self.assertEqual(excerpts, [])
            self.assertIn("达到时间预算", graph_insufficiency_message(report))


class RuntimeIntegrationTests(unittest.TestCase):
    def test_text_tool_uses_supplied_graph_plan_and_one_rerank(self):
        stubs = {}
        for name, symbol in (("knowledge_base", "KnowledgeBaseService"), ("vector_stores", "VectorStoreService")):
            module = ModuleType(name)
            setattr(module, symbol, type(symbol, (), {}))
            stubs[name] = module
        with patch.dict(sys.modules, stubs):
            import rag
            with patch.object(rag, "ChatOpenAI", return_value=SimpleNamespace(bind_tools=lambda tools: None)):
                service = rag.RagService()
        calls = {"recall": [], "rerank": []}
        plan = build_retrieval_plan(QUESTION, understanding())
        fields = {"adverse_event": "adverse_reactions", "efficacy": "indication", "administration": "dosage"}
        def search(query, k):
            task = next(t for t in plan["tasks"] if t["query"] == query)
            calls["recall"].append(query)
            return [doc(task["id"], fields[task["id"]])]
        def rerank(query, candidates, **kwargs):
            calls["rerank"].append(query)
            return candidates
        service.knowledge_service = SimpleNamespace(search_hybrid=search, rerank_documents=rerank)
        service.chroma_index_chunks = 3
        with patch.object(rag, "plan_medical_question", side_effect=AssertionError("must reuse graph plan")), contextlib.redirect_stdout(io.StringIO()):
            data = json.loads(service.tool_map["search_knowledge_base"].invoke({"question": QUESTION, "retrieval_plan": plan}))
        self.assertEqual(data["query_execution"]["plan_id"], plan["plan_id"])
        self.assertEqual(len(calls["recall"]), 3)
        self.assertEqual(len(calls["rerank"]), 1)
        self.assertEqual(data["status"], "ok")
        self.assertEqual({e["section_field"] for e in data["excerpts"]}, set(fields.values()))
        self.assertEqual(data["subqueries"], [t["query"] for t in plan["tasks"]])
        self.assertEqual(data["query_execution"]["after_provenance_uncovered_tasks"], [])
        timeout_report = {
            "plan_id": plan["plan_id"], "recall_calls": 3, "rerank_calls": 0,
            "seconds": .03, "timed_out": True,
            "uncovered_tasks": [t["id"] for t in plan["tasks"]],
        }
        with patch.object(rag, "execute_retrieval_plan", return_value=([], timeout_report)), contextlib.redirect_stdout(io.StringIO()):
            data = json.loads(service.tool_map["search_knowledge_base"].invoke({"question": QUESTION, "retrieval_plan": plan}))
        self.assertEqual(data["status"], "retrieval_timeout")
        self.assertFalse(data["found"])

    def test_actual_rerank_adapter_sets_timeout_and_preserves_full_text(self):
        chroma = ModuleType("langchain_chroma")
        chroma.Chroma = type("Chroma", (), {})
        jieba = ModuleType("jieba")
        jieba.lcut_for_search = lambda text: [text]
        source = Path(__file__).resolve().parents[1] / "knowledge_base.py"
        spec = importlib.util.spec_from_file_location("_tested_knowledge_base", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"langchain_chroma": chroma, "jieba": jieba}):
            spec.loader.exec_module(module)
        original = doc("x")
        original.page_content = "保留原始测试正文。" * 300
        result = SimpleNamespace(status_code=200, output={"results": [{"index": 0, "relevance_score": .9}]}, usage={"total_tokens": 12})
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-only"}), \
             patch.object(module.dashscope.TextReRank, "call", return_value=result) as call, contextlib.redirect_stdout(io.StringIO()):
            ranked = module.KnowledgeBaseService.__new__(module.KnowledgeBaseService).rerank_documents(
                QUESTION, [original], timeout=2, max_chars=1200)
        self.assertEqual(call.call_args.kwargs["timeout"], 2)
        self.assertLessEqual(len(call.call_args.kwargs["documents"][0]), 1200)
        self.assertEqual(ranked[0].page_content, original.page_content)
        # Validate the query-only embedding override without initializing a client.
        embedding_response = SimpleNamespace(status_code=200, output={"embeddings": [{"embedding": [.1,.2]}]})
        client = SimpleNamespace(call=lambda **kw: (self.assertEqual(kw["timeout"], 8), embedding_response)[1])
        fake_embedding = SimpleNamespace(client=client, model="test", dashscope_api_key="test-only")
        self.assertEqual(module.QueryTimeoutEmbeddings.embed_query(fake_embedding, "test"), [.1,.2])
        short = Document(page_content="测试 ![图](https://example.invalid/image.png) 连续   空格", metadata={"section_path": "测试章节"})
        text = module._prepare_rerank_document(short, 300)
        self.assertNotIn("https://", text)
        self.assertEqual(short.page_content, "测试 ![图](https://example.invalid/image.png) 连续   空格")


if __name__ == "__main__":
    unittest.main()
