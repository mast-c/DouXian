# -*- coding: utf-8 -*-
"""Synthetic source-backed SQLite test for v5 independent task execution.

Run from the delivered folder:
  python -m unittest -v test_semantic_graph_execution.py
No network, Neo4j or real project data required. External imports are replaced
only inside this isolated test process.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path


def load_graph_retriever():
    graph = types.ModuleType("medical_semantic_graph")
    graph.GraphBlocked = type("GraphBlocked", (Exception,), {})
    graph.Lexicon = lambda *a, **k: None
    graph.RELATIONS = {
        "CAUTION_MENTION": "注意事项提及",
        "ADVERSE_REACTION_MENTION": "不良反应提及",
        "INTERACTION_MENTION": "相互作用提及",
        "HAS_INGREDIENT": "包含成分",
        "CLASS_MEMBERSHIP_ASSERTION": "药品类别",
        "INDICATION_MENTION": "适应症提及",
        "HAS_SYMPTOM": "疾病症状",
    }
    graph.FIELD_RELATIONS = {
        "warnings": "CAUTION_MENTION",
        "adverse_reactions": "ADVERSE_REACTION_MENTION",
        "drug_interactions": "INTERACTION_MENTION",
        "indication": "INDICATION_MENTION",
    }
    graph.MECHANISM_RELATIONS = set()
    graph.MECHANISM_STATEMENT_RELATIONS = set()
    graph.digest = lambda s: "source_digest_123"
    graph.entity_id = lambda *a: "id"
    graph.class_assertion_spans = lambda *a: []
    graph.ingredient_spans = lambda *a: []
    graph.literal_entity_boundary = lambda *a, **k: True
    graph.qualifiers = lambda text: {}
    graph.verify_source = lambda root, source: None
    graph.index_path = lambda *a: None
    graph.open_graph = lambda *a: None
    sys.modules["medical_semantic_graph"] = graph

    uq = types.ModuleType("medical_query_understanding")
    uq.understand_question = lambda *a, **k: {}
    uq.active_anchors = lambda groups, understanding, nodes: groups
    uq.catalog_aliases_for_question = lambda *a, **k: {}
    sys.modules["medical_query_understanding"] = uq

    llm = types.ModuleType("medical_llm_query_understanding")
    llm.enhance_understanding = lambda q,u,m,**k: (u,{})
    llm.load_query_understanding_model_from_env = lambda: None
    llm.select_medical_query_model = lambda model: (model, "injected_app_model")
    llm._load_project_dotenv = lambda: None
    sys.modules["medical_llm_query_understanding"] = llm

    planner = types.ModuleType("medical_query_plan")
    planner.build_retrieval_plan = lambda q,u: {"plan_id": "fixture", "tasks": []}
    sys.modules["medical_query_plan"] = planner

    for name, exports in (
        ("medical_category_graph", {
            "category_only": lambda *a: False,
            "search_category_graph": lambda *a, **k: ([],{}),
            "extend_category_entities": lambda q,r,n,u: (n,u)
        }),
        ("medical_category_discovery", {
            "search_category_discovery": lambda *a, **k: ([],{})
        }),
    ):
        m=types.ModuleType(name)
        for key, value in exports.items():
            setattr(m,key,value)
        sys.modules[name] = m

    project=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location(
        "medical_semantic_retrieval_synthetic",
        str(project/"medical_semantic_retrieval.py"),
    )
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GraphTaskExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever=load_graph_retriever()
        cls.nodes={
            "D1":{"id":"D1","kind":"Drug","name":"布地奈德鼻喷雾剂"},
            "C1":{"id":"C1","kind":"Disease","name":"荨麻疹"},
            "S1":{"id":"S1","kind":"Symptom","name":"瘙痒"},
        }

    def setup_graph(self, malformed=False):
        connection=sqlite3.connect(":memory:")
        connection.row_factory=sqlite3.Row
        connection.executescript("""
        CREATE TABLE sources (
            id TEXT PRIMARY KEY,record_id TEXT,drug_id TEXT,field TEXT,
            text TEXT,file TEXT,url TEXT,authority TEXT,text_sha256 TEXT,
            byte_start INTEGER,byte_length INTEGER
        );
        CREATE TABLE edges (
            id TEXT PRIMARY KEY,src TEXT,dst TEXT,relation TEXT,source_id TEXT,
            quote_start INTEGER,quote_end INTEGER,mention_start INTEGER,
            mention_end INTEGER,qualifier_json TEXT,status TEXT
        );
        """)
        txt="速发或迟发的过敏反应，包括荨麻疹、皮疹等。"
        i=txt.index("荨麻疹")
        connection.execute(
            "INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            ("source1","r1","D1","warnings",txt,"fixture.jsonl","",
             "third_party_unverified","fake_sha256",0,0)
        )
        connection.execute(
            "INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("e1","D1","C1","CAUTION_MENTION","source1",0,len(txt),
             (i+1 if malformed else i),i+len("荨麻疹"),"{}","observed"),
        )
        connection.commit()
        return connection

    def task_plan(self):
        from medical_semantic_query_planner import build_semantic_task_plan
        groups=[
            {"surface":"布地奈德鼻喷雾剂","node_ids":["D1"],"kinds":["Drug"],
             "ambiguous":False},
            {"surface":"荨麻疹","node_ids":["C1"],"kinds":["Disease"],
             "ambiguous":False},
        ]
        understanding={
            "intent":{"adverse_event_question":True,"classification_question":True,
                      "fields":[]},
            "medicines":[{"surface":"布地奈德鼻喷雾剂","actual_patient_product_known":True}],
            "llm_query_understanding":{"status":"ok",
                                      "intents":["adverse_event","classification"]},
        }
        task_plan=build_semantic_task_plan(
            "布地奈德鼻喷雾剂会导致荨麻疹吗？它是什么类型的疾病？",
            understanding,groups,
            available_relations=set(self.retriever.RELATIONS)
        )
        return groups, understanding, task_plan

    def test_one_hop_adverse_evidence_survives_missing_disease_taxonomy(self):
        groups, understanding, task_plan = self.task_plan()
        conn=self.setup_graph()
        try:
            excerpts, report=self.retriever._run_semantic_task_plan(
                conn, Path(tempfile.gettempdir()), {"version":"fixture"},self.nodes,
                understanding,task_plan,
                max_paths=4,max_expansions=200,max_neighbors=20,max_seconds=3.0,
                report={"context_entities":groups,"semantic_sub_queries":[]},
            )
        finally:
            conn.close()
        self.assertEqual(len(excerpts),1)
        self.assertEqual(excerpts[0]["graph_path"]["hops"],1)
        self.assertEqual(excerpts[0]["task_kind"],"drug_adverse_event")
        self.assertEqual(report["direct_relation_paths"],1)
        self.assertEqual(report["multi_relation_paths"],0)
        self.assertEqual(report["task_coverage"][0]["status"],"graph_evidence_found")
        self.assertEqual(report["task_coverage"][1]["kind"],"disease_classification")
        self.assertEqual(report["task_coverage"][1]["status"],"needs_disease_classification_source")
        self.assertIn("荨麻疹", report["text_retrieval_requests"][1]["query"])
        self.assertTrue(report["partial_answer_only"])
        self.assertNotIn("皮肤疾病",str(excerpts))  # No invented classification claim.

    def test_public_search_routes_two_tasks_independently(self):
        import unittest.mock
        groups, understanding, task_plan = self.task_plan()
        conn = self.setup_graph()
        root = Path(tempfile.mkdtemp(prefix="semantic_graph_v5_"))
        dummy_index = root / "graph.sqlite"
        dummy_index.write_bytes(b"fixture")
        model = object()
        # Simulate the LLM's successful structured output, without sending
        # any external request. All graph-source verification still runs.
        parsed = {
            **understanding,
            "llm_medicine_mentions": [{
                "surface":"布地奈德鼻喷雾剂",
                "canonical":"布地奈德鼻喷雾剂",
                "catalog_kind":"Drug",
                "role":"mentioned",
                "actual_patient_product_known":True,
            }],
            "patient_context":[{
                "surface":"荨麻疹","kind":"disease_mention",
                "retrieval_terms":["荨麻疹"],"assertion":"user_reported",
                "role":"mentioned",
            }],
            "semantic_sub_tasks":[
                {"intent":"adverse_event",
                 "query":"布地奈德鼻喷雾剂和荨麻疹的不良反应原文",
                 "focus_entities":["布地奈德鼻喷雾剂","荨麻疹"]},
                {"intent":"classification",
                 "query":"荨麻疹属于什么疾病类型？",
                 "focus_entities":["荨麻疹"]},
            ],
        }
        self.retriever._catalog = lambda *a: (self.nodes, {}, None)
        self.retriever.index_path = lambda *a: dummy_index
        self.retriever.open_graph = lambda *a: (
            conn, {"version":"fixture","created_at":"fixture"}
        )
        self.retriever.select_medical_query_model = lambda injected: (
            model, "injected_app_model"
        )
        self.retriever.enhance_understanding = lambda q,u,m,**k: (
            parsed, {"status":"ok", "model_calls":1, "mode":"llm_first"}
        )
        result, trace=self.retriever.search_semantic_graph(
            "布地奈德鼻喷雾剂会导致荨麻疹吗？它是什么类型的疾病？",
            project_root=root, query_understanding_model=model,
            max_paths=4, max_expansions=200, max_neighbors=20, max_seconds=3,
        )
        self.assertEqual(trace["query_understanding_mode"],"llm_first")
        self.assertEqual(trace["status"],"ok")
        self.assertEqual(trace["direct_relation_paths"],1)
        self.assertEqual(len(result),1)
        self.assertEqual(trace["task_coverage"][0]["status"],"graph_evidence_found")
        self.assertEqual(trace["task_coverage"][1]["status"],
                         "needs_disease_classification_source")
        self.assertTrue(trace["partial_answer_only"])

    def test_interaction_missing_second_drug_does_not_recycle_unrelated_path(self):
        groups, understanding, _ = self.task_plan()
        known=groups[:1]
        partial={
            "intent":{"fields":[],"interaction_question":True},
            "medicines":[{"surface":"布地奈德鼻喷雾剂",
                          "actual_patient_product_known":True}],
            "llm_query_understanding":{"status":"ok","intents":["interaction"]},
            "llm_unlinked_medication_mentions":[
                {"surface":"伪麻黄碱","type":"ingredient","role":"mentioned"}
            ],
            "vague_medicines":[],
            "patient_context":[],
        }
        from medical_semantic_query_planner import build_semantic_task_plan
        tasks=build_semantic_task_plan(
            "布地奈德鼻喷雾剂与伪麻黄碱能一起用吗？",partial,known,
            available_relations=set(self.retriever.RELATIONS),
        )
        self.assertEqual(tasks["tasks"][0]["graph_strategy"],"text_only")
        self.assertEqual(tasks["tasks"][0]["kind"],"drug_interaction")
        conn=self.setup_graph()
        try:
            excerpts,report=self.retriever._run_semantic_task_plan(
                conn,Path(tempfile.gettempdir()),{"version":"fixture"},self.nodes,
                partial,tasks,max_paths=4,max_expansions=200,max_neighbors=20,
                max_seconds=3,report={"context_entities":known,"semantic_sub_queries":[]},
            )
        finally:
            conn.close()
        self.assertFalse(excerpts)
        self.assertEqual(report["status"],"source_retrieval_required")
        self.assertEqual(report["task_coverage"][0]["status"],"needs_text_retrieval")

    def test_irrelevant_three_hop_path_is_not_drug_adverse_evidence(self):
        """Neo4j might find A->itch<-otherDrug->urticaria, but no A->urticaria."""
        groups, understanding, plan = self.task_plan()
        fake_edges=[
            {"src":{"id":"D1"}, "dst":{"id":"S1"},
             "relation":"ADVERSE_REACTION_MENTION"},
            {"src":{"id":"X1"}, "dst":{"id":"S1"},
             "relation":"ADVERSE_REACTION_MENTION"},
            {"src":{"id":"X1"}, "dst":{"id":"C1"},
             "relation":"ADVERSE_REACTION_MENTION"},
        ]
        proof={"nodes":[self.nodes["D1"],self.nodes["S1"],
                         {"id":"X1","name":"另一种药","kind":"Drug"},self.nodes["C1"]],
               "edges":fake_edges,"relations":["ADVERSE_REACTION_MENTION"],"hops":3}
        self.assertFalse(self.retriever._task_has_grounded_relation(
            plan["tasks"][0],proof,groups))
        direct={"nodes":[self.nodes["D1"],self.nodes["C1"]],
                "edges":[{"src":self.nodes["D1"],"dst":self.nodes["C1"],
                          "relation":"CAUTION_MENTION"}],
                "relations":["CAUTION_MENTION"],"hops":1}
        self.assertTrue(self.retriever._task_has_grounded_relation(
            plan["tasks"][0],direct,groups))

    def test_neo4j_backend_routing_preserved_and_source_proof_still_required(self):
        """Mock Neo4j path discovery; verify the original SQLite proof runs."""
        from unittest.mock import patch
        import os
        groups, understanding, task_plan=self.task_plan()
        conn=self.setup_graph()
        edge=dict(conn.execute("SELECT e.*,s.record_id,s.drug_id FROM edges e "
                               "JOIN sources s ON s.id=e.source_id").fetchone())
        route={"node_ids":["D1","C1"],"edges":[edge],
               "relations":["CAUTION_MENTION"],"covered_anchors":[0,1],"score":100}
        calls=[]
        def mocked_neo4j(local_conn, nodes, search_plan, *, max_paths):
            calls.append(search_plan["requested_relations"])
            return [route], {"backend":"neo4j","candidate_paths":1}
        try:
            with patch.dict(os.environ,{"MEDICAL_SEMANTIC_GRAPH_BACKEND":"neo4j"}), \
                 patch.object(self.retriever,"_traverse_neo4j",mocked_neo4j):
                evidence,report=self.retriever._run_semantic_task_plan(
                    conn,Path(tempfile.gettempdir()),{"version":"fixture"},self.nodes,
                    understanding,task_plan,max_paths=4,
                    max_expansions=200,max_neighbors=20,max_seconds=3,
                    report={"context_entities":groups,"semantic_sub_queries":[]},
                )
        finally:
            conn.close()
        self.assertTrue(calls,"Should dispatch through current Neo4j backend")
        self.assertEqual(len(evidence),1)
        self.assertTrue(evidence[0]["graph_verified"])
        self.assertEqual(evidence[0]["graph_path"]["hops"],1)
        self.assertEqual(report["graph_backend"],"neo4j")
        self.assertEqual(report["task_coverage"][1]["status"],
                         "needs_disease_classification_source")

    def test_forged_source_quote_is_rejected(self):
        groups, understanding, task_plan = self.task_plan()
        conn=self.setup_graph(malformed=True)
        try:
            excerpts, report=self.retriever._run_semantic_task_plan(
                conn, Path(tempfile.gettempdir()), {"version":"fixture"},self.nodes,
                understanding,task_plan,
                max_paths=4,max_expansions=200,max_neighbors=20,max_seconds=3.0,
                report={"context_entities":groups,"semantic_sub_queries":[]},
            )
        finally:
            conn.close()
        self.assertEqual(len(excerpts),0)
        self.assertGreaterEqual(report["stats"].get("rejected_unverified_paths",0),1)
        self.assertEqual(report["task_coverage"][0]["status"],
                         "no_task_scoped_graph_evidence")


if __name__ == "__main__":
    unittest.main()
