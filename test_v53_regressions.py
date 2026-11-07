# -*- coding: utf-8 -*-
"""Offline GraphRAG stability regressions (no external services or real data)."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

# Load v5.2's original dependency stubs, then use the v5.3 local modules.
import test_v52_regressions  # noqa: F401
import medical_semantic_retrieval as retriever
from medical_semantic_graph import GraphBlocked
from medical_evidence_coverage import (
    GRAPH_TYPE, build_answer_targets, evidence_bindings,
    select_target_evidence,
)
from medical_graph_visibility import graph_visibility_report
from medical_graph_path_ui import render_verified_graph_paths


DRUG = {"id": "D1", "name": "测试药A", "kind": "Drug"}
DISEASE = {"id": "C1", "name": "测试症状B", "kind": "Disease"}
NODES = {"D1": DRUG, "C1": DISEASE}
QUOTATION = "本测试仅用于验证数据库引文位置，记载测试症状B。"
START = QUOTATION.index("测试症状B")


def sqlite_fixture():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE sources (
        id TEXT PRIMARY KEY, record_id TEXT, drug_id TEXT, field TEXT,
        text TEXT, file TEXT, url TEXT, authority TEXT, text_sha256 TEXT
    )""")
    c.execute("""CREATE TABLE edges (
        id TEXT PRIMARY KEY, src TEXT, dst TEXT, relation TEXT,
        source_id TEXT, quote_start INTEGER, quote_end INTEGER,
        mention_start INTEGER, mention_end INTEGER, qualifier_json TEXT,
        status TEXT
    )""")
    c.execute("INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)",(
        "s1", "record1", "D1", "warnings", QUOTATION,
        "test_source.jsonl", "", "third_party_unverified", "not_real_hash"
    ))
    for eid, position in (("bad_edge", START+1), ("good_edge", START)):
        c.execute("INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?)",(
            eid, "D1", "C1", "CAUTION_MENTION", "s1", 0,
            len(QUOTATION), position, START+len(DISEASE["name"]), "{}", "observed"
        ))
    c.commit()
    return c


def candidate(c, eid):
    edge = dict(c.execute("""SELECT e.*,s.record_id,s.drug_id,
        s.field AS section_field FROM edges e JOIN sources s
        ON s.id=e.source_id WHERE e.id=?""", (eid,)).fetchone())
    return {"node_ids": ["D1", "C1"], "edges": [edge],
            "relations": ["CAUTION_MENTION"], "covered_anchors": [0], "score": 1}


def graph_groups():
    return [{"surface": "测试药A", "node_ids": ["D1"],
             "kinds": ["Drug"], "ambiguous": False}]


class GraphVerificationTests(unittest.TestCase):
    def test_bad_edge_skipped_and_next_candidate_retained(self):
        c = sqlite_fixture()
        stats = {}
        try:
            bad = retriever._verified_candidate_or_none(
                c, Path("."), candidate(c, "bad_edge"), NODES, {}, stats
            )
            good = retriever._verified_candidate_or_none(
                c, Path("."), candidate(c, "good_edge"), NODES, {}, stats
            )
        finally:
            c.close()
        self.assertIsNone(bad)
        self.assertEqual(stats["rejected_invalid_local_edge"], 1)
        self.assertEqual(good["hops"], 1)
        self.assertEqual(good["edges"][0]["quote"], QUOTATION)
        self.assertTrue(good["verified_against_original"])

    def test_index_snapshot_mismatch_still_blocks_entire_search(self):
        c = sqlite_fixture()
        try:
            with patch.object(retriever, "_proof", side_effect=GraphBlocked(
                "Derived graph index changed; rebuild required"
            )):
                with self.assertRaises(GraphBlocked):
                    retriever._verified_candidate_or_none(
                        c, Path("."), candidate(c, "good_edge"), NODES, {}, {}
                    )
        finally:
            c.close()

    def test_valid_edge_not_promoted_without_quote(self):
        c = sqlite_fixture()
        try:
            good = candidate(c, "good_edge")
            good["edges"][0]["source_id"] = "nonexistent"
            result = retriever._verified_candidate_or_none(c, Path("."), good, NODES, {}, {})
        finally:
            c.close()
        self.assertIsNone(result)

    def test_normal_search_recovers_after_single_invalid_proof(self):
        c = sqlite_fixture()
        folder = Path(tempfile.mkdtemp(prefix="graph_v53_search_"))
        idx = folder/"index.sqlite"; idx.write_bytes(b"fixture")
        patient = {
            "intent": {"fields": ["warnings"], "adverse_event_question": True,
                       "interaction_question": False, "classification_question": False},
            "llm_query_understanding": {"status": "ok", "intents": ["medication_safety"]},
            "medicines": [{"surface": "测试药A", "candidate_drug_ids": ["D1"],
                           "candidate_drug_names": ["测试药A"],
                           "actual_patient_product_known": True}],
            "patient_context": [], "semantic_sub_queries": [],
            "vague_medicines": [], "vague_medicine_details": [],
            "unresolved_medication_mentions": [], "ambiguous_aliases": [],
            "route_conflicts": [], "alternative_medication_mentions": [],
        }
        retrieval_plan = {"status": "planned", "plan_id": "fixture", "tasks": [],
                          "category_discovery": None, "graph_relations": ["CAUTION_MENTION"],
                          "fields": ["warnings"]}
        # bypass the secondary LLM, and exercise only the real source proof.
        with patch.object(retriever,"open_graph",lambda *a:(c,{"version":"fixture","created_at":"test"})), \
             patch.object(retriever,"index_path",lambda *a:idx), \
             patch.object(retriever,"_catalog",lambda *a:(NODES,{},None)), \
             patch.object(retriever,"_semantic_catalog_payload",lambda *a:{}), \
             patch.object(retriever,"select_medical_query_model",lambda m:(object(),"injected_app_model")), \
             patch.object(retriever,"enhance_understanding",lambda *a,**kw:(patient,{"status":"ok","model_calls":1})), \
             patch.object(retriever,"extend_category_entities",lambda q,r,n,u:(n,u)), \
             patch.object(retriever,"_semantic_catalog_anchors",lambda *a:(graph_groups(),patient,[])), \
             patch.object(retriever,"refine_semantic_tasks_with_model",lambda *a:(patient,{"status":"not_needed"})), \
             patch.object(retriever,"build_retrieval_plan",lambda *a:retrieval_plan), \
             patch.object(retriever,"build_semantic_task_plan",lambda *a,**kw:{"tasks":[],"text_retrieval_requests":[]}), \
             patch.object(retriever,"category_only",lambda *a:False), \
             patch.object(retriever,"_traverse",lambda *a,**kw:([candidate(c,"bad_edge"),candidate(c,"good_edge")],{"candidate_paths":2})):
            rows, report = retriever.search_semantic_graph(
                "测试药A有什么注意事项？", project_root=folder,
                query_understanding_model=object(), max_seconds=3
            )
        self.assertEqual(report["status"],"ok")
        self.assertEqual(len(rows),1)
        self.assertEqual(report["direct_relation_paths"],1)
        self.assertEqual(report["multi_hop_paths"],0)
        self.assertEqual(report["stats"]["rejected_invalid_local_edge"],1)
        self.assertEqual(report["stats"]["candidate_paths_examined"],2)

    def test_neo4j_candidate_ordering_and_schema_safe_limit(self):
        c = sqlite_fixture()
        seen = {}
        class Session:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def run(self, query, **kwargs):
                seen["query"], seen["kwargs"] = query, kwargs
                return [{"node_ids":["D1","C1"],"edge_ids":["good_edge"],
                         "relations":["CAUTION_MENTION"],"hops":1}]
        class Driver:
            def session(self, database): return Session()
            def close(self): pass
        fake = types.ModuleType("neo4j")
        fake.GraphDatabase = types.SimpleNamespace(driver=lambda *a,**kw:Driver())
        plan = {"anchors":graph_groups()+[
            {"surface":"测试症状B","node_ids":["C1"],"kinds":["Disease"]}],
            "requested_relations":["CAUTION_MENTION"], "max_hops":3,
            "interaction_question":False,"drug_discovery":False,
            "patient_query":{"intent":{}}}
        try:
            with patch.dict(sys.modules,{"neo4j":fake}), \
                 patch.dict(os.environ,{"NEO4J_PASSWORD":"synthetic-test-password"}):
                candidates, stats = retriever._traverse_neo4j(c,NODES,plan,max_paths=3)
        finally:
            c.close()
        self.assertEqual(len(candidates),1)
        self.assertIn("ORDER BY hops ASC",seen["query"])
        self.assertIn("all(group_ids IN $anchor_groups",seen["query"])
        self.assertEqual(seen["kwargs"]["anchor_groups"],[["D1"],["C1"]])
        self.assertEqual(stats["raw_candidate_rows"],1)
        self.assertFalse(stats["neo4j_candidate_limit_hit"])

    def test_three_hops_same_relation_not_confused_with_multirelation(self):
        paths = [{"id":"p1","hops":1,"relations":["CAUTION_MENTION"]},
                 {"id":"p2","hops":3,"relations":["CAUTION_MENTION"]*3}]
        counts=retriever._verified_path_counts(paths)
        self.assertEqual(counts["verified_path_count"],2)
        self.assertEqual(counts["direct_relation_paths"],1)
        self.assertEqual(counts["multi_hop_paths"],1)
        self.assertEqual(counts["multi_relation_paths"],0)
        self.assertEqual(counts["same_relation_multi_hop_paths"],1)


class EvidenceCoverageTests(unittest.TestCase):
    def task_plan(self):
        return {"tasks":[{
            "id":"T1","kind":"drug_adverse_event", "query":"测试药A是否提及测试症状B？",
            "anchor_surfaces":["测试药A","测试症状B"],
            "graph_relations":["CAUTION_MENTION"],"graph_strategy":"direct",
        },{
            "id":"T2","kind":"disease_classification","query":"测试症状B是什么类型？",
            "anchor_surfaces":["测试症状B"],"graph_relations":[],
            "graph_strategy":"text_only",
        }]}

    def test_semantic_task_targets_created_without_legacy_plan(self):
        targets=build_answer_targets("测试",{"intent":{},"medicines":[]},None,
                                     semantic_task_plan=self.task_plan())
        self.assertEqual({t["semantic_task_id"] for t in targets},{"T1","T2"})
        self.assertEqual(len(targets),2)
        self.assertTrue(next(t for t in targets if t["semantic_task_id"]=="T2")["text_evidence_pending"])

    def test_proven_task_graph_binds_only_its_own_target(self):
        targets = build_answer_targets("测试",{"intent":{},"medicines":[]},None,
                                       semantic_task_plan=self.task_plan())
        path = {"id":"p1","nodes":[DRUG,DISEASE],"edges":[{
            "id":"e1","relation":"CAUTION_MENTION","src":DRUG,"dst":DISEASE,
        }],"hops":1,"relations":["CAUTION_MENTION"],"verified_against_original":True}
        graph={"evidence_type":GRAPH_TYPE,"source":"synthetic source","content":"source quote",
               "graph_verified":True,"graph_path":path,"task_id":"T1","id":1}
        ordinary={"evidence_type":"unverified_drug_source_section","source":"other",
                  "content":"Unrelated general text","section_field":"indication"}
        bindings=evidence_bindings(graph,targets)
        self.assertEqual({b["target_id"] for b in bindings},{"target_semantic_T1"})
        selected=select_target_evidence([ordinary],[graph],targets,cap=1)
        self.assertEqual(selected[0]["graph_path"]["id"],"p1")
        self.assertEqual(evidence_bindings({**graph,"graph_verified":False},targets),[])
        self.assertEqual(evidence_bindings({**graph,"task_id":"T2"},targets),[])

    def test_legacy_three_arg_target_api_remains_supported(self):
        self.assertEqual(build_answer_targets("测试",{"medicines":[]},None),[])

    def test_graph_path_visibility_separate_from_guard_cap(self):
        one={"id":"p1","hops":1,"relations":["CAUTION_MENTION"],
             "edges":[{"id":"e1"}],"verified_against_original":True}
        three={"id":"p2","hops":3,"relations":["CAUTION_MENTION"]*3,
               "edges":[{"id":"e2"}],"verified_against_original":True}
        rows=[{"graph_path":p,"graph_verified":True} for p in [one,three]]
        report={"status":"ok","storage":"neo4j_with_local_source_verification",
                "paths":[one,three],"stats":{"raw_candidate_rows":90,
                                              "rejected_invalid_local_edge":2},"truncated":True}
        data=graph_visibility_report(report,rows,rows[:1])
        self.assertEqual(data["retrieved_verified_paths"]["total"],2)
        self.assertEqual(data["retrieved_verified_paths"]["multi_hop"],1)
        self.assertEqual(data["retrieved_verified_paths"]["multi_relation"],0)
        self.assertEqual(data["accepted_into_generation"]["total"],1)
        self.assertEqual(data["not_selected_for_generation"],1)
        self.assertEqual(data["path_ids_not_selected_for_generation"],["p2"])

    def test_streamlit_renderer_rejects_unverified_path(self):
        class StubUI:
            def __init__(self): self.lines=[]
            def markdown(self,value): self.lines.append(value)
            def caption(self,value): self.lines.append(value)
            def info(self,value): self.lines.append(value)
            def warning(self,value): self.lines.append(value)
            def write(self,value): self.lines.append(value)
            def text(self,value): self.lines.append(value)
            @contextmanager
            def expander(self,label):
                self.lines.append(label)
                yield self
        valid={"id":"p1","hops":1,"relations":["CAUTION_MENTION"],
               "verified_against_original":True,"edges":[{
                   "src":DRUG,"dst":DISEASE,"relation":"CAUTION_MENTION",
                   "quote":"测试原文", "source_file":"fixture"
               }]}
        invalid={**valid,"id":"p2","verified_against_original":False}
        ui=StubUI()
        rendered=render_verified_graph_paths({"semantic_graph":{
            "status":"ok","paths":[valid,invalid]}},st=ui)
        self.assertEqual(rendered,1)
        self.assertTrue(any("单跳 1" in str(s) for s in ui.lines))
        self.assertTrue(any("测试原文" in str(s) for s in ui.lines))
        self.assertFalse(any("路径 2" in str(s) for s in ui.lines))

    def test_rag_pipeline_has_path_inventory_and_task_binding(self):
        src = (Path(__file__).resolve().parent / "rag.py").read_text(encoding="utf8")
        ast.parse(src)
        self.assertIn("semantic_task_plan=_semantic_report.get(\"semantic_task_plan\")",src)
        self.assertIn('trace["graph_path_visibility"] = graph_visibility_report(',src)


if __name__ == "__main__":
    unittest.main()
