# -*- coding: utf-8 -*-
"""Offline regression for existing Douxian evidence section 8-hop fix."""
import ast
from copy import deepcopy
import hashlib
import html
import json
import logging
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

TARGET=Path(__file__).with_name("豆馅.py")
LABELS={
    5:["DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug","SourceRecord","Section"],
    6:["DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug","DiseaseCandidate","DiseaseTerm","DiseaseDocument"],
    7:["DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug","DiseaseCandidate","DiseaseTerm","DiseaseCandidate","Drug"],
    8:["DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug","SourceRecord","Section","DiseaseCandidate","DiseaseTerm","DiseaseDocument"],
}
RELS={
    5:["EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA","MENTIONS_TERM","HAS_CANDIDATE","HAS_RECORD","HAS_SECTION"],
    6:["EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA","MENTIONS_TERM","HAS_CANDIDATE","HAS_CANDIDATE","MENTIONS_TERM","EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA"],
    7:["EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA","MENTIONS_TERM","HAS_CANDIDATE","HAS_CANDIDATE","MENTIONS_TERM","MENTIONS_TERM","HAS_CANDIDATE"],
    8:["EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA","MENTIONS_TERM","HAS_CANDIDATE","HAS_RECORD","HAS_SECTION","SUPPORTED_BY","MENTIONS_TERM","EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA"],
}

def fake(depth=8, i=1, *, with_disease=True):
    result={
       "hop_count":depth, "path_id":f"p{depth}_{i}",
       "node_ids":[f"node_{depth}_{i}_{n}" for n in range(depth+1)],
       "node_labels":list(LABELS[depth]),
       "relationship_types":list(RELS[depth]),
       "drug_name":"丙酸氟替卡松鼻喷雾剂",
       "related_disease_name":"枯草热",
       "source_status":"third_party_unverified",
       "clinical_verified":False,
       "treatment_relation":False,
       "graph_relationships_verified":True,
    }
    if with_disease:result["disease_name"]="过敏性鼻炎"
    return result

def load():
    source=TARGET.read_text("utf-8-sig")
    tree=ast.parse(source)
    functions={
       "_mh_safe","_mh_structurally_valid",
       "_mh_display_config","_mh_one_source_name",
       "_mh_paths_from_trace","_mh_node_name",
       "_mh_one_path_html","multihop_path_evidence_html",
       "populate_multihop_trace_for_display",
       "trace_details_html","graph_evidence_html",
       "trace_heading",
    }
    # Exclude trace_heading; it references app-specific retrieval_heading.
    functions.remove("trace_heading")
    constants={"_MH_PATH_LABELS","_MH_PATH_RELS","_MH_EDGE_DIRECTIONS",
               "_MH_CARD_TYPE","_MH_MAX_PATHS"}
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        if isinstance(node,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id in constants
            for t in node.targets
        ):
            nodes.append(node)
    program=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    ns={
      "html":html,"hashlib":hashlib,"json":json,"logging":logging,
      "os":os,"Path":Path,"re":__import__("re"),"urlsplit":urlsplit,
      "__file__":str(TARGET),
      "trace_heading":lambda trace:"已完成医疗检索",
      "_web_sources":lambda trace:[],"_web_queries":lambda trace:[],
      "retrieval_status_html":lambda trace:"",
      "elapsed_label":lambda ms:"10s",
      "describe_tool_event":lambda event:"✓ 已完成",
    }
    exec(compile(program,str(TARGET),"exec"),ns)
    return ns,source

class EightHopFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fn,cls.source=load()

    def test_file_ast(self):
        ast.parse(self.source)

    def test_previous_six_original_sources_still_visible(self):
        t={"knowledge_evidence":[{"id":i+1,"source":f"doc{i}",
              "content":"original evidence text"} for i in range(6)],
           "p3063_multihop_path_cards":{"paths":[fake(8,i+1) for i in range(4)]}}
        with patch.dict(os.environ,{
            "MEDICAL_P3063_PATH_MIN_HOPS":"8",
            "MEDICAL_P3063_PATH_MAX_HOPS":"8",
            "MEDICAL_P3063_PATH_MAX_PATHS":"4",
        }):
            result=self.fn["trace_details_html"](t)
        self.assertIn("实际命中的知识库证据（6 条）",result)
        self.assertEqual(result.count('class="kb-mh-card"'),4)

    def test_all_5_6_7_8_in_trace_only_8_displayed(self):
        t={"p3063_multihop_path_cards":{"paths":[]},
           "disease_encyclopedia":{"multi_hop_graph":{
               "verified_paths":[fake(5,1),fake(6,1),fake(7,1),fake(8,1)]
           }}}
        with patch.dict(os.environ,{
            "MEDICAL_P3063_PATH_MIN_HOPS":"8",
            "MEDICAL_P3063_PATH_MAX_HOPS":"8",
            "MEDICAL_P3063_PATH_MAX_PATHS":"4",
        }):
            result=self.fn["_mh_paths_from_trace"](t)
        self.assertEqual([p["hop_count"] for p in result],[8])

    def test_configurable_5_to_8_display(self):
        t={"p3063_multihop_path_cards":{"paths":[
            fake(5),fake(6),fake(7),fake(8)
        ]}}
        with patch.dict(os.environ,{
            "MEDICAL_P3063_PATH_MIN_HOPS":"5",
            "MEDICAL_P3063_PATH_MAX_HOPS":"8",
            "MEDICAL_P3063_PATH_MAX_PATHS":"16",
        }):
            result=self.fn["_mh_paths_from_trace"](t)
        self.assertEqual([x["hop_count"] for x in result],[5,6,7,8])

    def test_explicit_8_hops_overrides_range_config(self):
        t={"p3063_multihop_path_cards":{
            "frontend_question":"请展示8跳来源路径",
            "paths":[fake(5),fake(6),fake(8)]
        }}
        with patch.dict(os.environ,{
            "MEDICAL_P3063_PATH_MIN_HOPS":"5",
            "MEDICAL_P3063_PATH_MAX_HOPS":"8",
        }):
            result=self.fn["_mh_paths_from_trace"](t)
        self.assertEqual([p["hop_count"] for p in result],[8])

    def test_explicit_range_does_not_filter_to_8(self):
        t={"p3063_multihop_path_cards":{
            "frontend_question":"展示5～8跳全部路径",
            "paths":[fake(5),fake(6),fake(8)]
        }}
        with patch.dict(os.environ,{
            "MEDICAL_P3063_PATH_MIN_HOPS":"5",
            "MEDICAL_P3063_PATH_MAX_HOPS":"8",
        }):
            result=self.fn["_mh_paths_from_trace"](t)
        self.assertEqual({p["hop_count"] for p in result},{5,6,8})

    def test_p3063_full_path_wins_over_p3061_compact_duplicate(self):
        complete=fake(8,1,with_disease=True)
        compact=fake(8,1,with_disease=False)
        t={
            "p3063_multihop_path_cards":{"paths":[complete]},
            "disease_encyclopedia":{"multi_hop_graph":{
                "verified_paths":[compact]}}
        }
        result=self.fn["_mh_paths_from_trace"](t)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]["disease_name"],"过敏性鼻炎")

    def test_compact_path_uses_verified_single_matched_disease(self):
        compact=fake(8,1,with_disease=False)
        t={"disease_encyclopedia":{
            "multi_hop_graph":{
                "matched_diseases":["过敏性鼻炎"],
                "verified_paths":[compact],
            }
        }}
        result=self.fn["_mh_paths_from_trace"](t)
        self.assertEqual(result[0]["disease_name"],"过敏性鼻炎")
        self.assertNotIn("disease_name",compact)

    def test_no_guess_when_disease_match_ambiguous(self):
        t={"disease_encyclopedia":{"multi_hop_graph":{
            "matched_diseases":["过敏性鼻炎","糖尿病"],
            "verified_paths":[fake(8,1,with_disease=False)],
        }}}
        result=self.fn["_mh_paths_from_trace"](t)
        self.assertNotIn("disease_name",result[0])

    def test_capped_to_four_display_paths(self):
        t={"p3063_multihop_path_cards":{
            "paths":[fake(8,i) for i in range(1,12)]}}
        result=self.fn["_mh_paths_from_trace"](t)
        self.assertEqual(len(result),4)

    def test_html_contains_readable_p01_and_eight(self):
        t={"p3063_multihop_path_cards":{"paths":[fake(8)]}}
        text=self.fn["multihop_path_evidence_html"](t)
        self.assertIn("路径 P01",text)
        self.assertIn("8 跳",text)
        self.assertIn("过敏性鼻炎",text)
        self.assertIn("9 个节点",text)
        self.assertIn("非治疗因果路径",text)

    def test_no_treatment_relation_accepted(self):
        bad=fake(8)
        bad["relationship_types"][0]="TREATS"
        self.assertEqual(self.fn["_mh_paths_from_trace"](
            {"p3063_multihop_path_cards":{"paths":[bad]}}
        ),[])

    def test_substitute_dedicated_read_only_query_after_mixed_result(self):
        t={
          "knowledge_evidence":[{"id":1,"source":"original"}],
          "disease_encyclopedia":{"multi_hop_graph":{
              "verified_paths":[fake(5,1),fake(6,1),fake(7,1),fake(8,1)]
          }}
        }
        called=[]
        fake_adapter=types.ModuleType("medical_p3063_inline_path_evidence_v117")
        fake_adapter.attach_multihop_path_cards=lambda *a,**kw:None
        fake_core=types.ModuleType("medical_p3061_multihop_graph_v117")
        def query(q,**kwargs):
            called.append(kwargs)
            return [fake(8,i) for i in range(2,6)],{
                "status":"MULTIHOP_PATHS_VERIFIED",
                "query_count":1,"found_per_hops":{"8":4}}
        fake_core.retrieve_multihop=query
        with patch.dict(sys.modules,{
            "medical_p3063_inline_path_evidence_v117":fake_adapter,
            "medical_p3061_multihop_graph_v117":fake_core,
        }),patch.dict(os.environ,{
            "MEDICAL_P3063_INLINE_PATH_CARDS":"1",
            "MEDICAL_P3063_PATH_MIN_HOPS":"8",
            "MEDICAL_P3063_PATH_MAX_HOPS":"8",
            "MEDICAL_P3063_PATH_MAX_PATHS":"4",
        }):
            self.fn["populate_multihop_trace_for_display"](
                t,"哪些药品说明书提到了过敏性鼻炎？请展示8跳来源路径。"
            )
        self.assertEqual(len(called),1)
        self.assertEqual((called[0]["min_hops"],called[0]["max_hops"]),(8,8))
        self.assertEqual(len(self.fn["_mh_paths_from_trace"](t)),4)
        self.assertEqual(
            t["p3063_multihop_path_cards"]["status"],
            "MULTIHOP_SOURCE_PATHS_VERIFIED_FOR_DISPLAY"
        )
        self.assertEqual(len(t["knowledge_evidence"]),1)

    def test_no_duplicate_scans_when_cache_has_four_eight_paths(self):
        t={"p3063_multihop_path_cards":{
            "paths":[fake(8,i) for i in range(1,5)]}}
        fake_adapter=types.ModuleType("medical_p3063_inline_path_evidence_v117")
        fake_adapter.attach_multihop_path_cards=lambda *a,**kw:None
        fake_core=types.ModuleType("medical_p3061_multihop_graph_v117")
        def fail(*args,**kwargs):
            raise AssertionError("must not query Neo4j twice")
        fake_core.retrieve_multihop=fail
        with patch.dict(sys.modules,{
            "medical_p3063_inline_path_evidence_v117":fake_adapter,
            "medical_p3061_multihop_graph_v117":fake_core,
        }),patch.dict(os.environ,{"MEDICAL_P3063_INLINE_PATH_CARDS":"1"}):
            self.fn["populate_multihop_trace_for_display"](
                t,"哪些药品说明书提到了过敏性鼻炎？展示8跳路径"
            )

    def test_failure_does_not_delete_original_evidence(self):
        t={"knowledge_evidence":[{"id":1,"source":"medical.json"}]}
        fake_core=types.ModuleType("medical_p3061_multihop_graph_v117")
        fake_core.retrieve_multihop=lambda *a,**kw:(_ for _ in ()).throw(
            ConnectionError("Neo4j unavailable")
        )
        with patch.dict(sys.modules,{
            "medical_p3061_multihop_graph_v117":fake_core
        }),patch.dict(os.environ,{"MEDICAL_P3063_INLINE_PATH_CARDS":"1"}):
            self.fn["populate_multihop_trace_for_display"](
                t,"哪些药品说明书提到了过敏性鼻炎？"
            )
        self.assertEqual(t["knowledge_evidence"],[{"id":1,"source":"medical.json"}])

    def test_still_does_not_modify_models_provenance_scoring(self):
        s=self.source
        self.assertIn("excluded_from_generation",s)
        self.assertIn("excluded_from_citation_scoring",s)
        self.assertIn('multihop_html = multihop_path_evidence_html(trace)',s)


if __name__=="__main__":
    unittest.main()
