# -*- coding: utf-8 -*-
"""离线测试：豆馅 Streamlit 原证据区嵌入 5～8 跳图谱路径。

不会导入/执行 Streamlit 主程序、不会连接 Neo4j/Chroma。
在项目根目录解压后：
    python -m unittest -v test_douxian_multihop_frontend_v117.py
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import html
from html.parser import HTMLParser
import json
import logging
import os
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

TARGET=Path(
    os.environ.get("DOUXIAN_FRONTEND_TEST_TARGET")
    or Path(__file__).with_name("豆馅.py")
).resolve()

LABELS={
    5:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
       "SourceRecord","Section"),
    6:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
       "DiseaseCandidate","DiseaseTerm","DiseaseDocument"),
    7:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
       "DiseaseCandidate","DiseaseTerm","DiseaseCandidate","Drug"),
    8:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
       "SourceRecord","Section","DiseaseCandidate","DiseaseTerm",
       "DiseaseDocument"),
}


def fixture_path(hops=8,variant=1):
    return {
        "path_id":f"test_{hops}_{variant}",
        "hop_count":hops,
        "node_ids":[f"synthetic_{hops}_{variant}_node{i}"
                    for i in range(hops+1)],
        "node_labels":list(LABELS[hops]),
        "relationship_types":{
            5:["EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
               "MENTIONS_TERM","HAS_CANDIDATE","HAS_RECORD","HAS_SECTION"],
            6:["EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
               "MENTIONS_TERM","HAS_CANDIDATE","HAS_CANDIDATE",
               "MENTIONS_TERM","EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA"],
            7:["EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
               "MENTIONS_TERM","HAS_CANDIDATE","HAS_CANDIDATE",
               "MENTIONS_TERM","MENTIONS_TERM","HAS_CANDIDATE"],
            8:["EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
               "MENTIONS_TERM","HAS_CANDIDATE","HAS_RECORD","HAS_SECTION",
               "SUPPORTED_BY","MENTIONS_TERM",
               "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA"],
        }[hops],
        "disease_name":"过敏性鼻炎",
        "drug_name":"丙酸氟替卡松鼻喷雾剂",
        "related_disease_name":"枯草热" if hops>=6 else None,
        "related_drug_name":"虚构第二药品" if hops==7 else None,
        "candidate_quote":"数据仅提到两个术语",
        "source_status":"third_party_unverified",
        "clinical_verified":False,
        "graph_relationships_verified":True,
        "treatment_relation":False,
        "section_id":"section_synthetic" if hops in (5,8) else None,
        "section_field":"indication" if hops in (5,8) else None,
    }


def initial_trace(path_count=4,include_generic=True):
    real=[
        {"id":i+1,"source":f"真实文档{i+1}","section_path":"症状",
         "content":"原文提及但未经医学核验",
         "evidence_type":"unverified_disease_encyclopedia"}
        for i in range(6)
    ]
    paths=[fixture_path(8,i+1) for i in range(path_count)]
    if include_generic:
        for i,item in enumerate(paths):
            real.append({
                **item, "id":7+i,
                "evidence_type":"unverified_multihop_path_display_only",
                "content":"路径仅展示，不是医学引用",
            })
    return {
        "knowledge_evidence":real,
        "p3063_multihop_path_cards":{
            "status":"MULTIHOP_PATH_CARDS_APPENDED",
            "paths":paths,
        },
        "events":[],
    }


def load_functions():
    text=TARGET.read_text(encoding="utf-8-sig")
    tree=ast.parse(text)
    constants={
        "_MH_PATH_LABELS","_MH_PATH_RELS","_MH_EDGE_DIRECTIONS",
        "_MH_CARD_TYPE","_MH_MAX_PATHS",
    }
    names={
        "graph_evidence_html", "_mh_safe","_mh_structurally_valid",
        "_mh_paths_from_trace","_mh_node_name","_mh_one_path_html",
        "multihop_path_evidence_html","trace_details_html",
        "populate_multihop_trace_for_display",
        "_mh_trace_for_evidence_export",
    }
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id in constants
            for t in node.targets
        ):
            selected.append(node)
    expect=names-{n.name for n in selected if isinstance(n,ast.FunctionDef)}
    if expect:
        raise AssertionError(f"缺少前端集成函数：{sorted(expect)}")
    isolate=ast.fix_missing_locations(ast.Module(
        body=selected,type_ignores=[]
    ))
    scope={
        "__file__":str(TARGET),
        "html":html,"json":json,"hashlib":hashlib,
        "logging":logging,"os":os,"Path":Path,"urlsplit":urlsplit,
        "retrieval_status_html":lambda trace:"",
        "trace_heading":lambda trace:"医疗检索完成",
        "_web_sources":lambda trace:[],
        "_web_queries":lambda trace:[],
        "elapsed_label":lambda value:"3s",
        "describe_tool_event":lambda event:"已执行",
    }
    exec(compile(isolate,str(TARGET),"exec"),scope)
    return scope,text


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns,cls.source=load_functions()

    def test_main_app_ast_passes(self):
        ast.parse(self.source)

    def test_preserved_conversation_and_chat_store_logic(self):
        for value in ("ChatStore()", "RagService()",
                      "render_saved_trace(msg.get(\"trace\"))",
                      'st.write_stream(stream_with_timing(stream))',
                      "make_export_payload("):
            self.assertIn(value,self.source)

    def test_path_graph_directly_inside_original_trace_panel(self):
        body=self.ns["trace_details_html"](initial_trace())
        self.assertIn('实际命中的知识库证据（6 条）',body)
        self.assertIn('多跳图谱来源路径',body)
        self.assertIn('<section class="kb-mh-group"',body)
        self.assertIn('<details class="trace-panel">',body)
        self.assertLess(
            body.index('class="kb-evidence-list"'),
            body.index('class="kb-mh-group"')
        )
        self.assertLess(
            body.index('class="kb-mh-group"'),
            body.index('</div></details>',body.index('class="kb-mh-group"'))
        )

    def test_original_citations_six_and_graph_paths_four(self):
        body=self.ns["trace_details_html"](initial_trace())
        self.assertIn('实际命中的知识库证据（6 条）',body)
        self.assertEqual(body.count('class="kb-evidence-title"'),6)
        self.assertEqual(body.count('class="kb-mh-card"'),4)
        self.assertEqual(body.count('class="kb-mh-node kb-mh-node-type-'),36)
        self.assertEqual(body.count('class="kb-mh-edge"'),32)

    def test_all_depths_5_to_8_supported(self):
        for depth in (5,6,7,8):
            path=fixture_path(depth)
            self.assertTrue(self.ns["_mh_structurally_valid"](path))
            text=self.ns["_mh_one_path_html"](path,1)
            self.assertEqual(
                text.count('class="kb-mh-node kb-mh-node-type-'),
                depth+1
            )
            self.assertEqual(text.count('class="kb-mh-edge"'),depth)

    def test_duplicate_drug_disease_path_names_preserve_different_ids(self):
        paths=self.ns["_mh_paths_from_trace"](initial_trace())
        self.assertEqual(len(paths),4)
        self.assertEqual(len({x["path_id"] for x in paths}),4)

    def test_path_same_identity_in_two_trace_sources_deduplicated(self):
        obj=initial_trace()
        obj["disease_encyclopedia"]={
            "multi_hop_graph":{"verified_paths":deepcopy(
                obj["p3063_multihop_path_cards"]["paths"]
            )}
        }
        self.assertEqual(len(self.ns["_mh_paths_from_trace"](obj)),4)

    def test_p3061_verified_paths_used_without_p3063(self):
        obj={"disease_encyclopedia":{
            "multi_hop_graph":{
                "verified_paths":[fixture_path(6,1),fixture_path(8,2)],
                "status":"MULTIHOP_PATHS_VERIFIED",
            }
        }}
        data=self.ns["_mh_paths_from_trace"](obj)
        self.assertEqual([p["hop_count"] for p in data],[6,8])

    def test_old_plain_evidence_without_paths_is_unchanged(self):
        obj=initial_trace(0,include_generic=False)
        body=self.ns["trace_details_html"](obj)
        self.assertIn('实际命中的知识库证据（6 条）',body)
        self.assertNotIn('<section class="kb-mh-group"',body)

    def test_xss_escape_in_node_and_drug_name(self):
        obj=initial_trace(1,include_generic=False)
        item=obj["p3063_multihop_path_cards"]["paths"][0]
        item["drug_name"]="<script>alert(1)</script>"
        item["node_ids"][3]='node" onmouseover="alert(1)'
        body=self.ns["trace_details_html"](obj)
        self.assertIn("&lt;script&gt;",body)
        self.assertNotIn("<script>",body)
        self.assertNotIn('onmouseover="alert(1)"',body)

    def test_wrong_relationship_type_is_rejected(self):
        path=fixture_path(8)
        path["relationship_types"][1]="TREATS"
        self.assertFalse(self.ns["_mh_structurally_valid"](path))

    def test_clinical_verified_is_rejected(self):
        path=fixture_path(8)
        path["clinical_verified"]=True
        self.assertFalse(self.ns["_mh_structurally_valid"](path))

    def test_path_cycle_is_rejected(self):
        path=fixture_path(8)
        path["node_ids"][5]=path["node_ids"][1]
        self.assertFalse(self.ns["_mh_structurally_valid"](path))

    def test_display_only_path_rows_removed_from_export_gold(self):
        trace=initial_trace()
        filtered=self.ns["_mh_trace_for_evidence_export"](trace)
        self.assertEqual(len(filtered["knowledge_evidence"]),6)
        self.assertEqual(len(trace["knowledge_evidence"]),10)
        self.assertEqual(len(filtered["p3063_multihop_path_cards"]["paths"]),4)

    def test_no_path_cards_no_unnecessary_copy(self):
        trace=initial_trace(0,include_generic=False)
        self.assertIs(self.ns["_mh_trace_for_evidence_export"](trace),trace)

    def test_saved_trace_renders_paths_without_live_database(self):
        html_text=self.ns["trace_details_html"](initial_trace())
        self.assertIn("路径与上方原文证据分开展示",html_text)
        self.assertIn("Section ID：",html_text)

    def test_source_graph_slot_count_does_not_touch_citations(self):
        source=self.source
        self.assertIn("if item.get(\"evidence_type\") == _MH_CARD_TYPE:",source)
        self.assertIn(
            "multihop_html = multihop_path_evidence_html(trace)",source
        )

    def test_path_population_is_before_persisting_assistant_message(self):
        self.assertLess(
            self.source.index(
                "populate_multihop_trace_for_display(trace, pending)"
            ),
            self.source.index(
                'st.session_state["chat_store"].append_message(\\n'
            ) if False else self.source.rfind(
                'st.session_state["chat_store"].append_message('
            )
        )

    def test_drug_only_query_legacy_skip_gets_one_read_only_fallback(self):
        scope=self.ns
        calls=[]
        original=initial_trace(0,include_generic=False)
        original["disease_encyclopedia"]={"route":"DRUG_ONLY"}
        fake_adapter=types.ModuleType(
            "medical_p3063_inline_path_evidence_v117"
        )
        def attach(tr,q,project_root):
            tr["p3063_multihop_path_cards"]={
                "status":"SKIPPED_NO_DISEASE_ROUTE","paths":[]
            }
        fake_adapter.attach_multihop_path_cards=attach
        fake_graph=types.ModuleType("medical_p3061_multihop_graph_v117")
        def retrieve(q,**opts):
            calls.append(opts)
            return [fixture_path(8)],{
                "status":"MULTIHOP_PATHS_VERIFIED",
                "query_count":1,"found_per_hops":{"8":1}
            }
        fake_graph.retrieve_multihop=retrieve
        with patch.dict(sys.modules,{
            "medical_p3063_inline_path_evidence_v117":fake_adapter,
            "medical_p3061_multihop_graph_v117":fake_graph,
        }),patch.dict(os.environ,{
            "MEDICAL_P3063_INLINE_PATH_CARDS":"1",
            "MEDICAL_P3063_PATH_MIN_HOPS":"8",
            "MEDICAL_P3063_PATH_MAX_HOPS":"8",
        }):
            scope["populate_multihop_trace_for_display"](
                original,"哪些药品说明书提到了过敏性鼻炎？"
            )
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]["min_hops"],8)
        self.assertEqual(calls[0]["max_hops"],8)
        self.assertEqual(len(scope["_mh_paths_from_trace"](original)),1)
        self.assertEqual(len(original["knowledge_evidence"]),6)

    def test_disabled_flag_does_not_run_direct_neo4j(self):
        graph=types.ModuleType("medical_p3061_multihop_graph_v117")
        graph.retrieve_multihop=lambda *args,**kw:(
            self.fail("disabled mode must not access graph"))
        with patch.dict(sys.modules,{
            "medical_p3061_multihop_graph_v117":graph,
        }),patch.dict(os.environ,{
            "MEDICAL_P3063_INLINE_PATH_CARDS":"0",
        }):
            self.ns["populate_multihop_trace_for_display"](
                initial_trace(0),"哪些药品说明书提到了过敏性鼻炎？"
            )


if __name__=="__main__":
    unittest.main()
