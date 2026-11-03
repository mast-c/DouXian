# -*- coding: utf-8 -*-
"""P3-06.3C: offline test for NameError and sidebar cleanup; never touches user app."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import html
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import fix_douxian_re_sidebar_v117 as fix

MODULE_ROOT=Path(__file__).resolve().parent
FIXED=MODULE_ROOT/"修复文件"/"豆馅.py"

def fixture(depth=8,number=1):
    return {
        "hop_count":depth,
        "path_id":f"synthetic_{depth}_{number}",
        "node_ids":[f"syn_{depth}_{number}_{i}" for i in range(depth+1)],
        "node_labels":list({
            5:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
               "SourceRecord","Section"),
            6:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
               "DiseaseCandidate","DiseaseTerm","DiseaseDocument"),
            7:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
               "DiseaseCandidate","DiseaseTerm","DiseaseCandidate","Drug"),
            8:("DiseaseDocument","DiseaseTerm","DiseaseCandidate","Drug",
               "SourceRecord","Section","DiseaseCandidate","DiseaseTerm",
               "DiseaseDocument"),
        }[depth]),
        "relationship_types":list({
            5:("EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
               "MENTIONS_TERM","HAS_CANDIDATE","HAS_RECORD","HAS_SECTION"),
            6:("EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
               "MENTIONS_TERM","HAS_CANDIDATE","HAS_CANDIDATE",
               "MENTIONS_TERM","EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA"),
            7:("EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
               "MENTIONS_TERM","HAS_CANDIDATE","HAS_CANDIDATE",
               "MENTIONS_TERM","MENTIONS_TERM","HAS_CANDIDATE"),
            8:("EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA",
               "MENTIONS_TERM","HAS_CANDIDATE","HAS_RECORD","HAS_SECTION",
               "SUPPORTED_BY","MENTIONS_TERM",
               "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA"),
        }[depth]),
        "disease_name":"过敏性鼻炎",
        "related_disease_name":"枯草热",
        "drug_name":"丙酸氟替卡松鼻喷雾剂",
        "clinical_verified":False,
        "treatment_relation":False,
        "source_status":"third_party_unverified",
        "graph_relationships_verified":True,
    }


def isolated_frontend():
    source=FIXED.read_text("utf-8-sig")
    tree=ast.parse(source)
    methods={
        "_mh_display_config","_mh_one_source_name",
        "_mh_paths_from_trace","_mh_node_name",
        "_mh_one_path_html","multihop_path_evidence_html",
        "_mh_safe","_mh_structurally_valid",
        "trace_details_html","graph_evidence_html",
        "render_saved_trace",
    }
    constants={
        "_MH_PATH_LABELS","_MH_PATH_RELS","_MH_EDGE_DIRECTIONS",
        "_MH_CARD_TYPE","_MH_MAX_PATHS",
    }
    wanted=[]
    for node in tree.body:
        if isinstance(node,ast.Import) and any(
            part.name=="re" and part.asname is None for part in node.names
        ):
            wanted.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name in methods:
            wanted.append(node)
        elif isinstance(node,ast.Assign) and any(
            isinstance(target,ast.Name) and target.id in constants
            for target in node.targets
        ):
            wanted.append(node)
    ns={
        "__file__":str(FIXED),"html":html,"hashlib":hashlib,
        "logging":logging,"json":json,"os":os,"Path":Path,
        "urlsplit":urlsplit,
        "retrieval_status_html":lambda trace:"",
        "trace_heading":lambda trace:"医疗检索完成",
        "_web_sources":lambda trace:[],
        "_web_queries":lambda trace:[],
        "elapsed_label":lambda value:"0s",
        "describe_tool_event":lambda event:"检索完成",
        "render_html_fragment":lambda body:None,
    }
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=wanted,type_ignores=[])
    ),str(FIXED),"exec"),ns)
    return ns,source


class RegressionTests(unittest.TestCase):
    def test_actual_fixed_file_parses(self):
        ast.parse(FIXED.read_text("utf-8-sig"))

    def test_global_re_import_exists_in_complete_file(self):
        tree=ast.parse(FIXED.read_text("utf-8-sig"))
        self.assertTrue(any(
            isinstance(n,ast.Import)
            and any(i.name=="re" and i.asname is None for i in n.names)
            for n in tree.body
        ))

    def test_saved_chat_trace_no_name_error(self):
        ns,_=isolated_frontend()
        self.assertIn("re",ns)
        text=ns["trace_details_html"]({
            "events":[],"knowledge_evidence":[],"total_seconds":3
        })
        self.assertIsInstance(text,str)

    def test_render_saved_trace_no_name_error(self):
        ns,_=isolated_frontend()
        captured=[]
        ns["render_html_fragment"]=lambda body:captured.append(body)
        ns["render_saved_trace"]({
            "events":[],"knowledge_evidence":[],
            "disease_encyclopedia":{"route":"DISEASE_ONLY"}
        })
        self.assertEqual(len(captured),1)

    def test_four_eight_hop_paths_inside_original_evidence_area(self):
        ns,_=isolated_frontend()
        six=[
            {"id":i+1,"source":f"Source {i}","content":"Original text"}
            for i in range(6)
        ]
        trace={
            "knowledge_evidence":six,
            "p3063_multihop_path_cards":{
                "frontend_question":"请展示8跳来源路径",
                "paths":[fixture(8,i+1) for i in range(4)],
            },
            "events":[],
        }
        text=ns["trace_details_html"](trace)
        self.assertIn("实际命中的知识库证据（6 条）",text)
        self.assertEqual(text.count('class="kb-mh-card"'),4)
        self.assertEqual(text.count('class="kb-mh-edge"'),32)
        self.assertLess(text.index('class="kb-evidence-list"'),
                        text.index('class="kb-mh-group"'))

    def test_user_eight_hop_request_filters_old_mixed_trace(self):
        ns,_=isolated_frontend()
        trace={
            "p3063_multihop_path_cards":{
                "frontend_question":"哪些药品说明书提到了过敏性鼻炎？请展示8跳来源路径",
                "paths":[fixture(5),fixture(6),fixture(7),fixture(8)],
            }
        }
        with patch.dict(os.environ,{
            "MEDICAL_P3063_PATH_MIN_HOPS":"5",
            "MEDICAL_P3063_PATH_MAX_HOPS":"8",
        }):
            result=ns["_mh_paths_from_trace"](trace)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]["hop_count"],8)

    def test_unverified_relation_not_promoted(self):
        ns,_=isolated_frontend()
        path=fixture(8)
        path["relationship_types"][0]="TREATS"
        self.assertFalse(ns["_mh_structurally_valid"](path))

    def test_sidebar_page_exact_name_in_installer(self):
        self.assertEqual(fix.SIDEBAR_PAGE,"06_Multihop_Evidence.py")

    def test_dry_run_no_file_or_page_move(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            root.joinpath("pages").mkdir()
            main=root/"豆馅.py"
            main.write_bytes(FIXED.read_bytes().replace(
                b"import re\n",b"",1
            ))
            extra=root/"pages"/fix.SIDEBAR_PAGE
            extra.write_text("import streamlit as st",encoding="utf-8")
            before=main.read_bytes()
            result=fix.install(root,apply=False)
            self.assertEqual(result["status"],"READY_TO_APPLY")
            self.assertEqual(main.read_bytes(),before)
            self.assertTrue(extra.exists())

    def test_apply_one_line_preserves_everything_else_and_hides_page(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            main=root/"豆馅.py"
            root.joinpath("pages").mkdir()
            extra=root/"pages"/fix.SIDEBAR_PAGE
            keep=root/"pages"/"other_page.py"
            extra.write_text("# legacy multipage",encoding="utf-8")
            keep.write_text("# keep me",encoding="utf-8")
            before=FIXED.read_bytes().replace(b"import re\n",b"",1)
            before+=b"\n# personal user customization is retained\n"
            main.write_bytes(before)
            output=fix.install(root,apply=True)
            self.assertEqual(output["status"],"APPLIED")
            self.assertFalse(extra.exists())
            self.assertTrue(keep.exists())
            self.assertEqual(Path(output["saved_frontend"]).read_bytes(),before)
            self.assertTrue(Path(output["saved_sidebar_page"]).is_file())
            after=main.read_bytes()
            self.assertIn(b"import re\n",after)
            self.assertEqual(after.replace(b"import re\n",b"",1),before)
            ast.parse(main.read_text("utf-8-sig"))

    def test_second_install_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            main=root/"豆馅.py"
            main.write_bytes(FIXED.read_bytes())
            first=fix.install(root,apply=True)
            second=fix.install(root,apply=True)
            self.assertEqual(first["status"],"ALREADY_FIXED")
            self.assertEqual(second["status"],"ALREADY_FIXED")

    def test_preserves_windows_bom_crlf(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            main=root/"豆馅.py"
            body=FIXED.read_text("utf-8").replace(
                "import re\n","",1
            ).replace("\n","\r\n")
            main.write_bytes(b"\xef\xbb\xbf"+body.encode("utf-8"))
            output=fix.install(root,apply=True)
            self.assertEqual(output["status"],"APPLIED")
            after=main.read_bytes()
            self.assertTrue(after.startswith(b"\xef\xbb\xbf"))
            self.assertIn(b"import re\r\n",after)
            self.assertNotIn(b"\r\r\n",after)

    def test_unrecognized_frontend_safely_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            main=root/"豆馅.py"
            main.write_text("print('another app')\n",encoding="utf-8")
            with self.assertRaisesRegex(fix.FixBlocked,"未找到"):
                fix.install(root,apply=True)
            self.assertEqual(
                main.read_text("utf-8"),"print('another app')\n"
            )

if __name__=="__main__":
    unittest.main()
