# -*- coding: utf-8 -*-
"""P3-06.3 offline regression for display-only inline graph path cards."""
from __future__ import annotations

from copy import deepcopy
import ast
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import medical_p3063_inline_path_evidence_v117 as adapter
import medical_p3063_patch_rag_inline_evidence_v117 as patcher
from medical_p3061_multihop_graph_v117 import PATH_LABELS,PATH_TYPES


def fake_path(hops=8,index=1):
    return {
        "hop_count":hops,
        "path_id":f"synthetic_{hops}_{index}",
        "node_ids":[f"node_{hops}_{index}_{i}" for i in range(hops+1)],
        "node_labels":list(PATH_LABELS[hops]),
        "relationship_types":list(PATH_TYPES[hops]),
        "disease_name":"过敏性鼻炎",
        "drug_name":"丙酸氟替卡松鼻喷雾剂",
        "related_disease_name":"枯草热" if hops>=6 else None,
        "related_drug_name":"虚构药乙" if hops==7 else None,
        "section_field":"indication" if hops in (5,8) else None,
        "record_id":"recordA" if hops in (5,8) else None,
        "section_id":"sectionA" if hops in (5,8) else None,
        "candidate_quote":"来源候选提及过敏性鼻炎",
        "candidate_id":f"candidate_{index}",
        "section_text":"LONG_RAW_CONTENT_NOT_TO_EXPORT"*300,
        "source_status":"third_party_unverified",
        "clinical_verified":False,
        "treatment_relation":False,
        "clinical_truth_verified":False,
        "graph_relationships_verified":True,
    }


def initial_trace():
    return {
        "knowledge_evidence":[
            {"id":i+1,"source":f"drug_record_{i+1}",
             "content":f"original text {i+1}",
             "evidence_type":"unverified_drug_disease_candidate"}
            for i in range(6)
        ],
        "retrieval_pipeline":{"merge_count":6},
        "knowledge_sources":["source A"],
        "evidence_coverage":{"decision":"supported"},
        "disease_encyclopedia":{
            "route":"MIXED","status":"VERIFIED_ORIGINAL_TEXT",
            "graph_disease_bridge":{
                "status":"GRAPH_EXACT_OID_PROVENANCE_VERIFIED"
            }
        },
    }


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.trace=initial_trace()
        self.paths=[fake_path(8,i) for i in range(1,5)]
        self.calls=[]
        def fake_retrieve(question,**kwargs):
            self.calls.append((question,kwargs))
            return deepcopy(self.paths),{
                "status":"MULTIHOP_PATHS_VERIFIED",
                "query_count":1,
                "found_per_hops":{"8":4},
                "graph_audit_status":"FULL_GRAPH_LINK_AUDIT_OK",
                "max_hops_observed":8,
                "requested_hops":[8],
                "elapsed_ms":23.0
            }
        self.mock=patch.object(adapter,"retrieve_multihop",fake_retrieve)
        self.mock.start()
        self.env=patch.dict(os.environ,{
            "MEDICAL_P3063_INLINE_PATH_CARDS":"1",
            "MEDICAL_P3063_APPEND_GENERIC_CARDS":"1",
            "MEDICAL_P3063_PATH_MIN_HOPS":"8",
            "MEDICAL_P3063_PATH_MAX_HOPS":"8",
            "MEDICAL_P3063_PATH_MAX_PATHS":"4",
            "MEDICAL_P3063_PATH_PER_TEMPLATE":"4",
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.mock.stop()

    def append(self,question="哪些药品说明书提到了过敏性鼻炎？"):
        return adapter.attach_multihop_path_cards(
            self.trace,question,project_root=Path(__file__).resolve().parent
        )

    def test_five_six_seven_eight_are_allowed(self):
        self.assertEqual(adapter.TEMPLATES,{5,6,7,8})
        for h in (5,6,7,8):
            self.assertTrue(adapter._valid_path(fake_path(h)))

    def test_four_eight_hop_cards_after_six_evidence(self):
        result=self.append()
        self.assertEqual(result["display_path_card_count"],4)
        self.assertEqual(len(self.trace["knowledge_evidence"]),10)
        self.assertEqual(
            [c["id"] for c in self.trace["knowledge_evidence"][6:]],
            [7,8,9,10]
        )

    def test_original_evidence_unchanged(self):
        before=deepcopy(self.trace)
        self.append()
        self.assertEqual(self.trace["knowledge_evidence"][:6],
                         before["knowledge_evidence"])
        self.assertEqual(self.trace["retrieval_pipeline"],
                         before["retrieval_pipeline"])
        self.assertEqual(self.trace["knowledge_sources"],
                         before["knowledge_sources"])
        self.assertEqual(self.trace["evidence_coverage"],
                         before["evidence_coverage"])

    def test_self_rag_reassignment_reattaches_without_second_query(self):
        self.append()
        self.trace["knowledge_evidence"]=initial_trace()["knowledge_evidence"]
        self.append()
        self.assertEqual(len(self.calls),1)
        self.assertEqual(len(self.trace["knowledge_evidence"]),10)

    def test_repeated_hook_idempotent(self):
        self.append()
        self.append()
        self.assertEqual(len(self.calls),1)
        self.assertEqual(len(self.trace["knowledge_evidence"]),10)

    def test_same_drug_disease_paths_kept_separate(self):
        self.append()
        paths=self.trace["knowledge_evidence"][6:]
        self.assertEqual(len({x["path_id"] for x in paths}),4)
        self.assertEqual(len({x["drug_name"] for x in paths}),1)

    def test_cards_do_not_claim_medical_proof(self):
        self.append()
        for card in self.trace["knowledge_evidence"][6:]:
            self.assertEqual(card["evidence_type"],adapter.CARD_TYPE)
            self.assertFalse(card["counts_as_answer_evidence"])
            self.assertFalse(card["counts_as_retrieval_hit"])
            self.assertFalse(card["is_answer_citation"])
            self.assertFalse(card["clinical_verified"])
            self.assertFalse(card["treatment_relation"])
            self.assertIn("不代表该药治疗",card["content"])
            self.assertEqual(card["source_status"],"third_party_unverified")

    def test_each_card_has_actual_eight_steps(self):
        self.append()
        for card in self.trace["knowledge_evidence"][6:]:
            self.assertEqual(len(card["node_ids"]),9)
            self.assertEqual(len(card["relationship_types"]),8)
            self.assertIn("DiseaseDocument",card["content"])
            self.assertIn("Path ID",card["content"])

    def test_no_full_section_text_cached(self):
        self.append()
        meta=self.trace[adapter.TRACE_KEY]
        for item in meta["paths"]:
            self.assertNotIn("section_text",item)
        self.assertEqual(meta["source_evidence_count"],6)
        self.assertEqual(meta["displayed_card_count"],10)
        self.assertEqual(meta["graph_trace"]["query_count"],1)

    def test_zero_new_paths_no_fake_rows(self):
        self.paths=[]
        self.append()
        self.assertEqual(len(self.trace["knowledge_evidence"]),6)
        self.assertEqual(self.trace[adapter.TRACE_KEY]["path_cards_count"],0)

    def test_graph_driver_failure_leaves_originals(self):
        with patch.object(adapter,"retrieve_multihop",
                          side_effect=ConnectionError("fake outage")):
            self.append()
        self.assertEqual(len(self.trace["knowledge_evidence"]),6)
        self.assertEqual(
            self.trace[adapter.TRACE_KEY]["status"],
            "PATH_QUERY_BLOCKED_KEEP_ORIGINAL_EVIDENCE"
        )

    def test_invalid_treats_edge_rejected(self):
        invalid=fake_path()
        invalid["relationship_types"][4]="TREATS"
        self.paths=[invalid]
        self.append()
        self.assertEqual(len(self.trace["knowledge_evidence"]),6)

    def test_invalid_clinical_status_rejected(self):
        self.paths=[{**fake_path(),"clinical_verified":True}]
        self.append()
        self.assertEqual(len(self.trace["knowledge_evidence"]),6)

    def test_invalid_graph_verification_flag_rejected(self):
        self.paths=[{**fake_path(),"graph_relationships_verified":False}]
        self.append()
        self.assertEqual(len(self.trace["knowledge_evidence"]),6)

    def test_disabled_no_query_and_no_cards(self):
        with patch.dict(os.environ,{"MEDICAL_P3063_INLINE_PATH_CARDS":"0"}):
            result=self.append()
        self.assertEqual(result["status"],"DISABLED")
        self.assertEqual(len(self.calls),0)
        self.assertEqual(len(self.trace["knowledge_evidence"]),6)

    def test_missing_evidence_list_no_query(self):
        self.trace.pop("knowledge_evidence")
        self.assertEqual(self.append()["status"],"SKIPPED_NO_EVIDENCE_LIST")
        self.assertEqual(len(self.calls),0)

    def test_drug_only_route_no_graph_scan(self):
        self.trace["disease_encyclopedia"]={"route":"DRUG_ONLY"}
        self.append()
        self.assertEqual(len(self.calls),0)
        self.assertEqual(
            self.trace[adapter.TRACE_KEY]["status"],"SKIPPED_NO_DISEASE_ROUTE"
        )

    def test_all_depths_configurable(self):
        self.paths=[fake_path(h) for h in [5,6,7,8]]
        with patch.dict(os.environ,{
            "MEDICAL_P3063_PATH_MIN_HOPS":"5",
            "MEDICAL_P3063_PATH_MAX_HOPS":"8",
            "MEDICAL_P3063_PATH_MAX_PATHS":"16"
        }):
            self.append()
        depths=[p["hop_count"] for p in
                self.trace["knowledge_evidence"][6:]]
        self.assertEqual(depths,[5,6,7,8])
        self.assertEqual(self.calls[0][1]["max_paths"],16)

    def test_generic_cards_can_be_disabled_when_using_rich_ui(self):
        with patch.dict(os.environ,{
            "MEDICAL_P3063_APPEND_GENERIC_CARDS":"0"
        }):
            outcome=self.append()
        self.assertEqual(outcome["display_path_card_count"],0)
        self.assertEqual(outcome["paths_available_for_rich_ui"],4)
        self.assertEqual(len(self.trace["knowledge_evidence"]),6)
        self.assertEqual(self.trace[adapter.TRACE_KEY]["path_cards_count"],4)

    def test_clamps_invalid_env_limits(self):
        with patch.dict(os.environ,{
            "MEDICAL_P3063_PATH_MIN_HOPS":"100",
            "MEDICAL_P3063_PATH_MAX_HOPS":"-2",
            "MEDICAL_P3063_PATH_MAX_PATHS":"100",
            "MEDICAL_P3063_PATH_PER_TEMPLATE":"100"
        }):
            self.append()
        kw=self.calls[0][1]
        self.assertEqual((kw["min_hops"],kw["max_hops"]),(8,8))
        self.assertEqual(kw["max_paths"],16)
        self.assertEqual(kw["per_template"],6)

    def test_trace_hook_only_no_chroma_neo4j_writes(self):
        self.append()
        meta=self.trace[adapter.TRACE_KEY]
        self.assertEqual(meta["chroma_writes"],0)
        self.assertEqual(meta["graph_writes"],0)
        self.assertTrue(meta["excluded_from_generation"])
        self.assertTrue(meta["excluded_from_citation_scoring"])

    def test_rich_renderer_shows_multiple_cards_inside_existing_evidence(self):
        import sys
        import types
        self.append()
        calls={"containers":0,"graphs":0,"tables":0}
        fake=types.ModuleType("streamlit")
        fake.markdown=lambda *a,**kw:None
        fake.caption=lambda *a,**kw:None
        fake.graphviz_chart=lambda *a,**kw:calls.__setitem__(
            "graphs",calls["graphs"]+1
        )
        fake.dataframe=lambda *a,**kw:calls.__setitem__(
            "tables",calls["tables"]+1
        )
        fake.code=lambda *a,**kw:None
        fake.write=lambda *a,**kw:None
        fake.checkbox=lambda *a,**kw:False
        class Container:
            def __enter__(self):
                calls["containers"]+=1
                return self
            def __exit__(self,*args):
                return False
        fake.container=lambda *a,**kw:Container()
        with patch.dict(sys.modules,{"streamlit":fake}):
            count=adapter.render_rich_multihop_paths_inside_evidence(
                self.trace
            )
        self.assertEqual(count,4)
        self.assertEqual(calls["containers"],4)
        self.assertEqual(calls["graphs"],4)
        self.assertEqual(calls["tables"],0)

    def test_rich_renderer_without_cached_paths_no_output(self):
        import sys
        import types
        fake=types.ModuleType("streamlit")
        with patch.dict(sys.modules,{"streamlit":fake}):
            result=adapter.render_rich_multihop_paths_inside_evidence(
                initial_trace()
            )
        self.assertEqual(result,0)

    def test_inline_dot_renders_exact_eight_relationships(self):
        import re
        dot=adapter._inline_graph_dot(fake_path(8))
        self.assertEqual(
            len(re.findall(r"^n\d+ -> n\d+\s+\[",dot,re.MULTILINE)),
            8
        )
        self.assertIn("SUPPORTED_BY",dot)
        self.assertIn("枯草热",dot)

    def test_inline_dot_escapes_report_string(self):
        import re
        row=fake_path(8)
        row["drug_name"]='Injected"; new_node -> n0; "'
        dot=adapter._inline_graph_dot(row)
        self.assertNotIn("\nnew_node ->",dot)
        self.assertEqual(
            len(re.findall(r"^n\d+ -> n\d+\s+\[",dot,re.MULTILINE)),
            8
        )

    def test_rich_renderer_exists_and_is_lazy(self):
        src=Path(adapter.__file__).read_text("utf-8")
        self.assertIn("def render_rich_multihop_paths_inside_evidence(",src)
        self.assertIn("st.graphviz_chart(",src)
        self.assertNotIn("\nimport streamlit as st\n",src)


class ASTPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture="""from pathlib import Path
from medical_p3045_graph_disease_retrieval_v117 import (
    search_graph_enhanced_disease_encyclopedia
)
class RagService:
    def _call_tool(self, trace):
        trace["knowledge_evidence"]=[]
    def _stream_agent(self,inputs,session_id):
        trace={"knowledge_evidence":[]}
        question=inputs["input"]
        trace["knowledge_evidence"]=[
            {"id":i,"content":"original"} for i in range(1,3)
        ]
        if inputs.get("need_guard"):
            ui_evidence=[{"id":1,"content":"filtered"}]
            trace["knowledge_evidence"]=ui_evidence
        return trace
"""

    def test_inserts_two_hooks_in_medical_stream(self):
        data=patcher.transform(self.fixture)
        ast.parse(data)
        self.assertEqual(
            data.count("_p3063_attach_multihop_path_cards("),2
        )

    def test_does_not_touch_unrelated_tool_evidence_hook(self):
        data=patcher.transform(self.fixture)
        tree=ast.parse(data)
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef))
        tool=next(n for n in cls.body if n.name=="_call_tool")
        calls=[n for n in ast.walk(tool)
               if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name)
               and n.func.id=="_p3063_attach_multihop_path_cards"]
        self.assertEqual(calls,[])

    def test_requires_p3045_existing(self):
        with self.assertRaisesRegex(patcher.InlinePatchBlocked,"P3-04.5"):
            patcher.transform(
                self.fixture.replace(
                    'medical_p3045_graph_disease_retrieval_v117',
                    'other_runtime')
            )

    def test_double_patch_blocked(self):
        with self.assertRaisesRegex(patcher.InlinePatchBlocked,"重复"):
            patcher.transform(patcher.transform(self.fixture))

    def test_missing_second_replacement_blocks(self):
        source=self.fixture.replace(
            'trace["knowledge_evidence"]=ui_evidence',
            'trace["other"]=ui_evidence')
        with self.assertRaisesRegex(patcher.InlinePatchBlocked,"恰好2处"):
            patcher.transform(source)

    def _copy_dependencies(self,folder):
        root=Path(__file__).resolve().parent
        for name in [
            "medical_p3063_inline_path_evidence_v117.py",
            "medical_p3061_multihop_graph_v117.py",
            "medical_p3045_graph_disease_retrieval_v117.py",
        ]:
            shutil.copy2(root/name,folder/name)

    def test_dry_run_and_apply_with_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            dest=Path(folder)
            self._copy_dependencies(dest)
            rag=dest/"rag.py"
            rag.write_text(self.fixture,encoding="utf-8")
            before=rag.read_bytes()
            preview=patcher.patch(rag)
            self.assertEqual(preview["status"],"READY_TO_APPLY")
            self.assertEqual(rag.read_bytes(),before)
            saved=patcher.patch(rag,apply=True)
            self.assertEqual(saved["status"],"APPLIED")
            self.assertEqual(Path(saved["backup"]).read_bytes(),before)
            ast.parse(rag.read_text("utf-8"))
            with self.assertRaises(patcher.InlinePatchBlocked):
                patcher.patch(rag,apply=True)

    def test_windows_crlf_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            dest=Path(folder)
            self._copy_dependencies(dest)
            rag=dest/"rag.py"
            rag.write_bytes(
                self.fixture.replace("\n","\r\n").encode("utf-8")
            )
            patcher.patch(rag,apply=True)
            self.assertIn(b"\r\n",rag.read_bytes())
            self.assertNotIn(b"\r\r\n",rag.read_bytes())

    def test_with_p3061_import_keeps_prior_runtime(self):
        imported="""from medical_p3061_multihop_graph_v117 import (
    search_multihop_enhanced_disease_encyclopedia
)
"""
        result=patcher.transform(imported+self.fixture)
        ast.parse(result)
        self.assertIn("search_multihop_enhanced_disease_encyclopedia",result)
        self.assertEqual(
            result.count("_p3063_attach_multihop_path_cards("),2
        )

if __name__=="__main__":
    unittest.main()
