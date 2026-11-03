# -*- coding: utf-8 -*-
"""Offline P3-07.1 read-only interaction evidence tests."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import medical_p3071_drug_interaction_v117 as core
import medical_p3071_interaction_ui_v117 as ui
import medical_p3071_patch_douxian_v117 as installer


def row(name="布洛芬片",ingredient="ibuprofen",field="drug_interactions",
        text="本品与肝素、双香豆素等抗凝药同用时，可导致凝血酶原时间延长，增加出血倾向。",
        n=1):
    return dict(drug_id=f"drug_{ingredient}_{n}",drug_name=name,
                record_id=f"record_{ingredient}_{n}",
                section_id=f"section_{ingredient}_{n}",
                section_field=field,section_text=text,
                source_url="https://www.yaopinnet.com/huayao/test.htm",
                authority_status="third_party_unverified")


def provider(*,ingredient,limit,root):
    if ingredient=="ibuprofen":
        return [row(name="布洛芬片",n=1),
                row(name="布洛芬胶囊",n=2),
                row(name="布洛芬缓释胶囊",n=3)][:limit]
    return [row(
        name="华法林钠片",ingredient="warfarin",
        text="与非甾体抗炎药同用时可能增加出血风险。"
    )][:limit]


class Neo4jDriverContract(unittest.TestCase):
    """Regression: Query(timeout) may ONLY be used with session.run."""

    def test_query_is_read_only_session_run_not_execute_read(self):
        import sys
        import types
        log=[]
        class FakeQuery:
            def __init__(self,text,metadata=None,timeout=None):
                self.text=text
                self.timeout=timeout
        class FakeRow:
            def __init__(self,data):self._data=data
            def data(self):return dict(self._data)
        class Session:
            def __enter__(self):return self
            def __exit__(self,*x):return False
            def run(self,query,**kw):
                self.fail_if_not_session_query(query)
                log.append((query,kw))
                return [FakeRow(row())]
            @staticmethod
            def fail_if_not_session_query(query):
                if not isinstance(query,FakeQuery):
                    raise AssertionError("Use Session.run(Query(...))")
            def execute_read(self,*a,**kw):
                raise AssertionError("Transaction Query is unsupported")
        class Driver:
            def __enter__(self):return self
            def __exit__(self,*x):return False
            def session(self,**kw):
                log.append(("READ_SESSION",kw))
                return Session()
        mock=types.ModuleType("neo4j")
        mock.Query=FakeQuery
        mock.READ_ACCESS="READ_ACCESS"
        mock.GraphDatabase=types.SimpleNamespace(
            driver=lambda *a,**kw:Driver()
        )
        dotenv=types.ModuleType("dotenv")
        dotenv.load_dotenv=lambda *a,**kw:None
        with patch.dict(sys.modules,{"neo4j":mock,"dotenv":dotenv}),\
             patch.dict(os.environ,{
                 "NEO4J_URI":"bolt://example.local:7687",
                 "NEO4J_PASSWORD":"fake_password",
                 "NEO4J_USER":"neo4j"
             }):
            records=core._default_provider(
                ingredient="ibuprofen",limit=2,root=Path(".")
            )
        self.assertEqual(len(records),1)
        self.assertEqual(log[0][0],"READ_SESSION")
        self.assertEqual(log[0][1]["default_access_mode"],"READ_ACCESS")
        query,options=log[1]
        self.assertEqual(query.text,core.CYPHER_SECTION_QUERY)
        self.assertEqual(query.timeout,core.MAX_QUERY_SECONDS)
        self.assertEqual(options["ingredient"],"布洛芬")
        self.assertEqual(options["row_limit"],2)


class Tests(unittest.TestCase):
    def test_exact_chinese_pair(self):
        x=core.recognize_pair("服用华法林期间能不能吃布洛芬？")
        self.assertTrue(x["recognized_pair"])
        self.assertTrue(x["request_is_safety_or_interaction"])

    def test_reverse_pair(self):
        self.assertTrue(core.recognize_pair(
            "布洛芬能和华法林一起吃吗？"
        )["recognized_pair"])

    def test_english_pair(self):
        self.assertTrue(core.recognize_pair(
            "Using warfarin with ibuprofen?"
        )["recognized_pair"])

    def test_english_partial_alias_rejected(self):
        self.assertFalse(core.recognize_pair(
            "warfarins ibuprofen"
        )["recognized_pair"])

    def test_other_pairs_out_of_scope(self):
        for q in ("重症肌无力和左氧氟沙星","布洛芬和阿司匹林",
                  "只有华法林",""):
            self.assertFalse(core.recognize_pair(q)["recognized_pair"])

    def test_interaction_terms_detected(self):
        self.assertFalse(core.recognize_pair(
            "华法林；布洛芬"
        )["request_is_safety_or_interaction"])

    def test_drug_interaction_fields_whitelisted(self):
        self.assertEqual(core.PRIORITY_FIELDS,(
            "drug_interactions","warnings","contraindications"
        ))

    def test_class_risk_not_direct_pair(self):
        r=core.classify_leaflet_text(
            "双香豆素抗凝药同用增加出血风险","ibuprofen"
        )
        self.assertEqual(r["status"],"DRUG_CLASS_PLUS_RISK_IN_LEAFLET")
        self.assertFalse(r["other_drug_directly_mentioned"])

    def test_direct_pair_risk(self):
        r=core.classify_leaflet_text(
            "布洛芬与华法林联合增加出血风险","ibuprofen"
        )
        self.assertEqual(r["status"],
                         "DIRECT_OTHER_DRUG_PLUS_RISK_IN_LEAFLET")

    def test_direct_pair_without_risk(self):
        r=core.classify_leaflet_text("华法林被提到了","ibuprofen")
        self.assertEqual(r["status"],
                         "OTHER_DRUG_MENTION_WITHOUT_EXPLICIT_RISK")

    def test_no_pair_keyword(self):
        r=core.classify_leaflet_text("普通用法用量说明","ibuprofen")
        self.assertEqual(r["status"],"NO_PAIR_OR_CLASS_MENTION")

    def test_graph_two_actual_edges_not_drug_interaction(self):
        x=core._section_to_evidence(row(),"ibuprofen")
        self.assertEqual(x["neo4j_hops"],2)
        self.assertEqual(x["node_labels"],[
            "Drug","SourceRecord","Section"
        ])
        self.assertEqual(x["relationship_types"],[
            "HAS_RECORD","HAS_SECTION"
        ])
        self.assertFalse(x["graph_linked_path_is_drug_drug_interaction"])
        self.assertFalse(x["clinical_verified"])

    def test_exact_literal_source_quote(self):
        original=row()
        x=core._section_to_evidence(original,"ibuprofen")
        self.assertIn(x["quote"],original["section_text"])
        self.assertEqual(len(x["section_sha256"]),64)
        self.assertNotIn("section_text",x)

    def test_long_text_quote_bounded(self):
        original=row(text="Z"*500+"双香豆素抗凝药增加出血"+"X"*500)
        x=core._section_to_evidence(original,"ibuprofen")
        self.assertIn("双香豆素",x["quote"])
        self.assertLessEqual(len(x["quote"]),core.EXCERPT_CHARS)

    def test_wrong_ingredient_rejected(self):
        self.assertIsNone(core._section_to_evidence(row(),"warfarin"))

    def test_invalid_graph_identity_rejected(self):
        sample=row()
        sample["section_id"]=sample["record_id"]
        self.assertIsNone(core._section_to_evidence(sample,"ibuprofen"))

    def test_irrelevant_field_rejected(self):
        self.assertIsNone(core._section_to_evidence(
            row(field="indication"),"ibuprofen"
        ))

    def test_overlong_section_rejected(self):
        self.assertIsNone(core._section_to_evidence(
            row(text="x"*(core.SECTION_MAX_CHARS+1)),"ibuprofen"
        ))

    def test_four_local_sections_both_drugs(self):
        report=core.inspect_interaction(
            "华法林和布洛芬可以同服吗？",provider=provider
        )
        self.assertEqual(report["status"],
                         "LOCAL_CLASS_RISK_MENTION_FOUND")
        self.assertEqual(len(report["local_evidence"]),4)
        self.assertEqual(report["query_count"],2)
        self.assertEqual(report["graph_writes"],0)
        self.assertFalse(report["graph_paths_are_drug_drug_edges"])

    def test_official_references_nonlive(self):
        report=core.inspect_interaction(
            "华法林和布洛芬合用？",provider=provider
        )
        self.assertEqual(len(report["official_references"]),2)
        self.assertTrue(all(not r["live_runtime_verified"]
                            for r in report["official_references"]))
        self.assertFalse(report["research_interpretation"][
            "drug_drug_graph_relation_exists"
        ])

    def test_out_of_scope_does_not_query(self):
        calls=[]
        def spy(**kwargs):
            calls.append(kwargs)
            return []
        x=core.inspect_interaction("阿司匹林和布洛芬能同服吗？",
                                   provider=spy)
        self.assertEqual(x["status"],"OUT_OF_SCOPE_PAIR")
        self.assertEqual(calls,[])

    def test_duplicates_deduped(self):
        def dup(**kwargs):
            return [row(),deepcopy(row())] if kwargs["ingredient"]=="ibuprofen" else []
        x=core.inspect_interaction("华法林和布洛芬合用？",provider=dup)
        self.assertEqual(len(x["local_evidence"]),1)

    def test_provider_exceeds_budget_blocks(self):
        def overflow(**kwargs):
            return [row(n=i) for i in range(7)]
        x=core.inspect_interaction(
            "华法林和布洛芬合用？",provider=overflow
        )
        self.assertEqual(x["status"],"GRAPH_UNAVAILABLE")
        self.assertEqual(x["local_evidence"],[])

    def test_graph_unavailable_not_fabricated(self):
        def fail(**kw):
            raise ConnectionError("mock")
        x=core.inspect_interaction(
            "华法林和布洛芬合用？",provider=fail
        )
        self.assertEqual(x["status"],"GRAPH_UNAVAILABLE")
        self.assertEqual(len(x["official_references"]),2)
        self.assertEqual(x["local_evidence"],[])

    def test_partial_graph_failure(self):
        def partial(**kw):
            if kw["ingredient"]=="warfarin":raise ConnectionError("mock")
            return [row()]
        x=core.inspect_interaction(
            "华法林和布洛芬合用？",provider=partial
        )
        self.assertEqual(x["status"],"PARTIAL_GRAPH_FAILURE")
        self.assertEqual(len(x["local_evidence"]),1)

    def test_graph_cypher_read_only(self):
        query=core.CYPHER_SECTION_QUERY.upper()
        for token in (" CREATE "," MERGE "," DELETE "," SET ","TREATS",
                      "INTERACTS_WITH","BELONGS_TO_CLASS"):
            self.assertNotIn(token,query)
        self.assertIn("LIMIT $row_limit",core.CYPHER_SECTION_QUERY)
        self.assertIn("HAS_RECORD",query)
        self.assertIn("HAS_SECTION",query)

    def test_invalid_budget_blocked(self):
        with self.assertRaises(core.InteractionBlocked):
            core.inspect_interaction("华法林和布洛芬合用？",
                                    max_per_drug=999)

    def test_ui_disabled_does_not_scan(self):
        trace={"knowledge_evidence":[{"id":1}]}
        with patch.dict(os.environ,{core.ACTIVATE:"0"}):
            v=ui.prepare_interaction_trace_for_display(
                trace,"华法林和布洛芬可以同服吗？"
            )
        self.assertEqual(v["status"],"DISABLED")
        self.assertNotIn(ui.TRACE_KEY,trace)

    def test_ui_enabled_separate_evidence_not_citations(self):
        report=core.inspect_interaction(
            "华法林和布洛芬合用？",provider=provider
        )
        trace={"knowledge_evidence":[{"id":1,"content":"old"}]}
        with patch.dict(os.environ,{core.ACTIVATE:"1"}),patch.object(
            ui,"inspect_interaction",return_value=report
        ):
            ui.prepare_interaction_trace_for_display(
                trace,"华法林和布洛芬合用？"
            )
        self.assertEqual(len(trace["knowledge_evidence"]),1)
        self.assertIn(ui.TRACE_KEY,trace)
        output=ui.interaction_evidence_html(trace)
        self.assertIn("HAS_RECORD",output)
        self.assertIn("HAS_SECTION",output)
        self.assertNotIn("INTERACTS_WITH →",output)
        self.assertIn("dailymed.nlm.nih.gov",output)

    def test_html_escapes_untrusted_source_text(self):
        result=core.inspect_interaction(
            "华法林和布洛芬合用？",provider=provider
        )
        result["local_evidence"][0]["quote"]="<script>alert(1)</script>"
        result["official_references"][0]["url"]="https://bad.example/"
        out=ui.interaction_evidence_html({ui.TRACE_KEY:result})
        self.assertIn("&lt;script&gt;",out)
        self.assertNotIn("<script>",out)
        self.assertNotIn("https://bad.example/",out)

    def test_ui_avoids_repeat_lookup(self):
        trace={}
        with patch.dict(os.environ,{core.ACTIVATE:"1"}),patch.object(
            ui,"inspect_interaction",
            return_value={"stage":core.STAGE}
        ) as spy:
            ui.prepare_interaction_trace_for_display(
                trace,"华法林和布洛芬合用？"
            )
            ui.prepare_interaction_trace_for_display(
                trace,"华法林和布洛芬合用？"
            )
        self.assertEqual(spy.call_count,1)

    def test_ui_wrong_pair_leaves_no_cards(self):
        trace={}
        with patch.dict(os.environ,{core.ACTIVATE:"1"}):
            ui.prepare_interaction_trace_for_display(
                trace,"华法林和阿司匹林合用？"
            )
        self.assertEqual(ui.interaction_evidence_html(trace),"")

    def _fixture_frontend(self):
        return (
            "import re\n"
            "from pathlib import Path\n"
            "def trace_details_html(trace):\n"
            "    content=[]\n"
            "    multihop_html = multihop_path_evidence_html(trace)\n"
            "    if multihop_html:\n"
            "        content.append(multihop_html)\n"
            "    return ''.join(content)\n"
            "def populate_multihop_trace_for_display(trace,question):\n"
            "    return None\n"
            "def answer(trace,pending):\n"
            "    populate_multihop_trace_for_display(trace, pending)\n"
        )

    def test_selfcontained_ast_patch_fixture(self):
        old=self._fixture_frontend()
        out=installer.transform(old)
        ast.parse(out)
        self.assertEqual(out.count("interaction_evidence_html(trace)"),1)
        self.assertEqual(out.count("prepare_interaction_trace_for_display("),1)
        with self.assertRaises(installer.PatchBlocked):
            installer.transform(out)

    def test_real_previous_frontend_p3063c_ast_if_present(self):
        archive=Path(__file__).with_name(
            "豆馅_P3063C_修复报错并移除英文多跳页.zip"
        )
        if not archive.exists():
            old=self._fixture_frontend()
        else:
            with zipfile.ZipFile(archive) as z:
                old=z.read("修复文件/豆馅.py").decode("utf-8-sig")
        out=installer.transform(old)
        ast.parse(out)
        self.assertEqual(out.count("interaction_evidence_html(trace)"),1)
        self.assertEqual(out.count("prepare_interaction_trace_for_display("),1)

    def test_double_patch_refused(self):
        previous=installer.transform(self._fixture_frontend())
        with self.assertRaises(installer.PatchBlocked):
            installer.transform(previous)

    def test_patcher_unknown_frontend_refused(self):
        with self.assertRaises(installer.PatchBlocked):
            installer.transform("import re\ndef other(): pass\n")

    def test_apply_has_byte_exact_backup(self):
        before=self._fixture_frontend().encode("utf-8")
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            file=folder/"豆馅.py"
            file.write_bytes(before)
            for name in ("medical_p3071_drug_interaction_v117.py",
                         "medical_p3071_interaction_ui_v117.py"):
                shutil.copy2(Path(__file__).with_name(name),folder/name)
            dry=installer.patch(file)
            self.assertEqual(dry["status"],"READY_TO_APPLY")
            self.assertEqual(file.read_bytes(),before)
            done=installer.patch(file,apply=True)
            self.assertEqual(done["status"],"APPLIED")
            self.assertEqual(Path(done["backup"]).read_bytes(),before)
            ast.parse(file.read_text("utf-8-sig"))
            with self.assertRaises(installer.PatchBlocked):
                installer.patch(file,apply=True)

    def test_patch_preserves_crlf(self):
        before=self._fixture_frontend().replace("\n","\r\n").encode("utf-8")
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            file=folder/"豆馅.py"
            file.write_bytes(before)
            for name in ("medical_p3071_drug_interaction_v117.py",
                         "medical_p3071_interaction_ui_v117.py"):
                shutil.copy2(Path(__file__).with_name(name),folder/name)
            result=installer.patch(file,apply=True)
            self.assertEqual(result["status"],"APPLIED")
            self.assertIn(b"\r\n",file.read_bytes())
            self.assertNotIn(b"\r\r\n",file.read_bytes())


if __name__=="__main__":
    unittest.main()
