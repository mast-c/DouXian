# -*- coding: utf-8 -*-
"""P3-07.2 offline end-to-end tests on synthetic Neo4j-like source rows.

No Streamlit, live Neo4j, network, real patient data, or Chroma access.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from medical_p3072_corpus_index_v117 import (
    build_snapshot,alias_lexicon,open_snapshot,CorpusBlocked,
    Q_DRUGS,Q_SECTIONS
)
from medical_p3072_pair_retrieval_v117 import (
    inspect_pair,recognize_drugs,_load_aliases,
    _class_map,TRACE_KEY,ENABLE_FLAG
)
import medical_p3072_interaction_ui_v117 as ui
import medical_p3072_patch_douxian_v117 as patcher


DRUGS=[
    {"drug_id":"D01","drug_name":"华法林钠片"},
    {"drug_id":"D02","drug_name":"布洛芬片"},
    {"drug_id":"D03","drug_name":"布洛芬胶囊"},
    {"drug_id":"D04","drug_name":"氯雷他定片"},
    {"drug_id":"D05","drug_name":"左氧氟沙星片"},
    {"drug_id":"D06","drug_name":"吗啡片"},
    {"drug_id":"D07","drug_name":"未收录安全章节的药品片"},
]
SECTIONS=[
    ("D01","R01","S01","drug_interactions",
     "合用布洛芬可能增加出血风险。"),
    ("D02","R02","S02","drug_interactions",
     "本品与双香豆素等抗凝药同用可增加出血倾向。"),
    ("D03","R03","S03","warnings",
     "与抗凝药同用需注意；出血倾向。"),
    ("D04","R04","S04","drug_interactions",
     "【虚构测试文本】氯雷他定和左氧氟沙星同时使用应谨慎监测。"),
    ("D05","R05","S05","drug_interactions",
     "【虚构测试文本】没有明确提到氯雷他定。"),
    ("D06","R06","S06","contraindications",
     "【虚构测试文本】某些疾病禁用，未提及其他药品。"),
]
SYN_ROWS=[
    {
        "drug_id":drug,"drug_name":next(
            x["drug_name"] for x in DRUGS if x["drug_id"]==drug
        ),
        "record_id":record,"section_id":sid,
        "section_field":field,"section_text":text,
        "source_url":"https://example.org/synthetic-not-a-medical-source",
        "authority_status":"third_party_unverified",
    }
    for drug,record,sid,field,text in SECTIONS
]
DRUGS.sort(key=lambda x:x["drug_id"])
SYN_ROWS.sort(key=lambda x:(x["section_id"],x["drug_id"],x["record_id"]))


class FakePager:
    def __init__(self,drugs=None,sections=None):
        self.drugs=deepcopy(DRUGS if drugs is None else drugs)
        self.sections=deepcopy(SYN_ROWS if sections is None else sections)
        self.query_count=0

    def graph_counts(self):
        self.query_count+=1
        return {"total_nodes":len(self.drugs),"eligible_nodes":len(self.drugs)}

    def drug_page(self,cursor,batch):
        self.query_count+=1
        return [
            dict(x) for x in self.drugs if x["drug_id"]>cursor
        ][:batch]

    def section_page(self,cursor,batch):
        self.query_count+=1
        result=[]
        for row in self.sections:
            k=(row["section_id"],row["drug_id"],row["record_id"])
            if k>cursor:
                result.append(dict(row))
        return result[:batch]

    def close(self):
        pass


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir=tempfile.TemporaryDirectory()
        cls.root=Path(cls.dir.name)
        cls.target=cls.root/"p3072_test_all_drugs.sqlite"
        cls.meta=build_snapshot(
            cls.target,pager=FakePager(),batch=10
        )

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()
        _load_aliases.cache_clear()

    def search(self,text,**kw):
        return inspect_pair(
            text,root=self.root,index=self.target,**kw
        )

    def test_graph_total_drug_count_audited(self):
        self.assertEqual(self.meta["all_graph_drug_nodes"],7)
        self.assertEqual(self.meta["eligible_graph_drug_nodes"],7)
        self.assertEqual(self.meta["drug_nodes_missing_id_or_name"],0)

    def test_multi_page_full_inventory_scans_all_records(self):
        drugs=[
            {"drug_id":f"DRG_{i:04d}","drug_name":f"虚构药品测试{i:04d}片"}
            for i in range(1,42)
        ]
        sections=[
            {
                "drug_id":f"DRG_{i:04d}",
                "drug_name":f"虚构药品测试{i:04d}片",
                "record_id":f"REC_{i:04d}",
                "section_id":f"SEC_{i:04d}",
                "section_field":"drug_interactions",
                "section_text":f"synthetic interactions {i}",
                "source_url":"https://example.org/testing",
                "authority_status":"third_party_unverified",
            }
            for i in range(1,38)
        ]
        target=self.root/"p3072_multipage.sqlite"
        pager=FakePager(drugs=drugs,sections=sections)
        meta=build_snapshot(target,pager=pager,batch=10)
        self.assertEqual(meta["drugs_scanned"],41)
        self.assertEqual(meta["drugs_indexed"],41)
        self.assertEqual(meta["sections_scanned"],37)
        self.assertEqual(meta["sections_indexed"],37)
        self.assertEqual(meta["drugs_without_safety_sections"],4)
        self.assertGreaterEqual(meta["query_count"],10)
        db,_=open_snapshot(target)
        try:
            ids=[x[0] for x in db.execute(
                "SELECT drug_id FROM drugs ORDER BY drug_id"
            ).fetchall()]
        finally:
            db.close()
        self.assertEqual(ids[0],"DRG_0001")
        self.assertEqual(ids[-1],"DRG_0041")

    def test_all_drugs_including_missing_section(self):
        self.assertEqual(self.meta["drugs_indexed"],7)
        self.assertEqual(self.meta["drugs_with_safety_sections"],6)
        self.assertEqual(self.meta["drugs_without_safety_sections"],1)
        self.assertEqual(self.meta["sections_indexed"],6)
        self.assertEqual(self.meta["sections_overlong_skipped"],0)
        self.assertEqual(self.meta["status"],"complete")

    def test_index_is_readonly_on_query(self):
        before=hashlib.sha256(self.target.read_bytes()).hexdigest()
        self.search("华法林和布洛芬能一起吃吗？")
        after=hashlib.sha256(self.target.read_bytes()).hexdigest()
        self.assertEqual(before,after)

    def test_full_corpus_warfarin_ibuprofen(self):
        result=self.search("服用华法林期间能不能吃布洛芬？")
        self.assertEqual(
            result["name_resolution"]["status"],
            "EXACTLY_TWO_CATALOG_GROUPS_DETECTED"
        )
        self.assertEqual(result["status"],
                         "LOCAL_DIRECT_MENTION_WITH_WARNING_WORDS")
        self.assertEqual(
            {x["canonical_key"] for x in
             result["name_resolution"]["matches"]},
            {"华法林","布洛芬"}
        )
        self.assertGreaterEqual(len(result["source_evidence"]),1)
        self.assertFalse(result["interaction_exists_in_graph"])
        self.assertIsNone(result["pair_safe_to_combine"])

    def test_class_name_without_risk_term_is_not_warning(self):
        from medical_p3072_pair_retrieval_v117 import _analyze_section
        row={
            "section_text":"某些抗凝药的类别名称出现在资料中。",
            "section_field":"drug_interactions",
        }
        result=_analyze_section(
            row,opposite_aliases=["华法林"],reviewed_classes=[],
            own_key="布洛芬"
        )
        self.assertEqual(result["status"],
                         "UNRESOLVED_CLASS_MENTION_NOT_PAIR_EVIDENCE")
        self.assertEqual(result["risk_terms_near_match"],[])
        self.assertFalse(result["actual_interaction_graph_edge_verified"])

    def test_class_warning_unresolved_no_claim(self):
        result=self.search("华法林和布洛芬合用吗？")
        matches=[r for r in result["source_evidence"]
                 if r["owner_drug_name"]=="布洛芬片"]
        self.assertTrue(matches)
        self.assertEqual(
            matches[0]["analysis"]["status"],
            "UNRESOLVED_CLASS_WARNING_NOT_PAIR_EVIDENCE"
        )
        self.assertFalse(matches[0]["analysis"][
            "actual_interaction_graph_edge_verified"
        ])

    def test_generic_second_pair_beyond_hardcoded_example(self):
        report=self.search("氯雷他定和左氧氟沙星能一起吃吗？")
        self.assertEqual(report["name_resolution"]["status"],
                         "EXACTLY_TWO_CATALOG_GROUPS_DETECTED")
        self.assertTrue(any(
            r["analysis"]["status"]=="DIRECT_DRUG_NAME_PLUS_WARNING_WORDS"
            for r in report["source_evidence"]
        ))
        self.assertIsNone(report["medical_decision"])

    def test_second_pair_name_order_reversed(self):
        r=self.search("左氧氟沙星可以和氯雷他定一起服用吗？")
        self.assertEqual(r["name_resolution"]["detected_group_count"],2)

    def test_short_two_hanzi_ingredient_indexed(self):
        names=[x[0] for x in alias_lexicon("吗啡片")]
        self.assertIn("吗啡",names)
        result=self.search("布洛芬和吗啡合用吗？")
        self.assertEqual(result["name_resolution"]["status"],
                         "EXACTLY_TWO_CATALOG_GROUPS_DETECTED")

    def test_catalog_missing_two_drugs_safe_no(self):
        result=self.search("一种未知药甲和一种未知药乙可以一起吃吗？")
        self.assertEqual(result["status"],"NO_CATALOG_DRUG_DETECTED")
        self.assertEqual(result["source_evidence"],[])
        self.assertIsNone(result["medical_decision"])

    def test_one_drug_cannot_fake_pair(self):
        result=self.search("布洛芬是否安全？")
        self.assertEqual(result["status"],"ONLY_ONE_DRUG_DETECTED")

    def test_three_drugs_refused_without_pick_first_two(self):
        r=self.search("华法林、布洛芬和吗啡可以一起服用吗？")
        self.assertEqual(r["status"],"THREE_OR_MORE_DRUGS_NOT_A_PAIR")
        self.assertEqual(r["source_evidence"],[])

    def test_name_morphology_is_not_medical_equivalence(self):
        r=self.search("华法林钠片能和布洛芬胶囊同服吗？")
        self.assertTrue(r["name_resolution"]["catalog_name_morphology_only"])
        self.assertTrue(all(not m["name_resolution_is_clinically_verified"]
                            for m in r["name_resolution"]["matches"]))

    def test_snapshot_provenance_exact_hash(self):
        r=self.search("华法林和布洛芬合用？")
        original=r["source_evidence"][0]
        self.assertEqual(original["edge_types"],["HAS_RECORD","HAS_SECTION"])
        self.assertEqual(original["actual_graph_hops"],2)
        self.assertTrue(original["source_snapshot_not_live_reverified"])
        self.assertEqual(original["source_status"],"third_party_unverified")
        self.assertFalse(original["clinical_verified"])

    def test_no_approved_drug_drug_edge_in_report(self):
        r=self.search("华法林和布洛芬合用？")
        self.assertFalse(r["interaction_exists_in_graph"])
        self.assertIsNone(r["pair_safe_to_combine"])
        self.assertIsNone(r["medical_decision"])
        self.assertEqual(r["neo4j_writes"],0)
        self.assertEqual(r["chroma_writes"],0)

    def test_unknown_keyword_cannot_upgrade_class_to_pair(self):
        r=self.search("布洛芬和吗啡能联用吗？")
        self.assertNotIn(
            "LOCAL_DIRECT_MENTION_WITH_WARNING_WORDS",
            [e["analysis"]["status"] for e in r["source_evidence"]]
        )

    def test_source_row_count_and_coverage_exposed(self):
        r=self.search("华法林和布洛芬一起吃？")
        snapshot=r["index_snapshot"]
        self.assertEqual(snapshot["drugs_indexed"],7)
        self.assertEqual(snapshot["drugs_with_safety_sections"],6)
        self.assertTrue(snapshot["snapshot_may_be_stale"])
        self.assertEqual(r["source_rows_total"],3)

    def test_query_budget_truncation_explicit(self):
        r=self.search("华法林和布洛芬一起吃？",
                      max_sections_per_group=1)
        self.assertTrue(r["search_budget_truncated"])
        self.assertEqual(r["status"],"PARTIAL_COVERAGE_QUERY_BUDGET")
        self.assertEqual(r["source_rows_scanned"],2)

    def test_invalid_query_budget_rejected(self):
        with self.assertRaises(CorpusBlocked):
            self.search("华法林与布洛芬合用？",max_evidence=100)

    def test_curated_class_mapping_requires_reviewer_and_source(self):
        mapping=self.root/"reviewed_class.json"
        mapping.write_text(json.dumps({
            "schema_version":1,"entries":[{
                "canonical_key":"华法林",
                "class_terms":["抗凝药","香豆素"],
                "source_url":"https://example.org/manual-source",
                "reviewed_by":"research_curator",
                "status":"reviewed"
            }]
        },ensure_ascii=False),encoding="utf-8")
        report=self.search(
            "华法林和布洛芬能合用吗？",curated_map=mapping
        )
        found=[x for x in report["source_evidence"]
               if x["owner_drug_name"]=="布洛芬片"]
        self.assertTrue(found)
        self.assertEqual(
            found[0]["analysis"]["status"],
            "REVIEWED_CLASS_BRIDGE_WITH_WARNING_TERMS"
        )
        self.assertFalse(found[0]["reviewed_class_link"]["clinical_verified"])
        self.assertFalse(found[0]["reviewed_class_link"][
            "neo4j_class_edge_present"
        ])
        mapping.unlink()

    def test_no_reviewer_rejects_class_mapping(self):
        p=self.root/"invalid_class.json"
        p.write_text(json.dumps({
            "schema_version":1,
            "entries":[{"canonical_key":"华法林",
                        "class_terms":["抗凝药"],"status":"reviewed"}]
        },ensure_ascii=False),encoding="utf-8")
        self.assertEqual(_class_map(p),{})
        p.unlink()

    def test_ambiguous_name_refuses(self):
        rows=(("头孢","药甲","exact_formulation"),
              ("头孢","药乙","exact_formulation"),
              ("布洛芬","布洛芬","exact_formulation"))
        out=recognize_drugs("头孢和布洛芬能同服吗？",alias_rows=rows)
        self.assertEqual(out["status"],"AMBIGUOUS_CATALOG_ALIAS")

    def test_direct_quote_is_literal_in_source_text(self):
        r=self.search("华法林和布洛芬合用？")
        for evidence in r["source_evidence"]:
            self.assertIn("quote_start",evidence["analysis"])
            self.assertLessEqual(len(evidence["analysis"]["quote"]),420)

    def test_ui_new_panel_inside_evidence_html(self):
        r=self.search("华法林和布洛芬一起服用？")
        text=ui.interaction_full_corpus_html({TRACE_KEY:r})
        self.assertIn("全库药物相互作用来源检索",text)
        self.assertIn("HAS_RECORD",text)
        self.assertIn("HAS_SECTION",text)
        self.assertIn("未命中≠安全",text)
        self.assertNotIn("INTERACTS_WITH →",text)

    def test_ui_handles_missing_index_without_crashing(self):
        with patch.dict(os.environ,{
            ENABLE_FLAG:"1",
            "MEDICAL_P3072_INDEX_PATH":"stage4_reports/p3072_missing.sqlite"
        }):
            trace={"knowledge_evidence":[{"id":1,"content":"old"}]}
            result=ui.prepare_full_corpus_interaction_trace(
                trace,"华法林和布洛芬能同服吗？",root=self.root
            )
        self.assertEqual(
            result["status"],"INDEX_UNAVAILABLE_OR_QUERY_BLOCKED"
        )
        self.assertEqual(len(trace["knowledge_evidence"]),1)

    def test_ui_off_by_default(self):
        trace={"knowledge_evidence":[{"id":1}]}
        with patch.dict(os.environ,{ENABLE_FLAG:"0"}):
            r=ui.prepare_full_corpus_interaction_trace(
                trace,"华法林和布洛芬可以同服吗？",root=self.root
            )
        self.assertEqual(r["status"],"DISABLED")
        self.assertNotIn(TRACE_KEY,trace)

    def test_ui_no_drug_pair_intent_does_not_load_index(self):
        with patch.dict(os.environ,{ENABLE_FLAG:"1"}):
            trace={}
            r=ui.prepare_full_corpus_interaction_trace(
                trace,"布洛芬和华法林的化学结构图",root=self.root
            )
        self.assertEqual(r["status"],"NO_PAIR_SAFETY_INTENT")
        self.assertEqual(trace,{})

    def test_ui_trace_cache_avoids_second_query(self):
        with patch.dict(os.environ,{
            ENABLE_FLAG:"1",
            "MEDICAL_P3072_INDEX_PATH":str(self.target)
        }):
            trace={}
            first=ui.prepare_full_corpus_interaction_trace(
                trace,"华法林和布洛芬可以同服吗？",root=self.root
            )
            second=ui.prepare_full_corpus_interaction_trace(
                trace,"华法林和布洛芬可以同服吗？",root=self.root
            )
        self.assertIs(first,second)

    def test_ui_source_escaped_no_xss(self):
        r=self.search("华法林和布洛芬一起吃？")
        r["source_evidence"][0]["analysis"]["quote"]="<script>alert(1)</script>"
        t=ui.interaction_full_corpus_html({TRACE_KEY:r})
        self.assertIn("&lt;script&gt;",t)
        self.assertNotIn("<script>",t)

    def test_full_graph_cypher_is_readonly(self):
        for query in (Q_DRUGS,Q_SECTIONS):
            upper=query.upper()
            for keyword in ("CREATE ","MERGE ","DELETE "," DETACH "," SET ",
                            "INTERACTS_WITH","TREATS"):
                self.assertNotIn(keyword,upper)
            self.assertIn("LIMIT $batch",query)
        self.assertIn("WITH DISTINCT d,r,s",Q_SECTIONS)

    def test_second_build_without_replace_refused(self):
        with self.assertRaises(CorpusBlocked):
            build_snapshot(self.target,pager=FakePager(),batch=10)

    def test_replace_backs_up_old_index(self):
        target=self.root/"p3072_replace.sqlite"
        a=build_snapshot(target,pager=FakePager(),batch=10)
        b=build_snapshot(target,pager=FakePager(),batch=10,replace=True)
        self.assertTrue(Path(b["previous_snapshot_backup"]).is_file())
        self.assertEqual(a["sections_indexed"],b["sections_indexed"])

    def test_invalid_output_name_refused(self):
        with self.assertRaises(CorpusBlocked):
            build_snapshot(self.root/"chroma.sqlite",pager=FakePager(),batch=10)

    def test_mid_build_failure_does_not_publish(self):
        class BadPager(FakePager):
            def section_page(self,cursor,batch):
                raise ConnectionError("simulated graph connection lost")
        target=self.root/"p3072_fail.sqlite"
        with self.assertRaises(ConnectionError):
            build_snapshot(target,pager=BadPager(),batch=10)
        self.assertFalse(target.exists())

    def test_bad_sorted_graph_drugs_blocked(self):
        class BadPager(FakePager):
            def drug_page(self,cursor,batch):
                return list(reversed(super().drug_page(cursor,batch)))
        with self.assertRaises(CorpusBlocked):
            build_snapshot(self.root/"p3072_unsorted.sqlite",
                           pager=BadPager(),batch=10)

    def test_overlong_source_reported_as_excluded(self):
        rows=deepcopy(SYN_ROWS)
        rows[0]["section_text"]="Z"*200001
        out=build_snapshot(self.root/"p3072_long.sqlite",
                           pager=FakePager(sections=rows),batch=10)
        self.assertEqual(out["sections_overlong_skipped"],1)
        self.assertEqual(out["sections_indexed"],5)

    def test_graph_index_is_strict_snapshot_not_chroma_claim(self):
        self.assertEqual(self.meta["graph_source"],
                         "Neo4j Drug / SourceRecord / Section")
        self.assertFalse(self.meta["clinical_review_performed"])
        self.assertFalse(self.meta["source_rights_verified"])

    def test_cli_audit_verifies_every_snapshot_section(self):
        from medical_p3072_cli_v117 import main
        import contextlib
        import io
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            status=main(["audit","--root",str(self.root),
                         "--index",str(self.target)])
        self.assertEqual(status,0)
        self.assertIn("Section SHA256 verified: 6",output.getvalue())

    def test_cli_detects_snapshot_text_tampering(self):
        from medical_p3072_cli_v117 import main
        import contextlib
        import io
        import sqlite3
        target=self.root/"p3072_tampered.sqlite"
        shutil.copy2(self.target,target)
        with sqlite3.connect(target) as db:
            db.execute("""
UPDATE sections SET section_text='tampered note'
WHERE section_id='S01'
""")
            db.commit()
        db.close()  # SQLite context manager commits but does not close on Windows.
        out=io.StringIO()
        with contextlib.redirect_stderr(out):
            status=main(["audit","--index",str(target)])
        self.assertEqual(status,2)
        self.assertIn("哈希不一致",out.getvalue())

    def test_report_write_cannot_overwrite_previous_result(self):
        from medical_p3072_cli_v117 import _report_write
        target=self.root/"p3072_report_once.json"
        _report_write(target,{"status":"test"})
        with self.assertRaises(FileExistsError):
            _report_write(target,{"status":"override"})

    def test_rebuild_index_cache_mtime_key(self):
        before=_load_aliases(
            str(self.target),self.target.stat().st_mtime_ns
        )
        self.assertTrue(any(x[0]=="华法林" for x in before))
        self.assertTrue(any(x[0]=="吗啡" for x in before))


class ReadOnlyDriverContract(unittest.TestCase):
    def test_neoj4_query_object_always_on_session_run(self):
        import types
        from medical_p3072_corpus_index_v117 import Neo4jReadPager
        seen=[]
        class Query:
            def __init__(self,text,timeout=None):
                self.text=text
                self.timeout=timeout
        class ResultRow:
            def __init__(self,props):self.props=props
            def data(self):return self.props
        class Session:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def run(self,query,**kwargs):
                self_type=type(query).__name__
                if self_type!="Query":
                    raise AssertionError("Query must be on Session.run")
                seen.append((query.text,kwargs))
                if "total_nodes" in query.text:
                    return [ResultRow({
                        "total_nodes":2,"eligible_nodes":2
                    })]
                if "RETURN d.id AS drug_id" in query.text:
                    return [ResultRow(DRUGS[0])]
                return [ResultRow(SYN_ROWS[0])]
            def execute_read(self,*args,**kwargs):
                raise AssertionError("No ManagedTransaction with Query")
        class Driver:
            def session(self,**kwargs):
                self.opts=kwargs
                return Session()
            def close(self):pass
        d=Driver()
        fake=types.ModuleType("neo4j")
        fake.Query=Query
        fake.READ_ACCESS="READ_ACCESS"
        fake.GraphDatabase=types.SimpleNamespace(
            driver=lambda *args,**kwargs:d
        )
        dotenv=types.ModuleType("dotenv")
        dotenv.load_dotenv=lambda *args,**kwargs:None
        with patch.dict(sys.modules,{
            "neo4j":fake,"dotenv":dotenv
        }),patch.dict(os.environ,{
            "NEO4J_URI":"bolt://fake:7687",
            "NEO4J_PASSWORD":"not_real",
        }):
            reader=Neo4jReadPager(Path("."))
            self.assertEqual(reader.graph_counts()["total_nodes"],2)
            self.assertEqual(len(reader.drug_page("",10)),1)
            self.assertEqual(
                len(reader.section_page(("","",""),10)),1
            )
            self.assertEqual(d.opts["default_access_mode"],"READ_ACCESS")
            reader.close()
        self.assertEqual(len(seen),3)
        self.assertTrue(all(q for q,_ in seen))


class FrontendPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=(
            'import re\nfrom pathlib import Path\n'
            'def trace_details_html(trace):\n'
            '    content=[]\n'
            '    multihop_html = multihop_path_evidence_html(trace)\n'
            '    if multihop_html:\n'
            '        content.append(multihop_html)\n'
            '    return "".join(content)\n'
            'def populate_multihop_trace_for_display(trace,question):pass\n'
            'def app(trace,pending):\n'
            '    populate_multihop_trace_for_display(trace, pending)\n'
        )

    def test_actual_old_frontend_p3063c_patches(self):
        f=Path(__file__).with_name(
            "豆馅_P3063C_修复报错并移除英文多跳页.zip"
        )
        if f.exists():
            with zipfile.ZipFile(f) as z:
                src=z.read("修复文件/豆馅.py").decode("utf-8-sig")
        else:
            src=self.fixture
        after=patcher.transform(src)
        ast.parse(after)
        self.assertEqual(after.count("interaction_full_corpus_html(trace)"),1)
        self.assertEqual(after.count("prepare_full_corpus_interaction_trace("),1)
        self.assertEqual(after.count("multihop_html = multihop_path_evidence_html(trace)"),1)

    def test_with_old_p3071_already_installed(self):
        original=patcher.transform(self.fixture)
        self.assertIn("interaction_full_corpus_html",original)
        with self.assertRaises(patcher.PatchBlocked):
            patcher.transform(original)

    def test_ast_refuses_unknown_rag_entry(self):
        with self.assertRaises(patcher.PatchBlocked):
            patcher.transform("import re\nprint('unrelated')\n")

    def test_dry_run_and_apply_has_backup(self):
        root=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            target=folder/"豆馅.py"
            old=self.fixture.encode("utf-8")
            target.write_bytes(old)
            for file in ("medical_p3072_interaction_ui_v117.py",
                         "medical_p3072_corpus_index_v117.py",
                         "medical_p3072_pair_retrieval_v117.py"):
                shutil.copy2(root/file,folder/file)
            dry=patcher.patch(target,apply=False)
            self.assertEqual(dry["status"],"READY_TO_APPLY")
            self.assertEqual(target.read_bytes(),old)
            applied=patcher.patch(target,apply=True)
            self.assertEqual(applied["status"],"APPLIED")
            self.assertEqual(Path(applied["backup"]).read_bytes(),old)
            ast.parse(target.read_text("utf-8"))
            with self.assertRaises(patcher.PatchBlocked):
                patcher.patch(target,apply=True)

    def test_windows_crlf_and_bom_preserved(self):
        root=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            target=folder/"豆馅.py"
            original=b"\xef\xbb\xbf"+self.fixture.replace(
                "\n","\r\n"
            ).encode("utf-8")
            target.write_bytes(original)
            for file in ("medical_p3072_interaction_ui_v117.py",
                         "medical_p3072_corpus_index_v117.py",
                         "medical_p3072_pair_retrieval_v117.py"):
                shutil.copy2(root/file,folder/file)
            patcher.patch(target,apply=True)
            data=target.read_bytes()
            self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
            self.assertIn(b"\r\n",data)
            self.assertNotIn(b"\r\r\n",data)


if __name__=="__main__":
    unittest.main()
