# -*- coding: utf-8 -*-
import ast
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]

def getmodule(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

gate=getmodule("gate_test", ROOT/"medical_entity_gate.py")
quality=getmodule("quality_test", ROOT/"medical_answer_quality.py")
ev=getmodule("eval_test", ROOT/"medical_retrieval_eval.py")
ASK="阿托伐他汀钙片与动脉粥样硬化有什么关联？"

def evidence(drug,kind="json_neo4j_text_verified"):
    return {
        "drug_name":drug,
        "source":"药品资料："+drug,
        "section_path":drug+"/适应症",
        "content":"药品："+drug+"\n栏目：适应症\n证据原文：示例原文",
        "source_url":"https://example.org/1",
        "evidence_type":kind,
        "section_field":"indication",
    }

class TestDrugScope(unittest.TestCase):
    def test_real_screenshot_crossdrug_removed(self):
        items=[
            evidence("阿托伐他汀钙片","unverified_drug_disease_candidate"),
            evidence("阿托伐他汀钙片"),
            evidence("10%、20%、30%脂肪乳注射液"),
        ]
        filtered, report=gate.filter_evidence_for_query(ASK,items)
        self.assertEqual(len(filtered),2)
        self.assertEqual([r["id"] for r in filtered],[1,2])
        self.assertEqual(report["explicit_drugs"],["阿托伐他汀钙片"])
        self.assertEqual(report["removed_other_drugs"],1)
        self.assertFalse(any("脂肪乳" in r["content"] for r in filtered))

    def test_disease_only_keeps_distinct_drugs(self):
        items=[evidence("阿托伐他汀钙片"),evidence("10%、20%、30%脂肪乳注射液")]
        filtered, report=gate.filter_evidence_for_query("动脉粥样硬化的相关药品有哪些？",items)
        self.assertEqual(len(filtered),2)
        self.assertEqual(report["mode"],"unrestricted")

    def test_multi_drug_comparison_keeps_both(self):
        a="阿托伐他汀钙片"
        b="10%、20%、30%脂肪乳注射液"
        items=[evidence(a),evidence(b),evidence("一清胶囊")]
        filtered, report=gate.filter_evidence_for_query(
            f"比较{a}与{b}的适应症",items
        )
        self.assertEqual({r["drug_name"] for r in filtered},{a,b})
        self.assertEqual(report["removed_other_drugs"],1)

    def test_nested_drug_name_prefers_long_name(self):
        result=gate.query_drug_names(
            "阿托伐他汀钙片的适应症",
            [evidence("阿托伐他汀钙片"),evidence("阿托伐他汀")]
        )
        self.assertEqual(result,("阿托伐他汀钙片",))

    def test_legacy_source_without_drug_metadata_requires_exact_mention(self):
        items=[
            {"source":"notes.md","content":"和疾病有关的普通文本", "drug_name":""},
            {"source":"notes2.md","content":"阿托伐他汀钙片的完整原文", "drug_name":""},
        ]
        filtered,report=gate.filter_evidence_for_query(
            ASK,items,known_drugs=["阿托伐他汀钙片"]
        )
        self.assertEqual(len(filtered),1)
        self.assertEqual(report["removed_unattributed"],1)

    def test_unsupported_short_alias_does_not_guess_identity(self):
        items=[evidence("阿托伐他汀钙片"),evidence("一清胶囊")]
        _, report=gate.filter_evidence_for_query("阿托伐他汀的适应症",items)
        self.assertEqual(report["mode"],"unrestricted")

    def test_followup_self_rag_documents_scoped(self):
        docs=[
            SimpleNamespace(metadata={"drug_name":"一清胶囊"},page_content="药品：一清胶囊"),
            SimpleNamespace(metadata={"drug_name":"阿托伐他汀钙片"},page_content="药品：阿托伐他汀钙片"),
        ]
        selected=gate.filter_documents_for_drugs(docs,["阿托伐他汀钙片"])
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0].metadata["drug_name"],"阿托伐他汀钙片")

    def test_mechanism_refusal_does_not_appear_when_unasked(self):
        refused=["现有证据是否阐明该药干预动脉粥样硬化的具体作用机制或确切临床效果？"]
        result=quality.requested_refusal_questions(ASK,refused)
        self.assertEqual(result,[])
        self.assertEqual(
            quality.requested_refusal_questions(
                "该药的作用机制是什么？", refused
            ), refused
        )

    def test_targeted_other_refusal_is_retained(self):
        requested="这个药的禁忌与不良反应是什么？"
        refused=["现有证据缺乏禁忌证说明","不良反应资料未提供"]
        self.assertEqual(quality.requested_refusal_questions(requested,refused),refused)

    def test_eval_reports_before_after_contamination(self):
        docs=[
            SimpleNamespace(metadata={"drug_name":"10%、20%、30%脂肪乳注射液","section_field":"indication"},page_content="脂肪乳"),
            SimpleNamespace(metadata={"drug_name":"阿托伐他汀钙片","section_field":"indication"},page_content="阿托"),
        ]
        case={"id":"Q","question":ASK,"target_drugs":["阿托伐他汀钙片"],"expected_field":"indication"}
        row=ev.summarize_case(case,docs,top_k=5)
        self.assertEqual(row["raw"]["offtarget_chunks"],1)
        self.assertEqual(row["scoped"]["offtarget_chunks"],0)
        self.assertTrue(row["scoped"]["target_drug_field_hit"])
        totals=ev.aggregate_results([row])
        self.assertEqual(totals["scoped_target_drug_hit_rate"],1)
        self.assertEqual(totals["offtarget_chunk_rate_scoped"],0)

    def test_eval_cases_schema(self):
        cases=ev.read_cases(ROOT/"retrieval_eval_seed.jsonl")
        self.assertGreaterEqual(len(cases),12)
        self.assertTrue(all(x["annotation_status"]=="unreviewed" for x in cases))

    def test_v19_rag_integrated_at_all_relevant_paths(self):
        source=(ROOT/"rag.py").read_text(encoding="utf-8")
        ast.parse(source)
        for token in (
            "medical_named_drugs = query_drug_names(",
            "documents = filter_documents_for_drugs(",
            "merged_excerpts, entity_report = filter_evidence_for_query(",
            "filter_documents_for_drugs(",
            "trace[\"evidence_entity_gate\"] = entity_report",
            "requested_refusal_questions(",
            "collapse_duplicate_parent_sections(merged_excerpts)",
            "search_with_rerank(",
            "prepare_medical_documents(",
        ):
            self.assertIn(token,source)
        self.assertNotIn("MATCH (d:Drug) CREATE",source)

if __name__=="__main__":
    unittest.main()
