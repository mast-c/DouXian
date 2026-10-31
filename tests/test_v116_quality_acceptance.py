# -*- coding: utf-8 -*-
"""V1.16 工程验收测试：真实证据索引、栏目引用、同 Section 重复、单轮导出。"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tests"))
from test_v115_source_overview import (
    results as simulated_results, DRUGS, QUERY, answer as overview,
    quality as source_quality, section as fake_section,
)

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

tool=import_file("quality_acceptance_v116",ROOT/"medical_quality_acceptance.py")


def sample_payload():
    items,_=overview.scope_graph_discovered_drugs(simulated_results(),DRUGS)
    body,details=overview.build_multi_drug_source_answer(
        QUERY,items,DRUGS,["adverse_reactions","warnings"]
    )
    refs={
        i["id"]:(
            f"[查看原文](https://example.org/{i['drug_name']})",
            i["section_path"]
        )
        for i in items
    }
    full=body+source_quality.render_compact_references(body,refs)
    trace={
        "knowledge_evidence":items,
        "medical_answer_mode":"structured_disease_drug_source_overview",
        "multi_drug_coverage":details,
        "multi_drug_scope":{
            "listed_drugs":DRUGS,
            "supplement_fields":["adverse_reactions","warnings"],
            "max_listed_drugs":4,
            "removed_off_scope":1,
        },
        "retrieval_pipeline":{
            "schema_version":1,
            "chroma_hybrid":{"parent_verified":2,"returned":2},
            "neo4j_direct":{"returned":12},
            "merge":{"retained":12,"removed_by_entity_filter":1,
                     "removed_by_dedup":1},
        },
        "unused_internal_secret":"should_not_export",
    }
    return tool.make_export_payload(QUERY,full,trace)


class TestV116Pass(unittest.TestCase):
    def test_realistic_v115_four_drug_two_field_answer_passes(self):
        report=tool.analyze_export(sample_payload())
        self.assertEqual(report["status"],"PASS",report["issues"])
        self.assertEqual(report["metrics"]["evidence_cards"],12)
        self.assertEqual(report["metrics"]["definite_section_duplicates"],0)
        self.assertEqual(report["metrics"]["multi_drug"]["expected_cells"],8)
        self.assertEqual(report["metrics"]["multi_drug"]["correctly_cited_cells"],8)
        self.assertEqual(report["metrics"]["multi_drug"]["false_missing_cells"],0)
        self.assertEqual(report["metrics"]["reported_neo4j_parent_verified"],2)
        self.assertEqual(report["summary"]["failures"],0)

    def test_disease_candidate_and_section_same_source_are_not_duplicates(self):
        evidence=sample_payload()["trace"]["knowledge_evidence"]
        repeats,_=tool.section_duplicate_groups(evidence)
        self.assertEqual(repeats,[])

    def test_same_text_but_different_drugs_not_duplicates(self):
        evidence=sample_payload()["trace"]["knowledge_evidence"]
        a=next(x for x in evidence if x["id"]==5)
        b=next(x for x in evidence if x["id"]==6)
        self.assertIn("偶见皮疹",a["content"])
        self.assertIn("偶见皮疹",b["content"])
        repeat,_=tool.section_duplicate_groups([a,b])
        self.assertEqual(repeat,[])

    def test_only_this_turn_and_approved_fields_exported(self):
        data=sample_payload()
        self.assertNotIn("unused_internal_secret",data["trace"])
        self.assertNotIn("events",data["trace"])
        self.assertIn("knowledge_evidence",data["trace"])
        self.assertEqual(data["question"],QUERY)
        self.assertEqual(data["schema_version"],1)

    def test_plain_text_original_field_detection(self):
        data=sample_payload()
        sections=data["trace"]["knowledge_evidence"]
        self.assertIn("对大环内酯类",tool.extracted_source_text(sections[10]))
        self.assertIn("特殊人群应遵医嘱",tool.extracted_source_text(sections[11]))
        self.assertFalse(tool.extracted_source_text(sections[0]))

    def test_single_source_references_not_counted_twice(self):
        data=sample_payload()
        body, count=tool.split_answer_and_reference_footer(data["answer"])
        self.assertEqual(count,1)
        self.assertEqual(tool.cited_numbers(body),set(range(1,13)))
        self.assertEqual(tool.cited_numbers(data["answer"]),set(range(1,13)))

    def test_chroma_count_is_not_same_as_final_retain(self):
        data=sample_payload()
        report=tool.analyze_export(data)
        self.assertEqual(report["metrics"]["reported_neo4j_parent_verified"],2)
        self.assertEqual(report["metrics"]["reported_merged_evidence"],12)
        self.assertEqual(report["metrics"]["evidence_cards"],12)

class TestInvalidReports(unittest.TestCase):
    def _errors(self, payload):
        result=tool.analyze_export(payload)
        return result,{row["code"] for row in result["issues"]}

    def test_other_drug_chroma_not_accepted_as_answer_evidence(self):
        data=sample_payload()
        unrelated=fake_section(13,"七味解痛口服液","warnings",
                               "注意清淡饮食",verified=True)
        data["trace"]["knowledge_evidence"].append(unrelated)
        data["trace"]["retrieval_pipeline"]["merge"]["retained"]=13
        result,codes=self._errors(data)
        self.assertEqual(result["status"],"FAIL")
        self.assertIn("off_scope_drug_in_evidence",codes)

    def test_actual_field_available_but_wrongly_declared_missing(self):
        data=sample_payload()
        data["answer"]=data["answer"].replace(
            "1．对大环内酯类过敏者也可能不耐受本品。2．用药期间宜随访肝功能。 [11]",
            "本轮没有获得该药对应栏目的可引用原文；"
            "不能据此认定该药没有相关风险或限制。",
        )
        result,codes=self._errors(data)
        self.assertIn("false_missing_source_claim",codes)
        self.assertIn("field_evidence_not_cited",codes)
        self.assertGreater(result["metrics"]["multi_drug"]["false_missing_cells"],0)

    def test_known_section_cited_to_another_drug_is_failure(self):
        data=sample_payload()
        data["answer"]=data["answer"].replace(
            "2．用药期间宜随访肝功能。 [11]",
            "2．用药期间宜随访肝功能。 [9]",
        )
        result,codes=self._errors(data)
        self.assertEqual(result["status"],"FAIL")
        self.assertIn("cross_drug_or_field_citation",codes)

    def test_missing_two_drug_answer_sections_fail(self):
        data=sample_payload()
        before=data["answer"]
        start=before.index("### 万通炎康片")
        stop=before.index("### 丙酸交沙霉素颗粒")
        data["answer"]=before[:start]+before[stop:]
        result,codes=self._errors(data)
        self.assertEqual(result["status"],"FAIL")
        self.assertIn("drug_omitted_from_answer",codes)

    def test_citation_99_not_in_evidence(self):
        data=sample_payload()
        # 只有回答正文中的新引用才检查有效性；参考资料尾注不属于正文。
        data["answer"] = data["answer"].replace(
            "参考资料（同一来源的不同片段合并显示）：",
            "无证据断言。[99]\n\n参考资料（同一来源的不同片段合并显示）：",
            1
        )
        result,codes=self._errors(data)
        self.assertIn("invalid_citation",codes)

    def test_two_reference_footers_are_failure(self):
        data=sample_payload()
        data["answer"] += "\n\n参考资料：\n[1] 来源"
        result,codes=self._errors(data)
        self.assertIn("duplicated_reference_footer",codes)

    def test_manual_approval_claim_not_sustained_by_automatic_evidence(self):
        data=sample_payload()
        data["answer"]="根据审核通过的证据，"+data["answer"]
        result,codes=self._errors(data)
        self.assertIn("false_medical_approval_wording",codes)

    def test_duplicate_direct_section_and_verified_chroma_same_record_is_failure(self):
        data=sample_payload()
        direct=copy.deepcopy(data["trace"]["knowledge_evidence"][8])
        verified=fake_section(13,DRUGS[0],"warnings","原文补充",verified=True)
        verified["record_id"]=direct["record_id"]
        verified["section_id"]=direct["section_id"]
        data["trace"]["knowledge_evidence"].append(verified)
        data["trace"]["retrieval_pipeline"]["merge"]["retained"]=13
        result,codes=self._errors(data)
        self.assertIn("duplicate_cross_source_section",codes)

    def test_same_parent_different_chroma_chunks_warn_not_definite_duplicate(self):
        data=sample_payload()
        chroma=copy.deepcopy(data["trace"]["knowledge_evidence"][11])
        chroma["id"]=13
        chroma["content"]+="\n另一处不同原文 Chunk"
        data["trace"]["knowledge_evidence"].append(chroma)
        data["trace"]["retrieval_pipeline"]["merge"]["retained"]=13
        result,codes=self._errors(data)
        self.assertIn("same_parent_multiple_chunks",codes)
        self.assertNotIn("duplicate_cross_source_section",codes)

    def test_without_record_id_cannot_claim_exact_dedup(self):
        data=sample_payload()
        data["trace"]["knowledge_evidence"][8].pop("record_id")
        result,codes=self._errors(data)
        self.assertIn("missing_parent_identifiers",codes)
        self.assertNotIn("duplicate_cross_source_section",codes)

    def test_non_contiguous_ids_are_failure(self):
        data=sample_payload()
        data["trace"]["knowledge_evidence"][4]["id"]=41
        result,codes=self._errors(data)
        self.assertIn("evidence_ids_not_contiguous",codes)

    def test_same_quote_three_disease_candidates_warn_display_not_delete(self):
        data=sample_payload()
        first=copy.deepcopy(data["trace"]["knowledge_evidence"][0])
        second=copy.deepcopy(first)
        second["disease_term"]="咽炎"
        second["id"]=13
        data["trace"]["knowledge_evidence"].append(second)
        data["trace"]["retrieval_pipeline"]["merge"]["retained"]=13
        report,codes=self._errors(data)
        self.assertIn("repeated_candidate_quote_cards",codes)
        self.assertEqual(
            report["metrics"]["same_quote_candidate_card_groups"],1
        )
        self.assertNotIn("duplicate_cross_source_section",codes)

    def test_no_evidence_is_not_pass(self):
        data=tool.make_export_payload("问题","回答",{})
        result,codes=self._errors(data)
        self.assertIn("no_evidence_cards",codes)
        self.assertNotEqual(result["status"],"PASS")

    def test_legacy_source_without_record_id_can_still_be_analyzed(self):
        data=sample_payload()
        for item in data["trace"]["knowledge_evidence"]:
            item.pop("record_id",None)
        report=tool.analyze_export(data)
        self.assertEqual(report["status"],"WARN")
        self.assertGreater(report["summary"]["warnings"],0)

    def test_unsafe_or_incomplete_input_fails_fast(self):
        with self.assertRaisesRegex(ValueError,"trace"):
            tool.analyze_export({"question":"hi"})

class TestCliUI(unittest.TestCase):
    def test_v116_trace_has_section_record_id(self):
        code=(ROOT/"rag.py").read_text(encoding="utf-8")
        ast.parse(code)
        self.assertIn('"record_id": str(item.get("record_id") or "")',code)
        self.assertIn('"candidate_id": str(item.get("candidate_id") or "")',code)
        self.assertIn('trace["knowledge_evidence"] = [',code)

    def test_streamlit_export_is_user_opt_in_and_scope_limited(self):
        code=(ROOT/"豆馅.py").read_text(encoding="utf-8")
        ast.parse(code)
        self.assertIn("from medical_quality_acceptance import make_export_payload",code)
        self.assertIn('with st.expander("检索质量验收（单轮数据导出）", expanded=False)',code)
        self.assertIn('st.download_button(',code)
        self.assertIn('last_question_for_acceptance',code)

    def test_cli_valid_sample_outputs_report(self):
        data=sample_payload()
        with tempfile.TemporaryDirectory() as tmp:
            inp=Path(tmp)/"trace.json"
            out=Path(tmp)/"quality.json"
            inp.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
            rc=tool.main(["--input",str(inp),"--output",str(out)])
            self.assertEqual(rc,0)
            result=json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["status"],"PASS")
            self.assertEqual(result["summary"]["failures"],0)

    def test_cli_does_not_overwrite_existing_report_implicitly(self):
        data=sample_payload()
        with tempfile.TemporaryDirectory() as tmp:
            inp=Path(tmp)/"trace.json"
            out=Path(tmp)/"quality.json"
            inp.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
            out.write_text("{}")
            with self.assertRaises(SystemExit):
                tool.main(["--input",str(inp),"--output",str(out)])
            self.assertEqual(out.read_text(),"{}")

if __name__=="__main__":
    unittest.main()
