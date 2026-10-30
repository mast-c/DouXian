"""Regression tests: no live Qwen calls, no Neo4j writes."""
import csv
import json
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from kg.indication_v31_checkpoint import run, evaluate, filter_selection, model_validated, scope_rule_v31, build_anchored_focus
from kg.disease_stage3 import hash_id

def write_jsonl(path, records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")

class V31Test(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)
        self.graph=self.root/"graph"
        self.src=self.root/"source.csv"
        self.out=self.root/"out.csv"

        # Synthetic text; not clinical facts or approved labels.
        examples=[
          ("本品适用于强直性脊椎炎。","脊椎炎","rejected"),
          ("不作为治疗原发性高血压的首选药物，但当噻嗪类药物疗效不佳时，本类药物尤为适用。",
           "原发性高血压","rejected"),
          ("本品适用于肝硬化的治疗。","肝硬化","approved"),
          ("泌尿系统感染：肾盂肾炎、复杂性尿路感染等。","尿路感染","approved"),
          ("患者伴有肾功能不全时，本品用于高血压。","肾功能不全","rejected"),
          ("本品明确不适用于脑膜炎。","脑膜炎","rejected"),
        ]
        records=[]
        self.rows=[]
        for index,(sentence,term,status) in enumerate(examples):
            sec=f"s{index}"
            start=sentence.index(term)
            end=start+len(term)
            records.append({
                "id":sec, "record_id":"record_1",
                "field":"indication","text":sentence
            })
            self.rows.append({
                "candidate_id":hash_id("DiseaseCandidate",sec,term,start,end),
                "drug_name":"虚构药甲",
                "disease_name":term,
                "source_record_id":"record_1",
                "section_id":sec,
                "source_url":"https://example.org/fiction",
                "evidence_quote":sentence,
                "quote_start":0,
                "quote_end":len(sentence),
                "disease_start":start,
                "disease_end":end,
                "context_flag":"manual_semantic_review",
                "review_status":status,
                "relation_type":"treatment" if status=="approved" else "",
                "reviewer":"Codex-医学文本审核",
                "review_note":"合成文本演示",
            })
        write_jsonl(self.graph/"drugs.jsonl",[
            {"id":"drug_1","name":"虚构药甲"}
        ])
        write_jsonl(self.graph/"source_records.jsonl",[
            {"id":"record_1","drug_id":"drug_1",
             "source_url":"https://example.org/fiction"}
        ])
        write_jsonl(self.graph/"sections.jsonl",records)
        self.write_csv(self.src,self.rows)

    @staticmethod
    def write_csv(path,rows):
        with path.open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def read_csv(path):
        with path.open("r",encoding="utf-8-sig",newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def response(row, *, decision="accept", target=None, polarity="positive",
                 qualifiers=None, role="indication_target", specificity="exact",
                 quote=None):
        return {
            "decision":decision,
            "relation_type":"treatment" if decision=="accept" else None,
            "target_role":role,
            "specificity":specificity,
            "polarity":polarity,
            "exact_disease_surface":target if target is not None else row["disease_name"],
            "qualifiers":[] if qualifiers is None else qualifiers,
            "supporting_quote":row["evidence_quote"] if quote is None else quote,
            "explanation":"合成文本证据分类",
            "needs_human_review":True,
        }

    def simulated(self, rows, responses, *, retries=0):
        self.write_csv(self.src, rows)
        with patch("kg.indication_v3.llm_endpoint",side_effect=responses):
            report=run(
                self.src,self.graph,self.out,
                mode="llm",model="mock",
                endpoint="http://localhost:8123/v1/chat/completions",
                max_llm=len(rows),
                max_retries=retries
            )
        return report,self.read_csv(self.out)

    def test_more_specific_target_is_not_unqualified_accept(self):
        row=self.rows[0]
        report, result=self.simulated(
            [row], [self.response(
                row, target="强直性脊椎炎",specificity="broader_than_evidence")]
        )
        x=result[0]
        self.assertEqual(x["v31_model_raw_decision"],"accept")
        self.assertEqual(x["v31_specific_target_surface"],"强直性脊椎炎")
        self.assertEqual(x["v31_candidate_decision"],"uncertain")
        self.assertEqual(x["v31_text_relation_status"],
                         "specific_target_proposed_review")
        self.assertEqual(x["review_status"],"rejected")
        self.assertIn("unqualified_candidate_broader_than_specific_source",
                      x["v31_override_reasons_json"])
        self.assertEqual(report["counts"]["llm_valid"],1)

    def test_conditional_support_is_not_unqualified_accept(self):
        row=self.rows[1]
        quot1="不作为治疗原发性高血压的首选药物"
        quot2="噻嗪类药物疗效不佳"
        report, result=self.simulated(
            [row], [self.response(
                row, polarity="conditional", qualifiers=[
                  {"type":"not_first_line","value":quot1},
                  {"type":"prior_therapy_failure","value":quot2},
                ]
            )]
        )
        x=result[0]
        self.assertEqual(x["v31_model_raw_decision"],"accept")
        self.assertEqual(x["v31_candidate_decision"],"uncertain")
        self.assertEqual(x["v31_text_relation_status"],
                         "conditional_positive_proposal")
        self.assertEqual(x["v31_candidate_scope"],"conditional_only")
        self.assertEqual(x["review_status"],"rejected")

    def test_bad_quote_retries_without_fabricating_evidence(self):
        row=self.rows[2]
        bogus=self.response(row, quote="模型改写了原文")
        corrected=self.response(row)
        report,result=self.simulated([row],[bogus,corrected],retries=1)
        x=result[0]
        self.assertEqual(report["counts"]["api_requests"],2)
        self.assertEqual(report["counts"]["llm_valid"],1)
        self.assertEqual(x["v31_llm_status"],"validated_model_output")
        self.assertEqual(x["v31_evidence_validated"],"true")

    def test_wrong_target_is_preserved_and_unresolved(self):
        row=self.rows[3]
        wrong=self.response(row,target="肾盂肾炎")
        report,result=self.simulated([row],[wrong],retries=0)
        x=result[0]
        self.assertEqual(x["v31_llm_status"],"failed_needs_manual_review")
        self.assertEqual(x["v31_candidate_decision"],"uncertain")
        self.assertIn("model disease surface",x["v31_validation_error"])
        self.assertIn("肾盂肾炎",x["v31_model_raw_json"])
        self.assertEqual(x["v31_evidence_validated"],"false")
        self.assertEqual(report["counts"]["llm_errors"],1)

    def test_patient_condition_not_treatment_target(self):
        row=self.rows[4]
        response=self.response(row,decision="reject",
                               role="patient_condition",polarity="uncertain")
        _,result=self.simulated([row],[response])
        self.assertEqual(result[0]["v31_candidate_decision"],"reject")
        self.assertEqual(result[0]["v31_proposed_relation"],"")

    def test_source_integrity_mismatch_aborts_before_calls(self):
        row=dict(self.rows[2])
        row["evidence_quote"]="篡改来源"
        self.write_csv(self.src,[row])
        with patch("kg.indication_v3.llm_endpoint") as f:
            with self.assertRaisesRegex(RuntimeError,"Preflight failed"):
                run(self.src,self.graph,self.out,mode="llm",model="mock",
                    endpoint="http://localhost:8000/v1/chat/completions")
            f.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_non_ascii_key_fails_before_calls(self):
        self.write_csv(self.src,[self.rows[2]])
        with patch.dict(os.environ,{"DASHSCOPE_API_KEY":"中文密钥"}):
            with patch("kg.indication_v3.llm_endpoint") as f:
                with self.assertRaisesRegex(ValueError,"ASCII"):
                    run(self.src,self.graph,self.out,mode="llm",model="mock",
                        endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                        api_key_env="DASHSCOPE_API_KEY")
                f.assert_not_called()

    def test_select_ids_subsets_without_modifying_annotations(self):
        self.write_csv(self.src,self.rows)
        sel=self.root/"ids.csv"
        self.write_csv(sel,[{"candidate_id":self.rows[1]["candidate_id"]}])
        report=run(self.src,self.graph,self.out,mode="rules",ids_from=sel)
        self.assertEqual(report["selected_rows"],1)
        items=self.read_csv(self.out)
        self.assertEqual(items[0]["candidate_id"],self.rows[1]["candidate_id"])
        self.assertEqual(items[0]["review_status"],"rejected")
        self.assertEqual(items[0]["v31_candidate_decision"],"uncertain")

    def test_dev_evaluate_counts_do_not_claim_clinical_accuracy(self):
        row=self.rows[2]
        self.simulated([row],[self.response(row)])
        rep=evaluate(self.out,self.src)
        self.assertEqual(rep["evaluated_rows"],1)
        self.assertIn("not independent expert gold",rep["reference"])
        self.assertEqual(
            rep["candidate_decision_matrix"]["approved|accept"],1)

    def test_selected_mention_not_confused_with_second_disease_mention(self):
        quote="尿路感染可见于鉴别诊断；本品仅用于复杂性尿路感染。"
        a=quote.index("尿路感染")
        row={
            "disease_name":"尿路感染",
            "evidence_quote":quote,
            "quote_start":"0",
            "disease_start":str(a),
            "disease_end":str(a+len("尿路感染"))
        }
        raw=self.response(
            row,
            target="复杂性尿路感染",
            quote=quote
        )
        with self.assertRaisesRegex(ValueError,"selected disease span"):
            model_validated(raw,row)

    def test_extended_boundary_guard_complex_urinary_infection(self):
        sentence="用于复杂性尿路感染。"
        a=sentence.index("尿路感染")
        rule={
            "specificity":"exact",
            "risk_flags":[],
            "target_role":"indication_target",
            "polarity":"positive"
        }
        verified={"text":sentence,"ds":a,"de":a+len("尿路感染")}
        result=scope_rule_v31(rule,verified)
        self.assertEqual(result["specificity"],"broader_than_evidence")
        self.assertIn("candidate_specificity_extended_v31",result["risk_flags"])

    def test_repeated_disease_name_second_occurrence_is_anchored(self):
        sentence="恶性淋巴瘤，非何杰金淋巴瘤和蕈样肉芽肿；"
        term="淋巴瘤"
        start=sentence.rindex(term)
        row={
            "disease_name":term,"evidence_quote":sentence,
            "quote_start":"0", "disease_start":str(start),
            "disease_end":str(start+len(term))
        }
        proof={"ds":start,"de":start+len(term),"qs":0}
        focus=build_anchored_focus(row,proof)
        self.assertEqual(focus["ordinal"],2)
        self.assertEqual(focus["occurrence_count"],2)
        self.assertEqual(focus["source_segment_hint"],"非何杰金淋巴瘤")
        self.assertIn("非何杰金【目标词开始】淋巴瘤【目标词结束】",
                      focus["marked_quote_for_disambiguation"])
        wrong=self.response(row,target="恶性淋巴瘤")
        with self.assertRaisesRegex(ValueError,"selected disease span"):
            model_validated(wrong,row)
        correct=self.response(row,target="非何杰金淋巴瘤",
                              specificity="broader_than_evidence")
        self.assertEqual(
            model_validated(correct,row)["exact_disease_surface"],
            "非何杰金淋巴瘤"
        )

    def test_full_surface_guard_does_not_trust_model_exact_specificity(self):
        sentence="本品适用于强直性脊椎炎。"
        term="脊椎炎"
        start=sentence.index(term)
        row=dict(self.rows[0])
        row.update({
            "evidence_quote":sentence,
            "disease_name":term,
            "disease_start":start,
            "disease_end":start+len(term),
            "quote_start":0,
            "quote_end":len(sentence),
            "section_id":"s0",
            "candidate_id":hash_id("DiseaseCandidate","s0",term,start,start+len(term))
        })
        # The existing synthetic graph already has this text; here the
        # response deliberately mislabels specificity as exact.
        self.write_csv(self.src,[row])
        report,items=self.simulated(
            [row],[self.response(row,target="强直性脊椎炎",specificity="exact")]
        )
        self.assertEqual(items[0]["v31_model_raw_specificity"],"exact")
        self.assertEqual(items[0]["v31_candidate_decision"],"uncertain")
        self.assertEqual(items[0]["v31_text_relation_status"],
                         "specific_target_proposed_review")
    def test_checkpoint_stop_after_and_resume_without_duplicate_calls(self):
        rows=self.rows[:3]
        self.write_csv(self.src,rows)
        answers=[self.response(r) for r in rows]
        options={
            "mode":"llm","model":"mock",
            "endpoint":"http://localhost:8123/v1/chat/completions",
            "max_llm":3,"max_retries":0,"timeout":20
        }
        with patch("kg.indication_v3.llm_endpoint",side_effect=answers) as m:
            first=run(self.src,self.graph,self.out,stop_after=2,**options)
            self.assertEqual(first["completed_rows"],2)
            self.assertEqual(first["remaining_rows"],1)
            self.assertFalse(first["complete"])
            self.assertTrue(self.out.with_suffix(".v31_checkpoint.json").exists())
            self.assertEqual(len(self.read_csv(self.out)),2)
            second=run(self.src,self.graph,self.out,resume=True,**options)
            self.assertEqual(second["completed_rows"],3)
            self.assertTrue(second["complete"])
            self.assertEqual(m.call_count,3)
        self.assertEqual(
            [x["candidate_id"] for x in self.read_csv(self.out)],
            [r["candidate_id"] for r in rows]
        )
        self.assertEqual(second["counts"]["llm_attempted"],3)
        self.assertEqual(second["counts"]["llm_valid"],3)

    def test_checkpoint_crash_keyboardinterrupt_resume(self):
        rows=self.rows[:3]
        self.write_csv(self.src,rows)
        options={
            "mode":"llm","model":"mock",
            "endpoint":"http://localhost:8123/v1/chat/completions",
            "max_llm":3,"max_retries":0,"timeout":20
        }
        with patch("kg.indication_v3.llm_endpoint",
                   side_effect=[self.response(rows[0]),KeyboardInterrupt]) as m:
            with self.assertRaises(KeyboardInterrupt):
                run(self.src,self.graph,self.out,**options)
            self.assertEqual(m.call_count,2)
        self.assertEqual(len(self.read_csv(self.out)),1)
        with patch("kg.indication_v3.llm_endpoint",
                   side_effect=[self.response(rows[1]),self.response(rows[2])]) as m:
            report=run(self.src,self.graph,self.out,resume=True,**options)
            self.assertEqual(m.call_count,2)
        self.assertEqual(report["counts"]["api_requests"],3)
        self.assertEqual(len(self.read_csv(self.out)),3)

    def test_resume_rejects_changed_prompt_or_config(self):
        rows=self.rows[:2]
        self.write_csv(self.src,rows)
        options={
            "mode":"llm","model":"mock",
            "endpoint":"http://localhost:8123/v1/chat/completions",
            "max_llm":2,"max_retries":0,"timeout":20
        }
        with patch("kg.indication_v3.llm_endpoint",side_effect=[
            self.response(rows[0])
        ]):
            run(self.src,self.graph,self.out,stop_after=1,**options)
        with self.assertRaisesRegex(ValueError,"differ from checkpoint"):
            run(self.src,self.graph,self.out,
                resume=True,**{**options,"model":"other_model"})
        with self.assertRaisesRegex(ValueError,"differ from checkpoint"):
            run(self.src,self.graph,self.out,
                resume=True,**{**options,"timeout":30})
        self.assertEqual(len(self.read_csv(self.out)),1)

    def test_resume_rejects_csv_source_mutation(self):
        rows=self.rows[:2]
        self.write_csv(self.src,rows)
        with patch("kg.indication_v3.llm_endpoint",side_effect=[
            self.response(rows[0])
        ]):
            run(self.src,self.graph,self.out,stop_after=1,
                mode="llm",model="mock",
                endpoint="http://localhost:8123/v1/chat/completions",
                max_llm=2,max_retries=0)
        changed=[dict(r) for r in rows]
        changed[0]["review_note"]="trial modified after checkpoint"
        self.write_csv(self.src,changed)
        with self.assertRaisesRegex(ValueError,"differ from checkpoint"):
            run(self.src,self.graph,self.out,resume=True,
                mode="llm",model="mock",
                endpoint="http://localhost:8123/v1/chat/completions",
                max_llm=2,max_retries=0)

    def test_partial_checkpoint_cannot_be_evaluated_as_full_set(self):
        rows=self.rows[:3]
        self.write_csv(self.src,rows)
        report=run(self.src,self.graph,self.out,
                   mode="rules",stop_after=1)
        self.assertFalse(report["complete"])
        with self.assertRaisesRegex(ValueError,"incomplete"):
            evaluate(self.out,self.src)

    def test_failed_row_is_persisted_and_not_retried_on_resume(self):
        rows=self.rows[:2]
        self.write_csv(self.src,rows)
        options={
            "mode":"llm","model":"mock",
            "endpoint":"http://localhost:8123/v1/chat/completions",
            "max_llm":2,"max_retries":0
        }
        with patch("kg.indication_v3.llm_endpoint",
                   side_effect=[RuntimeError("simulated timeout")]):
            partial=run(self.src,self.graph,self.out,stop_after=1,**options)
        self.assertEqual(partial["counts"]["llm_errors"],1)
        self.assertEqual(self.read_csv(self.out)[0]["v31_llm_status"],
                         "failed_needs_manual_review")
        with patch("kg.indication_v3.llm_endpoint",
                   side_effect=[self.response(rows[1])]) as mock:
            final=run(self.src,self.graph,self.out,resume=True,**options)
            mock.assert_called_once()
        self.assertEqual(final["counts"]["llm_errors"],1)
        self.assertEqual(final["counts"]["llm_valid"],1)
        self.assertEqual(final["counts"]["api_requests"],2)

    def test_no_auto_reclassification_from_negative_quote(self):
        row=self.rows[5]
        _,res=self.simulated([row],[self.response(row)])
        self.assertEqual(res[0]["v31_candidate_decision"],"uncertain")
        self.assertIn("conflicting_target_role_or_negative_polarity",
                      res[0]["v31_override_reasons_json"])

if __name__=="__main__":
    unittest.main()
