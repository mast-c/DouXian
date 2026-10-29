import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from kg.indication_v3 import (
    run_classification, rule_classify, validate_llm, postprocess_llm,
    evaluate, extract_context
)
from kg.disease_stage3 import hash_id

def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")

class IndicationV3Test(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.graph=self.base/"graph"
        write_jsonl(self.graph/"drugs.jsonl",[
            {"id":"drug_1","name":"示例甲"}
        ])
        write_jsonl(self.graph/"source_records.jsonl",[
            {"id":"src_1","drug_id":"drug_1","source_url":"https://example.org/source"}
        ])
        self.texts=[
            "本品适用于强直性脊椎炎。",
            "本品不适用于脑膜炎的治疗。",
            "本品用于高血压，但当伴有肾功能不全时更应注意。",
            "对其他药物治疗无效的反流性食管炎有效。",
            "用于缓解恶心和呕吐。",
        ]
        self.diseases=["脊椎炎","脑膜炎","肾功能不全","反流性食管炎","恶心和呕吐"]
        self.rows=[]
        sections=[]
        for i,(text,term) in enumerate(zip(self.texts,self.diseases),1):
            sec_id=f"section_{i}"
            ds=text.index(term)
            de=ds+len(term)
            sections.append({"id":sec_id,"record_id":"src_1",
                             "field":"indication","text":text})
            self.rows.append({
                "candidate_id":hash_id("DiseaseCandidate",sec_id,term,ds,de),
                "drug_name":"示例甲","disease_name":term,
                "source_record_id":"src_1","section_id":sec_id,
                "source_url":"https://example.org/source",
                "evidence_quote":text,"quote_start":"0","quote_end":str(len(text)),
                "disease_start":str(ds),"disease_end":str(de),
                "context_flag":"manual_semantic_review","review_status":"pending",
                "relation_type":"","reviewer":"","review_note":"",
            })
        write_jsonl(self.graph/"sections.jsonl",sections)
        self.csv=self.base/"source.csv"
        self.out=self.base/"pred.csv"
        self.write_csv(self.rows)

    def write_csv(self,rows):
        with self.csv.open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def predict(self):
        run_classification(self.csv,self.graph,self.out)
        with self.out.open("r",encoding="utf-8-sig",newline="") as f:
            return list(csv.DictReader(f))

    def test_specific_disease_cannot_be_unqualified(self):
        result=self.predict()
        first=result[0]
        self.assertEqual(first["v3_specificity"],"broader_than_evidence")
        self.assertIn("candidate_broader_than_specific_disease",
                      json.loads(first["v3_risk_flags"]))
        self.assertEqual(first["review_status"],"pending")
        self.assertEqual(first["v3_needs_human_review"],"true")

    def test_negation_and_patient_condition_roles(self):
        result=self.predict()
        self.assertEqual(result[1]["v3_proposed_decision"],"reject")
        self.assertEqual(result[1]["v3_target_role"],"exclusion")
        self.assertEqual(result[2]["v3_proposed_decision"],"reject")
        self.assertEqual(result[2]["v3_target_role"],"patient_condition")
        # Previous-therapy failure is a qualification, not a ban on use.
        self.assertNotEqual(result[3]["v3_polarity"],"negative")

    def test_source_integrity_guard(self):
        bad=[dict(r) for r in self.rows]
        bad[2]["evidence_quote"]="内容被篡改"
        self.write_csv(bad)
        with self.assertRaisesRegex(RuntimeError,"integrity"):
            run_classification(self.csv,self.graph,self.out)
        self.assertFalse(self.out.exists())

    def test_llm_evidence_validation(self):
        output={
            "decision":"accept","relation_type":"treatment",
            "target_role":"indication_target","specificity":"exact",
            "polarity":"positive","exact_disease_surface":"脑膜炎",
            "qualifiers":[],"supporting_quote":"脑膜炎",
            "explanation":"直接匹配", "needs_human_review":True,
        }
        self.assertEqual(
            validate_llm(output,"本品不适用于脑膜炎的治疗。","脑膜炎")["decision"],
            "accept"
        )
        output["qualifiers"]=[{"type":"population","value":"不存在的儿童限制"}]
        with self.assertRaisesRegex(ValueError,"qualifier"):
            validate_llm(output,"本品不适用于脑膜炎的治疗。","脑膜炎")

    def test_rule_overrides_bad_model_accept(self):
        result=self.predict()
        rule={
            "specificity":result[0]["v3_specificity"],
            "target_role":"indication_target","polarity":"positive",
            "risk_flags":["candidate_broader_than_specific_disease"]
        }
        accepted={"decision":"accept","relation_type":"treatment"}
        guard=postprocess_llm(accepted,rule)
        self.assertEqual(guard["decision"],"uncertain")
        self.assertTrue(guard["needs_human_review"])

    def test_conditional_appearance_is_not_a_causal_rejection(self):
        text=("在高血压阶梯疗法中，当噻嗪类药物疗效不佳，"
              "尤其当伴有肾功能不全或出现高血压危象时，本类药物尤为适用。")
        ds=text.index("高血压危象")
        evidence={"text":text,"ds":ds,"de":ds+len("高血压危象")}
        row={"disease_name":"高血压危象",
             "context_flag":"negation_or_restriction_check",
             "evidence_quote":text}
        output=rule_classify(row,evidence)
        self.assertNotEqual(output["decision"],"reject")
        self.assertNotEqual(output["target_role"],"adverse_effect_or_cause")
        self.assertEqual(output["polarity"],"conditional")

    def test_mock_llm_is_never_auto_approved(self):
        def fake_llm(endpoint, model, system, payload, key,
                     timeout=45, use_json_mode=True):
            return {
                "decision":"accept",
                "relation_type":"treatment",
                "target_role":"indication_target",
                "specificity":"exact","polarity":"positive",
                "exact_disease_surface":payload["disease_name"],
                "qualifiers":[],
                "supporting_quote":payload["disease_name"],
                "explanation":"仅提供候选建议",
                "needs_human_review":True,
            }
        with patch("kg.indication_v3.llm_endpoint",side_effect=fake_llm):
            report=run_classification(
                self.csv,self.graph,self.out,mode="llm",
                model="fake-model",
                endpoint="http://localhost:8000/v1/chat/completions",
                max_llm=2,limit=2
            )
        self.assertEqual(report["counts"]["llm_valid"],2)
        with self.out.open("r",encoding="utf-8-sig",newline="") as f:
            predictions=list(csv.DictReader(f))
        self.assertEqual(predictions[0]["v3_proposed_decision"],"uncertain")
        self.assertEqual(predictions[1]["v3_proposed_decision"],"uncertain")
        self.assertTrue(all(x["review_status"]=="pending" for x in predictions))
        self.assertTrue(all(x["v3_needs_human_review"]=="true" for x in predictions))

    def test_exact_offset_for_repeated_mentions(self):
        text="适用于高血压。对其他治疗失败的高血压可考虑。"
        start=text.rfind("高血压")
        context,relative_start,relative_end=extract_context(
            text,start,start+len("高血压")
        )
        self.assertEqual(context[relative_start:relative_end],"高血压")
        self.assertIn("治疗失败",context)

if __name__=="__main__":
    unittest.main()
