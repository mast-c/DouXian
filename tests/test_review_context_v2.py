import csv
import json
from pathlib import Path
import tempfile
import unittest

from kg.review_context_v2 import (
    export_reviewed, import_reviewed, REQUIRED_FIELDS, SCHEMA_VERSION
)
from kg.disease_stage3 import hash_id

def save(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")

class QualifiedReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.graph=self.root/"graph"
        self.output=self.root/"pilot"
        self.csv_path=self.root/"pilot.csv"
        self.drug_id="drug_demo_a"
        self.record_id="src_demo_a"
        self.section_id="section_demo_a"
        self.source="在其他方案无效时，本品可用于疾病甲的治疗。"
        disease="疾病甲"
        ds=self.source.index(disease)
        de=ds+len(disease)
        save(self.graph/"drugs.jsonl",[
            {"id":self.drug_id,"name":"虚构示例药甲"}])
        save(self.graph/"source_records.jsonl",[
            {"id":self.record_id,"drug_id":self.drug_id,
             "source_url":"https://example.org/demo"}])
        save(self.graph/"sections.jsonl",[
            {"id":self.section_id,"record_id":self.record_id,
             "field":"indication","text":self.source}])
        self.row={
            "candidate_id":hash_id(
                "DiseaseCandidate",self.section_id,disease,ds,de
            ),
            "drug_name":"虚构示例药甲",
            "disease_name":disease,
            "source_record_id":self.record_id,
            "section_id":self.section_id,
            "source_url":"https://example.org/demo",
            "evidence_quote":self.source,
            "quote_start":"0",
            "quote_end":str(len(self.source)),
            "disease_start":str(ds),
            "disease_end":str(de),
            "context_flag":"negation_or_restriction_check",
            "review_status":"approved",
            "relation_type":"treatment",
            "reviewer":"Codex-医学文本审核",
            "review_note":"在既往治疗无效的条件下使用，保留限制条件",
        }

    def write_csv(self, rows):
        with self.csv_path.open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(self.row))
            writer.writeheader()
            writer.writerows(rows)

    def test_conditional_approved_is_preserved(self):
        self.write_csv([self.row])
        report=export_reviewed(self.csv_path,self.graph,self.output)
        self.assertEqual(report["accepted_claims"],1)
        self.assertEqual(report["context_sensitive_approved"],1)
        with (self.output/"pilot_indication_claims.jsonl").open(
                "r",encoding="utf-8") as f:
            claim=json.loads(f.readline())
        self.assertTrue(claim["requires_condition_check"])
        self.assertEqual(claim["expert_review_status"],"pending")
        self.assertEqual(claim["clinical_status"],"not_validated")
        self.assertEqual(claim["annotation_method"],
                         "AI_assisted_not_expert_verified")
        self.assertIn("既往治疗无效",claim["qualification_note"])
        preview=import_reviewed(self.output,apply=False)
        self.assertEqual(preview["provisional_claim_count"],1)

    def test_missing_approval_note_is_blocked(self):
        self.write_csv([{**self.row,"review_note":""}])
        with self.assertRaisesRegex(ValueError,"reviewer, note"):
            export_reviewed(self.csv_path,self.graph,self.output)
        self.assertFalse((self.output/"pilot_indication_claims.jsonl").exists())

    def test_modified_source_quote_is_blocked(self):
        self.write_csv([{**self.row,"evidence_quote":"篡改证据"}])
        with self.assertRaisesRegex(ValueError,"Evidence text"):
            export_reviewed(self.csv_path,self.graph,self.output)

    def test_rejected_cannot_import_as_claim(self):
        self.write_csv([{**self.row,"review_status":"rejected",
                         "review_note":"审核拒绝","relation_type":""}])
        result=export_reviewed(self.csv_path,self.graph,self.output)
        self.assertEqual(result["accepted_claims"],0)
        self.assertEqual(result["decisions"]["rejected"],1)

if __name__=="__main__":
    unittest.main()
