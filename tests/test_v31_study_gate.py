"""Offline tests: all cases are synthetic; no live holdout or API calls."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import v31_study_gate as gate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home=Path(self.tmp.name)
        self.code=self.home/"frozen_code.py"
        self.code.write_text('VERSION = "mock_v31"\n',encoding="utf-8")
        self.prompt=self.home/"prompt.md"
        self.prompt.write_text("Stable, no new test-time tuning.",encoding="utf-8")
        self.dev=self.home/"dev60.csv"
        self.devrows=[self.srcrow(f"d-{j}") for j in range(60)]
        self.write_rows(self.dev, self.devrows)
        self.dev_report=self.home/"dev60.v31_report.json"
        self.dev_report.write_text(json.dumps({
            "version":"mock_v31","model":"mock",
            "complete":True,"selected_rows":60,"completed_rows":60,
            "output_sha256":gate.sha(self.dev),
            "prompt_sha256":gate.sha(self.prompt),
            "input_sha256":"fake_source_sha",
            "source_graph_sha256":"fake_graph_sha",
            "counts":{"llm_valid":58}
        }),encoding="utf-8")
        self.freeze=self.home/"freeze.json"
        self.priority=self.home/"priority.csv"
        self.write_rows(self.priority,[
            {**r,"model_raw_decision":"accept",
             "original_ai_review_status":"approved"} for r in self.devrows[:21]
        ])
        self.holdout=self.home/"holdout300.csv"
        self.holdoutrows=[{
            **self.srcrow(f"h-{j}"),
            "review_status":"approved",
            "relation_type":"treatment",
            "gold_classification":"SUPPORTED_UNQUALIFIED",
            "gold_relation_type":"treatment",
        } for j in range(300)]
        self.write_rows(self.holdout,self.holdoutrows)

    @staticmethod
    def srcrow(cid):
        quote="示例药品仅用于甲疾病。"
        target="甲疾病"
        ds=quote.index(target)
        return {
            "candidate_id":cid,
            "drug_name":"测试药",
            "disease_name":target,
            "source_record_id":"record1",
            "section_id":"section1",
            "source_url":"https://example.test/label",
            "evidence_quote":quote,
            "quote_start":"0",
            "quote_end":str(len(quote)),
            "disease_start":str(ds),
            "disease_end":str(ds+len(target)),
            "context_flag":"synthetic",
            "review_status":"approved",
            "relation_type":"treatment",
            "reviewer":"test",
            "review_note":"test only"
        }

    @staticmethod
    def write_rows(path,rows):
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def freezed(self):
        from argparse import Namespace
        gate.freeze(Namespace(
            code=str(self.code),
            prompt=str(self.prompt),
            dev_csv=str(self.dev),
            dev_report=str(self.dev_report),
            out=str(self.freeze),
            model="mock",
            endpoint="https://example.test/v1/chat/completions",
            timeout=90,
            max_retries=0
        ))

    def make_holdout(self):
        from argparse import Namespace
        self.freezed()
        self.input=self.home/"inference_only.csv"
        self.blind=self.home/"gold_blind.csv"
        gate.prepare_holdout(Namespace(
            freeze=str(self.freeze),
            input=str(self.holdout),
            out=str(self.input),
            annotation_out=str(self.blind),
            expect=300
        ))

    def test_freeze_is_valid_but_not_clinical_approval(self):
        self.freezed()
        frozen=gate.verify_freeze(self.freeze)
        self.assertEqual(frozen["status"],"engineering_baseline_only")
        self.assertFalse(frozen["clinical_release_ready"])
        self.assertTrue(frozen["metadata"]["frozen_before_holdout"])
        self.assertEqual(frozen["prompt_sha256"],gate.sha(self.prompt))

    def test_freeze_rejects_changed_prompt_hash(self):
        self.freezed()
        self.prompt.write_text("Modified prompt after freeze",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"prompt file has changed"):
            gate.verify_freeze(self.freeze)

    def test_blind_review_forms_never_include_llm_predictions(self):
        from argparse import Namespace
        out=self.home/"review"
        gate.make_review_forms(Namespace(
            priority=str(self.priority),out_dir=str(out)
        ))
        _,rows=gate.load_csv(out/"reviewer_A_21_blind.csv")
        self.assertEqual(len(rows),21)
        self.assertNotIn("model_raw_decision",rows[0])
        self.assertNotIn("original_ai_review_status",rows[0])
        self.assertEqual(rows[0]["gold_classification"],"")

    def test_prepare_holdout_strips_all_gold_or_development_labels(self):
        self.make_holdout()
        cols,inf=gate.load_csv(self.input)
        self.assertEqual(len(inf),300)
        self.assertNotIn("gold_classification",cols)
        self.assertEqual(inf[0]["review_status"],"unlabeled")
        self.assertEqual(inf[0]["relation_type"],"")
        _,blind=gate.load_csv(self.blind)
        self.assertEqual(blind[0]["gold_classification"],"")

    def test_holdout_cannot_overlap_development_ids(self):
        self.freezed()
        self.holdoutrows[0]["candidate_id"]=self.devrows[0]["candidate_id"]
        self.write_rows(self.home/"overlap.csv",self.holdoutrows)
        from argparse import Namespace
        with self.assertRaisesRegex(ValueError,"overlaps development"):
            gate.prepare_holdout(Namespace(
                freeze=str(self.freeze),input=str(self.home/"overlap.csv"),
                out=str(self.home/"inference.csv"),
                annotation_out=str(self.home/"anno.csv"),expect=300
            ))

    def test_score_300_synthetic_only_with_adjudicated_gold(self):
        from argparse import Namespace
        self.make_holdout()
        _,inferences=gate.load_csv(self.input)
        predicted=[]
        for i,r in enumerate(inferences):
            d=dict(r)
            d.update({
                "v31_llm_status":"validated_model_output",
                "v31_candidate_decision":"accept" if i<2 else "uncertain",
                "v31_proposed_relation":"treatment" if i<2 else "",
                "v31_specific_target_surface":r["disease_name"] if i<2 else ""
            })
            predicted.append(d)
        predictions=self.home/"pred.csv"
        self.write_rows(predictions,predicted)
        prep=self.input.with_suffix(".holdout_preflight.json")
        fp=json.loads(prep.read_text())
        fr=json.loads(self.freeze.read_text())
        rpath=self.home/"pred.v31_report.json"
        rpath.write_text(json.dumps({
            "complete":True,"selected_rows":300,
            "output_sha256":gate.sha(predictions),
            "input_sha256":gate.sha(self.input),
            "model":fr["model"],
            "prompt_sha256":fr["prompt_sha256"]
        }),encoding="utf-8")
        gold=[]
        _,blind=gate.load_csv(self.blind)
        for i,r in enumerate(blind):
            g=dict(r)
            g.update({
                "gold_classification":"SUPPORTED_UNQUALIFIED" if i<2 else "NOT_SUPPORTED",
                "gold_exact_target":r["disease_name"] if i<2 else "",
                "gold_target_type":"Disease" if i<2 else "",
                "gold_relation_type":"treatment" if i<2 else "",
                "gold_polarity":"positive" if i<2 else "",
                "gold_qualifiers_json":"[]",
                "source_identity_verified":"unknown",
                "annotator_name":"synthetic",
                "annotator_role":"test_only"
            })
            gold.append(g)
        gpath=self.home/"gold.csv"
        self.write_rows(gpath,gold)
        report=gate.score_holdout(Namespace(
            freeze=str(self.freeze),
            preflight=str(prep),pred=str(predictions),
            pred_report=str(rpath),gold=str(gpath),
            out=str(self.home/"score.json")
        ))
        self.assertEqual(report["holdout_size"],300)
        self.assertEqual(report["counts"]["accepted"],2)
        self.assertEqual(report["counts"]["accepted_matches_gold_unqualified"],2)
        self.assertEqual(report["source_identity_not_verified_among_accepts"],2)

    def test_validate_gold_rejects_nonverbatim_target(self):
        r={
            "candidate_id":"test","gold_classification":"SUPPORTED_UNQUALIFIED",
            "gold_exact_target":"不在原文的诊断",
            "evidence_quote":"仅有甲疾病",
            "gold_target_type":"Disease","gold_relation_type":"treatment",
            "gold_polarity":"positive","gold_qualifiers_json":"[]",
            "source_identity_verified":"unknown",
            "annotator_name":"test","annotator_role":"test"
        }
        self.assertTrue(any("verbatim" in e for e in gate.validate_gold(r)))

if __name__=="__main__":
    unittest.main()
