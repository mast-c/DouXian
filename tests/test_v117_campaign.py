# -*- coding: utf-8 -*-
"""V1.17 持续评测：纯离线集成回归，不能把模拟样本当真实医疗QA结果。"""
from __future__ import annotations
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def import_module(filename,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    instance=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instance)
    return instance

m=import_module("medical_quality_campaign.py","quality_campaign_v117")
checker=import_module("medical_quality_acceptance.py","quality_checker_v116")

def payload():
    p=json.loads((ROOT/"sample_synthetic_v116.json").read_text(encoding="utf-8"))
    p.pop("demo_warning",None)
    p.pop("is_synthetic",None)
    p["scope_notice"]="合成样例用于离线测试，不代表本机实际数据库或医学证据"
    return p

def write_sample(directory:Path,filename:str,*,add_invalid_citation=False,
                 wrong_drug=False):
    data=payload()
    if add_invalid_citation:
        data["answer"]=data["answer"].replace(
            "参考资料（同一来源的不同片段合并显示）：",
            "本轮存在模拟无效引用[99]\\n\\n参考资料（同一来源的不同片段合并显示）：",
            1
        )
    if wrong_drug:
        e=data["trace"]["knowledge_evidence"]
        e[-1]["drug_name"]="离题的其他药物"
    path=directory/filename
    path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
    return path

def rows_from_manifest(manifest, *, reviewed=0,overstated=False):
    out=[]
    for i,case in enumerate(manifest["cases"]):
        row={k:"" for k in m.REVIEW_COLUMNS}
        row.update({
            "case_id":case["case_id"],
            "source_filename":case["source_filename"],
            "source_sha256":case["source_sha256"],
            "question":case["question"],
            "answer_excerpt":case["answer_excerpt"],
            "engineering_status":case["engineering_status"],
            "engineering_issue_codes":"|".join(case["engineering_issue_codes"]),
            "unverified_disease_candidate_count":str(case["unverified_disease_candidate_count"]),
            "review_focus":case["review_focus"],
            "review_status":"unreviewed",
        })
        if i<reviewed:
            row.update({
                "review_status":"reviewed",
                "evidence_support":"supported",
                "relation_safety":"overstated" if overstated and i==0 else "adequately_caveated",
                "medical_safety":"needs_correction" if overstated and i==0 else "acceptable",
                "source_authority":"third_party_unverified",
                "reviewer":"人工标注员A",
                "review_note":"逐条检查原文，对照关系类型",
            })
        out.append(row)
    return out

def write_review(path:Path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=m.REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:m._safe_csv(row.get(key,"")) for key in m.REVIEW_COLUMNS})

class TestCollect(unittest.TestCase):
    def test_end_to_end_batch_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            trace_dir=base/"exports"
            trace_dir.mkdir()
            write_sample(trace_dir,"medical_trace_001.json")
            write_sample(trace_dir,"medical_trace_002.json",add_invalid_citation=True)
            (trace_dir/"medical_trace_003.json").write_bytes(
                (trace_dir/"medical_trace_001.json").read_bytes()
            )
            report=m.collect(trace_dir,base/"campaign",snapshot_label="test-v117")
            self.assertEqual(report["schema_version"],1)
            self.assertEqual(report["summary"]["valid_unique_runs"],2)
            self.assertEqual(report["summary"]["invalid_or_duplicate_files"],1)
            self.assertEqual(
                report["summary"]["engineering_status_counts"],
                {"PASS":1,"WARN":0,"FAIL":1}
            )
            self.assertEqual(report["summary"]["invalid_body_citations"],1)
            self.assertEqual(report["summary"]["multi_drug_expected_cells"],16)
            self.assertEqual(report["summary"]["multi_drug_correctly_cited_cells"],16)
            self.assertGreater(report["summary"]["unverified_candidate_cards"],0)
            self.assertEqual(
                report["invalid_files"][0]["kind"],"duplicated_export"
            )
            self.assertTrue(
                (base/"campaign/engineering_manifest_v117.json").exists()
            )
            with (base/"campaign/manual_review_v117.csv").open(
                encoding="utf-8-sig",newline=""
            ) as f:
                reviews=list(csv.DictReader(f))
            self.assertEqual(len(reviews),2)
            self.assertTrue(all(x["review_status"]=="unreviewed" for x in reviews))
            self.assertTrue(all(x["evidence_support"]=="" for x in reviews))

    def test_recollect_refuses_to_overwrite_manual_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            source=base/"traces"
            source.mkdir()
            write_sample(source,"medical_trace_001.json")
            m.collect(source,base/"results")
            target=base/"results/manual_review_v117.csv"
            target.write_text("人工审核不能丢失",encoding="utf-8")
            with self.assertRaises(FileExistsError):
                m.collect(source,base/"results")
            self.assertEqual(target.read_text(encoding="utf-8"),"人工审核不能丢失")

    def test_no_exports_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                m.evaluate_files(Path(tmp))

    def test_invalid_pattern_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):
                m.evaluate_files(root,pattern="../*.json")

    def test_same_question_different_trace_is_not_deduped(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            write_sample(d,"medical_trace_001.json")
            write_sample(d,"medical_trace_002.json",wrong_drug=True)
            rows,skipped=m.evaluate_files(d)
            self.assertEqual(len(rows),2)
            self.assertEqual(len(skipped),0)
            self.assertEqual(rows[0]["question"],rows[1]["question"])
            self.assertNotEqual(rows[0]["source_sha256"],rows[1]["source_sha256"])

    def test_synthetic_marker_rejected_even_if_renamed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"medical_trace_001.json"
            path.write_bytes((ROOT/"sample_synthetic_v116.json").read_bytes())
            rows,issues=m.evaluate_files(Path(tmp))
            self.assertEqual(rows,[])
            self.assertEqual(issues[0]["kind"],"invalid_export")
            self.assertIn("合成/演示数据",issues[0]["reason"])

    def test_malformed_and_empty_trace_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"medical_trace_001.json"
            path.write_text("{bad json}",encoding="utf-8")
            rows,issues=m.evaluate_files(Path(tmp))
            self.assertEqual(rows,[])
            self.assertEqual(issues[0]["kind"],"invalid_export")
            data=payload()
            data["trace"].pop("knowledge_evidence")
            path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
            rows,issues=m.evaluate_files(Path(tmp))
            self.assertEqual(rows,[])
            self.assertIn("evidence",issues[0]["reason"])

    def test_file_byte_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"medical_trace_001.json"
            source.write_bytes(b"a"*64)
            with patch.object(m,"MAX_FILE_BYTES",30):
                with self.assertRaisesRegex(ValueError,"15MB"):
                    m._load_export(source)

    def test_disallow_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            real=write_sample(d,"medical_trace_001.json")
            sym=d/"medical_trace_002.json"
            try:
                sym.symlink_to(real)
            except (OSError,NotImplementedError):
                self.skipTest("filesystem does not support symlink")
            rows,bad=m.evaluate_files(d)
            self.assertEqual(len(rows),1)
            self.assertEqual(len(bad),1)
            self.assertIn("符号链接",bad[0]["reason"])

    def test_csv_formula_cells_escaped(self):
        for cell in ("=1+2","+2","-1","@command","\tSUM(A1)","regular","中文内容"):
            actual=m._read_csv(m._safe_csv(cell))
            self.assertEqual(actual,cell)

class TestRetrievalReportAttachment(unittest.TestCase):
    @staticmethod
    def sample(mode,*,snapshot="fixed-chroma-snapshot",
               hit=0.5,gold="section_good"):
        return {
            "mode":mode,"collection":"rag",
            "collection_fingerprint":snapshot,
            "summary":{
                "top_k":5,"positive_reviewed_cases":20,
                "raw":{
                    "section_hit_at_5":hit,
                    "mean_section_recall_at_5":hit,
                    "wrong_drug_chunk_rate":0.1,
                    "average_returned_chunks":5.0,
                },
                "scoped":{
                    "section_hit_at_5":hit,
                    "mean_section_recall_at_5":hit,
                    "wrong_drug_chunk_rate":0.0,
                    "average_returned_chunks":4.0,
                },
            },
            "results":[{"case_id":"MED001","gold_section_ids":[gold]}],
        }

    def test_two_reports_comparable_same_snapshot_and_gold(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            files=[p/"bm25.json",p/"hybrid.json"]
            for path,data in zip(files,[
                self.sample("bm25",hit=0.6),
                self.sample("hybrid-rerank",hit=0.9),
            ]):
                path.write_text(json.dumps(data),encoding="utf-8")
            result=m.attach_retrieval_reports(files)
            self.assertEqual(
                result["comparison_status"],
                "COMPARABLE_SAME_SNAPSHOT_AND_GOLD"
            )
            self.assertEqual(result["comparison"]["gold_cases"],1)
            self.assertEqual(
                result["comparison"]["comparisons"]["scoped"]["section_hit_at_5"]["difference"],
                0.3
            )
            self.assertTrue(result["sample_sets_independent_of_generation_quality"])

    def test_different_chroma_snapshot_not_comparable(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            files=[p/"bm25.json",p/"hybrid.json"]
            data=[self.sample("bm25"),self.sample("hybrid",snapshot="changed")]
            for path,obj in zip(files,data):
                path.write_text(json.dumps(obj),encoding="utf-8")
            result=m.attach_retrieval_reports(files)
            self.assertEqual(result["comparison_status"],"NOT_COMPARABLE")
            self.assertIn("不可公平比较",result["warning"])

    def test_different_gold_not_comparable(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            files=[p/"bm25.json",p/"hybrid.json"]
            for path,obj in zip(files,[
                self.sample("bm25"),self.sample("hybrid",gold="other")]):
                path.write_text(json.dumps(obj),encoding="utf-8")
            result=m.attach_retrieval_reports(files)
            self.assertEqual(result["comparison_status"],"NOT_COMPARABLE")

    def test_invalid_or_more_than_two_reports_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"invalid.json"
            p.write_text("{}",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"缺少 V1.13"):
                m.attach_retrieval_reports([p])
        with self.assertRaisesRegex(ValueError,"最多导入"):
            m.attach_retrieval_reports([Path("a"),Path("b"),Path("c")])

    def test_question_bank_covers_requested_error_modes(self):
        lines=(ROOT/"evaluation_questions_v117.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        cases=[json.loads(x) for x in lines if x.strip()]
        self.assertEqual(len(cases),18)
        self.assertEqual(len(set(x["id"] for x in cases)),18)
        self.assertTrue(all(x["annotation_status"]=="unreviewed" for x in cases))
        self.assertTrue(all(x["clinical_truth_verified"] is False for x in cases))
        focus={x["review_focus"] for x in cases}
        self.assertTrue({"risk_vs_treatment","component_vs_product",
                         "multi_drug_full_coverage","official_approval",
                         "individualized_medication"}.issubset(focus))


class TestManualScoring(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name)
        trace_dir=self.base/"exports"
        trace_dir.mkdir()
        write_sample(trace_dir,"medical_trace_001.json")
        self.manifest=m.collect(trace_dir,self.base/"campaign")
        self.campaign_path=self.base/"campaign/engineering_manifest_v117.json"
        self.reviews_path=self.base/"campaign/manual_review_v117.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_annotations_do_not_claim_accuracy(self):
        rows,counts=m.parse_reviews(self.reviews_path,self.manifest)
        self.assertEqual(counts["unreviewed"],1)
        report=m.manual_review_summary(self.manifest,rows)
        self.assertEqual(report["quality_status"],"NO_HUMAN_REVIEW")
        self.assertIsNone(report["human_review_rates"]["fully_supported_fraction"])
        self.assertEqual(report["review_counts"]["reviewed"],0)

    def test_reviewed_but_small_sample_remains_insufficient(self):
        rows=rows_from_manifest(self.manifest,reviewed=1)
        write_review(self.reviews_path,rows)
        parsed,_=m.parse_reviews(self.reviews_path,self.manifest)
        report=m.manual_review_summary(self.manifest,parsed,min_reviewed=20)
        self.assertEqual(report["quality_status"],"INSUFFICIENT_SAMPLE")
        self.assertEqual(report["human_review_rates"]["fully_supported_fraction"],1)
        self.assertEqual(report["human_review_rates"]["third_party_only_fraction"],1)

    def test_overstated_or_unsafe_blocks_baseline(self):
        rows=rows_from_manifest(self.manifest,reviewed=1,overstated=True)
        write_review(self.reviews_path,rows)
        parsed,_=m.parse_reviews(self.reviews_path,self.manifest)
        report=m.manual_review_summary(self.manifest,parsed,min_reviewed=1)
        self.assertEqual(report["quality_status"],"REVIEW_AND_FIX")
        self.assertEqual(report["human_review_rates"]["overstated_relation_fraction"],1)
        self.assertEqual(len(report["critical_cases"]),1)

    def test_human_uncertainty_does_not_become_approved_baseline(self):
        rows=rows_from_manifest(self.manifest,reviewed=1)
        rows[0]["evidence_support"]="uncertain"
        result=m.manual_review_summary(self.manifest,rows,min_reviewed=1)
        self.assertEqual(result["quality_status"],"REVIEW_UNCERTAIN")
        self.assertEqual(result["unresolved_case_ids"],[rows[0]["case_id"]])
        self.assertEqual(result["human_review_rates"]["fully_supported_fraction"],0)

    def test_engineering_fail_has_priority_even_if_reviewed(self):
        bad=copy.deepcopy(self.manifest)
        bad["summary"]["engineering_status_counts"]["FAIL"]=1
        rows=rows_from_manifest(bad,reviewed=1)
        result=m.manual_review_summary(bad,rows,min_reviewed=1)
        self.assertEqual(result["quality_status"],"ENGINEERING_NEEDS_FIX")

    def test_replaying_same_question_20_times_cannot_pass_sample_size(self):
        exemplar=copy.deepcopy(self.manifest["cases"][0])
        cloned=[]
        for i in range(20):
            c=copy.deepcopy(exemplar)
            c["case_id"]=f"Q{i:03d}"
            c["source_sha256"]=f"{i:064d}"
            cloned.append(c)
        manifest={
            "schema_version":1,
            "cases":cloned,
            "summary":{"engineering_status_counts":{"PASS":20,"WARN":0,"FAIL":0}},
        }
        rows=rows_from_manifest(manifest,reviewed=20)
        result=m.manual_review_summary(manifest,rows,min_reviewed=20)
        self.assertEqual(result["quality_status"],"INSUFFICIENT_SAMPLE")
        self.assertEqual(result["review_counts"]["reviewed"],20)
        self.assertEqual(result["review_counts"]["reviewed_unique_questions"],1)

    def test_successful_minimum_is_only_recorded_baseline(self):
        rows=rows_from_manifest(self.manifest,reviewed=1)
        result=m.manual_review_summary(self.manifest,rows,min_reviewed=1)
        self.assertEqual(result["quality_status"],"BASELINE_RECORDED")
        self.assertTrue(any("不代表" in note for note in result["notes"]))

    def test_label_validation_rejects_unverified_values(self):
        row=rows_from_manifest(self.manifest,reviewed=1)
        row[0]["relation_safety"]="unknown_probability"
        write_review(self.reviews_path,row)
        with self.assertRaisesRegex(ValueError,"relation_safety"):
            m.parse_reviews(self.reviews_path,self.manifest)

    def test_official_claim_requires_specific_review_note(self):
        rows=rows_from_manifest(self.manifest,reviewed=1)
        rows[0]["source_authority"]="verified_official"
        rows[0]["review_note"]=""
        write_review(self.reviews_path,rows)
        with self.assertRaisesRegex(ValueError,"具体产品"):
            m.parse_reviews(self.reviews_path,self.manifest)

    def test_tampered_manifest_summary_rejected(self):
        manifest=copy.deepcopy(self.manifest)
        manifest["summary"]["engineering_status_counts"]["PASS"]=0
        tampered=self.base/"tampered_manifest.json"
        tampered.write_text(json.dumps(manifest,ensure_ascii=False),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"统计"):
            m.load_manifest(tampered)

    def test_reviewer_required(self):
        rows=rows_from_manifest(self.manifest,reviewed=1)
        rows[0]["reviewer"]=""
        write_review(self.reviews_path,rows)
        with self.assertRaisesRegex(ValueError,"审核人"):
            m.parse_reviews(self.reviews_path,self.manifest)

    def test_exclusion_requires_reason(self):
        rows=rows_from_manifest(self.manifest)
        rows[0]["review_status"]="excluded"
        write_review(self.reviews_path,rows)
        with self.assertRaisesRegex(ValueError,"review_note"):
            m.parse_reviews(self.reviews_path,self.manifest)

    def test_tampered_sha_rejected(self):
        rows=rows_from_manifest(self.manifest)
        rows[0]["source_sha256"]="0000000"
        write_review(self.reviews_path,rows)
        with self.assertRaisesRegex(ValueError,"SHA256"):
            m.parse_reviews(self.reviews_path,self.manifest)

    def test_modified_question_rejected(self):
        rows=rows_from_manifest(self.manifest)
        rows[0]["question"]="不同的诊疗问题"
        write_review(self.reviews_path,rows)
        with self.assertRaisesRegex(ValueError,"question"):
            m.parse_reviews(self.reviews_path,self.manifest)

    def test_run_score_returns_nonclinical_label(self):
        rows=rows_from_manifest(self.manifest,reviewed=1)
        write_review(self.reviews_path,rows)
        dest=self.base/"manual_result.json"
        report=m.run_score(self.campaign_path,self.reviews_path,dest,
                           min_reviewed=1)
        self.assertEqual(report["quality_status"],"BASELINE_RECORDED")
        self.assertTrue(dest.is_file())
        self.assertEqual(
            json.loads(dest.read_text(encoding="utf-8"))["review_counts"]["reviewed"],1
        )
        with self.assertRaises(FileExistsError):
            m.run_score(self.campaign_path,self.reviews_path,dest,min_reviewed=1)

    def test_duplicate_case_in_review_csv_rejected(self):
        rows=rows_from_manifest(self.manifest)
        write_review(self.reviews_path,rows+rows)
        with self.assertRaisesRegex(ValueError,"重复"):
            m.parse_reviews(self.reviews_path,self.manifest)

    def test_partial_review_nonempty_unreviewed_not_counted(self):
        sample=rows_from_manifest(self.manifest)
        sample[0]["reviewer"]="填了姓名但没有将状态改为 reviewed"
        write_review(self.reviews_path,sample)
        parsed,_=m.parse_reviews(self.reviews_path,self.manifest)
        result=m.manual_review_summary(self.manifest,parsed,min_reviewed=1)
        self.assertIsNone(result["human_review_rates"]["fully_supported_fraction"])

if __name__=="__main__":
    unittest.main()
