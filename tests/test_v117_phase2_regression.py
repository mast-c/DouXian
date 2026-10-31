# -*- coding: utf-8 -*-
"""V1.17 阶段2质量回归工具测试：不调用数据库和Qwen。"""
import json
from pathlib import Path
import tempfile
import unittest

from v117_phase2_regression import (
    audit_traces, canonical, create_plan, load_questions, write_new
)


class Phase2RegressionTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.questions = self.root / "questions.jsonl"
        items = [
            {"id": f"QA{i:02d}", "question": f"测试问题{i}？"}
            for i in range(1, 21)
        ]
        self.questions.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n",
            encoding="utf-8",
        )
        self.trace_dir = self.root / "traces"
        self.trace_dir.mkdir()

    def make_trace(self, filename, question, **extra):
        sample = {
            "schema_version": 1,
            "question": question, "answer": "这是占位测试回答，不属于真实基线。",
            "trace": {"knowledge_evidence": [], "evidence_coverage": {"decision": "refuse"}},
            **extra,
        }
        (self.trace_dir / filename).write_text(
            json.dumps(sample, ensure_ascii=False), encoding="utf-8"
        )

    def test_plan_has_20_unique_questions(self):
        self.assertEqual(len(load_questions(self.questions)), 20)
        result = create_plan(self.questions, self.root / "new_plan")
        self.assertEqual(result["unique_questions"], 20)

    def test_plan_cannot_overwrite_previous(self):
        directory = self.root / "run"
        create_plan(self.questions, directory)
        with self.assertRaises(FileExistsError):
            create_plan(self.questions, directory)

    def test_canonical_matches_campaign_whitespace_rules(self):
        self.assertEqual(canonical("儿童  扁桃体炎 ？"), "儿童 扁桃体炎 ？")

    def test_audit_matches_by_question_not_filename(self):
        self.make_trace("medical_trace_888 (1).json", "测试问题6？")
        result = audit_traces(self.questions, self.trace_dir)
        self.assertEqual(result["covered_unique_questions"], 1)
        self.assertEqual(result["cases"]["QA06"][0]["file"],
                         "medical_trace_888 (1).json")
        self.assertEqual(result["status"], "INCOMPLETE_OR_NEEDS_ATTENTION")

    def test_audit_excludes_synthetic_flagged_traces(self):
        self.make_trace("medical_trace_bad.json", "测试问题2？", is_synthetic=True)
        result = audit_traces(self.questions, self.trace_dir)
        self.assertEqual(result["covered_unique_questions"], 0)
        self.assertEqual(len(result["invalid_files"]), 1)

    def test_duplicate_runs_are_not_20_unique_questions(self):
        self.make_trace("medical_trace_001.json", "测试问题1？")
        self.make_trace("medical_trace_002.json", "测试问题1？",
                        answer="另一轮回答")
        result = audit_traces(self.questions, self.trace_dir)
        self.assertIn("QA01", result["repeated_question_files"])

    def test_unknown_question_detected(self):
        self.make_trace("medical_trace_999.json", "不在题集中的问题？")
        result = audit_traces(self.questions, self.trace_dir)
        self.assertEqual(len(result["unexpected_questions"]), 1)

    def test_complete_with_20_distinct_runs(self):
        for i in range(1, 21):
            self.make_trace(f"medical_trace_{i:03d}.json", f"测试问题{i}？")
        result = audit_traces(self.questions, self.trace_dir)
        self.assertEqual(result["status"], "READY_FOR_COLLECT")
        self.assertEqual(len(result["missing_questions"]), 0)

    def test_reports_never_overwrite(self):
        out = self.root / "report.json"
        write_new(out, "{}")
        with self.assertRaises(FileExistsError):
            write_new(out, "{}")


if __name__ == "__main__":
    unittest.main()
