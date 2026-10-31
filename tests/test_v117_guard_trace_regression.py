# V1.17 medical trace and guard regression tests; offline, stdlib only.
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from evidence_guard import validate_judge_output, assess_evidence
from self_rag_retry import retry_insufficient_evidence
from medical_quality_acceptance import make_export_payload


class GuardTraceRegression(unittest.TestCase):
    def setUp(self):
        self.short_content = (
            "【药品资料原文章节｜来源未完成医学权威核验】\\n"
            "药品：万通炎康片\\n栏目：不良反应\\n原文：尚不明确。\\n"
            "重要：这只是第三方来源保存的原文。"
        )
        self.short = {
            "id": 1, "content": self.short_content,
            "evidence_type": "unverified_drug_source_section",
            "drug_name": "万通炎康片",
            "section_field": "adverse_reactions",
        }

    def _judge(self, quote, item):
        return validate_judge_output({
            "aspects": [{
                "question": "原文如何记载不良反应？",
                "status": "supported", "reason": "逐字记载",
                "evidence": [{"id": 1, "quote": quote}],
            }]
        }, [item])

    def test_complete_short_medical_source_quote_is_allowed(self):
        report = self._judge("尚不明确。", self.short)
        self.assertEqual(report["decision"], "answer")
        self.assertEqual(report["evidence_ids"], [1])

    def test_short_incomplete_or_unattributed_quote_is_not_allowed(self):
        self.assertEqual(self._judge("尚不明确", self.short)["decision"], "refuse")
        other = dict(self.short, evidence_type="unverified_drug_disease_candidate")
        self.assertEqual(self._judge("尚不明确。", other)["decision"], "refuse")

    def test_007_guard_can_separate_supported_fact_from_unsupported_claim(self):
        def judge(_prompt, payload_text):
            payload = json.loads(payload_text)
            self.assertEqual(payload["knowledge_evidence"][0]["section_field"],
                             "adverse_reactions")
            return {
                "aspects": [
                    {"question": "资料记载的状态是什么？",
                     "status": "supported", "reason": "原文已有明确文字",
                     "evidence": [{"id": 1, "quote": "尚不明确。"}]},
                    {"question": "能否证明绝对没有不良反应？",
                     "status": "insufficient", "reason": "不能从资料状态推导临床结论",
                     "evidence": []},
                ]
            }
        outcome = assess_evidence(
            model=None, question="不良反应写着尚不明确，是不是没有不良反应？",
            excerpts=[self.short], ask_fn=judge
        )
        self.assertEqual(outcome["decision"], "partial")
        self.assertEqual(outcome["evidence_ids"], [1])

    def test_013_guard_can_preserve_risk_context_without_treatment_claim(self):
        quote = "2．高胆固醇血症并有动脉粥样硬化危险的病人。"
        evidence = [{
            "id": 2,
            "content": "药品：阿托伐他汀钙片\\n栏目：适应症\\n证据原文：" + quote,
            "evidence_type": "json_neo4j_text_verified",
            "drug_name": "阿托伐他汀钙片", "parent_verified": True,
        }]
        result = validate_judge_output({
            "aspects": [
                {"question": "原文中描述的风险人群关联是什么？",
                 "status": "supported", "evidence": [{"id": 2, "quote": quote}]},
                {"question": "这能否证明对该疾病的直接治疗作用？",
                 "status": "insufficient", "evidence": []}
            ]
        }, evidence)
        self.assertEqual(result["decision"], "partial")
        self.assertEqual(result["evidence_ids"], [2])

    def test_retry_new_evidence_keeps_parent_identity(self):
        question = "本轮有哪些可描述的事实？"
        initial = {
            "decision": "refuse",
            "aspects": [{"question": question, "status": "insufficient"}],
            "evidence_ids": [], "answer_questions": [],
            "refused_questions": [question]
        }
        doc = SimpleNamespace(
            page_content="药品：甲药\\n证据原文：明确记录。",
            metadata={
                "source": "drug-source", "section_path": "甲药 / 适应症",
                "evidence_type": "json_neo4j_text_verified",
                "drug_name": "甲药", "source_url": "https://example.com/a",
                "record_id": "recordA", "section_id": "sectionB",
                "section_field": "indication", "provenance_verified": True,
            }
        )
        def judge_fn(**kwargs):
            additional = kwargs["excerpts"][-1]
            self.assertEqual(additional["record_id"], "recordA")
            self.assertEqual(additional["section_id"], "sectionB")
            self.assertEqual(additional["drug_name"], "甲药")
            self.assertTrue(additional["parent_verified"])
            return {
                "decision": "answer",
                "aspects": [{"question": question, "status": "supported"}],
                "answer_questions": [question], "refused_questions": [],
                "evidence_ids": [2]
            }
        result = retry_insufficient_evidence(
            model=None, question=question,
            excerpts=[{"id": 1, "source": "old", "content": "无关内容"}],
            coverage=initial, original_queries=[],
            search_fn=lambda _: [doc], judge_fn=judge_fn,
            rewrite_fn=lambda *args: ["甲药的适应症与具体来源如何记载？"],
        )
        self.assertEqual(result["status"], "improved")
        self.assertEqual(result["excerpts"][1]["record_id"], "recordA")

    def test_quality_export_includes_retry_diagnostics(self):
        payload = make_export_payload("q", "a", {
            "self_rag_retry": {"status": "no_new_evidence", "new_chunks": 0}
        })
        self.assertEqual(payload["trace"]["self_rag_retry"]["status"],
                         "no_new_evidence")

    def test_rag_final_ui_evidence_keeps_critical_keys(self):
        source = Path(__file__).resolve().parents[1] / "rag.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        candidate_dicts = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "append" and isinstance(node.func.value, ast.Name)                and node.func.value.id == "ui_evidence" and node.args                and isinstance(node.args[0], ast.Dict):
                candidate_dicts.append(node.args[0])
        self.assertTrue(candidate_dicts)
        self.assertTrue(any({
            "record_id", "section_id", "section_field", "candidate_id"
        }.issubset({key.value for key in d.keys if isinstance(key, ast.Constant)})
                            for d in candidate_dicts))


if __name__ == "__main__":
    unittest.main()
