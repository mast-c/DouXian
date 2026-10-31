# V1.17 层次化儿童用药安全回复；全部离线，只读、无云端依赖。
import ast
import unittest
from pathlib import Path

from medical_safety_response import (
    is_selfmedication_aspect,
    pediatric_antibiotic_selfmedication_notice,
)
from medical_quality_acceptance import make_export_payload

QUESTION = "儿童扁桃体炎可以自己服用丙酸交沙霉素颗粒吗？"
EVIDENCE = [
    {
        "drug_name": "丙酸交沙霉素颗粒",
        "section_field": "contraindications",
        "content": "对本品、红霉素或其他大环内酯类抗生素过敏者禁用。",
    },
    {
        "drug_name": "丙酸交沙霉素颗粒",
        "section_field": "dosage",
        "content": "小儿按体重一日30mg/kg，分3～4次服用。",
    },
]


class PediatricAntibioticSafetyTests(unittest.TestCase):
    def test_025_produces_plain_uncited_safety_notice(self):
        note = pediatric_antibiotic_selfmedication_notice(QUESTION, EVIDENCE)
        self.assertIn("不建议", note)
        self.assertIn("自行给儿童使用抗菌药", note)
        self.assertIn("不是本轮药品说明书的引文", note)
        self.assertNotIn("30mg/kg", note)
        self.assertNotIn("[1]", note)

    def test_only_drug_information_question_does_not_trigger(self):
        self.assertEqual(
            pediatric_antibiotic_selfmedication_notice(
                "丙酸交沙霉素颗粒的适应症是什么？", EVIDENCE
            ),
            "",
        )

    def test_adult_self_medication_does_not_trigger_child_specific_notice(self):
        self.assertEqual(
            pediatric_antibiotic_selfmedication_notice(
                "成人可以自己服用丙酸交沙霉素颗粒吗？", EVIDENCE
            ),
            "",
        )

    def test_no_antibiotic_evidence_no_antibiotic_classification(self):
        self.assertEqual(
            pediatric_antibiotic_selfmedication_notice(
                QUESTION, [{"drug_name": "丙酸交沙霉素颗粒", "content": "一种药品。"}]
            ),
            "",
        )

    def test_aspect_detection_keeps_non_safety_uncertainties(self):
        self.assertTrue(is_selfmedication_aspect("儿童能否在未经医嘱下自行服用该药？"))
        self.assertFalse(is_selfmedication_aspect("该药的儿童剂量如何？"))
        self.assertFalse(is_selfmedication_aspect("该药的适应症是什么？"))

    def test_trace_export_records_non_evidence_safety_notice(self):
        payload = make_export_payload("问", "答", {
            "medical_safety_guidance": {
                "applied": True,
                "category": "pediatric_antibiotic_self_medication",
            }
        })
        self.assertTrue(payload["trace"]["medical_safety_guidance"]["applied"])

    def test_rag_has_pre_generation_notice_and_retains_guard(self):
        source = (Path(__file__).resolve().parents[1] / "rag.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("pediatric_antibiotic_selfmedication_notice(", source)
        self.assertIn('if safety_notice:', source)
        self.assertIn("yield notice_text", source)
        self.assertIn('if coverage["decision"] in ("refuse", "error"):', source)
        self.assertIn("if not is_selfmedication_aspect(text)", source)


if __name__ == "__main__":
    unittest.main()
