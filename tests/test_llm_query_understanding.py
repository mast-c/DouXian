"""Tests for bounded, source-grounded LLM query understanding."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_llm_query_understanding import enhance_understanding


class FakeModel:
    def __init__(self, payload):
        self.payload = payload
        self.bind_args = None
        self.messages = None

    def bind(self, **kwargs):
        self.bind_args = kwargs
        return self

    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": json.dumps(self.payload, ensure_ascii=False)})()


class LLMQueryUnderstandingTests(unittest.TestCase):
    def base(self):
        return {"medicines": [], "patient_context": [], "intent": {
            "interaction_question": False, "risk_question": False,
            "drug_discovery_question": False, "symptom_question": False,
            "adverse_event_question": False, "efficacy_question": False,
            "mechanism_question": False, "classification_question": False,
            "fields": [],
        }}

    def test_llm_adds_implicit_symptom_and_population_and_safe_retrieval_intents(self):
        question = "我有点烧，怀宝宝了要怎么处理"
        model = FakeModel({"intents": ["symptom_treatment", "medication_safety"],
                           "entities": [
                               {"type": "symptom", "surface": "有点烧", "canonical": "发热"},
                               {"type": "population", "surface": "怀宝宝", "canonical": "妊娠"},
                           ]})
        with patch.dict("os.environ", {"MEDICAL_LLM_QUERY_UNDERSTANDING": "1"}):
            merged, report = enhance_understanding(
                question, self.base(), model, allowed_entity_names={"发热", "妊娠"},
            )
        self.assertEqual(report["status"], "ok")
        self.assertTrue(merged["intent"]["drug_discovery_question"])
        self.assertTrue(merged["intent"]["risk_question"])
        self.assertIn("contraindications", merged["intent"]["fields"])
        self.assertEqual({c["surface"] for c in merged["patient_context"]}, {"有点烧", "怀宝宝"})
        self.assertTrue(all(not c["diagnosis_asserted"] for c in merged["patient_context"]))
        self.assertEqual(model.bind_args, {"temperature": 0, "max_tokens": 600})

    def test_unquoted_or_unknown_entity_cannot_be_injected(self):
        question = "我不舒服怎么办"
        model = FakeModel({"intents": ["drug_discovery"], "entities": [
            {"type": "symptom", "surface": "发烧", "canonical": "发热"},
            {"type": "symptom", "surface": "不舒服", "canonical": "未知疾病"},
            {"type": "drug", "surface": "对乙酰氨基酚", "canonical": "对乙酰氨基酚"},
        ]})
        merged, report = enhance_understanding(
            question, self.base(), model, allowed_entity_names={"发热"},
        )
        self.assertTrue(merged["intent"]["drug_discovery_question"])
        self.assertEqual(merged["patient_context"], [])
        self.assertEqual(report["accepted_entities"], [])

    def test_assertion_and_other_person_context_are_not_upgraded(self):
        question = "如果我发烧，我妈妈怀孕时该怎么办"
        model = FakeModel({"intents": ["symptom_treatment"], "entities": [
            {"type": "symptom", "surface": "发烧", "canonical": "发热"},
            {"type": "population", "surface": "怀孕", "canonical": "妊娠"},
        ]})
        merged, _ = enhance_understanding(
            question, self.base(), model, allowed_entity_names={"发热", "妊娠"},
        )
        assertions = {item["surface"]: item["assertion"] for item in merged["patient_context"]}
        self.assertEqual(assertions, {"发烧": "questioned", "怀孕": "other_person"})

    def test_simple_named_medicine_question_skips_api(self):
        understanding = self.base()
        understanding["medicines"] = [{"surface": "甲试药片"}]
        understanding["intent"]["risk_question"] = True
        merged, report = enhance_understanding("甲试药片注意什么", understanding, FakeModel({}))
        self.assertIs(merged, understanding)
        self.assertEqual(report["model_calls"], 0)

    def test_symptom_only_discovery_question_uses_api_even_after_rule_match(self):
        understanding = self.base()
        understanding["patient_context"] = [{"surface": "发烧", "kind": "symptom_word"}]
        understanding["intent"]["drug_discovery_question"] = True
        model = FakeModel({"intents": [], "entities": []})
        _, report = enhance_understanding("发烧了应该吃什么药", understanding, model)
        self.assertEqual(report["model_calls"], 1)

    def test_api_failure_preserves_rule_parse(self):
        class Broken:
            def bind(self, **kwargs):
                return self
            def invoke(self, messages):
                raise TimeoutError()
        understanding = self.base()
        merged, report = enhance_understanding("有点烧怎么办", understanding, Broken())
        self.assertIs(merged, understanding)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error_type"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
