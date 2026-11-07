# -*- coding: utf-8 -*-
"""Offline smoke tests for the semantic query planner (NO API or graph required).

Run in the actual medical project:
    python -m unittest -v test_llm_query_understanding.py
"""
import json
import unittest
import os
from unittest.mock import patch
from types import SimpleNamespace
from medical_llm_query_understanding import (
    enhance_understanding, should_enhance,
)


class MockModel:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def bind(self, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=json.dumps(self.answer, ensure_ascii=False))


CATALOG = {
    "Drug": {"布地奈德鼻喷雾剂", "某药片"},
    "Ingredient": {"伪麻黄碱", "甲硝唑"},
    "Symptom": {"发热"},
    "Population": {"妊娠"},
    "Disease": {"过敏性鼻炎"},
    "_aliases": {},
}


class QueryPlannerRegressionTests(unittest.TestCase):
    def test_unlisted_compound_wording_does_not_get_blocked(self):
        question = "我在用布地奈德鼻喷雾剂，另一种综合药里头有伪麻黄碱，会不会冲突？"
        model = MockModel({
            "intents": ["interaction", "medication_safety"],
            "entities": [
                {"type": "drug", "surface": "布地奈德鼻喷雾剂", "role": "current"},
                {"type": "ingredient", "surface": "伪麻黄碱", "role": "proposed"},
            ],
            "unknown_products": [{
                "surface": "综合药", "kind": "compound", "contains": ["伪麻黄碱"],
            }],
            "relations": [{
                "type": "contains_ingredient", "subject": "综合药", "object": "伪麻黄碱",
            }],
            "sub_queries": ["布地奈德鼻喷雾剂与伪麻黄碱相互作用资料"],
        })
        self.assertTrue(should_enhance(question, {
            "intent": {"interaction_question": True}, "medicines": [1, 2],
        }))
        result, report = enhance_understanding(question, {}, model, allowed_entity_names=CATALOG)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(model.calls, 1)
        self.assertTrue(result["intent"]["interaction_question"])
        self.assertEqual({v["surface"] for v in result["llm_medicine_mentions"]},
                         {"布地奈德鼻喷雾剂", "伪麻黄碱"})
        self.assertEqual(result["vague_medicine_details"][0]["named_ingredients"], ["伪麻黄碱"])
        self.assertTrue(result["needs_product_details"])

    def test_unknown_formulation_is_not_replaced_with_a_different_drug(self):
        question = "甲硝唑与未知加强版药片是否能一起用？"
        model = MockModel({
            "intents": ["interaction"],
            "entities": [
                {"type": "ingredient", "surface": "甲硝唑", "role": "current"},
                {"type": "drug", "surface": "未知加强版药片", "role": "proposed"},
            ], "unknown_products": [], "relations": [], "sub_queries": [],
        })
        result, report = enhance_understanding(question, {}, model, allowed_entity_names=CATALOG)
        self.assertEqual(report["status"], "ok")
        self.assertEqual([v["surface"] for v in result["llm_medicine_mentions"]], ["甲硝唑"])
        self.assertEqual(result["llm_unlinked_medication_mentions"][0]["surface"], "未知加强版药片")

    def test_negated_ingredient_not_treated_as_active(self):
        question = "这个复方药不含伪麻黄碱"
        model = MockModel({
            "intents": ["medication_safety"],
            "entities": [{"type": "ingredient", "surface": "伪麻黄碱", "role": "negated"}],
            "unknown_products": [{"surface": "复方药", "kind": "compound", "contains": []}],
            "relations": [], "sub_queries": [],
        })
        result, _ = enhance_understanding(question, {}, model, allowed_entity_names=CATALOG)
        self.assertTrue(result["llm_negated_medication_mentions"])
        self.assertEqual(result["vague_medicine_details"][0]["named_ingredients"], [])

    def test_new_clinical_expression_maps_to_soft_candidate(self):
        question = "怀上宝宝后烧得厉害，想了解一般风险"
        model = MockModel({
            "intents": ["symptom_treatment", "medication_safety"],
            "entities": [
                {"type": "population", "surface": "怀上宝宝", "canonical": "妊娠"},
                {"type": "symptom", "surface": "烧得厉害", "canonical": "发热"},
            ],
            "unknown_products": [], "relations": [], "sub_queries": [],
        })
        result, _ = enhance_understanding(question, {}, model, allowed_entity_names=CATALOG)
        contexts = {v["surface"]: v["retrieval_terms"] for v in result["patient_context"]}
        self.assertIn("妊娠", contexts["怀上宝宝"])
        self.assertIn("发热", contexts["烧得厉害"])
        self.assertTrue(all(not x["diagnosis_asserted"] for x in result["patient_context"]))

    def test_http_adapter_thinking_and_timeout_are_forwarded(self):
        from medical_llm_query_understanding import _OpenAICompatibleModel

        seen = {}
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): return False
            def read(self, size): return b'{"choices":[{"message":{"content":"{}"}}]}'

        def fake_openurl(request, timeout):
            seen["payload"] = json.loads(request.data.decode("utf-8"))
            seen["timeout"] = timeout
            return FakeResponse()

        model = _OpenAICompatibleModel(
            "https://example.org/compatible-mode/v1",
            "local-unit-test-key",
            "model-name",
            timeout=60,
        )
        with patch.dict(os.environ, {"MEDICAL_QUERY_LLM_ENABLE_THINKING": "0"}), \
             patch("medical_llm_query_understanding.urllib.request.urlopen", fake_openurl):
            result = model.bind(max_tokens=100).invoke([
                {"role": "user", "content": "hi"}
            ])
        self.assertEqual(result["content"], "{}")
        self.assertEqual(seen["timeout"], 60)
        self.assertIs(seen["payload"]["enable_thinking"], False)
        self.assertEqual(seen["payload"]["max_tokens"], 100)

    def test_non_json_response_retries_once_and_recovers(self):
        class FirstBadThenGood:
            def __init__(self):
                self.calls = 0
            def bind(self, **kwargs):
                return self
            def invoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return {"content": "",
                            "__llm_meta__": {"finish_reason": "length",
                                             "content_length": 0,
                                             "reasoning_present": True}}
                return {"content": json.dumps({
                    "intents": ["medication_safety"],
                    "entities": [{"type": "drug", "surface": "布地奈德鼻喷雾剂"}],
                    "unknown_products": [],
                    "relations": [],
                    "sub_queries": [],
                }, ensure_ascii=False)}

        model = FirstBadThenGood()
        with patch.dict(os.environ, {"MEDICAL_QUERY_LLM_JSON_RETRY": "1"}):
            understood, report = enhance_understanding(
                "布地奈德鼻喷雾剂有什么注意事项？", {}, model,
                allowed_entity_names=CATALOG,
            )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["model_calls"], 2)
        self.assertEqual(report["parse_retries"], 1)
        self.assertEqual(model.calls, 2)
        self.assertEqual(understood["llm_medicine_mentions"][0]["surface"],
                         "布地奈德鼻喷雾剂")

    def test_empty_output_error_is_diagnostic_but_does_not_log_prompt(self):
        class EmptyModel:
            def bind(self, **kwargs):
                return self
            def invoke(self, messages):
                return {"content": "", "__llm_meta__": {
                    "finish_reason": "length", "completion_tokens": 1600,
                    "reasoning_present": True, "content_length": 0
                }}
        with patch.dict(os.environ, {"MEDICAL_QUERY_LLM_JSON_RETRY": "0"}):
            _, report = enhance_understanding(
                "这是一段不应出现在异常信息中的私密描述", {}, EmptyModel(),
                allowed_entity_names=CATALOG,
            )
        self.assertEqual(report["error_category"], "invalid_model_json")
        self.assertEqual(report["model_calls"], 1)
        self.assertIn("finish_reason", report["error"])
        self.assertNotIn("私密描述", report["error"])

    def test_env_dedicated_model_overrides_injected_app_model(self):
        from medical_llm_query_understanding import select_medical_query_model

        env_model, injected = object(), object()
        with patch.dict(os.environ, {"MEDICAL_QUERY_LLM_PREFER_ENV": "1"}), \
             patch("medical_llm_query_understanding._load_project_dotenv"), \
             patch("medical_llm_query_understanding.load_query_understanding_model_from_env",
                   return_value=env_model):
            chosen, source = select_medical_query_model(injected)
        self.assertIs(chosen, env_model)
        self.assertEqual(source, "env_http_adapter")

        with patch.dict(os.environ, {"MEDICAL_QUERY_LLM_PREFER_ENV": "0"}), \
             patch("medical_llm_query_understanding._load_project_dotenv"), \
             patch("medical_llm_query_understanding.load_query_understanding_model_from_env",
                   return_value=env_model) as loader:
            chosen, source = select_medical_query_model(injected)
            loader.assert_not_called()
        self.assertIs(chosen, injected)
        self.assertEqual(source, "injected_app_model")

    def test_http_json_mode_and_thinking_flags_reach_request_payload(self):
        from medical_llm_query_understanding import _OpenAICompatibleModel
        seen = {}

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): return False
            def read(self, size):
                return b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}],"usage":{}}'

        def fake_openurl(request, timeout):
            seen["payload"] = json.loads(request.data.decode("utf-8"))
            seen["timeout"] = timeout
            return FakeResponse()

        model = _OpenAICompatibleModel("https://test.maas.aliyuncs.com/compatible-mode/v1",
                                       "unit-test-key", "qwen-test", timeout=60)
        env = {
            "MEDICAL_QUERY_LLM_JSON_MODE": "1",
            "MEDICAL_QUERY_LLM_ENABLE_THINKING": "0",
            "MEDICAL_QUERY_LLM_SEND_TEMPERATURE": "0",
        }
        with patch.dict(os.environ, env), \
             patch("medical_llm_query_understanding.urllib.request.urlopen", fake_openurl):
            result = model.bind(max_tokens=2500).invoke([{"role":"user","content":"输出 JSON"}])
        self.assertEqual(seen["payload"]["response_format"], {"type":"json_object"})
        self.assertIs(seen["payload"]["enable_thinking"], False)
        self.assertNotIn("temperature", seen["payload"])
        self.assertEqual(seen["payload"]["max_tokens"], 2500)
        self.assertEqual(result["__llm_meta__"]["json_mode_sent"], True)
        self.assertIs(result["__llm_meta__"]["enable_thinking_sent"], False)
        self.assertEqual(seen["timeout"], 60)

    def test_non_json_result_exposes_finish_reason_without_content(self):
        class EmptyModel:
            def bind(self, **kwargs): return self
            def invoke(self, messages):
                return {
                    "content": "",
                    "__llm_meta__": {
                        "finish_reason": "length",
                        "content_length": 0,
                        "reasoning_present": True,
                        "json_mode_sent": True,
                        "enable_thinking_sent": False
                    }
                }
        with patch.dict(os.environ, {"MEDICAL_QUERY_LLM_JSON_RETRY": "0"}):
            _, report = enhance_understanding("不能出现在日志里的问题描述", {},
                                              EmptyModel(), allowed_entity_names=CATALOG)
        self.assertEqual(report["error_category"], "invalid_model_json")
        self.assertEqual(report["response_metadata"]["finish_reason"], "length")
        self.assertTrue(report["response_metadata"]["json_mode_sent"])
        self.assertNotIn("不能出现在日志里", str(report))

    def test_ambiguous_brand_cannot_be_selected_by_model_guess(self):
        question = "甲药是啥"
        model = MockModel({
            "intents": ["source_lookup"],
            "entities": [{
                "type": "drug", "surface": "甲药", "canonical": "某药片",
            }], "unknown_products": [], "relations": [], "sub_queries": [],
        })
        catalog = dict(CATALOG)
        catalog["Drug"] = {"某药片", "乙药片"}
        catalog["_aliases"] = {"甲药": [
            {"canonical": "某药片", "kind": "Drug", "basis": "source_catalog_alias"},
            {"canonical": "乙药片", "kind": "Drug", "basis": "source_catalog_alias"},
        ]}
        result, report = enhance_understanding(question, {}, model, allowed_entity_names=catalog)
        self.assertEqual(report["status"], "ok")
        self.assertFalse(result["llm_medicine_mentions"])
        self.assertEqual(result["llm_unlinked_medication_mentions"][0]["surface"], "甲药")


if __name__ == "__main__":
    unittest.main()
