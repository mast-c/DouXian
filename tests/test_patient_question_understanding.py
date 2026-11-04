"""Offline regressions for patient wording, ambiguity and statement attribution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_patient_context import patient_context_summary
from medical_query_understanding import understand_question


class PatientQuestionUnderstandingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        exports = self.root / "kg_output_clean_v2"
        exports.mkdir()
        self.products = ["布地奈德鼻喷雾剂", "盐酸西替利嗪片", "氯雷他定片",
                         "甲试药片", "甲试药注射液", "复方甲试药片"]
        rows = [{"id": f"d{i}", "name": name, "aliases": []} for i, name in enumerate(self.products)]
        (exports / "drugs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        self.nodes = {row["id"]: {**row, "kind": "Drug"} for row in rows}
        for i, name in enumerate(["青光眼", "鼻炎", "过敏性鼻炎"]):
            self.nodes[f"c{i}"] = {"id": f"c{i}", "kind": "Disease", "name": name}

    def tearDown(self):
        self.temp.cleanup()

    def understand(self, question):
        return understand_question(question, root=self.root, nodes=self.nodes)

    def test_addition_keeps_current_proposed_and_disease_roles(self):
        q = "患者有严重的青光眼，同时正在服用西替利嗪，因为鼻炎鼻塞严重，能不能加上布地奈德喷雾剂？"
        result = self.understand(q)
        a, b = result["medicines"]
        self.assertEqual((a["surface"], a["role"]), ("西替利嗪", "current"))
        self.assertEqual((b["surface"], b["role"]), ("布地奈德喷雾剂", "proposed"))
        self.assertEqual(b["candidate_drug_names"], ["布地奈德鼻喷雾剂"])
        self.assertFalse(b["actual_patient_product_known"])
        self.assertFalse(b["clinical_equivalence_asserted"])
        self.assertTrue(result["intent"]["interaction_question"])
        self.assertFalse(result["intent"]["adverse_event_question"])
        self.assertIn("青光眼", [c["surface"] for c in result["patient_context"]])

    def test_typo_in_parenthetical_alternative_is_not_silently_corrected(self):
        result = self.understand("正在服用西替利嗪（或弗雷他定），能不能加上布地奈德喷雾剂？")
        unresolved = result["unresolved_medication_mentions"]
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["surface"], "弗雷他定")
        self.assertIn("氯雷他定", unresolved[0]["suggestions"])
        self.assertEqual(unresolved[0]["role"], "alternative")
        self.assertFalse(unresolved[0]["automatic_correction"])
        self.assertFalse(any("氯雷他定片" in m["candidate_drug_names"] for m in result["medicines"]))
        result = self.understand("西替利嗪（或者弗雷他定）能不能加布地奈德喷雾剂")
        self.assertEqual(result["unresolved_medication_mentions"][0]["surface"], "弗雷他定")

    def test_spray_symptoms_and_duration_preserve_uncertainty(self):
        q = "我喷了布地奈德鼻喷雾剂好几天了，鼻子还是堵，而且今天鼻子里好像有点疼，这是药不管用还是我有其他问题"
        result = self.understand(q)
        medicine, = result["medicines"]
        self.assertEqual(medicine["role"], "taken")
        self.assertTrue(medicine["actual_patient_product_known"])
        self.assertEqual(medicine["route_hint"], "spray")
        self.assertTrue(result["intent"]["adverse_event_question"])
        self.assertTrue(result["intent"]["efficacy_question"])
        self.assertFalse(result["intent"]["interaction_question"])
        self.assertFalse(result["intent"]["drug_causality_asserted"])
        self.assertEqual(result["timing"][0]["surface"], "好几天")
        self.assertEqual(result["timing"][0]["medicine_surface"], medicine["surface"])
        self.assertFalse(result["timing"][0]["exact_duration_known"])
        self.assertIsNone(result["timing"][1]["medicine_surface"])
        self.assertIn("鼻腔疼痛", [t for c in result["patient_context"] for t in c["retrieval_terms"]])

    def test_known_name_accepts_duration_wording(self):
        for duration in ["好几天", "三天", "3天", "一周", "两个星期", "半个月", "已经两周"]:
            with self.subTest(duration=duration):
                result = self.understand(f"我用了布地奈德鼻喷雾剂{duration}了，还是不管用")
                self.assertEqual(result["medicines"][0]["surface"], "布地奈德鼻喷雾剂")
                self.assertTrue(result["intent"]["efficacy_question"])

    def test_colloquial_spray_forms_are_only_search_candidates(self):
        for suffix in ("喷雾剂", "喷雾", "喷剂", "鼻喷剂", "鼻喷"):
            with self.subTest(suffix=suffix):
                medicine, = self.understand("准备加上布地奈德" + suffix + "，行不行")["medicines"]
                self.assertEqual(medicine["candidate_drug_names"], ["布地奈德鼻喷雾剂"])
                self.assertFalse(medicine["actual_patient_product_known"])
                self.assertEqual(medicine["role"], "proposed")

    def test_exact_duration_is_preserved_without_making_up_a_number(self):
        result = self.understand("我用了甲试药片3天了")
        self.assertEqual(result["timing"][0]["surface"], "3天")
        self.assertTrue(result["timing"][0]["exact_duration_known"])

    def test_medication_slots_cover_unknown_addition_switch_and_spray(self):
        for prefix in ["准备加上", "想换成", "我喷了"]:
            with self.subTest(prefix=prefix):
                result = self.understand(prefix + "未知试药喷雾剂，能用吗")
                self.assertTrue(result["unresolved_medication_mentions"])
                self.assertFalse(result["medicines"])

    def test_preexisting_symptom_is_not_assumed_to_be_drug_reaction(self):
        result = self.understand("我鼻塞严重，正在服用西替利嗪，能不能加上布地奈德喷雾剂")
        self.assertFalse(result["intent"]["adverse_event_question"])
        self.assertTrue(result["intent"]["interaction_question"])

    def test_negated_symptom_is_not_a_reported_adverse_event(self):
        result = self.understand("我吃了甲试药片，没有头晕，也没有恶心")
        self.assertFalse(result["intent"]["adverse_event_question"])
        symptoms = [c for c in result["patient_context"] if c["kind"] == "symptom_word"]
        self.assertTrue(symptoms)
        self.assertTrue(all(c["assertion"] == "negated" for c in symptoms))

    def test_negated_stopped_and_alternative_medicines_are_not_current(self):
        for q, role in [("我没有吃过甲试药片", "not_taken"),
                        ("我停用了甲试药片", "stopped"),
                        ("我停了甲试药片3天了", "stopped"),
                        ("我能吃甲试药片吗", "proposed"),
                        ("甲试药片能用吗", "proposed"),
                        ("西替利嗪（或氯雷他定）能用吗", "alternative")]:
            with self.subTest(q=q):
                result = self.understand(q)
                self.assertEqual(result["medicines"][-1]["role"], role)

    def test_shared_action_and_postposed_action_are_understood(self):
        result = self.understand("我正在吃西替利嗪和氯雷他定，会有问题吗")
        self.assertEqual([m["role"] for m in result["medicines"]], ["current", "current"])
        result = self.understand("甲试药片吃了三天了，胃疼")
        self.assertEqual(result["medicines"][0]["role"], "taken")
        self.assertTrue(result["intent"]["adverse_event_question"])
        self.assertIn("胃痛", [t for c in result["patient_context"] for t in c["retrieval_terms"]])

    def test_relative_dates_attach_only_to_explicit_action(self):
        result = self.understand("我昨天吃了甲试药片，今天胃不舒服")
        self.assertEqual(result["timing"][0]["medicine_surface"], "甲试药片")
        self.assertIsNone(result["timing"][1]["medicine_surface"])

    def test_negation_and_family_history_are_preserved(self):
        for q, assertion in [("我没有青光眼，能喷布地奈德鼻喷雾剂吗", "negated"),
                             ("我妈妈有青光眼，我用了甲试药片", "other_person"),
                             ("如果有青光眼，能用甲试药片吗", "questioned")]:
            with self.subTest(q=q):
                condition = next(c for c in self.understand(q)["patient_context"] if c["surface"] == "青光眼")
                self.assertEqual(condition["assertion"], assertion)
                self.assertFalse(condition["diagnosis_asserted"])

    def test_unknown_longer_names_do_not_match_known_substrings(self):
        for q in ["未知甲试药片怎么吃", "我吃了甲试药片钠", "准备加上复方布地奈德喷雾剂"]:
            with self.subTest(q=q):
                self.assertFalse(self.understand(q)["medicines"])

    def test_spray_scope_excludes_oral_and_injected_products(self):
        result = self.understand("我喷了甲试药，今天头晕")
        self.assertTrue(result["route_conflicts"])
        self.assertFalse(result["medicines"][0]["candidate_drug_names"])

    def test_repeated_name_keeps_each_mention_role(self):
        result = self.understand("我停用了甲试药片，现在想再用甲试药片，行不行")
        medicine, = result["medicines"]
        self.assertEqual([m["role"] for m in medicine["mentions"]], ["stopped", "proposed"])

    def test_asking_about_a_medicine_does_not_mean_proposing_to_take_it(self):
        for prefix in ("我想问", "我想了解", "我想知道"):
            with self.subTest(prefix=prefix):
                result = self.understand(prefix + "甲试药片是干什么的")
                self.assertEqual(result["medicines"][0]["role"], "mentioned")

    def test_offsets_and_generation_summary_reference_patient_statements(self):
        result = self.understand("我喷了布地奈德鼻喷雾剂好几天了，今天鼻子疼")
        for item in result["medicines"] + result["timing"]:
            self.assertEqual(result["normalized_question"][item["start"]:item["end"]], item["surface"])
        summary = patient_context_summary(result)
        self.assertEqual(summary["medicines"][0]["role"], "taken")
        self.assertIn("不是医学证据", summary["meaning"])


if __name__ == "__main__":
    unittest.main()
