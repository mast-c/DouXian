# -*- coding: utf-8 -*-
"""Offline tests: no model API, Neo4j, SQLite source index or secrets required.

Run: python -m unittest -v test_semantic_query_planner.py
"""
import unittest
import os
import json
from unittest.mock import patch

from medical_semantic_query_planner import build_semantic_task_plan


RELATIONS = {
    "ADVERSE_REACTION_MENTION", "CAUTION_MENTION",
    "CONTRAINDICATION_MENTION", "INTERACTION_MENTION",
    "CLASS_MEMBERSHIP_ASSERTION", "INDICATION_MENTION", "HAS_SYMPTOM",
}
DRUG = {"surface": "布地奈德鼻喷雾剂", "node_ids": ["D1"],
        "kinds": ["Drug"], "ambiguous": False}
URTICARIA = {"surface": "荨麻疹", "node_ids": ["C1"],
             "kinds": ["Disease"], "ambiguous": False}
OTHER = {"surface": "伪麻黄碱", "node_ids": ["I1"],
         "kinds": ["Ingredient"], "ambiguous": False}
FALLBACK = {"llm_query_understanding": {"status": "ok", "intents": []},
            "intent": {}}


class TaskPlanTests(unittest.TestCase):
    def parse(self, intents, groups, **extras):
        parsed = {"llm_query_understanding": {"status": "ok", "intents": intents},
                  "intent": {}, **extras}
        return build_semantic_task_plan(
            "测试问题", parsed, groups, available_relations=RELATIONS
        )

    def test_drug_adverse_event_and_disease_classification_split(self):
        planned = self.parse(["adverse_event", "classification"], [DRUG, URTICARIA])
        a,b=planned["tasks"]
        self.assertEqual(planned["mode"], "semantic_multi_task")
        self.assertEqual(a["kind"], "drug_adverse_event")
        self.assertEqual(a["anchor_surfaces"], ["布地奈德鼻喷雾剂", "荨麻疹"])
        self.assertEqual(a["graph_strategy"], "direct")
        self.assertEqual(a["max_hops"], 1)
        self.assertIn("CAUTION_MENTION", a["graph_relations"])
        self.assertEqual(b["kind"], "disease_classification")
        self.assertEqual(b["anchor_surfaces"], ["荨麻疹"])
        self.assertEqual(b["graph_strategy"], "text_only")
        self.assertNotIn("CLASS_MEMBERSHIP_ASSERTION", b["graph_relations"])
        self.assertTrue(any(x["kind"]=="disease_classification"
                            for x in planned["text_retrieval_requests"]))

    def test_drug_classification_is_separate_graph_relation(self):
        plan=self.parse(["classification"], [DRUG])
        task=plan["tasks"][0]
        self.assertEqual(task["kind"], "drug_classification")
        self.assertEqual(task["graph_relations"], ["CLASS_MEMBERSHIP_ASSERTION"])
        self.assertEqual(task["graph_strategy"], "direct")

    def test_classification_only_drops_generic_disease_symptom_intent(self):
        parsed = {"llm_query_understanding": {
                      "status": "ok", "intents": ["disease_symptoms", "classification"]},
                  "intent": {}}
        plan = build_semantic_task_plan(
            "荨麻疹属于什么疾病类型？", parsed, [URTICARIA],
            available_relations={"HAS_SYMPTOM", "DISEASE_CATEGORY_MENTION"},
        )
        self.assertEqual([task["kind"] for task in plan["tasks"]],
                         ["disease_classification"])
        both = build_semantic_task_plan(
            "荨麻疹属于什么疾病类型？有哪些症状？", parsed, [URTICARIA],
            available_relations={"HAS_SYMPTOM", "DISEASE_CATEGORY_MENTION"},
        )
        self.assertEqual([task["kind"] for task in both["tasks"]],
                         ["disease_classification", "disease_symptoms"])

    def test_drug_interaction_requires_both_catalog_anchors(self):
        pair=self.parse(["interaction"], [DRUG, OTHER])["tasks"][0]
        self.assertEqual(pair["anchor_surfaces"], ["布地奈德鼻喷雾剂", "伪麻黄碱"])
        self.assertEqual(pair["graph_strategy"], "connected")
        single=self.parse(["interaction"], [DRUG])["tasks"][0]
        self.assertEqual(single["graph_strategy"], "text_only")

    def test_safety_does_not_overwrite_targeted_adverse_event(self):
        plan=self.parse(["adverse_event","medication_safety","classification"],
                        [DRUG, URTICARIA])
        self.assertEqual([t["kind"] for t in plan["tasks"]],
                         ["drug_adverse_event","disease_classification"])

    def test_llm_referent_task_only_changes_query_not_graph_identification(self):
        plan=self.parse(["adverse_event","classification"],[DRUG,URTICARIA],
            semantic_sub_tasks=[
              {"intent":"classification","query":"荨麻疹的疾病分类证据是什么？",
               "focus_entities":["荨麻疹"]},
              {"intent":"adverse_event","query":"布地奈德鼻喷雾剂的不良反应荨麻疹原文",
               "focus_entities":["布地奈德鼻喷雾剂","荨麻疹"]}
            ])
        self.assertEqual(plan["tasks"][0]["query"],
                         "布地奈德鼻喷雾剂的不良反应荨麻疹原文")
        self.assertEqual(plan["tasks"][1]["query"],
                         "荨麻疹的疾病分类证据是什么？")

    def test_unlinked_disease_classification_not_pretended_verified(self):
        plan=self.parse(["classification"], [DRUG],
            patient_context=[{"surface":"荨麻疹","kind":"disease_mention",
                              "role":"mentioned"}],
            semantic_sub_tasks=[{"intent":"classification","query":"荨麻疹什么类型",
                                 "focus_entities":["荨麻疹"]}])
        # Without a linked Disease node, still request disease-classification
        # source material; never use the DrugClass edge.
        self.assertEqual(plan["tasks"][0]["kind"], "disease_classification")
        self.assertEqual(plan["tasks"][0]["graph_strategy"], "text_only")
        self.assertEqual(plan["tasks"][0]["query"], "荨麻疹什么类型")

    def test_coreference_chooses_urticaria_not_background_rhinitis(self):
        rhinitis={"surface":"过敏性鼻炎","node_ids":["C2"],
                  "kinds":["Disease"],"ambiguous":False}
        plan=self.parse(["adverse_event","classification"],[DRUG,rhinitis,URTICARIA],
            semantic_sub_tasks=[
              {"intent":"adverse_event",
               "query":"布地奈德鼻喷雾剂可能导致荨麻疹的原文？",
               "focus_entities":["布地奈德鼻喷雾剂","荨麻疹"]},
              {"intent":"classification",
               "query":"荨麻疹属于什么疾病类型？",
               "focus_entities":["荨麻疹"]}
            ])
        first, second=plan["tasks"]
        self.assertEqual(first["anchor_surfaces"], ["布地奈德鼻喷雾剂","荨麻疹"])
        self.assertEqual(second["anchor_surfaces"], ["荨麻疹"])

    def test_ambiguous_disease_referent_stays_unresolved(self):
        rhinitis={"surface":"过敏性鼻炎","node_ids":["C2"],
                  "kinds":["Disease"],"ambiguous":False}
        plan=self.parse(["classification"],[DRUG,rhinitis,URTICARIA])
        self.assertEqual(plan["tasks"][0]["kind"],"unresolved_classification")
        self.assertEqual(plan["tasks"][0]["graph_strategy"],"text_only")

    def test_old_llm_subqueries_disambiguate_target_disease(self):
        rhinitis={"surface":"过敏性鼻炎","node_ids":["C2"],
                  "kinds":["Disease"],"ambiguous":False}
        planned=self.parse(
            ["adverse_event","classification"],[DRUG,rhinitis,URTICARIA],
            semantic_sub_queries=[
                "布地奈德鼻喷雾剂与荨麻疹之间的说明书证据",
                "荨麻疹的疾病分类是什么"
            ],
        )
        self.assertEqual(planned["tasks"][0]["anchor_surfaces"],
                         ["布地奈德鼻喷雾剂","荨麻疹"])
        self.assertEqual(planned["tasks"][1]["anchor_surfaces"],["荨麻疹"])
        self.assertEqual(planned["text_retrieval_requests"][1]["executed"],False)
        self.assertIn("dataset_categories.sqlite",
                      planned["text_retrieval_requests"][1]["suggested_source"])

    def test_three_drugs_no_arbitrary_pair(self):
        third={"surface":"华法林钠片","node_ids":["D3"],
               "kinds":["Drug"],"ambiguous":False}
        task=self.parse(["interaction"],[DRUG,OTHER,third])["tasks"][0]
        self.assertEqual(task["graph_strategy"],"text_only")
        self.assertEqual(task["anchor_surfaces"],[])

    def test_second_llm_pass_recovers_missing_classification_intent(self):
        from medical_semantic_query_planner import refine_semantic_tasks_with_model
        question="布地奈德鼻喷雾剂会导致荨麻疹吗？它是什么类型的疾病"
        base={"llm_query_understanding":{"status":"ok","intents":["adverse_event"]},
              "intent":{"adverse_event_question":True},
              "patient_context":[{"surface":"荨麻疹","kind":"disease_mention"}],
              "semantic_sub_queries":[]}
        class MockModel:
            def __init__(self): self.calls=0
            def bind(self,**kwargs):return self
            def invoke(self,messages):
                self.calls+=1
                return {"content":json.dumps({
                    "intents":["adverse_event","classification"],
                    "sub_tasks":[
                        {"intent":"adverse_event","query":"布地奈德鼻喷雾剂与荨麻疹不良反应",
                         "focus_entities":["布地奈德鼻喷雾剂","荨麻疹"]},
                        {"intent":"classification","query":"荨麻疹属于什么类型的疾病",
                         "focus_entities":["荨麻疹"]}
                    ]
                },ensure_ascii=False)}
        model=MockModel()
        with patch.dict(os.environ,{"MEDICAL_SEMANTIC_TASK_SECOND_PASS":"auto"}):
            enhanced,meta=refine_semantic_tasks_with_model(
                question,base,[DRUG,URTICARIA],model)
        self.assertEqual(meta["status"],"ok")
        self.assertEqual(model.calls,1)
        self.assertTrue(enhanced["intent"]["classification_question"])
        plan=build_semantic_task_plan(question,enhanced,[DRUG,URTICARIA],
                                      available_relations=RELATIONS)
        self.assertEqual([t["kind"] for t in plan["tasks"]],
                         ["drug_adverse_event","disease_classification"])

    def test_second_llm_pass_failure_does_not_erase_first_pass(self):
        from medical_semantic_query_planner import refine_semantic_tasks_with_model
        class BadModel:
            def bind(self,**k):return self
            def invoke(self,m):return {"content":"No valid JSON"}
        base={"llm_query_understanding":{"intents":["adverse_event"]},
              "intent":{"adverse_event_question":True}}
        with patch.dict(os.environ,{"MEDICAL_SEMANTIC_TASK_SECOND_PASS":"1"}):
            result,meta=refine_semantic_tasks_with_model(
                "布地奈德与荨麻疹吗？是什么病",base,[DRUG,URTICARIA],BadModel())
        self.assertIs(result,base)
        self.assertEqual(meta["status"],"fallback_to_first_pass")
        self.assertEqual(meta["model_calls"],1)

    def test_unknown_phrasing_needs_no_regex_added(self):
        plan=self.parse(["adverse_event","classification"],[DRUG,URTICARIA],
            semantic_sub_tasks=[
              {"intent":"adverse_event","query":"吃完这个会起风团么，查原文",
               "focus_entities":["布地奈德鼻喷雾剂","荨麻疹"]}
            ])
        self.assertEqual(len(plan["tasks"]), 2)
        self.assertEqual(plan["tasks"][0]["graph_strategy"], "direct")


if __name__ == "__main__":
    unittest.main()
