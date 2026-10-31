# -*- coding: utf-8 -*-
"""阶段2：医学提示词 / 旧计算机规则隔离 / 四药列表结构化回归。"""
import ast
import json
from pathlib import Path
import re
import unittest

from medical_overview_answer import build_multi_drug_source_answer

BASE = Path(__file__).resolve().parents[1]
RAG_PATH = BASE / "rag.py"


class MedicalPromptAndRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = RAG_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.code)
        cls.prompt = next(
            n.value.value
            for n in cls.tree.body
            if isinstance(n, ast.Assign)
            for target in n.targets
            if isinstance(target, ast.Name)
            and target.id == "MEDICAL_KB_ANSWER_PROMPT"
            and isinstance(n.value, ast.Constant)
        )
        env = {"re": re}
        needed = []
        for node in cls.tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_MEDICAL_INTENT_HINT"
                for t in node.targets
            ):
                needed.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in (
                "_has_medical_intent", "_fast_plan_paired_methods"
            ):
                needed.append(node)
        exec(compile(ast.Module(body=needed, type_ignores=[]),
                     "<isolated-legacy-rules>", "exec"), env)
        cls.fast_plan = staticmethod(env["_fast_plan_paired_methods"])

    def test_unverified_source_cannot_claim_official_approval(self):
        self.assertIn("法定适应症", self.prompt)
        self.assertIn("主要获批适应症", self.prompt)
        self.assertIn("当前证据无法核实批准状态", self.prompt)

    def test_no_cross_formulation_numeric_hallucination(self):
        self.assertIn("剂型", self.prompt)
        self.assertIn("腹泻", self.prompt)
        self.assertIn("具体次数阈值", self.prompt)

    def test_single_intent_not_expanded_into_four_aspects(self):
        self.assertIn("不擅自扩展成无关的三四个方面", self.prompt)

    def test_old_computer_fast_plan_retained_for_computer_questions(self):
        queries = self.fast_plan(
            "贝叶斯网有哪些精确推断和近似推断方法？它们分别采用什么基本原理？"
        )
        self.assertEqual(len(queries), 4)

    def test_old_computer_fast_plan_cannot_take_over_medical_question(self):
        self.assertEqual(self.fast_plan(
            "药物治疗有哪些抗生素和中成药方法？它们分别采用什么原理？"
        ), [])

    def test_20_medical_baseline_questions_do_not_enter_old_technical_fast_path(self):
        manifest_path = BASE / "engineering_manifest_v117.json"
        if not manifest_path.exists():
            self.skipTest("未提供实际20题Manifest，跳过原始数据比对")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["cases"]), 20)
        self.assertTrue(all(
            self.fast_plan(c["question"]) == []
            for c in manifest["cases"]
        ))

    def test_old_technical_evidence_boost_is_gated_from_medical_queries(self):
        selector = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_select_diverse_documents"
        )
        self.assertTrue(any(
            "_has_medical_intent" in ast.unparse(node.test)
            for node in ast.walk(selector)
            if isinstance(node, ast.If)
        ))

    def test_list_only_overview_now_uses_deterministic_branch(self):
        candidates = [
            {
                "id": i + 1, "drug_name": drug,
                "evidence_type": "unverified_drug_disease_candidate",
                "disease_term": "扁桃体炎",
                "evidence_quote": "本轮资料中提及扁桃体炎，仅作为未验证候选。",
            }
            for i, drug in enumerate(("一清胶囊", "一清颗粒",
                                      "万通炎康片", "丙酸交沙霉素颗粒"))
        ]
        rendered, summary = build_multi_drug_source_answer(
            "请列出数据库中所有治疗扁桃体炎的药品并保证没有遗漏。",
            candidates, [x["drug_name"] for x in candidates], [],
        )
        self.assertEqual(len(summary["listed_drugs"]), 4)
        for drug in (x["drug_name"] for x in candidates):
            self.assertIn("### " + drug, rendered)
        self.assertIn("不是完整的治疗药物清单", rendered)
        self.assertIn("候选原文", rendered)
        self.assertNotIn("**不良反应**", rendered)

        # 检查RAG调用分支不再需要 requested_details 非空。
        conditions = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.If):
                continue
            is_overview = "overview_question" in ast.unparse(node.test)
            calls = any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "build_multi_drug_source_answer"
                for statement in node.body
                for child in ast.walk(statement)
            )
            if is_overview and calls:
                conditions.append(ast.unparse(node.test))
        self.assertTrue(conditions)
        self.assertTrue(all("requested_details" not in cond for cond in conditions))

    def test_source_view_still_does_not_claim_verified_treatment(self):
        self.assertIn("待人工审核", self.prompt)  # 来源等级仍是强制限制
        self.assertIn("本轮资料记载的用途", self.prompt)


if __name__ == "__main__":
    unittest.main()
