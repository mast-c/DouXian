#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Second-stage evidence guard quality regression, offline and API-free.

Run: python test_evidence_guard_v2.py
"""
import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new = load_module("guard_v2", HERE/"evidence_guard.py")
old = load_module("guard_v1", HERE/"original_backup"/"evidence_guard.py")

EVIDENCE = [
    {"id":1,"source":"机器学习.md","section_path":"14.4.1",
     "content":"精确推断的实质是一类动态规划算法, "
               "它利用图模型所描述的条件独立性来削减计算目标概率值所需的计算量。"
               "变量消去法是最直观的精确推断算法。"},
    {"id":2,"source":"机器学习.md","section_path":"14.5",
     "content":"近似推断方法大致可分为两大类: 第一类是采样(sampling), "
               "通过使用随机化方法完成近似; 第二类是使用确定性近似完成近似推断, "
               "典型代表为变分推断(variational inference)."},
    {"id":3,"source":"神经网络与深度学习.md","section_path":"11.3.2",
     "content":"近似推断主要有以下三种方法：环路信念传播、变分推断和采样法。"
               "变分推断引入一个简单的变分分布来近似复杂的条件概率。"},
    {"id":4,"source":"机器学习.md","section_path":"14.5.1",
     "content":"概率图模型中最常用的采样技术是马尔可夫链蒙特卡罗(MCMC)方法。"},
]
QUESTION="贝叶斯网有哪些精确推断和近似推断方法？它们分别采用什么基本思路？"


def aspect(text,status="supported",*citations):
    return {"question":text,"status":status,
            "evidence":[{"id":id_,"quote":quote} for id_,quote in citations],
            "reason":"示例审核结论"}


def reply(*aspects):
    return {"aspects":list(aspects)}


class GuardV2Tests(unittest.TestCase):
    def test_original_and_v2_agree_for_exact_quotes(self):
        data=reply(
            aspect("精确推断", "supported",
                   (1,"精确推断的实质是一类动态规划算法")),
            aspect("近似推断", "supported",
                   (2,"第一类是采样(sampling)"),
                   (3,"变分推断引入一个简单的变分分布")),
        )
        a=old.validate_judge_output(data,EVIDENCE)
        b=new.validate_judge_output(data,EVIDENCE)
        for key in ("decision","evidence_ids","answer_questions","refused_questions"):
            self.assertEqual(a[key],b[key])
        self.assertEqual(b["decision"],"answer")

    def test_punctuation_is_accepted_only_when_typographic(self):
        quote="精确推断的实质是一类动态规划算法， 它利用图模型所描述的条件独立性"
        original=old.validate_judge_output(
            reply(aspect("精确推断","supported",(1,quote))),EVIDENCE
        )
        revised=new.validate_judge_output(
            reply(aspect("精确推断","supported",(1,quote))),EVIDENCE
        )
        self.assertEqual(original["decision"],"refuse")
        self.assertEqual(revised["decision"],"answer")
        self.assertEqual(revised["aspects"][0]["quotes"][0]["quote"],
                         "精确推断的实质是一类动态规划算法, "
                         "它利用图模型所描述的条件独立性")
        self.assertIn(
            revised["aspects"][0]["quotes"][0]["quote"],
            EVIDENCE[0]["content"]
        )

    def test_changed_digit_fails(self):
        docs=[{"id":1,"content":"训练误差是 10%，测试误差是 15%。"}]
        proposed=reply(aspect("准确率","supported",
                              (1,"训练误差是 20%")))
        res=new.validate_judge_output(proposed,docs)
        self.assertEqual(res["decision"],"refuse")
        self.assertEqual(res["evidence_ids"],[])
        self.assertIn("quote_not_contiguous_in_source",
            [d["reason"] for d in res["validation_diagnostics"][0]["invalid_citations"]])

    def test_changed_math_operator_fails(self):
        docs=[{"id":1,"content":"当 x >= 10 时模型收敛。"}]
        res=new.validate_judge_output(
            reply(aspect("收敛","supported",(1,"当 x <= 10 时模型收敛"))),docs
        )
        self.assertEqual(res["decision"],"refuse")

    def test_no_fuzzy_semantic_matching(self):
        res=new.validate_judge_output(
            reply(aspect("精确推断","supported",
                         (1,"变量消除算法可以精确推断概率"))),EVIDENCE
        )
        self.assertEqual(res["decision"],"refuse")

    def test_one_invalid_citation_blocks_entire_aspect(self):
        res=new.validate_judge_output(
            reply(aspect("一个需要多证据的结论","supported",
                         (1,"变量消去法是最直观的精确推断算法"),
                         (2,"不存在的对比关系描述"))),EVIDENCE
        )
        self.assertEqual(res["decision"],"refuse")
        self.assertEqual(res["evidence_ids"],[])
        self.assertEqual(res["validation_diagnostics"][0]["verified_citations"],1)
        self.assertEqual(res["validation_diagnostics"][0]["final_status"],
                         "insufficient")

    def test_missing_citation_fails_closed(self):
        result=new.validate_judge_output(
            reply(aspect("方法","supported")),EVIDENCE
        )
        self.assertEqual(result["decision"],"refuse")
        self.assertEqual(result["evidence_ids"],[])

    def test_unsupported_id_fails_closed(self):
        result=new.validate_judge_output(
            reply(aspect("方法","supported",(99,"变量消去法是最直观的精确推断算法"))),EVIDENCE
        )
        self.assertEqual(result["decision"],"refuse")

    def test_insufficient_not_automatically_approved_by_quote(self):
        result=new.validate_judge_output(
            reply(aspect("方法","insufficient",
                         (1,"变量消去法是最直观的精确推断算法"))),EVIDENCE
        )
        self.assertEqual(result["decision"],"refuse")
        self.assertEqual(result["evidence_ids"],[])

    def test_multi_aspect_partial_matches_original(self):
        data=reply(
            aspect("精确方法","supported",(1,"变量消去法是最直观的精确推断算法")),
            aspect("具体证明","insufficient")
        )
        res=new.validate_judge_output(data,EVIDENCE)
        self.assertEqual(res["decision"],"partial")
        self.assertEqual(res["evidence_ids"],[1])

    def test_malformed_json_fails_closed(self):
        result=new.validate_judge_output("irrelevant text",EVIDENCE)
        self.assertEqual(result["decision"],"error")

    def test_raw_quotes_not_exposed_in_diagnostic(self):
        quote="完全不存在于书中的敏感文本片段123"
        result=new.validate_judge_output(
            reply(aspect("内容","supported",(1,quote))),EVIDENCE
        )
        self.assertNotIn(quote,json.dumps(result["validation_diagnostics"],
                                         ensure_ascii=False))

    def test_assess_with_fake_llm_function(self):
        captured={}
        def judge(prompt,payload):
            captured["prompt"]=prompt
            captured["payload"]=json.loads(payload)
            return json.dumps(reply(
                aspect("精确推断","supported",
                       (1,"变量消去法是最直观的精确推断算法"))
            ),ensure_ascii=False)
        answer=new.assess_evidence(
            model=None,question=QUESTION,excerpts=EVIDENCE,
            retrieval_queries=["精确推断","近似推断"],ask_fn=judge
        )
        self.assertEqual(answer["decision"],"answer")
        self.assertEqual(len(captured["payload"]["knowledge_evidence"]),4)
        self.assertIn("原样复制",captured["prompt"])

    def test_no_evidence_fails_closed(self):
        result=new.assess_evidence(None,QUESTION,[],ask_fn=lambda *x: "{}")
        self.assertEqual(result["decision"],"refuse")

    def test_v1_prompt_switch(self):
        with patch.dict(os.environ,{"RAG_GUARD_PROMPT_VERSION":"v1"}):
            self.assertEqual(new._judge_prompt(),new.GUARD_SYSTEM_PROMPT_V1)

    def test_v2_prompt_default(self):
        with patch.dict(os.environ,{"RAG_GUARD_PROMPT_VERSION":""}):
            self.assertEqual(new._judge_prompt(),new.GUARD_SYSTEM_PROMPT_V2)

    def test_no_extra_speed_options_by_default(self):
        with patch.dict(os.environ,{"RAG_GUARD_MODEL":"",
                                    "RAG_GUARD_MAX_TOKENS":"",
                                    "RAG_GUARD_DISABLE_THINKING":"0"}):
            self.assertEqual(new._judge_model_kwargs(),{"temperature":0})

    def test_provider_thinking_switch_opt_in(self):
        with patch.dict(os.environ,{"RAG_GUARD_DISABLE_THINKING":"1",
                                    "RAG_GUARD_MODEL":"qwen-test",
                                    "RAG_GUARD_MAX_TOKENS":"1300"}):
            self.assertEqual(new._judge_model_kwargs(),
                             {"temperature":0,"model":"qwen-test",
                              "max_tokens":1300,
                              "extra_body":{"enable_thinking":False}})

    def test_malformed_max_tokens_throws_and_assess_fails_closed(self):
        with patch.dict(os.environ,{"RAG_GUARD_MAX_TOKENS":"not-a-number"}):
            with self.assertRaises(ValueError):
                new._judge_model_kwargs()

    def test_real_baseline_excerpts_can_verify_citations(self):
        data=json.loads((Path("/mnt/data")/"rag_baseline_results.json")
                        .read_text(encoding="utf-8"))
        sample=data["samples"][3]
        docs=[]
        for item in sample["trace"]["knowledge_evidence"]:
            docs.append({"id":item["id"],"content":item["content"]})
        requested=reply(
            aspect("精确方法","supported",
                   (2,"变量消去法是最直观的精确推断算法")),
            aspect("近似方法","supported",
                   (1,"第一类是采样(sampling)"),
                   (4,"变分推断（Variational Inference）是引入一个变分分布"))
        )
        result=new.validate_judge_output(requested,docs)
        self.assertEqual(result["decision"],"answer")
        self.assertEqual(result["evidence_ids"],[1,2,4])

    def test_quote_whitespace_and_fullwidth_punctuation(self):
        docs=[{"id":1,"content":"目标方法（Inference）：样本均值可用于估计期望。"}]
        result=new.validate_judge_output(
            reply(aspect("估计期望","supported",
                         (1,"目标方法(Inference):样本均值可用于估计期望。"))),docs
        )
        self.assertEqual(result["decision"],"answer")


if __name__=="__main__":
    unittest.main(verbosity=2)
