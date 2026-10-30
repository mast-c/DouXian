# -*- coding: utf-8 -*-
"""医疗问答 V1.8 纯离线回归：不调用 Chroma、Neo4j 或在线模型。"""
import ast
import importlib.util
import json
import os
import re
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "medical_answer_quality_target", BASE/"medical_answer_quality.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
gate_spec = importlib.util.spec_from_file_location(
    "medical_entity_gate_target", BASE/"medical_entity_gate.py"
)
entity_gate = importlib.util.module_from_spec(gate_spec)
gate_spec.loader.exec_module(entity_gate)

URL="https://example.org/drug"

class TestOutputQuality(unittest.TestCase):
    def test_same_source_multiple_evidence_ids_grouped(self):
        ref = {
            1: (f"[查看药品资料来源]({URL})","Neo4j / 疾病提及候选"),
            2: (f"[查看药品资料来源]({URL})","Neo4j / 适应症原文"),
        }
        result = module.render_compact_references("原文显示患者风险[1]，另见适应症[2]。",ref)
        self.assertEqual(result.count(URL),1)
        self.assertIn("[1]、[2]",result)
        self.assertIn("不同片段",result)

    def test_two_source_links_remain_distinct(self):
        refs={
            1: (f"[查看资料]({URL})", "适应症"),
            2: ("[查看资料](https://other.example/drug)","禁忌"),
        }
        result=module.render_compact_references("见[1]与[2]", refs)
        self.assertEqual(result.count("https://"),2)

    def test_reference_numbers_are_not_invented(self):
        result=module.render_compact_references("只能见[1]和[6]",{1:("来源A","章节1")})
        self.assertIn("引用核对提醒",result)
        self.assertIn("[6]", result)
        self.assertNotIn("[2]",result)

    def test_broken_svg_link_across_stream_chunks_removed(self):
        chunks = [
            "结论：相关证据[1]。\n### 小结[",
            "svg](http://localhost:8501/#1)\n",
            "风险条件，不等于治疗效果。[2]\n",
        ]
        filt=module.MedicalAnswerStreamFilter()
        actual="".join(filt.feed(x) for x in chunks)+filt.finish()
        self.assertNotIn("[svg](",actual)
        self.assertIn("相关证据[1]",actual)
        self.assertIn("风险条件",actual)
        self.assertIn("[2]",actual)

    def test_bare_bullets_are_suppressed(self):
        chunks=["-\n-\n", "**正文**\n", "  *  \n", "有效内容[1]"]
        filt=module.MedicalAnswerStreamFilter()
        actual="".join(filt.feed(x) for x in chunks)+filt.finish()
        self.assertEqual(actual, "**正文**\n有效内容[1]")

    def test_legitimate_markdown_is_retained(self):
        text="## 原文依据\n- 第一条：患者需满足条件[1]\n- 第二条：风险证据[2]\n"
        filt=module.MedicalAnswerStreamFilter()
        self.assertEqual(filt.feed(text)+filt.finish(),text)

    def test_long_unbroken_stream_line_progresses(self):
        filt=module.MedicalAnswerStreamFilter(long_line_threshold=400,reserve=200)
        output=filt.feed("A"*410)
        self.assertEqual(len(output),210)
        self.assertEqual(len(filt.finish()),200)

    def test_no_references_no_fake_footer(self):
        self.assertEqual(module.render_compact_references("您好", {1:("A","B")}), "")


class TestRagIntegration(unittest.TestCase):
    def test_updated_branch_and_existing_components(self):
        content=(BASE/"rag.py").read_text(encoding="utf-8")
        ast.parse(content)
        self.assertIn("if only_unverified_graph:",content)
        self.assertNotIn("if only_unverified_graph or (graph_candidate_only",content)
        self.assertNotIn('graph_notice = (',content)
        self.assertIn("MedicalAnswerStreamFilter()",content)
        self.assertIn("return render_compact_references(answer, reference_map)",content)
        self.assertIn("collapse_duplicate_parent_sections(merged_excerpts)",content)
        self.assertIn("prepare_medical_documents(",content)
        self.assertIn('names = ["search_drug_graph", "search_knowledge_base"]',content)
        self.assertIn('MEDICAL_KB_ANSWER_PROMPT',content)
        # 清掉原来计算机教程遗留的算法质量检查。
        self.assertNotIn("Gibbs 采样通常是 MCMC",content)

    def test_mixed_candidate_and_verified_original_goes_to_llm(self):
        raw=(BASE/"rag.py").read_text(encoding="utf-8")
        tree=ast.parse(raw)
        service=next(n for n in tree.body
                     if isinstance(n, ast.ClassDef) and n.name=="RagService")
        method=next(n for n in service.body
                    if isinstance(n,ast.FunctionDef) and n.name=="_stream_agent")
        def message(**kwargs):
            return SimpleNamespace(**kwargs)

        prompts=[]
        class FakeChat:
            def bind(self,**kwargs):
                return self
            def stream(self,messages):
                prompts.append(messages)
                answer=(
                    "**结论：**资料提到有动脉粥样硬化危险的患者[1]，"
                    "但不能据此判断直接治疗。[2]\n"
                    "-\n"
                    "原文与候选来自同一网页。[1][2]"
                    "[svg](http://localhost:8501/#1)\n"
                    "参考资料：模型自己生成的冗余页脚\n"
                )
                for i in range(0,len(answer),7):
                    yield SimpleNamespace(content=answer[i:i+7])
        graph={
            "id":1,
            "source":"药品候选记录",
            "source_url":URL,
            "section_path":"Neo4j / 疾病提及候选",
            "evidence_type":"unverified_drug_disease_candidate",
            "content":"疾病提及候选；待人工审核；原文提及动脉粥样硬化危险",
            "drug_name":"阿托伐他汀钙片",
            "disease_term":"动脉粥样硬化",
            "evidence_quote":"有动脉粥样硬化危险的病人。",
            "candidate_status":"pending_human_review",
        }
        chroma={
            "id":1,
            "source":"药品原文",
            "source_url":URL,
            "section_path":"Neo4j / 适应症",
            "section_id":"sec_1",
            "record_id":"record_1",
            "evidence_type":"json_neo4j_text_verified",
            "content":"药品：阿托伐他汀钙片\n栏目：适应症\n证据原文：高胆固醇血症并有动脉粥样硬化危险的病人。",
        }
        call_names=[]
        # 真实 V1.8 截图的失败模式：脂肪乳注射液虽然提到
        # 动脉粥样硬化且原文对齐成功，但不是用户询问的药品。
        unrelated={
            **chroma,
            "source":"药品资料：10%、20%、30%脂肪乳注射液",
            "source_url":"https://example.org/unrelated",
            "drug_name":"10%、20%、30%脂肪乳注射液",
            "content":"药品：10%、20%、30%脂肪乳注射液\\n"
                     "栏目：适应症\\n证据原文：脂肪乳资料同样出现动脉粥样硬化",
            "section_path":"10%、20%、30%脂肪乳注射液 / 适应症",
            "section_id":"sec_2",
        }
        def fake_tool(name,args,trace,progress):
            call_names.append(name)
            excerpts=([graph] if name=="search_drug_graph"
                      else [chroma,unrelated])
            return (json.dumps({
                "found": True,"status":"ok","excerpts":excerpts,
                "sources":[item["source"] for item in excerpts],
                "subqueries":[args["question"]]
            },ensure_ascii=False),True)
        context={
            "Iterator":__import__("typing").Iterator,
            "json":json, "re":re, "time":time,"os":os,
            "get_history":lambda sid: SimpleNamespace(messages=[]),
            "requires_web_search":lambda question: False,
            "build_candidate_answer":lambda items: "不应提前走纯候选回答",
            "collapse_duplicate_parent_sections":lambda xs: xs,
            "render_kb_references":module.render_compact_references,
            "MedicalAnswerStreamFilter":module.MedicalAnswerStreamFilter,
            "requested_refusal_questions":module.requested_refusal_questions,
            "filter_evidence_for_query": entity_gate.filter_evidence_for_query,
            "filter_documents_for_drugs":entity_gate.filter_documents_for_drugs,
            "inspect_fast_kb_answer":lambda answer,refs: {
                "status":"pass","valid_citation_ids":[1,2],
                "invalid_citation_ids":[],"warnings":[],
            },
            "SystemMessage":message,"HumanMessage":message,
            "MEDICAL_KB_ANSWER_PROMPT":"医疗引用规则",
            "SYSTEM_PROMPT":"医疗助手",
            "MEDICAL_FAST_QUALITY_PROMPT":"医疗快速模式",
            "_plain_content":lambda v: str(v or ""),
            "_reference_label":lambda item,source: f"[查看药品资料来源]({URL})",
        }
        exec(compile(ast.Module(body=[method],type_ignores=[]),
                     "<offline-rag-v18>", "exec"),context)
        fake_self=SimpleNamespace(
            _ui_history=lambda h:[],
            _call_tool=fake_tool,
            chat_model=FakeChat(),
            knowledge_service=None,
        )
        data={
            "input":"阿托伐他汀钙片用于治疗动脉粥样硬化吗？",
            "mode":"知识库","history":[], "trace":{"events":[],"sources":[]},
        }
        with patch.dict(os.environ,{"DOUXIAN_KB_MODE":"fast"}):
            answer="".join(context["_stream_agent"](fake_self,data,"sess"))
        self.assertEqual(call_names,["search_drug_graph","search_knowledge_base"])
        self.assertTrue(prompts,"双路证据应交给生成模型")
        self.assertNotIn("不应提前走纯候选回答",answer)
        self.assertNotIn("模型自己生成的冗余页脚",answer)
        self.assertNotIn("[svg](",answer)
        self.assertNotIn("\n-\n",answer)
        self.assertNotIn("图谱说明：",answer)
        self.assertIn("原文与候选来自同一网页",answer)
        self.assertEqual(answer.count(URL),1)
        self.assertIn("[1]、[2]",answer)
        self.assertIn("资料提到有动脉粥样硬化危险",answer)
        self.assertNotIn("脂肪乳",answer)
        self.assertEqual(len(data["trace"]["knowledge_evidence"]),2)
        self.assertEqual(
            data["trace"]["evidence_entity_gate"]["removed_other_drugs"],1
        )
        self.assertEqual(
            data["trace"]["retrieval_summary"]["search_knowledge_base"]["hits"],1
        )
        self.assertEqual(
            data["trace"]["retrieval_summary"]["search_knowledge_base"]["hits_before_entity_gate"],2
        )

if __name__=="__main__":
    unittest.main()
