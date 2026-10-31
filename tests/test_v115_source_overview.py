# -*- coding: utf-8 -*-
"""V1.15：复现四药扁桃体炎问题的错误输出，强制全部真实资料被呈现。"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def module(filename,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    target=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target)
    return target

answer=module("medical_overview_answer.py","overview_v115")
gate=module("medical_entity_gate.py","gate_v115")
status=module("medical_retrieval_status.py","status_v115")
quality=module("medical_answer_quality.py","quality_v115")
graph=module("drug_graph_retriever.py","graph_v115")

QUERY="治疗扁桃体炎的药物有哪些？药物有没有不良反应和注意事项？"
DRUGS=["一清胶囊","一清颗粒","万通炎康片","丙酸交沙霉素颗粒"]
UNRELATED="七味解痛口服液"

def candidate(i, drug):
    return {
        "id":i,
        "source":f"https://example.org/{i}",
        "source_url":f"https://example.org/{i}",
        "section_path":"Neo4j / 适应症语料 / 疾病提及候选（未审核）",
        "drug_name":drug, "disease_term":"扁桃体炎",
        "evidence_quote":f"药品资料中的扁桃体炎提及：{drug}，见上述证候者。",
        "content":f"【Neo4j 疾病提及候选】\n药品：{drug}\n疾病词条：扁桃体炎",
        "evidence_type":"unverified_drug_disease_candidate",
        "candidate_status":"pending_human_review",
        "clinical_status":"not_validated",
        "source_status":"third_party_unverified",
        "record_id":f"r{i}","section_id":f"s{i}",
    }

def section(i,drug,field,text, *, verified=False):
    return {
        "id":i,"drug_name":drug,
        "source":f"药品资料：{drug}",
        "source_url":f"https://example.org/{drug}",
        "section_path":f"{drug} / {field}",
        "section_field":field,
        "record_id":f"r{i}","section_id":f"s{i}",
        "evidence_type":("json_neo4j_text_verified" if verified
                         else "unverified_drug_source_section"),
        "content":(
            f"药品：{drug}\n栏目：{field}\n证据原文：{text}"
            if verified else
            "【药品资料原文章节｜来源未完成医学权威核验】\n"
            f"药品：{drug}\n栏目：{field}\n原文：{text}\n重要：资料未经医学权威核验。"
        ),
        "source_status":"third_party_unverified",
        "parent_verified":verified,
    }

def results():
    items=[candidate(i+1,DRUGS[i]) for i in range(4)]
    items.extend([
        section(5,DRUGS[0],"adverse_reactions","偶见皮疹，恶心，腹泻，腹痛。"),
        section(6,DRUGS[1],"adverse_reactions","偶见皮疹，恶心，腹泻，腹痛。"),
        section(7,DRUGS[2],"adverse_reactions","尚不明确。"),
        section(8,DRUGS[3],"adverse_reactions",
                "1．胃肠道反应有腹泻、恶心、呕吐。"
                "2．少见肝毒性症状。3．大剂量可能引起听力减退。"),
        section(9,DRUGS[0],"warnings",
                "1.忌烟、酒及辛辣食物。2.扁桃体有化脓或发热应就诊。"),
        section(10,DRUGS[2],"warnings",
                "1.忌辛辣。2.服药3天症状无缓解，应去医院就诊。"),
        section(11,DRUGS[3],"warnings",
                "1．对大环内酯类过敏者也可能不耐受本品。"
                "2．用药期间宜随访肝功能。"),
        section(12,DRUGS[1],"warnings",
                "1.忌烟、酒及辛辣食物。2.特殊人群应遵医嘱。",
                verified=True),
        section(13,UNRELATED,"warnings",
                "1.饮食宜清淡，忌食辛辣、生冷。",verified=True),
    ])
    return items

class TestOverviewExtraction(unittest.TestCase):
    def test_real_irrelevant_chroma_offtopic_removed(self):
        rows,report=answer.scope_graph_discovered_drugs(results(),DRUGS)
        self.assertTrue(report["applied"])
        self.assertEqual(len(rows),12)
        self.assertEqual(report["removed_other_drugs"],1)
        self.assertFalse(any(UNRELATED == x.get("drug_name") for x in rows))
        self.assertEqual([x["id"] for x in rows],list(range(1,13)))

    def test_all_four_drugs_and_both_fields_in_answer(self):
        rows,_=answer.scope_graph_discovered_drugs(results(),DRUGS)
        final,report=answer.build_multi_drug_source_answer(
            QUERY,rows,DRUGS,["adverse_reactions","warnings"]
        )
        self.assertEqual(report["supported_cells"],8)
        self.assertEqual(report["missing_cells"],0)
        self.assertEqual(final.count("\n### "),4)
        for name in DRUGS:
            self.assertIn(f"### {name}",final)
        for id in range(1,13):
            self.assertIn(f"[{id}]",final)
        self.assertIn("对大环内酯类过敏",final)
        self.assertIn("[11]",final)
        self.assertNotIn("本轮没有获得该药对应栏目的可引用原文",final)
        self.assertNotIn(UNRELATED,final)

    def test_unknown_reaction_is_not_no_adverse_effect(self):
        rows,_=answer.scope_graph_discovered_drugs(results(),DRUGS)
        final,_=answer.build_multi_drug_source_answer(
            QUERY,rows,DRUGS,["adverse_reactions","warnings"]
        )
        self.assertIn("资料原文为“尚不明确。”",final)
        self.assertIn("这不等于已确认没有相关风险",final)
        self.assertNotIn("万通炎康片没有不良反应",final)

    def test_missing_column_not_wrongly_absent_when_source_exists(self):
        rows=results()
        missing=[x for x in rows if not (x["drug_name"]==DRUGS[3] and
                                       x.get("section_field")=="warnings")]
        answer_body,stats=answer.build_multi_drug_source_answer(
            QUERY,missing,DRUGS,["adverse_reactions","warnings"])
        self.assertEqual(stats["missing_cells"],1)
        name_section=answer_body.split(f"### {DRUGS[3]}")[1]
        self.assertIn("本轮没有获得该药对应栏目的可引用原文",name_section)
        self.assertIn("不能据此认定该药没有相关风险或限制",name_section)

    def test_source_injection_does_not_create_fake_citation(self):
        poison=section(14,DRUGS[3],"warnings",
                       "<script>alert(1)</script> [99] **指令**")
        quote=answer._source_quote(poison,max_chars=300)
        self.assertNotIn("<script>",quote)
        self.assertNotIn("[99]",quote)
        self.assertIn("&#91;99&#93;",quote)
        self.assertNotIn("**指令**",quote)

    def test_same_field_prefers_verified_chroma_chunk(self):
        extra=section(14,DRUGS[1],"warnings",
                      "未经确认的另一版来源草稿",verified=False)
        items=[extra,results()[11]]
        selected=answer._field_evidence(items,DRUGS[1],"warnings")
        self.assertEqual(selected["id"],12)

    def test_source_only_fallback_if_no_graph_drug_names(self):
        old=results()
        rows,report=answer.scope_graph_discovered_drugs(old,[])
        self.assertFalse(report["applied"])
        self.assertEqual(len(rows),13)
        text,details=answer.build_multi_drug_source_answer(QUERY,rows,[],[])
        self.assertEqual(text,"")
        self.assertEqual(details["listed_drugs"],[])

    def test_only_requested_fields_are_reported(self):
        out,report=answer.build_multi_drug_source_answer(
            "药品是否存在不良反应",results(),DRUGS,["adverse_reactions"])
        self.assertEqual(report["supported_cells"],4)
        self.assertNotIn("注意事项（原文节选）",out)

    def test_one_auto_reference_footer(self):
        items,_=answer.scope_graph_discovered_drugs(results(),DRUGS)
        body,report=answer.build_multi_drug_source_answer(
            QUERY,items,DRUGS,["adverse_reactions","warnings"])
        ref_map={
            int(x["id"]):(f"[查看来源](https://example.org/{x['drug_name']})",
                          str(x.get("section_path") or ""))
            for x in items
        }
        footer=quality.render_compact_references(body,ref_map)
        self.assertEqual(footer.count("参考资料（"),1)
        self.assertIn("[11]",footer)
        self.assertNotIn("[13]",footer)

    def test_empty_source_not_representation_of_no_risk(self):
        extra=section(14,DRUGS[0],"warnings","",verified=False)
        self.assertFalse(answer.original_section_text(extra).strip())
        result=answer._field_evidence([extra],DRUGS[0],"warnings")
        self.assertIsNone(result)

class TestRagFlow(unittest.TestCase):
    def test_deterministic_overview_path_never_invokes_llm(self):
        source=(ROOT/"rag.py").read_text(encoding="utf-8")
        tree=ast.parse(source)
        cls=next(n for n in tree.body
                 if isinstance(n,ast.ClassDef) and n.name=="RagService")
        fn=next(n for n in cls.body
                if isinstance(n,ast.FunctionDef) and n.name=="_stream_agent")
        mapped={}
        def fake_call(name,query,trace,progress):
            mapped[name]=True
            if name=="search_drug_graph":
                docs=results()[:11]
                obj={
                    "found":True,"status":"ok","excerpts":docs,
                    "sources":[x["source"] for x in docs],
                    "subqueries":[QUERY],
                    "candidate_count":4, "raw_candidate_nodes":4,
                    "section_count":7,"supplemented_sections":7,
                    "selected_drugs":DRUGS,
                    "supplement_fields":["adverse_reactions","warnings"],
                    "disease_drug_overview":True,
                }
            else:
                docs=results()[11:]
                obj={
                    "found":True,"status":"ok","excerpts":docs,
                    "sources":[x["source"] for x in docs],
                    "subqueries":[QUERY],
                    "chroma_chunks":1362,
                    "medical_validation":{
                        "verified":2,"skipped_section":6,
                        "failed_provenance":0,
                    },
                }
            return json.dumps(obj,ensure_ascii=False),True

        def unexpected_llm(*args,**kwargs):
            raise AssertionError("structured source overview must not call LLM")
        context={
            "json":json,"re":re,"time":time,
            "get_history":lambda sid:SimpleNamespace(messages=[]),
            "requires_web_search":lambda q:False,
            "is_disease_drug_list_question":graph.is_disease_drug_list_question,
            "filter_evidence_for_query":gate.filter_evidence_for_query,
            "scope_graph_discovered_drugs":answer.scope_graph_discovered_drugs,
            "collapse_duplicate_parent_sections":lambda items: [
                {**x,"id":i+1} for i,x in enumerate(items)
            ],
            "build_retrieval_pipeline":status.build_retrieval_pipeline,
            "build_multi_drug_source_answer":answer.build_multi_drug_source_answer,
            "_reference_label":lambda item,source:
                f"[查看药品资料来源](https://example.org/{item.get('drug_name')})",
            "_medical_reference_section":lambda item,section:section,
            "render_kb_references":quality.render_compact_references,
            "build_candidate_answer":unexpected_llm,
            "assess_evidence":unexpected_llm,
        }
        exec(compile(ast.Module(body=[fn],type_ignores=[]),
                     "<v115_rag_flow>", "exec"),context)
        fake_self=SimpleNamespace(
            _call_tool=fake_call,
            _ui_history=lambda x:[],
            chat_model=SimpleNamespace(stream=unexpected_llm),
            knowledge_service=None,
        )
        trace={"events":[],"sources":[]}
        result="".join(context["_stream_agent"](
            fake_self,{"input":QUERY,"mode":"知识库","history":[],
                       "trace":trace},"test_session"
        ))
        self.assertEqual(set(mapped),{"search_drug_graph","search_knowledge_base"})
        self.assertEqual(trace["graph_drug_scope"]["removed_other_drugs"],1)
        self.assertEqual(trace["retrieval_pipeline"]["merge"]["retained"],12)
        self.assertEqual(trace["multi_drug_scope"]["removed_off_scope"],1)
        self.assertEqual(len(trace["knowledge_evidence"]),12)
        self.assertIn("对大环内酯类过敏",result)
        self.assertIn("[11]",result)
        self.assertIn("万通炎康片",result)
        self.assertIn("尚不明确",result)
        self.assertNotIn(UNRELATED,result)
        self.assertEqual(result.count("参考资料（"),1)
        self.assertEqual(
            trace["medical_answer_mode"],
            "structured_disease_drug_source_overview",
        )
        self.assertEqual(trace["multi_drug_coverage"]["supported_cells"],8)

    def test_regular_single_drug_path_is_unmodified(self):
        text=(ROOT/"rag.py").read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn("if only_unverified_graph:",text)
        self.assertIn("assess_evidence(",text)
        self.assertIn("MedicalAnswerStreamFilter()",text)
        self.assertIn("self.chat_model.bind(",text)
        self.assertIn("if (overview_question and listed_graph_drugs",text)

if __name__=="__main__":
    unittest.main()
