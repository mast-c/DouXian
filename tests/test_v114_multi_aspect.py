# -*- coding: utf-8 -*-
"""V1.14 回归：疾病药品多维问题、原文引用完整展示、尾注去重。"""
from __future__ import annotations

import ast
import html
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def module(filename,name):
    spec=importlib.util.spec_from_file_location(name, ROOT/filename)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

graph=module("drug_graph_retriever.py","druggraph_v114")
quality=module("medical_answer_quality.py","quality_v114")
status=module("medical_retrieval_status.py","status_v114")

QUESTION="治疗扁桃体炎的药物有哪些？药物有没有不良反应和注意事项？"
DRUGS=["一清胶囊","一清颗粒","万通炎康片","丙酸交沙霉素颗粒"]


class TestRouting(unittest.TestCase):
    def test_union_fields_for_complex_question(self):
        fields=graph.fields_for_question(QUESTION)
        self.assertIn("indication",fields)
        self.assertIn("adverse_reactions",fields)
        self.assertIn("warnings",fields)

    def test_first_match_is_not_only_field(self):
        self.assertEqual(
            graph.fields_for_question("某药的不良反应和注意事项有哪些？"),
            ["adverse_reactions","warnings"]
        )
        self.assertEqual(
            graph.fields_for_question("某药的禁忌和药物相互作用"),
            ["warnings","contraindications","drug_interactions"]
        )

    def test_disease_drug_list_only_expected_questions(self):
        self.assertTrue(graph.is_disease_drug_list_question(QUESTION))
        self.assertFalse(graph.is_disease_drug_list_question(
            "阿托伐他汀钙片与动脉粥样硬化有什么关联？"))
        self.assertFalse(graph.is_disease_drug_list_question(
            "阿托伐他汀钙片与脂肪乳注射液的适应症有什么不同？"))

    def test_4_drugs_2_columns_exact_match_read_only(self):
        retriever=graph.DrugGraphRetriever(driver=object())
        calls=[]
        def run(cypher, **params):
            calls.append((cypher,params))
            if cypher != graph._SECTION_BY_DRUG_NAME_AND_FIELD:
                raise AssertionError("unexpected Cypher")
            return [{
                "drug_name":params["drug_name"],
                "record_id":f"r-{params['drug_name']}",
                "section_id":f"s-{params['drug_name']}-{params['field']}",
                "source_url":"https://example.org/medical",
                "authority_status":"third_party_unverified",
                "section_field":params["field"],
                "section_text":params["drug_name"]+" "+params["field"]+" 的原文",
            }]
        retriever._run=run
        rows=retriever.search_fields_for_drugs(
            DRUGS,["adverse_reactions","warnings"],
            max_drugs=4,max_sections=8,
        )
        self.assertEqual(len(rows),8)
        self.assertEqual({r["drug_name"] for r in rows},set(DRUGS))
        self.assertEqual(
            {field for field in ("adverse_reactions","warnings")},
            set(r["section_field"] for r in rows)
        )
        self.assertTrue(all(
            r["evidence_type"]=="unverified_drug_source_section"
            and "第三方" in r["content"]
            for r in rows
        ))
        self.assertEqual(len(calls),8)
        self.assertTrue(all(
            "MATCH" in cypher and "CREATE" not in cypher
            and params["limit"]==1
            for cypher,params in calls
        ))

    def test_no_missing_field_is_invented(self):
        retriever=graph.DrugGraphRetriever(driver=object())
        retriever._run=lambda cypher,**params: []
        self.assertEqual(
            retriever.search_fields_for_drugs(DRUGS,["adverse_reactions","warnings"]),
            []
        )

    def test_graph_tool_actual_multi_drug_question(self):
        tree=ast.parse((ROOT/"rag.py").read_text(encoding="utf-8"))
        service=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="RagService")
        init=next(n for n in service.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
        fn=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name=="search_drug_graph")
        fn.decorator_list=[]
        class FakeGraph:
            def __init__(self):
                self.calls=[]
            def search(self,q,limit,explicit_disease_only=False):
                self.calls.append(("search",limit,explicit_disease_only))
                out=[]
                for i,name in enumerate(DRUGS):
                    out.append({
                        "drug_name":name,"record_id":"r"+str(i),
                        "section_id":"s"+str(i),
                        "candidate_id":"c"+str(i),
                        "evidence_quote":"原文提及扁桃体炎",
                        "disease_term":"扁桃体炎",
                        "source_url":"https://example.org/"+str(i),
                        "candidate_status":"pending_human_review",
                        "clinical_status":"not_validated",
                        "source_status":"third_party_unverified",
                        "source":"https://example.org/"+str(i),
                        "section_path":"Neo4j / 疾病候选",
                        "content":"仅为疾病候选，不是治疗适应症",
                        "evidence_type":"unverified_drug_disease_candidate",
                    })
                # 其中一个药有第二条候选，但不会挤掉其他3个药。
                out.insert(1,{**out[0],"candidate_id":"c_extra","section_id":"s_extra"})
                return out
            def search_sections(self,q,limit):
                self.calls.append(("sections",limit))
                return []
            def search_fields_for_drugs(self,names,fields,**kwargs):
                self.calls.append(("supplement",names,fields))
                return [{
                    "source":"药品资料 · "+name,
                    "drug_name":name,
                    "section_field":field,
                    "record_id":"record"+name,
                    "section_id":"section"+name+field,
                    "source_url":"https://example.org/"+name,
                    "evidence_type":"unverified_drug_source_section",
                    "content":f"药品{name}的{field}原文章节，未经医学核验",
                    "section_path":name+"/"+field,
                } for field in fields for name in names]
        fake=FakeGraph()
        namespace={
            "self":SimpleNamespace(graph_retriever=fake),
            "json":json, "re":re,
            "compact_same_quote_candidates":graph.compact_same_quote_candidates,
            "fields_for_question":graph.fields_for_question,
            "is_disease_drug_list_question":graph.is_disease_drug_list_question,
        }
        exec(compile(ast.Module(body=[fn],type_ignores=[]),
                     "<multi_tool>","exec"),namespace)
        data=json.loads(namespace["search_drug_graph"](QUESTION))
        self.assertTrue(data["disease_drug_overview"])
        self.assertEqual(data["candidate_count"],4)
        self.assertEqual(data["section_count"],8)
        self.assertEqual(data["supplemented_sections"],8)
        self.assertEqual(data["selected_drugs"],DRUGS)
        self.assertEqual(len(data["excerpts"]),12)
        self.assertEqual(
            [name for action,names,*_ in fake.calls if action=="supplement" for name in names],
            DRUGS
        )

    def test_model_footer_with_parenthesis_is_removed_across_chunks(self):
        target="正文说明药品存在风险[8]。\n\n参考资料（同一来源的不同片段合并显示）：\n[8] http://bad.example\n"
        filt=quality.MedicalAnswerStreamFilter()
        res="".join(
            filt.feed(target[i:i+5]) for i in range(0,len(target),5)
        )+filt.finish()
        self.assertIn("正文说明药品存在风险[8]",res)
        self.assertNotIn("参考资料（",res)
        self.assertNotIn("bad.example",res)
        refs=quality.render_compact_references(res,{
            8:("[查看原始来源](https://example.org/8)","注意事项")
        })
        combined=res+refs
        self.assertEqual(combined.count("参考资料（"),1)
        self.assertIn("[8]",combined)

    def test_model_footer_same_line_also_removed(self):
        filt=quality.MedicalAnswerStreamFilter()
        payload="正文有依据[4]。\n参考资料（同一来源的不同片段合并显示）：[4] 伪造网址"
        output="".join(filt.feed(payload[i:i+7]) for i in range(0,len(payload),7))+filt.finish()
        self.assertNotIn("伪造网址",output)
        self.assertEqual(output.strip(),"正文有依据[4]。")

    def test_standalone_reference_word_in_prose_not_cut(self):
        filt=quality.MedicalAnswerStreamFilter()
        sentence="参考资料是否可靠仍需要核实。[1]\n"
        self.assertEqual(filt.feed(sentence)+filt.finish(),sentence)

    def test_all_8_evidence_cards_rendered(self):
        source=(ROOT/"豆馅.py").read_text(encoding="utf-8")
        tree=ast.parse(source)
        fn=next(n for n in tree.body
                if isinstance(n,ast.FunctionDef) and n.name=="trace_details_html")
        ns={
            "trace_heading":lambda trace:"检索完成",
            "retrieval_status_html":lambda trace:"",
            "_web_sources":lambda trace:[],
            "_web_queries":lambda trace:[],
            "graph_evidence_html":lambda item,idx:f"<div>[{idx}] 候选</div>",
            "html":html,
            "urlsplit":urlsplit,
            "describe_tool_event":lambda event:"已完成",
            "elapsed_label":lambda value:"1s",
        }
        exec(compile(ast.Module(body=[fn],type_ignores=[]),
                     "<ui-render>","exec"),ns)
        trace={
            "events":[{"type":"end","tool":"search_drug_graph","ok":True}],
            "multi_drug_scope":{
                "listed_drugs":DRUGS,
                "supplement_fields":["adverse_reactions","warnings"],
                "max_listed_drugs":4,
            },
            "knowledge_evidence":[
                {
                    "id":i,"source":"来源"+str(i),
                    "section_path":"栏目"+str(i),
                    "content":"资料"+str(i),
                    "preview":"中文摘要",
                    "source_url":"https://example.org",
                    "evidence_type":"json_neo4j_text_verified",
                    "parent_verified":True,
                } for i in range(1,9)
            ],
        }
        markup=ns["trace_details_html"](trace)
        self.assertIn("实际命中的知识库证据（8 条）",markup)
        self.assertIn("[8] 来源8",markup)
        self.assertIn("[1] 来源1",markup)
        self.assertIn("本轮仅列出前 4 种匹配药品资料",markup)
        self.assertIn("疾病候选提及不等于已批准",markup)

    def test_backend_no_limited_five_evidence_trace(self):
        code=(ROOT/"rag.py").read_text(encoding="utf-8")
        self.assertNotIn("for item in merged_excerpts[:5]",code)
        self.assertIn("evidence_cap = 16 if overview_question else 8",code)
        self.assertIn("max_sections=8",code)
        self.assertIn("按本轮实际找到的药品逐项回答",code)


if __name__=="__main__":
    unittest.main()
