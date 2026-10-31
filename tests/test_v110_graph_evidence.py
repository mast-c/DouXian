# -*- coding: utf-8 -*-
"""V1.10 回归：同句多疾病候选、双药适应症比较、医学来源引用。"""
import ast
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = "阿托伐他汀钙片"
B = "10%、20%、30%脂肪乳注射液"
D = "动脉粥样硬化"
SAME_QUOTE = "卵磷脂可辅助治疗动脉粥样硬化，脂肪肝，以及小儿湿疹，神经衰弱症。"

def import_file(name,filename):
    spec=importlib.util.spec_from_file_location(name, ROOT/filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

graph=import_file("medical_graph_v110_test","drug_graph_retriever.py")
quality=import_file("medical_quality_v110_test","medical_answer_quality.py")


def record(*, drug=B, term=D, candidate_id="c1",
           record_id="r1", section_id="s1", quote=SAME_QUOTE):
    return {
        "drug_name":drug, "term_name":term, "candidate_id":candidate_id,
        "evidence_quote":quote, "candidate_status":"pending_human_review",
        "clinical_status":"not_validated", "source_status":"third_party_unverified",
        "context_flag":"manual_semantic_review",
        "source_url":"https://example.org/drug",
        "record_id":record_id,"section_id":section_id,
    }


def section(drug,id):
    return {
        "drug_name":drug,
        "record_id":"r"+id,"section_id":"s"+id,
        "source_url":"https://example.org/"+id,
        "authority_status":"third_party_unverified",
        "section_field":"indication",
        "section_text": f"{drug}说明书适应症原文",
    }


class FakeGraph:
    """真实 DrugGraphRetriever 的参数化只读查询替身，覆盖每个查询分支。"""
    def __init__(self):
        self.instance=graph.DrugGraphRetriever(driver=object())
        self.calls=[]
        self.instance._run=self.run

    def run(self,cypher,**params):
        self.calls.append((cypher,params))
        question=params.get("question","")
        if cypher==graph._DRUG_IDENTIFY:
            return [
                {"id":drug,"name":drug}
                for drug in (B,A)
                if drug in question
            ]
        if cypher==graph._TERM_IDENTIFY:
            # 只对问题中出现的疾病（且已剔除完整药名）作显式匹配。
            return ([{"id":"d1","name":D}] if D in question else [])
        if cypher==graph._SECTION_BY_DRUG:
            drug=params["id"]
            return [
                section(drug,drug[-3:]+"1"),
                section(drug,drug[-3:]+"2"),
                section(drug,drug[-3:]+"3"),
            ][:params["limit"]]
        if cypher==graph._DRUG_CANDIDATES:
            return [
                record(candidate_id="c1",term=D),
                record(candidate_id="c2",term="脂肪肝"),
                record(candidate_id="c3",term="小儿湿疹"),
            ][:params["limit"]]
        if cypher==graph._PAIR_CANDIDATES:
            return [record(drug=params["id"],term=D,candidate_id="pair1")]
        if cypher==graph._TERM_CANDIDATES:
            return []
        raise AssertionError("unknown Cypher")

    def execute_tool(self,question):
        """抽取 rag.py 真实闭包工具逻辑，不实例化外部 LLM。"""
        tree=ast.parse((ROOT/"rag.py").read_text(encoding="utf-8"))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="RagService")
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
        tool=next(n for n in init.body
                  if isinstance(n,ast.FunctionDef) and n.name=="search_drug_graph")
        tool.decorator_list=[]
        namespace={
            "json":json,"re":__import__("re"),
            "self":type("DummyService",(),{"graph_retriever":self.instance})(),
            "compact_same_quote_candidates":graph.compact_same_quote_candidates,
            "is_disease_drug_list_question":graph.is_disease_drug_list_question,
            "fields_for_question":graph.fields_for_question,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[tool],type_ignores=[])),
                     "<graph-tool>", "exec"),namespace)
        return json.loads(namespace["search_drug_graph"](question))


class TestDiseaseCandidates(unittest.TestCase):
    def test_same_quote_three_disease_terms_shown_once(self):
        items=[
            graph.format_candidate(record(term=D,candidate_id="c1")),
            graph.format_candidate(record(term="小儿湿疹",candidate_id="c2")),
            graph.format_candidate(record(term="神经衰弱",candidate_id="c3")),
        ]
        merged=graph.compact_same_quote_candidates(items)
        self.assertEqual(len(merged),1)
        self.assertEqual(merged[0]["candidate_ids"],["c1","c2","c3"])
        self.assertEqual(len(merged[0]["disease_terms"]),3)
        self.assertEqual(merged[0]["content"].count(SAME_QUOTE),1)
        self.assertIn("同句多个候选",merged[0]["content"])
        self.assertEqual(merged[0]["clinical_status"],"not_validated")

    def test_same_quote_different_sections_not_merged(self):
        items=[
            graph.format_candidate(record(section_id="s1",candidate_id="c1")),
            graph.format_candidate(record(section_id="s2",candidate_id="c2")),
        ]
        self.assertEqual(len(graph.compact_same_quote_candidates(items)),2)

    def test_same_quote_different_review_states_not_merged(self):
        first=graph.format_candidate(record(candidate_id="c1"))
        second=graph.format_candidate(record(term="小儿湿疹",candidate_id="c2"))
        second["candidate_status"]="reviewed"
        self.assertEqual(len(graph.compact_same_quote_candidates([first,second])),2)

    def test_no_binding_ids_no_unsafe_merging(self):
        first=graph.format_candidate(record(candidate_id="c1"))
        second=graph.format_candidate(record(term="小儿湿疹",candidate_id="c2"))
        first["section_id"]=""
        second["section_id"]=""
        self.assertEqual(len(graph.compact_same_quote_candidates([first,second])),2)

    def test_only_explicit_disease_can_return_specific_candidate(self):
        mock=FakeGraph()
        no=mock.instance.search(f"{A}的适应症是什么？", explicit_disease_only=True)
        yes=mock.instance.search(f"{A}与{D}有什么关联？", explicit_disease_only=True)
        self.assertEqual(no, [])
        self.assertEqual(len(yes),1)
        self.assertEqual(yes[0]["disease_term"],D)

    def test_medication_name_not_disease_term(self):
        mock=FakeGraph()
        mock.instance.search(f"比较{A}与{B}的适应症", explicit_disease_only=True)
        query=[p["question"] for q,p in mock.calls if q==graph._TERM_IDENTIFY][-1]
        self.assertNotIn(A,query)
        self.assertNotIn(B,query)

    def test_two_drugs_get_sections_fairly(self):
        mock=FakeGraph()
        items=mock.instance.search_sections(f"比较{A}与{B}的适应症", limit=4)
        self.assertEqual(len(items),4)
        self.assertEqual(sum(x["drug_name"]==A for x in items),2)
        self.assertEqual(sum(x["drug_name"]==B for x in items),2)
        per_drug_limit=[p["limit"] for q,p in mock.calls if q==graph._SECTION_BY_DRUG]
        self.assertEqual(per_drug_limit,[2,2])

    def test_actual_rag_tool_comparison_returns_sections_not_candidates(self):
        mock=FakeGraph()
        result=mock.execute_tool(f"{A}与{B}的适应症有何不同？")
        self.assertEqual(result["candidate_count"],0)
        self.assertEqual(result["raw_candidate_nodes"],0)
        self.assertEqual(result["section_count"],4)
        self.assertEqual({x["drug_name"] for x in result["excerpts"]},{A,B})
        self.assertTrue(all(x["evidence_type"]=="unverified_drug_source_section"
                            for x in result["excerpts"]))

    def test_actual_rag_tool_explicit_open_discovery_quote_compacted(self):
        mock=FakeGraph()
        result=mock.execute_tool(f"{B}的说明书提及了哪些疾病？")
        self.assertEqual(result["raw_candidate_nodes"],3)
        self.assertEqual(result["candidate_count"],1)
        self.assertEqual(result["grouped_same_quote_nodes"],2)
        grouped=[x for x in result["excerpts"]
                 if x["evidence_type"]=="unverified_drug_disease_candidate"]
        self.assertEqual(len(grouped),1)
        self.assertEqual(len(grouped[0]["candidate_ids"]),3)

    def test_source_citation_distinguishes_chroma_graph_and_candidate(self):
        source=(ROOT/"rag.py").read_text(encoding="utf-8")
        tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                and n.name=="_medical_reference_section")
        ns={"Any":__import__("typing").Any}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"<refs>","exec"),ns)
        f=ns["_medical_reference_section"]
        result=f({"evidence_type":"json_neo4j_text_verified","parent_verified":True},"药名 / 适应症")
        self.assertIn("Chroma Hybrid 召回",result)
        self.assertIn("Neo4j Section 原文核对",result)
        only=f({"evidence_type":"unverified_drug_disease_candidate"},"DiseaseCandidate")
        self.assertIn("未经医学验证",only)

    def test_wrong_medical_review_claim_rewritten_without_losing_citations(self):
        filter=quality.MedicalAnswerStreamFilter()
        response=(filter.feed("根据审核通")+filter.feed("过的证据，两药不同 [1][2]。\\n")
                  +filter.finish())
        self.assertNotIn("审核通过的证据",response)
        self.assertIn("[1][2]",response)
        self.assertIn("程序级原文核对",response)

    def test_model_prompt_disallows_component_as_product_indication(self):
        source=(ROOT/"rag.py").read_text(encoding="utf-8")
        self.assertIn("不能因此声称包含该成分的整种制剂具有相同获批适应症",source)
        self.assertIn("禁止在回答中使用“审核通过的证据”",source)
        self.assertIn('explicit_disease_only=not candidate_discovery',source)

if __name__=="__main__":
    unittest.main()
