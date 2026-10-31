# -*- coding: utf-8 -*-
"""V1.11：真实检索返回数 vs 原文核验 vs 去重保留的回归测试。"""
import ast
import html
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def import_module(filename, name):
    spec=importlib.util.spec_from_file_location(name, ROOT/filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

status=import_module("medical_retrieval_status.py","medical_status_v111_test")

def item(tool, name, verified=False):
    return {
        "_retrieval_tool":tool, "drug_name":name,
        "content":"适应症原文",
        "parent_verified":verified,
    }

GRAPH="search_drug_graph"
CHROMA="search_knowledge_base"
DRUG_A="阿托伐他汀钙片"
DRUG_B="10%、20%、30%脂肪乳注射液"

def typical_case():
    reports={
        GRAPH: {
            "status":"ok", "hits":0, "hits_before_entity_gate":2,
            "section_count":2, "candidate_count":0,
            "raw_candidate_nodes":0, "grouped_same_quote_nodes":0,
        },
        CHROMA: {
            "status":"ok", "hits":2, "hits_before_entity_gate":2,
            "chroma_chunks":88,
            "medical_validation": {
                "verified":2, "failed_provenance":0, "skipped_section":6,
                "legacy_unverified":0,
            },
        }
    }
    before=[
        item(GRAPH,DRUG_A),item(GRAPH,DRUG_B),
        item(CHROMA,DRUG_A,True),item(CHROMA,DRUG_B,True),
    ]
    final=before[2:]
    return reports,before,final

class TestRetrievalStages(unittest.TestCase):
    def test_screenshot_scenario_never_says_graph_not_used(self):
        reports,before,final=typical_case()
        flow=status.build_retrieval_pipeline(
            reports,before,before,final,
            {"explicit_drugs":[DRUG_A,DRUG_B]},
        )
        self.assertEqual(flow["neo4j_direct"]["returned"],2)
        self.assertEqual(flow["neo4j_direct"]["retained"],0)
        self.assertEqual(flow["chroma_hybrid"]["returned"],2)
        self.assertEqual(flow["chroma_hybrid"]["parent_verified"],2)
        self.assertEqual(flow["chroma_hybrid"]["retained"],2)
        self.assertEqual(flow["merge"]["removed_by_dedup"],2)
        self.assertEqual(flow["merge"]["retained"],2)
        self.assertEqual(flow["merge"]["retained_neo4j_verified"],2)
        trace={"retrieval_pipeline":flow}
        heading=status.retrieval_heading(trace)
        self.assertIn("合并去重后 2 条",heading)
        self.assertIn("Neo4j 原文核验 2 条",heading)
        self.assertNotIn("Neo4j 0 条",heading)
        markup=status.retrieval_status_html(trace)
        self.assertIn("Neo4j 直接检索",markup)
        self.assertIn("去重后独立保留 0 条",markup)
        self.assertIn("同原文合并 2 条",markup)
        self.assertIn("Chroma 的 2 条文本证据通过了 Neo4j",markup)
        self.assertIn("Neo4j 直接返回过 2 条证据",markup)

    def test_graph_zero_can_still_have_parent_verification(self):
        reports,before,final=typical_case()
        reports[GRAPH]["hits_before_entity_gate"]=0
        flow=status.build_retrieval_pipeline(reports,before[2:],before[2:],final,{})
        self.assertEqual(flow["neo4j_direct"]["returned"],0)
        self.assertEqual(flow["chroma_hybrid"]["parent_verified"],2)
        self.assertIn("Neo4j 原文核验 2 条",status.retrieval_heading(
            {"retrieval_pipeline":flow}
        ))

    def test_actual_graph_failure_is_not_described_as_no_match(self):
        reports,before,final=typical_case()
        reports[GRAPH]={"status":"error","hits":0}
        flow=status.build_retrieval_pipeline(reports,before[2:],before[2:],final,{})
        text=status.retrieval_status_html({"retrieval_pipeline":flow})
        self.assertIn("直接检索失败",text)
        self.assertIn("执行失败",text)
        self.assertIn("（部分通道异常）",status.retrieval_heading(
            {"retrieval_pipeline":flow}))

    def test_chroma_failure_retains_graph_and_marks_failure(self):
        reports,before,final=typical_case()
        reports[CHROMA]={"status":"error","hits":0}
        flow=status.build_retrieval_pipeline(reports,before[:2],before[:2],before[:2],{})
        self.assertEqual(flow["neo4j_direct"]["retained"],2)
        self.assertEqual(flow["chroma_hybrid"]["returned"],0)
        self.assertEqual(flow["chroma_hybrid"]["parent_verified"],0)
        self.assertIn("Chroma 文本检索失败",status.retrieval_status_html(
            {"retrieval_pipeline":flow}))

    def test_explicit_entity_filter_kept_separate_from_dedup(self):
        reports,before,final=typical_case()
        scoped=[before[0],before[2]]
        final=[before[2]]
        flow=status.build_retrieval_pipeline(
            reports,before,scoped,final,{"explicit_drugs":[DRUG_A]}
        )
        self.assertEqual(flow["merge"]["removed_by_entity_filter"],2)
        self.assertEqual(flow["merge"]["removed_by_dedup"],1)
        self.assertEqual(flow["merge"]["retained"],1)

    def test_same_quote_candidate_count_not_miscounted_as_sources(self):
        reports,before,final=typical_case()
        reports[GRAPH].update({
            "hits_before_entity_gate":1,
            "candidate_count":1,"raw_candidate_nodes":3,
            "grouped_same_quote_nodes":2, "section_count":0,
        })
        flow=status.build_retrieval_pipeline(reports,before,before,final,{})
        self.assertEqual(flow["neo4j_direct"]["raw_candidate_nodes"],3)
        self.assertEqual(flow["neo4j_direct"]["candidate_cards"],1)
        self.assertEqual(flow["neo4j_direct"]["grouped_same_quote_nodes"],2)

    def test_bad_or_missing_legacy_trace_is_safe(self):
        for trace in (None, {}, {"retrieval_summary":{GRAPH:{"hits":0}}},
                      {"retrieval_pipeline":{"schema_version":999}}):
            self.assertIsNone(status.retrieval_heading(trace))
            self.assertEqual(status.retrieval_status_html(trace),"")

    def test_invalid_counts_do_not_crash_or_claim_hits(self):
        flow=status.build_retrieval_pipeline({
            GRAPH: {"hits_before_entity_gate":"not a number","status":"invalid_response"},
            CHROMA: {"hits":"nan","status":"empty_collection",
                     "medical_validation":{"verified":True, "failed_provenance":-1}},
        }, [], [], [], {})
        self.assertEqual(flow["neo4j_direct"]["returned"],0)
        self.assertEqual(flow["chroma_hybrid"]["parent_verified"],0)
        self.assertEqual(flow["chroma_hybrid"]["parent_rejected"],0)

    def test_no_patient_or_untrusted_query_is_in_status_markup(self):
        reports,before,final=typical_case()
        flow=status.build_retrieval_pipeline(
            reports,before,before,final,
            {"explicit_drugs":["<img src=x onerror=alert(1)>"]}
        )
        markup=status.retrieval_status_html({"retrieval_pipeline":flow})
        self.assertNotIn("<img",markup)
        self.assertNotIn("alert(1)",markup)

class TestIntegration(unittest.TestCase):
    def test_backend_snapshots_all_real_stages(self):
        code=(ROOT/"rag.py").read_text(encoding="utf-8")
        ast.parse(code)
        needed=[
            "from medical_retrieval_status import build_retrieval_pipeline",
            "pre_entity_excerpts = list(merged_excerpts)",
            "post_entity_excerpts = list(merged_excerpts)",
            "merged_excerpts = collapse_duplicate_parent_sections(merged_excerpts)",
            'trace["retrieval_pipeline"] = build_retrieval_pipeline(',
            '"section_count": data.get("section_count", 0)',
        ]
        self.assertTrue(all(token in code for token in needed))

    def test_streamlit_renders_new_status_and_has_old_trace_fallback(self):
        code=(ROOT/"豆馅.py").read_text(encoding="utf-8")
        ast.parse(code)
        for text in (
            "from medical_retrieval_status import retrieval_heading, retrieval_status_html",
            "new_heading = retrieval_heading(trace)",
            "retrieval_panel = retrieval_status_html(trace)",
            "旧会话统计：Neo4j 独立保留",
            ".kb-retrieval-metrics",
        ):
            self.assertIn(text,code)

    def test_legacy_heading_marks_postdedup_count_as_legacy(self):
        code=(ROOT/"豆馅.py").read_text(encoding="utf-8")
        parsed=ast.parse(code)
        target=next(n for n in parsed.body
                    if isinstance(n,ast.FunctionDef) and n.name=="trace_heading")
        ns={
            "retrieval_heading":status.retrieval_heading,
            "_web_sources":lambda t:[],
            "_web_queries":lambda t:[],
        }
        exec(compile(ast.Module(body=[target],type_ignores=[]),
                     "<legacy-status-heading>", "exec"),ns)
        trace={
            "events":[
                {"type":"end","ok":True,"tool":GRAPH},
                {"type":"end","ok":True,"tool":CHROMA},
            ],
            "retrieval_summary":{
                GRAPH:{"status":"ok","hits":0},
                CHROMA:{
                    "status":"ok","hits":2,
                    "medical_validation":{"verified":2},
                },
            },
        }
        label=ns["trace_heading"](trace)
        self.assertIn("旧会话独立证据",label)
        self.assertIn("Neo4j 0 条",label)
        self.assertIn("Neo4j 原文核验 2 条",label)

    def test_actual_streamlit_heading_and_panel(self):
        code=(ROOT/"豆馅.py").read_text(encoding="utf-8")
        parsed=ast.parse(code)
        funcs=[
            node for node in parsed.body
            if isinstance(node,ast.FunctionDef)
            and node.name in ("trace_heading","trace_details_html")
        ]
        self.assertEqual(len(funcs),2)
        ns={
            "retrieval_heading":status.retrieval_heading,
            "retrieval_status_html":status.retrieval_status_html,
            "html":html,
            "_web_sources":lambda t:[],
            "_web_queries":lambda t:[],
            "elapsed_label":lambda t:"53s",
            "describe_tool_event":lambda event:event.get("tool",""),
        }
        exec(compile(ast.Module(body=funcs,type_ignores=[]),
                     "<status-ui-functions>", "exec"),ns)
        reports,before,final=typical_case()
        trace={
            "retrieval_pipeline":status.build_retrieval_pipeline(
                reports,before,before,final,{}
            ),
            "events":[
                {"type":"end","ok":True,"tool":GRAPH},
                {"type":"end","ok":True,"tool":CHROMA},
            ],
            "total_seconds":53,
        }
        self.assertIn("合并去重后 2 条证据",ns["trace_heading"](trace))
        self.assertNotIn("Neo4j 0 条",ns["trace_heading"](trace))
        render=ns["trace_details_html"](trace)
        for content in ("检索与原文核验统计",
                        "Neo4j 直接检索","Chroma 文本证据",
                        "Neo4j 原文核验","同原文合并 2 条"):
            self.assertIn(content,render)
        self.assertNotIn("Neo4j：已命中",render)
        self.assertLess(render.index("检索与原文核验统计"),
                        render.index("Neo4j 直接检索"))
        self.assertEqual(render.count("检索与原文核验统计"),1)

if __name__=="__main__":
    unittest.main()
