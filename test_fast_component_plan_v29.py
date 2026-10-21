# -*- coding: utf-8 -*-
"""豆馅 V2.9 离线回归：真实 rag.py 的 AST、规则正反例、Qwen 调用计数、原检索链路。

没有外部依赖，不调用 DashScope。可在项目 data 目录执行。
"""
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
from unittest.mock import patch

from fast_component_planner_v29 import plan_two_layer_aspects

ROOT = Path(__file__).resolve().parent
RAG = ROOT / "rag.py"
if not RAG.exists() or "DOUXIAN_FAST_COMPONENT_PLAN" not in RAG.read_text(encoding="utf-8"):
    RAG = ROOT / "rag_V2_9_双概念快速查询规划.py"

B04 = "卷积神经网络中，卷积层与池化层分别完成什么运算？它们怎样影响特征表示？"

SRC=RAG.read_text(encoding="utf-8")
TREE=ast.parse(SRC, filename=str(RAG))
RAG_CLS=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=="RagService")
DEF_DECOMPOSE=next(n for n in RAG_CLS.body if isinstance(n,ast.FunctionDef) and n.name=="decompose_queries")
DEF_METHODS=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=="_fast_plan_paired_methods")
DEF_INIT=next(n for n in RAG_CLS.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
DEF_KB_SEARCH=next(n for n in ast.walk(DEF_INIT)
                   if isinstance(n,ast.FunctionDef) and n.name=="search_knowledge_base")
DEF_KB_SEARCH.decorator_list=[]


class FakeMessage:
    def __init__(self,content):
        self.content=content


class FakePlannerModel:
    def __init__(self):
        self.calls=0
        self.prompts=[]
    def bind(self, **kwargs):
        assert kwargs.get("temperature")==0
        return self
    def invoke(self,messages):
        self.calls += 1
        self.prompts.append(messages)
        return SimpleNamespace(content=json.dumps({
            "queries":[
                "旧版Qwen规划卷积层操作",
                "旧版Qwen规划池化层操作",
                "旧版Qwen规划卷积层特征",
                "旧版Qwen规划池化层特征",
            ]
        },ensure_ascii=False))


def isolated_rag_class():
    proxy_cls=ast.ClassDef(name="RagClass",bases=[],keywords=[],
                           body=[DEF_DECOMPOSE],decorator_list=[])
    isolated=ast.fix_missing_locations(ast.Module(body=[DEF_METHODS,proxy_cls],
                                                 type_ignores=[]))
    namespace={"os":os, "json":json, "re":re,
               "HumanMessage":FakeMessage,"SystemMessage":FakeMessage,
               "_plain_content":lambda v:v}
    exec(compile(isolated,filename=str(RAG),mode="exec"),namespace)
    return namespace["RagClass"]


MockRag=isolated_rag_class()


def _plan_run(question=B04, component_flag="1", kb_mode="fast",
              master="1", graph="off"):
    rag=MockRag()
    rag.chat_model=FakePlannerModel()
    buf=StringIO()
    flags={
        "DOUXIAN_FAST_QUERY_PLAN":master,
        "DOUXIAN_FAST_COMPONENT_PLAN":component_flag,
        "DOUXIAN_KB_MODE":kb_mode,
        "DOUXIAN_GRAPH_MODE":graph,
    }
    with patch.dict(os.environ, flags):
        with redirect_stdout(buf):
            queries=rag.decompose_queries(question)
    return queries,rag.chat_model.calls,buf.getvalue()


def test_exact_B04_full_coverage():
    results=plan_two_layer_aspects(B04)
    assert len(results)==4
    assert len(set(results))==4
    for concept in ("卷积层","池化层"):
        subset=[q for q in results if concept in q]
        assert len(subset)==2
        assert any("运算" in q for q in subset)
        assert any("特征表示" in q for q in subset)
    assert all("卷积神经网络中" in q for q in results)


def test_wording_variants_and_reversed_components():
    samples=[
        "卷积层和汇聚层分别进行哪些操作？两者对特征图有什么影响？",
        "CNN中，池化层与卷积层各自执行什么计算？它们如何影响特征表达？",
        "请问，在卷积神经网络中，卷积层和池化层分别做什么运算？各自如何改变特征的表示？",
        "卷积层与池化层分别完成什么运算，它们怎样影响特征表示？",
    ]
    for question in samples:
        planned=plan_two_layer_aspects(question)
        assert len(planned)==4,(question,planned)
        assert len({q for q in planned if "运算" in q})==2


def test_unsupported_questions_are_not_oversimplified():
    negative=[
        "卷积层与池化层有什么区别？",
        "卷积层与池化层分别完成什么运算？它们怎样影响特征表示？请给出公式与Python实现。",
        "卷积层与池化层分别完成什么运算？它们怎样影响特征表示？还要对比Transformer。",
        "卷积层与池化层分别完成什么运算？它们怎样影响特征表示？如果使用步长3呢？",
        "卷积层和汇聚层有何区别？它们有何关系？",
        "池化层与汇聚层分别完成什么运算？它们怎样影响特征表示？",
        "卷积层与卷积层分别完成什么运算？它们怎样影响特征表示？",
        "随机森林与Bagging分别完成什么运算？它们怎样影响特征表示？",
        "贝叶斯网有哪些精确推断和近似推断方法？它们分别采用什么基本思路？",
        "",
        " " * 100,
        "x" * 150,
    ]
    for question in negative:
        assert plan_two_layer_aspects(question)==[],question
    assert plan_two_layer_aspects(None)==[]
    assert plan_two_layer_aspects(123)==[]


def test_on_skips_qwen_completely():
    result,count,logs=_plan_run(component_flag="1")
    assert count==0
    assert len(result)==4
    assert result==plan_two_layer_aspects(B04)
    assert "[Query Planner] component_fast_path=on, skipped Qwen" in logs


def test_off_uses_original_qwen():
    result,count,logs=_plan_run(component_flag="0")
    assert count==1,(result,count)
    assert result==[
        "旧版Qwen规划卷积层操作","旧版Qwen规划池化层操作",
        "旧版Qwen规划卷积层特征","旧版Qwen规划池化层特征",
    ]
    assert "[Query Planner] component_fast_path=on" not in logs


def test_master_disabled_does_not_use_new_rule():
    result,count,_=_plan_run(component_flag="1",master="0")
    assert count==1 and len(result)==4


def test_strict_mode_does_not_use_new_rule():
    result,count,_=_plan_run(component_flag="1",kb_mode="strict")
    assert count==1 and len(result)==4


def test_other_questions_fallback_to_qwen():
    question="卷积层与池化层有什么区别？"
    result,count,logs=_plan_run(question,component_flag="1")
    # 原有 decompose_queries 的复合问题识别会将“区别”视为复合。
    assert count==1,(result,logs)
    assert len(result)==4


def test_B01_existing_method_fast_rule_unchanged():
    q="贝叶斯网有哪些精确推断和近似推断方法？它们分别采用什么基本思路？"
    queries,count,logs=_plan_run(q,component_flag="1")
    assert count==0
    assert len(queries)==4,queries
    assert "精确推断" in queries[0] and "近似推断" in queries[2]
    assert "[Query Planner] fast_path=on, skipped Qwen" in logs


def test_graph_mode_does_not_change_planner_itself():
    off,c_off,_=_plan_run(graph="off")
    assist,c_assist,_=_plan_run(graph="assist")
    assert off==assist
    assert c_off==c_assist==0


def test_default_feature_off_from_actual_code():
    assert 'os.getenv("DOUXIAN_FAST_COMPONENT_PLAN", "0")' in SRC
    assert 'os.getenv("DOUXIAN_KB_MODE", "strict")' in SRC
    assert len(re.findall(r"search_with_rerank\(",SRC))>=2
    assert "DOUXIAN_GRAPH_MODE" in SRC
    assert "DOUXIAN_FAST_QUALITY_PROMPT" in SRC


def test_real_search_wrapper_same_four_rerank_calls_and_provenance():
    # 使用真实生产 search_knowledge_base 函数 AST；外部网络调用一律替换为虚拟模块。
    count_by_mode={}
    for enabled in ("0","1"):
        q_list,count,log=_plan_run(component_flag=enabled)
        calls=[]
        def fake_rerank(**kwargs):
            calls.append(kwargs)
            i=len(calls)
            return [SimpleNamespace(
                page_content=f"来自 Chroma 的原始 Chunk {i}",
                metadata={"source":"神经网络与深度学习.md",
                          "section_path":f"第5章/5.2.{i} 教材章节",
                          "chunk_id":f"real-index-doc-{i}"}
            )]
        svc=SimpleNamespace(
            knowledge_service=SimpleNamespace(search_with_rerank=fake_rerank),
            decompose_queries=lambda question,queries=q_list:queries,
        )
        ns={
            "os":os,"json":json,"time":time, "Path":Path, "self":svc,
            "__file__":str(RAG),
            "_select_diverse_documents":lambda groups,**kwargs: [
                doc for group in groups for doc in group
            ][:5],
        }
        fn_ast=ast.fix_missing_locations(ast.Module(body=[DEF_KB_SEARCH],type_ignores=[]))
        exec(compile(fn_ast,str(RAG),"exec"),ns)
        with patch.dict(os.environ,{"DOUXIAN_GRAPH_MODE":"off"}):
            with redirect_stdout(StringIO()):
                payload=json.loads(ns["search_knowledge_base"](B04))
        assert len(calls)==4,calls
        assert all(call["recall_k"]==20 and call["top_n"]==10 for call in calls)
        assert len(payload["excerpts"])==4
        assert all(item["chunk_id"].startswith("real-index-doc-")
                   for item in payload["excerpts"])
        assert all(item["content"].startswith("来自 Chroma 的原始 Chunk")
                   for item in payload["excerpts"])
        assert payload["graph_assist"]["status"]=="off"
        count_by_mode[enabled]=(count,len(calls),len(payload["excerpts"]))
    assert count_by_mode=={"0":(1,4,4),"1":(0,4,4)},count_by_mode


if __name__=="__main__":
    tests=sorted((k,v) for k,v in globals().items()
                 if k.startswith("test_") and callable(v))
    for name,fn in tests:
        fn()
        print("通过：",name)
    print(f"✅ V2.9 查询规划 {len(tests)} 组离线测试通过；没有模型/API 调用。")
