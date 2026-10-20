"""Stage 1: behavior-preserving timing instrumentation regression tests.

Run: python test_baseline_instrumentation.py
No real model / Chroma / DashScope API calls are made.
"""
import ast
import importlib.util
import json
import sys
import time
import types
import unittest
from contextvars import ContextVar
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODE = {"guard": "full", "kb_count": 4, "auto_tool": False, "empty_docs": False}


def install_fake_dependencies():
    def pkg(name):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
        return m

    pkg("langchain_community")
    pkg("langchain_community.embeddings").DashScopeEmbeddings = type(
        "DashScopeEmbeddings", (), {}
    )
    core = pkg("langchain_core")
    core.Document = None
    documents = pkg("langchain_core.documents")

    class Doc:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    documents.Document = Doc

    msgs = pkg("langchain_core.messages")
    for name in ("AIMessage", "HumanMessage", "SystemMessage", "ToolMessage"):
        def init(self, content="", **kwargs):
            self.content = content
            self.tool_calls = kwargs.get("tool_calls") or []
            self.kwargs = kwargs
        setattr(msgs, name, type(name, (), {"__init__": init}))

    tools = pkg("langchain_core.tools")
    class FakeTool:
        def __init__(self, func, name):
            self.func = func
            self.name = name
        def invoke(self, args):
            return self.func(**args)
    tools.tool = lambda name: (lambda func: FakeTool(func, name))

    openai = pkg("langchain_openai")
    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.stream_rounds = 0
        def bind_tools(self, available):
            return self
        def bind(self, **kwargs):
            return self
        def invoke(self, messages):
            return types.SimpleNamespace(content=json.dumps({
                "queries": ["精确推断方法和原理", "近似推断方法和原理"]
            }, ensure_ascii=False))
        def stream(self, messages):
            self.stream_rounds += 1
            # In automatic agent mode the fake first round chooses a tool.
            is_auto = any(
                "当前服务器日期" in str(getattr(m, "content", ""))
                for m in messages
            )
            if MODE["auto_tool"] and is_auto and self.stream_rounds == 1:
                yield types.SimpleNamespace(
                    content="", tool_call_chunks=[{"id":"t1"}],
                    tool_calls=[{"name":"search_knowledge_base",
                                 "args":{"question":"精确推断方法和原理"},
                                 "id":"call_1"}]
                )
                return
            if is_auto:
                yield types.SimpleNamespace(
                    content="直接回答内容。", tool_call_chunks=[],
                    tool_calls=[], invalid_tool_calls=[]
                )
            else:
                yield types.SimpleNamespace(
                    content="变量消去法与信念传播。[1]",
                    tool_call_chunks=[], tool_calls=[], invalid_tool_calls=[]
                )
    openai.ChatOpenAI = FakeLLM

    pkg("dotenv").load_dotenv = lambda *args, **kwargs: None

    knowledge = pkg("knowledge_base")
    knowledge.KB_TIMING_CALLBACK = ContextVar("kb_timing_callback_fake", default=None)
    def emit(stage, started, **details):
        cb = knowledge.KB_TIMING_CALLBACK.get()
        if cb:
            cb(stage, time.perf_counter()-started, details)
    knowledge._emit_kb_timing = emit

    class FakeKB:
        instances = []
        def __init__(self):
            self.calls=[]
            self.__class__.instances.append(self)
        def build_bm25_index(self):
            return MODE["kb_count"]
        def build_hybrid_retriever(self):
            pass
        def search_with_rerank(self, query, recall_k=10, top_n=3):
            self.calls.append((query,recall_k,top_n))
            emit("kb.hybrid_search", time.perf_counter(), query=query, k=recall_k)
            emit("kb.rerank_api", time.perf_counter(), query=query,
                 candidates=recall_k, top_n=top_n)
            if MODE["empty_docs"]:
                return []
            docs = [
                Doc("章节：第14章 / 14.4\n变量消去法利用动态规划完成精确推断。",
                    {"source":"机器学习.md", "section_path":"14.4 变量消去", "chunk_id":"k1"}),
                Doc("章节：第14章 / 14.5\n近似推断可以采用 MCMC。",
                    {"source":"机器学习.md", "section_path":"14.5 MCMC采样", "chunk_id":"k2"}),
                Doc("章节：第14章 / 14.4\n信念传播也是精确推断。",
                    {"source":"机器学习.md", "section_path":"14.4 信念传播", "chunk_id":"k3"}),
            ]
            return docs[:top_n]
    knowledge.KnowledgeBaseService = FakeKB

    guard = pkg("evidence_guard")
    def assess_evidence(model, question, excerpts, retrieval_queries=None):
        status = MODE["guard"]
        return {
            "decision": status,
            "evidence_ids": list(range(1, len(excerpts)+1)),
            "aspects": [{"status":"supported", "question":question,
                         "evidence_ids":[1]}],
            "refused_questions": ["尚未覆盖的内容"] if status=="partial" else [],
        }
    guard.assess_evidence = assess_evidence

    pkg("config_data")
    history = pkg("file_history_store")
    class Store:
        messages = []
        def add_messages(self, messages):
            self.messages = list(messages)
    history.get_history = lambda session_id: Store()
    search = pkg("search_tool")
    search.DISPLAY_SOURCE_LIMIT = 5
    def web_search(query):
        return json.dumps({
            "ok": True, "answer": "已核实的联网回答。",
            "sources": [{"title": "Official", "url":"https://example.org/official"}],
            "queries":[query], "search_calls":1
        }, ensure_ascii=False)
    search.web_search = FakeTool(web_search, "web_search")
    pkg("vector_stores").VectorStoreService = type("VectorStoreService", (), {})
    return knowledge


KB = install_fake_dependencies()


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


original = load_module("rag_original_for_tests", HERE/"original_backup"/"rag.py")
timed = load_module("rag_timed_for_tests", HERE/"rag.py")


def run(mod, text, mode="知识库", **options):
    KB.KnowledgeBaseService.instances.clear()
    trace = {}
    service = mod.RagService()
    response = service.chain.invoke({
        "input": text, "mode": mode, "trace": trace, "history": [],
    })
    queries = [
        call
        for instance in KB.KnowledgeBaseService.instances
        for call in instance.calls
    ]
    return response, trace, queries


class BaselineTests(unittest.TestCase):
    def setUp(self):
        MODE.update({"guard":"full","kb_count":4,"auto_tool":False,"empty_docs":False})

    def assert_same_behavior(self, text, mode="知识库"):
        a, ta, qa = run(original, text, mode=mode)
        b, tb, qb = run(timed, text, mode=mode)
        self.assertEqual(a, b, "新旧版本回答内容不一致")
        self.assertEqual(qa, qb, "新旧版本检索的查询和 k 不一致")
        normalized = lambda events: [
            {k:v for k,v in x.items() if k!="duration"}
            for x in events
        ]
        self.assertEqual(normalized(ta["events"]), normalized(tb["events"]))
        return b, tb

    def test_python_syntax(self):
        for name in ("rag.py","knowledge_base.py"):
            ast.parse((HERE/name).read_text(encoding="utf-8"))
            ast.parse((HERE/"original_backup"/name).read_text(encoding="utf-8"))

    def test_kb_single_question_output_unchanged(self):
        answer, t = self.assert_same_behavior("精确推断是什么？")
        self.assertIn("参考资料", answer)
        self.assertIn("llm.evidence_guard", t["timings"])
        self.assertIn("llm.answer_generation", t["timings"])
        self.assertIn("kb.hybrid_search", t["timings"])
        self.assertIn("kb.rerank_api", t["timings"])
        self.assertIn("turn_total", t["timings"])

    def test_kb_compound_query_output_and_k_unchanged(self):
        answer, t = self.assert_same_behavior(
            "精确推断和近似推断有什么区别？"
        )
        self.assertEqual(len(t["timing_details"]) > 3, True)
        self.assertIn("kb.query_decomposition", t["timings"])

    def test_kb_empty_collection_output_unchanged(self):
        MODE["kb_count"]=0
        answer, t = self.assert_same_behavior("精确推断是什么？")
        self.assertIn("知识库",answer)
        self.assertIn("kb.bm25_total",t["timings"])

    def test_kb_empty_docs_output_unchanged(self):
        MODE["empty_docs"]=True
        answer, _ = self.assert_same_behavior("精确推断是什么？")
        self.assertIn("知识库",answer)

    def test_partial_rejection_output_unchanged(self):
        MODE["guard"]="partial"
        answer, t = self.assert_same_behavior("精确推断是什么？")
        self.assertIn("当前知识库证据不足", answer)
        self.assertEqual(t["evidence_coverage"]["decision"],"partial")

    def test_full_rejection_output_unchanged(self):
        MODE["guard"]="refuse"
        answer, _ = self.assert_same_behavior("精确推断是什么？")
        self.assertIn("证据不足",answer)

    def test_web_mode_output_unchanged(self):
        answer, t=self.assert_same_behavior("2026年最新新闻是什么？",mode="联网")
        self.assertIn("已核实",answer)
        self.assertIn("tool.web_search",t["timings"])

    def test_auto_direct_output_unchanged(self):
        answer, t=self.assert_same_behavior("简单问好",mode="自动")
        self.assertEqual(answer,"直接回答内容。")
        self.assertIn("llm.agent_round",t["timings"])

    def test_auto_tool_output_unchanged(self):
        MODE["auto_tool"]=True
        answer, t=self.assert_same_behavior(
            "贝叶斯网是什么？",mode="自动"
        )
        self.assertIn("直接回答内容",answer)
        self.assertIn("tool.search_knowledge_base",t["timings"])

    def test_trace_stage_details_and_ttft(self):
        _, t, _ = run(timed,"精确推断是什么？")
        self.assertIsInstance(t["timings"],dict)
        self.assertGreaterEqual(t["timings"]["turn_total"],0)
        self.assertIn("answer_ttft_seconds",t)
        self.assertGreaterEqual(t["answer_ttft_seconds"],0)
        stages={e["stage"] for e in t["timing_details"]}
        self.assertIn("kb.query_decomposition",stages)
        self.assertIn("llm.evidence_guard",stages)




class KnowledgeBaseMethodParityTests(unittest.TestCase):
    """直接比较两版真实 KB 方法（用假的检索器和假的 Rerank API）。"""

    @staticmethod
    def extract_methods(file_path):
        source=Path(file_path).read_text(encoding="utf-8")
        tree=ast.parse(source)
        cls=next(x for x in tree.body
                 if isinstance(x,ast.ClassDef) and x.name=="KnowledgeBaseService")
        methods={m.name:m for m in cls.body if isinstance(m,ast.FunctionDef)}
        lines=source.splitlines()
        required=("search_hybrid","search_with_rerank")
        code="class KBOnly:\n"
        for name in required:
            node=methods[name]
            code+="\n".join(lines[node.lineno-1:node.end_lineno])+"\n\n"
        return code

    def test_search_and_rerank_real_methods_unchanged(self):
        import os
        from http import HTTPStatus
        from unittest.mock import patch

        audit_calls=[]
        timing_calls=[]
        class FakeRerank:
            @classmethod
            def call(cls, **kwargs):
                audit_calls.append({
                    "model":kwargs["model"],
                    "query":kwargs["query"],
                    "documents":kwargs["documents"],
                    "top_n":kwargs["top_n"],
                })
                return types.SimpleNamespace(
                    status_code=HTTPStatus.OK,
                    output={"results":[
                        {"index":i,"relevance_score":1/(i+1)}
                        for i in range(kwargs["top_n"])
                    ]}
                )
        class Retriever:
            def __init__(self,docs):
                self.docs=docs
                self.queries=[]
            def invoke(self,q):
                self.queries.append(q)
                return self.docs

        def build(path):
            namespace={
                "Document":sys.modules["langchain_core.documents"].Document,
                "os":os, "time":time, "HTTPStatus":HTTPStatus,
                "dashscope":types.SimpleNamespace(TextReRank=FakeRerank,
                                                  base_http_api_url=None),
                "_emit_kb_timing":lambda stage,start,**kw:
                    timing_calls.append(stage),
            }
            code=self.extract_methods(path)
            exec(compile(code,str(path),"exec"),namespace)
            service=namespace["KBOnly"]()
            service.hybrid_retriever=Retriever([
                namespace["Document"](
                    "证据内容 "+str(i),
                    {"source":"机器学习.md","chunk_id":str(i)}
                ) for i in range(5)
            ])
            return service

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY":"dummy-test-token"}):
            old=build(HERE/"original_backup"/"knowledge_base.py")
            new=build(HERE/"knowledge_base.py")
            original_docs=old.search_with_rerank(
                "算法原理",recall_k=4,top_n=3
            )
            instrumented_docs=new.search_with_rerank(
                "算法原理",recall_k=4,top_n=3
            )
        normalize=lambda docs:[
            (d.page_content,d.metadata) for d in docs
        ]
        self.assertEqual(normalize(original_docs),normalize(instrumented_docs))
        self.assertEqual(old.hybrid_retriever.queries,new.hybrid_retriever.queries)
        self.assertEqual(audit_calls[0],audit_calls[1])
        self.assertEqual(audit_calls[0]["top_n"],3)
        self.assertEqual(audit_calls[0]["model"],"qwen3-vl-rerank")
        self.assertIn("kb.hybrid_search",timing_calls)
        self.assertIn("kb.rerank_api",timing_calls)


if __name__=="__main__":
    unittest.main(verbosity=2)
