"""豆馅 V2.4 端到端路径离线测试：从真实 _stream_agent AST 执行，无模型/API依赖。"""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
TARGET = HERE / "rag.py"
if not TARGET.exists() or 'DOUXIAN_KB_MODE' not in TARGET.read_text(encoding='utf-8'):
    TARGET = HERE / "rag_快速与严格双模式_V2_4.py"
src = TARGET.read_text(encoding="utf-8")
module = ast.parse(src, filename=str(TARGET))
rag_cls = next(n for n in module.body
               if isinstance(n, ast.ClassDef) and n.name == "RagService")
method = next(n for n in rag_cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "_stream_agent")
fake_cls = ast.ClassDef(
    name="MockRag", bases=[], keywords=[], body=[method],
    decorator_list=[]
)
isolated = ast.fix_missing_locations(ast.Module(body=[fake_cls], type_ignores=[]))


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self):
        self.prompts = []
        self.options = []

    def bind(self, **options):
        self.options.append(options)
        return self

    def stream(self, prompts):
        self.prompts.append(prompts)
        yield SimpleNamespace(content="变量消去可计算后验边际概率 [1]。")


class Counters:
    def __init__(self):
        self.guard = 0
        self.retry = 0


COUNTERS = Counters()


def mock_guard(**kwargs):
    COUNTERS.guard += 1
    assert kwargs["question"] == "贝叶斯网的精确推断方法是什么？"
    return {
        "decision": "partial",
        "aspects": [
            {"question": "贝叶斯网的精确推断方法是什么？",
             "status": "supported", "evidence_ids": [1]},
            {"question": "教材是否讨论2025年国际竞赛准确率？",
             "status": "insufficient", "evidence_ids": []},
        ],
        "evidence_ids": [1],
        "answer_questions": ["贝叶斯网的精确推断方法是什么？"],
        "refused_questions": ["教材是否讨论2025年国际竞赛准确率？"],
    }


def mock_retry(**kwargs):
    COUNTERS.retry += 1
    return {
        "coverage": kwargs["coverage"],
        "excerpts": kwargs["excerpts"],
        "retry_queries": [],
        "new_chunks": 0,
        "status": "unchanged",
    }


def mock_history(session_id):
    return SimpleNamespace(messages=[], add_messages=lambda msg: None)


def mock_references(answer, refs):
    assert 1 in refs and refs[1][0] == "机器学习.md"
    return "\n参考资料：[1] 机器学习.md" if "[1]" in answer else ""


symbols = {
    "os": os, "json": json, "re": re, "time": time,
    "Iterator": Iterator,
    "get_history": mock_history,
    "HumanMessage": FakeMessage,
    "SystemMessage": FakeMessage,
    "AIMessage": FakeMessage,
    "SYSTEM_PROMPT": "你是豆馅。",
    "requires_web_search": lambda q: False,
    "_plain_content": lambda content: content if isinstance(content, str) else "",
    "render_kb_references": mock_references,
    "assess_evidence": mock_guard,
    "retry_insufficient_evidence": mock_retry,
    "reassess_fixed_aspects": lambda **k: {},
    "_today": lambda: "2026-10-20",
}
# V2.5 fast 模式会调用新的纯 Python 质检函数，独立 AST 测试须一并载入。
quality_helper = next(
    (n for n in module.body
     if isinstance(n, ast.FunctionDef) and n.name == "inspect_fast_kb_answer"),
    None,
)
if quality_helper is not None:
    helper_module = ast.fix_missing_locations(
        ast.Module(body=[quality_helper], type_ignores=[])
    )
    exec(compile(helper_module, str(TARGET), "exec"), symbols)

exec(compile(isolated, str(TARGET), "exec"), symbols)
MockRag = symbols["MockRag"]


def execute(mode="fast", found=True):
    COUNTERS.guard = 0
    COUNTERS.retry = 0
    svc = MockRag()
    svc.chat_model = FakeModel()
    svc._ui_history = lambda h: []
    svc.knowledge_service = SimpleNamespace(search_with_rerank=lambda **kwargs: [])
    payload = {
        "found": found,
        "excerpts": [
            {
                "source": "机器学习.md",
                "section_path": "14.4.1 变量消去",
                "content": "变量消去通过求和消去变量计算后验边际概率。",
            }
        ] if found else [],
        "subqueries": ["贝叶斯网的精确推断方法是什么？"],
    }
    svc._call_tool = lambda name, args, trace, progress: (
        json.dumps(payload, ensure_ascii=False), True
    )
    trace = {}
    inputs = {
        "input": "贝叶斯网的精确推断方法是什么？",
        "mode": "知识库",
        "trace": trace,
        "history": [],
    }
    console = StringIO()
    with patch.dict(os.environ, {"DOUXIAN_KB_MODE": mode}):
        with redirect_stdout(console):
            output = "".join(svc._stream_agent(inputs, session_id="local-test"))
    return output, trace, svc.chat_model, console.getvalue()


def test_fast_skips_both_external_qwen_judges():
    text, trace, model, log = execute("fast")
    assert COUNTERS.guard == 0 and COUNTERS.retry == 0, (COUNTERS.guard, COUNTERS.retry)
    assert trace["evidence_coverage"]["decision"] == "unchecked", trace
    assert trace["evidence_coverage"]["aspects"] == [], trace
    assert trace["evidence_coverage"]["evidence_ids"] == [1]
    assert trace["self_rag_retry"]["status"] == "skipped_fast_mode", trace
    assert len(trace["knowledge_evidence"]) == 1
    assert "变量消去可计算后验边际概率 [1]" in text, text
    assert "参考资料：[1]" in text
    assert "快速模式：本次未执行独立证据校验" in text
    assert "skipped (KB fast mode" in log


def test_fast_prompt_does_not_falsely_claim_evidence_is_verified():
    _, _, model, _ = execute("fast")
    assert len(model.prompts) == 1
    system, user = model.prompts[0][0], model.prompts[0][-1]
    assert "未执行独立的证据充分性审核" in system.content
    assert "已通过程序级证据预检查" not in system.content
    assert "未进行独立证据审核" in user.content
    assert "机器学习.md" in user.content


def test_strict_still_calls_guard_and_selfrag():
    text, trace, model, log = execute("strict")
    assert COUNTERS.guard == 1 and COUNTERS.retry == 1
    assert trace["evidence_coverage"]["decision"] == "partial", trace
    assert trace["self_rag_retry"]["status"] == "unchanged", trace
    assert "教材是否讨论2025年国际竞赛准确率？" in text
    assert "当前知识库证据不足" in text
    assert "已通过程序级证据预检查" in model.prompts[0][0].content


def test_default_mode_is_strict():
    # No need to mock model: verify the production branch uses strict by default.
    assert 'os.getenv("DOUXIAN_KB_MODE", "strict")' in src
    execute("not-fast")
    assert COUNTERS.guard == 1 and COUNTERS.retry == 1


def test_no_documents_is_still_handled_without_model_call():
    text, trace, model, log = execute("fast", found=False)
    assert "没有检索到相关文档" in text
    assert not model.prompts
    assert COUNTERS.guard == 0 and COUNTERS.retry == 0


def test_search_and_prompt_options_unchanged():
    assert "max_documents=5, subqueries=queries" in src
    assert "recall_k=20," in src and "top_n=10," in src
    assert "DOUXIAN_FAST_QUERY_PLAN" in src
    assert 'extra_body={"enable_thinking": False}' in src
    assert "[RAG耗时] 最终回答首字等待" in src
    assert "render_kb_references(" in src


if __name__ == "__main__":
    test_functions = [value for key, value in sorted(globals().items())
                      if key.startswith("test_") and callable(value)]
    for test in test_functions:
        test()
        print("通过：", test.__name__)
    print(f"\n✅ 快速/严格双模式：{len(test_functions)} 组离线链路测试通过。")
