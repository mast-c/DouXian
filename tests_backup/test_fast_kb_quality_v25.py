"""V2.5 快速模式轻量校验：独立纯函数 + 模拟整条流式链路。"""
import ast
from pathlib import Path
import re
from unittest.mock import patch
from types import SimpleNamespace
import test_kb_fast_strict_v24 as core

HERE = Path(__file__).resolve().parent
FILE = HERE / "rag.py"
if not FILE.exists() or "def inspect_fast_kb_answer" not in FILE.read_text(encoding="utf-8"):
    FILE = HERE / "rag_V2_5_快速模式轻量质检.py"
source = FILE.read_text(encoding="utf-8")
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == "inspect_fast_kb_answer")
mini = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
ns = {"re": re}
exec(compile(mini, str(FILE), "exec"), ns)
inspect = ns["inspect_fast_kb_answer"]

REFS = {
    1: ("机器学习.md", "变量消去"),
    2: ("机器学习.md", "MCMC"),
    3: ("神经网络与深度学习.md", "信念传播"),
    4: ("机器学习.md", "变分推断"),
    5: ("机器学习.md", "贝叶斯网"),
}

ANSWER = (
    "### 精确推断\n"
    "- **变量消去法** [1]\n"
    "- **信念传播算法** [3]\n"
    "### 近似推断\n"
    "**代表方法：**\n"
    "- **MCMC采样 (Markov Chain Monte Carlo)** [2]\n"
    "- **吉布斯采样 (Gibbs Sampling)** [5]\n"
    "- **变分推断** [4]\n"
    "基于大数定律，通过抽取大量独立样本来近似目标期望 [2]。\n"
)


def test_catches_known_algorithm_concept_issues_without_calling_model():
    result = inspect(ANSWER, REFS)
    assert result["status"] == "warn"
    assert result["valid_citation_ids"] == [1,2,3,4,5]
    assert result["invalid_citation_ids"] == []
    assert len(result["warnings"]) == 2, result
    assert any("Gibbs" in w for w in result["warnings"])
    assert any("独立采样" in w for w in result["warnings"])


def test_valid_hierarchy_and_related_samples_do_not_trigger():
    answer = (
        "- MCMC [2]\n"
        "  - Gibbs Sampling 是其中的具体算法 [5]\n"
        "- 变分推断 [4]\n"
        "MCMC相邻样本不是独立样本，通常存在相关性 [2]。\n"
    )
    result = inspect(answer, REFS)
    assert result["status"] == "pass", result
    assert result["warnings"] == []


def test_invalid_citation_and_no_valid_citation():
    result = inspect("变量消去使用消息传递 [99]。", REFS)
    assert result["status"] == "warn"
    assert result["invalid_citation_ids"] == [99]
    assert result["valid_citation_ids"] == []
    assert "引用" in result["warnings"][0]

    result = inspect("变量消去使用求和消去变量 [2][999]。", REFS)
    assert result["valid_citation_ids"] == [2]
    assert result["invalid_citation_ids"] == [999]
    assert result["warnings"] == []  # 无效编号由已有参考资料渲染函数提示。


def test_exact_repeated_long_line():
    line = (
        "- **实验说明：** 已经得到的变量后验概率"
        "可以通过局部边际分布进行解释，这是一条足够长的文本 [1]。"
    )
    result = inspect(line + "\n" + line, REFS)
    assert result["status"] == "warn"
    assert any("重复" in x for x in result["warnings"])
    result = inspect("**基本思路**\n**基本思路**\n有效证据 [1]。", REFS)
    assert result["status"] == "pass"


def test_streaming_fast_mode_stores_trace_and_shows_caution_only_once():
    class MockModel(core.FakeModel):
        def stream(self, prompts):
            self.prompts.append(prompts)
            yield SimpleNamespace(content=(
                "根据教材可得：[1]\n"
                "- **MCMC采样** [1]\n"
                "- **Gibbs采样** [1]\n"
                "抽取大量独立样本用来估计期望。"
            ))

    with patch.object(core, "FakeModel", MockModel):
        text, trace, _, output = core.execute("fast")
    q = trace["fast_quality_check"]
    assert q["status"] == "warn",q
    assert len(q["warnings"]) == 2,q
    assert text.count("快速模式轻量检查提醒") == 1
    assert text.count("参考资料：") == 1
    assert text.count("快速模式：本次未执行独立证据校验") == 1
    assert "[RAG轻量质检]" in output


def test_streaming_strict_mode_never_runs_extra_qc():
    text, trace, _, output = core.execute("strict")
    assert "fast_quality_check" not in trace
    assert "快速模式轻量检查提醒" not in text
    assert "[RAG轻量质检]" not in output


def test_streaming_fast_mode_clean_answer_does_not_add_warning():
    text, trace, _, output = core.execute("fast")
    assert trace["fast_quality_check"]["status"] == "pass"
    assert trace["fast_quality_check"]["warnings"] == []
    assert "快速模式轻量检查提醒" not in text


def test_its_local_only_and_rest_of_pipeline_preserved():
    assert "model=self.chat_model" in source  # 只用于原有审查等调用
    assert "def inspect_fast_kb_answer(" in source
    assert "max_documents=5, subqueries=queries" in source
    assert 'extra_body={"enable_thinking": False}' in source
    assert "DOUXIAN_FAST_QUERY_PLAN" in source
    assert "DOUXIAN_KB_MODE" in source
    assert "_footer_markers = (" in source


if __name__ == "__main__":
    tests = [
        (name, value) for name,value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for name,fn in tests:
        fn()
        print("通过：",name)
    print(f"✅ V2.5 轻量规则质检：{len(tests)} 组离线测试通过（无模型调用）")
