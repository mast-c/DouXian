"""V2.4.1 尾注去重回归：真实流式入口 + 不同拆包粒度。"""
import test_kb_fast_strict_v24 as core
from types import SimpleNamespace
from unittest.mock import patch


def run_mocked_reply(reply, chunk_width=1, mode="fast"):
    class StreamingModel(core.FakeModel):
        def stream(self, prompts):
            self.prompts.append(prompts)
            for i in range(0, len(reply), chunk_width):
                yield SimpleNamespace(content=reply[i:i+chunk_width])

    with patch.object(core, "FakeModel", StreamingModel):
        return core.execute(mode)


def test_model_reference_list_is_filtered_even_one_char_per_chunk():
    model_reply = (
        "变量消去通过因子求和计算边际概率 [1]。"
        "\n\n*⚡ 快速模式：本次未执行独立证据校验，请核对引用原文。*"
        "\n\n参考资料："
        "\n\n[1] 机器学习.md / 旧的模型自带引用"
    )
    text, trace, model, stdout = run_mocked_reply(model_reply, 1, "fast")
    assert text.count("参考资料：") == 1, text
    assert text.count("快速模式：本次未执行独立证据校验") == 1, text
    assert "旧的模型自带引用" not in text
    assert "\\n\\n" not in text  # 严禁字面量转义泄露到 UI
    assert "*⚡ 快速模式" in text
    assert "[RAG生成] 已过滤模型自行追加" in stdout
    assert trace["evidence_coverage"]["decision"] == "unchecked"


def test_model_only_references_also_filtered():
    for width in (1, 4, 13, 200):
        model_reply = (
            "变量消去的基本思路是求和消去非目标变量 [1]。"
            "\n\n参考资料："
            "\n\n[7] 编造的教材章节"
        )
        text, _, _, stdout = run_mocked_reply(model_reply, width, "fast")
        assert text.count("参考资料：") == 1
        assert "编造的教材章节" not in text
        assert "[1] 机器学习.md" in text


def test_short_answer_without_model_footer_retained():
    original_reply = "变量消去通过消去非目标变量计算边际概率 [1]。"
    for width in (1, 2, 100):
        text, _, _, stdout = run_mocked_reply(original_reply, width, "fast")
        assert original_reply in text
        assert text.count("参考资料：") == 1
        assert text.count("快速模式：本次未执行独立证据校验") == 1
        assert "[RAG生成] 已过滤" not in stdout


def test_strict_mode_model_footer_filtered_but_partial_refusal_remains():
    fake = ("变量消去利用变量独立性 [1]。"
            "\n\n参考资料：\n\n[42] 无效的模型生成章节")
    text, trace, _, stdout = run_mocked_reply(fake, 2, "strict")
    assert text.count("参考资料：") == 1
    assert "无效的模型生成章节" not in text
    assert "当前知识库证据不足" in text
    assert "快速模式：本次未执行独立证据校验" not in text
    assert trace["evidence_coverage"]["decision"] == "partial"


def test_plain_reference_word_in_prose_is_not_footer():
    phrase = "有关参考资料：可使用知识库引用 [1]。"
    text, _, _, _ = run_mocked_reply(phrase, 1, "fast")
    assert phrase in text
    assert text.count("参考资料：") == 2  # 正文中的字词 + 后附自动生成的引用


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print("通过：", test.__name__)
    print(f"✅ V2.4.1 流式尾注去重：{len(tests)} 组离线测试通过。")
