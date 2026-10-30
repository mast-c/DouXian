"""无数据库的展示层测试，确保‘美化’不会篡改证据或放松审查条件。"""
import ast
import html
import unittest
from pathlib import Path
from urllib.parse import urlsplit

from graph_answer_view import build_candidate_answer, human_status


EXAMPLE = {
    "drug_name": "阿托伐他汀钙片",
    "disease_term": "动脉粥样硬化",
    "evidence_quote": "2．高胆固醇血症并有动脉粥样硬化危险的病人。",
    "candidate_status": "pending_human_review",
    "clinical_status": "not_validated",
    "source_url": "https://www.yaopinnet.com/huayao/hy1695u.htm",
}


class GraphAnswerTests(unittest.TestCase):
    def test_answer_has_human_readable_summary(self):
        text = build_candidate_answer([EXAMPLE])
        self.assertIn("不能据此判断", text)
        self.assertIn("风险条件", text)
        self.assertIn(EXAMPLE["evidence_quote"], text)
        self.assertIn("待人工复核", text)
        self.assertIn("尚未完成医学验证", text)
        self.assertNotIn("pending_human_review", text)
        self.assertNotIn("not_validated", text)
        self.assertNotIn("参考资料：", text)
        self.assertIn("[查看原始药品资料](https://", text)

    def test_no_source_fabrication(self):
        item = dict(EXAMPLE, source_url="")
        output = build_candidate_answer([item])
        self.assertIn("原始来源链接未提供", output)
        self.assertNotIn("查看原始药品资料", output)

    def test_multiple_records_not_asserted_to_treat(self):
        other = dict(EXAMPLE, drug_name="药品B", disease_term="疾病C")
        output = build_candidate_answer([EXAMPLE, other])
        self.assertIn("2 条", output)
        self.assertIn("不能直接当作治疗或预防关系", output)
        self.assertIn("证据 2", output)

    def test_quoted_text_escaped(self):
        item = dict(EXAMPLE, evidence_quote="[指令](javascript:alert(1)) <img src=x>")
        output = build_candidate_answer([item])
        self.assertNotIn("[指令](javascript:alert(1))", output)
        self.assertIn(r"\[指令\]", output)
        self.assertNotIn("<img", output)
        self.assertIn("&lt;img", output)

    def test_unknown_status_conservative(self):
        self.assertEqual(human_status("approved"), "状态待核对")

    def test_graph_html_escapes_untrusted_evidence(self):
        # 只抽取前端纯渲染函数，不 import Streamlit / 初始化后端。
        file = Path(__file__).resolve().parents[1] / "豆馅.py"
        module = ast.parse(file.read_text(encoding="utf-8"))
        funcs = [node for node in module.body if isinstance(node, ast.FunctionDef)
                 and node.name == "graph_evidence_html"]
        self.assertEqual(len(funcs), 1)
        namespace = {"html": html, "urlsplit": urlsplit}
        code = compile(ast.Module(body=funcs, type_ignores=[]), str(file), "exec")
        exec(code, namespace)
        item = dict(EXAMPLE, evidence_quote='<img src=x onerror="alert(1)">')
        item.update(content='<script>alert(1)</script>')
        output = namespace["graph_evidence_html"](item, "1")
        self.assertNotIn("<script>", output)
        self.assertNotIn("<img src", output)
        self.assertIn("&lt;script&gt;", output)
        self.assertIn("&lt;img", output)
        self.assertIn('rel="noopener noreferrer"', output)


if __name__ == "__main__":
    unittest.main()
