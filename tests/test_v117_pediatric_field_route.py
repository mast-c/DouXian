# 仅使用标准库；不连接 Neo4j / Chroma / DashScope。
import unittest
from drug_graph_retriever import DrugGraphRetriever, fields_for_question


class FieldRoutingRegression(unittest.TestCase):
    def test_025_pediatric_self_medication_routes_four_sections(self):
        question = "儿童扁桃体炎可以自己服用丙酸交沙霉素颗粒吗？"
        self.assertEqual(fields_for_question(question), [
            "indication", "warnings", "contraindications", "dosage"
        ])

    def test_pediatric_dosage_only_remains_specific(self):
        self.assertEqual(fields_for_question("丙酸交沙霉素颗粒的小儿用法用量是什么？"),
                         ["dosage"])

    def test_regular_indication_question_unchanged(self):
        self.assertEqual(fields_for_question("阿托伐他汀钙片的适应症是什么？"),
                         ["indication"])

    def test_non_pediatric_self_medication_prioritizes_safety_without_dose(self):
        self.assertEqual(fields_for_question("我可以自己服用某某药吗？"), [
            "indication", "warnings", "contraindications"
        ])

    def test_existing_contraindication_question_unchanged(self):
        self.assertEqual(fields_for_question("丙酸交沙霉素颗粒的禁忌是什么？"),
                         ["contraindications"])

    def test_search_sections_passes_the_four_fields_to_neo4j(self):
        # 模拟已有 1 个 Drug、4 个 Section；验证路由与返回标识。
        retriever = object.__new__(DrugGraphRetriever)
        calls = []
        name = "丙酸交沙霉素颗粒"
        field_names = ("indication", "warnings", "contraindications", "dosage")
        sections = [
            {
                "drug_name": name,
                "record_id": "record-1",
                "source_url": "https://example.org/example",
                "authority_status": "third_party_unverified",
                "section_id": f"section-{field}",
                "section_field": field,
                "section_text": f"【模拟】{field}原文，仅用于路由测试。",
            }
            for field in field_names
        ]
        def fake_run(cypher, **params):
            calls.append(params)
            if "question" in params:
                return [{"id": "drug-1", "name": name}]
            return [row for row in sections if row["section_field"] in params["fields"]][
                :params["limit"]
            ]
        retriever._run = fake_run
        results = retriever.search_sections(
            "儿童扁桃体炎可以自己服用丙酸交沙霉素颗粒吗？",
            limit=4
        )
        self.assertEqual({x["section_field"] for x in results}, set(field_names))
        self.assertEqual(len(results), 4)
        self.assertEqual(calls[-1]["limit"], 4)
        self.assertTrue(all(x["record_id"] == "record-1" for x in results))
        self.assertTrue(all(x["source_status"] == "third_party_unverified" for x in results))


if __name__ == "__main__":
    unittest.main()
