import importlib.util
import unittest
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "seed_medical_chroma.py"
spec = importlib.util.spec_from_file_location("medical_seed", TARGET)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SAMPLE = {
    "drug_id": "drug_abc",
    "drug_name": "阿托伐他汀钙片",
    "record_id": "sr_123",
    "source_url": "https://www.yaopinnet.com/huayao/hy1695u.htm",
    "authority_status": "third_party_unverified",
    "section_id": "section_1",
    "field": "indication",
    "text": "高胆固醇血症并有动脉粥样硬化危险的病人。",
}

class TestMedicalSeed(unittest.TestCase):
    def test_default_read_only_cli_has_apply_opt_in(self):
        code = TARGET.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--apply", action="store_true"', code)
        self.assertIn("if args.apply and not args.allow_cloud_embedding", code)
        self.assertIn("if not args.apply:", code)
        self.assertIn("return", code)
        self.assertNotIn("CREATE (", mod.CYPHER)
        self.assertNotIn("DELETE ", mod.CYPHER)

    def test_preserves_original_and_source_warning(self):
        documents = mod.build_documents([SAMPLE], max_records=1)
        self.assertEqual(len(documents), 1)
        contents = documents[0]["markdown"]
        self.assertIn("阿托伐他汀钙片", contents)
        self.assertIn("动脉粥样硬化危险", contents)
        self.assertIn("来源记录ID：sr_123", contents)
        self.assertIn(SAMPLE["source_url"], contents)
        self.assertIn("未经医学权威核验", contents)

    def test_deduplicate_section_ids(self):
        item = dict(SAMPLE)
        documents = mod.build_documents([item, dict(item)], max_records=1)
        self.assertEqual(documents[0]["markdown"].count(item["text"]), 1)
        self.assertEqual(documents[0]["sections"], 1)

    def test_same_record_fields_form_single_document(self):
        second = dict(SAMPLE, field="contraindications", section_id="section_2",
                      text="若有已知过敏反应应核对原始资料。")
        docs = mod.build_documents([SAMPLE, second])
        self.assertEqual(len(docs), 1)
        self.assertIn("## 适应症", docs[0]["markdown"])
        self.assertIn("## 禁忌", docs[0]["markdown"])
        self.assertEqual(docs[0]["sections"], 2)

    def test_ids_stable_and_records_separated(self):
        second = dict(SAMPLE, record_id="sr_456")
        docs = mod.build_documents([SAMPLE, second], max_records=2)
        self.assertEqual(len(docs), 2)
        self.assertNotEqual(docs[0]["filename"], docs[1]["filename"])
        self.assertEqual(docs[0]["filename"],
                         mod.build_documents([SAMPLE])[0]["filename"])

    def test_unknown_field_not_embedded(self):
        self.assertEqual(mod.build_documents([dict(SAMPLE, field="secret")]), [])

    def test_truncation_is_disclosed(self):
        long = dict(SAMPLE, text="原" * 400)
        doc = mod.build_documents([long], max_chars_per_field=200)[0]["markdown"]
        self.assertIn("仅导入了该字段的前一部分原文", doc)
        self.assertLess(doc.count("原"), 400)


if __name__ == "__main__":
    unittest.main()
