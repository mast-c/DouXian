import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "medical_chroma_acceptance.py"
spec = importlib.util.spec_from_file_location("medical_acceptance_target", PATH)
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)

DRUG = "一清胶囊"
CONTENT = (
    f"药品：{DRUG}\n栏目：适应症\n"
    "资料状态：第三方来源尚未完成医学权威核验\n"
    "证据原文：文字片段仅用于测试。"
)
SECTION = "前文。" + "文字片段仅用于测试。" + "后文。"

def make_meta():
    quote = "文字片段仅用于测试。"
    start = SECTION.index(quote)
    return {
        "chunk_id": "medical_v2_123",
        "data_origin": target.ORIGIN,
        "drug_id": "drug_1", "record_id": "record_1",
        "section_id": "section_1", "section_field": "indication",
        "drug_name": DRUG,
        "source_url": "https://example.org/medicine",
        "source_status": "third_party_unverified",
        "json_path": "indication",
        "section_sha256": hashlib.sha256(SECTION.encode()).hexdigest(),
        "content_sha256": hashlib.sha256(CONTENT.encode()).hexdigest(),
        "quote_start": start,
        "quote_end": start+len(quote),
        "forced_cut": False,
    }

class TestLocalChecks(unittest.TestCase):
    def test_valid_json_entry(self):
        self.assertEqual(target.inspect_chunk("medical_v2_123", CONTENT, make_meta()), [])

    def test_wrong_content_hash_is_detected(self):
        meta = make_meta()
        meta["content_sha256"] = "f" * 64
        self.assertTrue(any("正文 SHA256" in x for x in target.inspect_chunk("medical_v2_123", CONTENT, meta)))

    def test_bad_url_and_offsets_are_detected(self):
        meta = make_meta()
        meta["source_url"] = "javascript:alert(1)"
        meta["quote_end"] = meta["quote_start"]-3
        result = target.inspect_chunk("medical_v2_123", CONTENT, meta)
        self.assertTrue(any("来源 URL" in x for x in result))
        self.assertTrue(any("位置" in x for x in result))

    def test_missing_keys_detected(self):
        meta = make_meta()
        meta.pop("section_id")
        self.assertTrue(any("section_id" in x for x in target.inspect_chunk("medical_v2_123", CONTENT, meta)))

    def test_legacy_document_is_not_claimed_verified(self):
        report, samples, errors = target.audit_records([("legacy_1", "一般教材", {"source":"textbook.md"})],parent_sample_size=1)
        self.assertEqual(report["legacy_or_other"], 1)
        self.assertEqual(report["json_metadata_valid"], 0)
        self.assertFalse(samples)
        self.assertFalse(errors)

    def test_mixed_collection_counts(self):
        meta = make_meta()
        rows = [
          ("medical_v2_123", CONTENT, meta),
          ("legacy_1", "普通笔记", {"source":"old.md"}),
        ]
        report, samples, issues = target.audit_records(rows,parent_sample_size=2)
        self.assertEqual(report["scanned"], 2)
        self.assertEqual(report["json_structured_chunks"], 1)
        self.assertEqual(report["json_metadata_valid"], 1)
        self.assertEqual(report["drug_count"], 1)
        self.assertEqual(len(samples), 1)
        self.assertFalse(issues)

    def test_forced_cut_is_counted(self):
        meta = make_meta()
        meta["forced_cut"] = True
        report,_,issues = target.audit_records([("medical_v2_123", CONTENT, meta)])
        self.assertEqual(report["forced_cuts"],1)
        self.assertFalse(issues)

    def test_rejects_duplicate_ids(self):
        m1 = make_meta()
        report,_,issues = target.audit_records([
            ("medical_v2_123",CONTENT,m1),
            ("medical_v2_123",CONTENT,m1),
        ])
        self.assertTrue(any("重复" in x for x in issues))

    def test_new_metadata_copy_is_not_clinical_validation(self):
        meta = make_meta()
        report,_,issues = target.audit_records([("medical_v2_123",CONTENT,meta)])
        self.assertNotIn("clinically_approved", report)
        self.assertNotIn("treats_disease", report)
        self.assertFalse(issues)

    def test_tool_never_calls_embedding_or_writes(self):
        txt = PATH.read_text(encoding="utf-8")
        self.assertNotIn(".add_texts(",txt)
        self.assertNotIn(".upsert(",txt)
        self.assertNotIn(".delete(",txt)
        self.assertNotIn("embed_documents(",txt)
        self.assertNotIn(".get_or_create_collection(",txt)

if __name__=="__main__":
    unittest.main()
