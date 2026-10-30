# -*- coding: utf-8 -*-
import hashlib
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    # dataclasses refer to the module's sys.modules entry
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

chunker = load("medical_json_chunker")
probe = load("medical_hybrid_probe")

DRUG = "阿托伐他汀钙片"
URL = "https://www.yaopinnet.com/huayao/hy1695u.htm"
INDICATION = "1．原发性高胆固醇血症和混合性高脂血症。 2．高胆固醇血症并有动脉粥样硬化危险的病人。"


def row(field, text, record="source_1", section="section_1", url=URL):
    return {
        "drug_id": "drug_1",
        "drug_name": DRUG,
        "record_id": record,
        "section_id": section,
        "source_url": url,
        "source_status": "third_party_unverified",
        "field": field,
        "section_text": text,
    }


class TestSemanticChunker(unittest.TestCase):
    def test_short_indication_remains_one_entire_chunk(self):
        raw = chunker.extract_raw_chunks(
            DRUG, {"indication": INDICATION}, ("indication",),
            chunk_size=1000, overlap=200
        )
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0].quote, INDICATION)
        self.assertFalse(raw[0].forced_cut)

    def test_json_list_items_are_separate_chunks(self):
        vals = ["1．过敏者禁用", "2．活动性肝病禁用", "3．孕妇慎用"]
        raw = chunker.extract_raw_chunks(
            DRUG, {"contraindications": vals}, ("contraindications",),
            chunk_size=1000, overlap=200
        )
        self.assertEqual([item.quote for item in raw], vals)
        self.assertEqual([item.json_path for item in raw],
                         ["contraindications[0]", "contraindications[1]",
                          "contraindications[2]"])

    def test_long_text_windows_preserve_verbatim_and_limit(self):
        text = "每一段包含医学条件，不可直接改变含义。" * 190
        raw = chunker.extract_raw_chunks(
            DRUG, {"warnings": text}, ("warnings",),
            chunk_size=1000, overlap=200
        )
        self.assertGreater(len(raw), 2)
        self.assertEqual(raw[0].item_start, 0)
        self.assertEqual(raw[-1].item_end, len(text))
        for item in raw:
            self.assertEqual(item.quote, text[item.item_start:item.item_end])
            self.assertLessEqual(
                len(chunker.document_text(DRUG, "warnings", item.quote)), 1000
            )
        for earlier, later in zip(raw, raw[1:]):
            self.assertLess(later.item_start, earlier.item_end)

    def test_non_sentence_fallback_is_marked(self):
        raw = chunker.extract_raw_chunks(
            DRUG, {"warnings": "长" * 1300}, ("warnings",),
            chunk_size=1000, overlap=200
        )
        self.assertTrue(any(item.forced_cut for item in raw))

    def test_exact_section_align_and_metadata(self):
        raw = chunker.extract_raw_chunks(
            DRUG, {"indication": INDICATION}, ("indication",),
            chunk_size=1000, overlap=200
        )
        stext = "适应症：\n" + INDICATION
        linked, errors = chunker.align_to_neo4j(
            raw, [row("indication", stext)], URL
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(linked), 1)
        meta = chunker.make_chroma_record(
            linked[0], chunk_index=0, chunk_total=1,
            json_filename="drug_db_v2_cleaned.json"
        )
        self.assertEqual(stext[meta["metadata"]["quote_start"]:
                                meta["metadata"]["quote_end"]], INDICATION)
        self.assertEqual(meta["metadata"]["section_id"], "section_1")
        self.assertEqual(meta["metadata"]["section_field"], "indication")
        self.assertEqual(meta["metadata"]["source_url"], URL)
        self.assertTrue(all(isinstance(v, (str, int, float, bool))
                            for v in meta["metadata"].values()))
        record = row("indication", stext)
        verdict = probe.verify_hit(meta["metadata"], meta["text"], {
            "full_section": stext,
            "drug_name": DRUG, "field": "indication",
            "source_url": URL,
        })
        self.assertTrue(verdict["ok"], verdict)

    def test_wrong_source_url_fails_closed(self):
        raw = chunker.extract_raw_chunks(DRUG, {"indication": INDICATION},
                                          ("indication",), chunk_size=1000, overlap=200)
        linked, errors = chunker.align_to_neo4j(raw, [
            row("indication", INDICATION, url="https://wrong.example/entry")
        ], URL)
        self.assertFalse(linked)
        self.assertEqual(len(errors), 1)

    def test_ambiguous_matches_do_not_get_fake_section_id(self):
        raw = chunker.extract_raw_chunks(DRUG, {"indication": INDICATION},
                                          ("indication",), chunk_size=1000, overlap=200)
        linked, errors = chunker.align_to_neo4j(raw, [
            row("indication", INDICATION, section="s1"),
            row("indication", INDICATION, record="s2", section="s2"),
        ], URL)
        self.assertFalse(linked)
        self.assertEqual(len(errors), 1)

    def test_different_sha_or_offset_rejected(self):
        raw = chunker.extract_raw_chunks(DRUG, {"indication": INDICATION},
                                          ("indication",), chunk_size=1000, overlap=200)
        linked, _ = chunker.align_to_neo4j(raw, [row("indication", INDICATION)], URL)
        meta = chunker.make_chroma_record(linked[0], chunk_index=0, chunk_total=1,
                                          json_filename="drug_db_v2_cleaned.json")
        changed = dict(meta["metadata"], quote_start=1)
        self.assertFalse(probe.verify_hit(changed, meta["text"], {
            "full_section": INDICATION, "drug_name": DRUG,
            "field": "indication", "source_url": URL
        })["ok"])
        self.assertFalse(probe.verify_hit(meta["metadata"], meta["text"], {
            "full_section": INDICATION + "新增内容", "drug_name": DRUG,
            "field": "indication", "source_url": URL
        })["ok"])

    def test_chroma_record_id_is_stable(self):
        raw = chunker.extract_raw_chunks(DRUG, {"indication": INDICATION},
                                          ("indication",), chunk_size=1000, overlap=200)
        linked, _ = chunker.align_to_neo4j(raw, [row("indication", INDICATION)], URL)
        a = chunker.make_chroma_record(linked[0], chunk_index=0, chunk_total=1,
                                       json_filename="drug_db_v2_cleaned.json")
        b = chunker.make_chroma_record(linked[0], chunk_index=0, chunk_total=1,
                                       json_filename="drug_db_v2_cleaned.json")
        self.assertEqual(a, b)

    def test_preview_no_database_writes(self):
        code = (ROOT / "medical_json_index.py").read_text(encoding="utf-8")
        self.assertIn("if not args.verify_neo4j and not args.apply:", code)
        self.assertIn("if args.apply and not args.allow_cloud_embedding", code)
        self.assertIn("if not args.apply:", code)
        self.assertIn('if prior is None:', code)
        self.assertIn('chroma.add_texts(', code)
        self.assertNotIn('chroma.delete(', code)
        self.assertNotIn('CREATE (', code)
        self.assertNotIn('DETACH DELETE', code)


if __name__ == "__main__":
    unittest.main()
