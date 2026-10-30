# -*- coding: utf-8 -*-
"""V1.5 离线验收：Fake Neo4j / Fake Chroma，不发模型请求。"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import medical_json_bulk_index as mod
from medical_json_chunker import DEFAULT_FIELDS

MEDICINE = "阿托伐他汀钙片"
URL = "https://www.yaopinnet.com/huayao/hy1695u.htm"
DRUG = {
    "indication": "1．高胆固醇血症。 2．高胆固醇血症并有动脉粥样硬化危险的病人。",
    "contraindications": [
        "1．对本品过敏者",
        "2．活动型肝病",
        "3．孕妇及围产期妇女",
    ],
    "adverse_reactions": "轻微不良反应示例。",
    "warnings": "注意事项示例。",
    "drug_interactions": "",
    "source": URL,
}


def neo4j_rows(drug=DRUG, name=MEDICINE) -> list[dict]:
    rows=[]
    for idx, field in enumerate(DEFAULT_FIELDS):
        value=drug.get(field)
        if not value:
            continue
        section = "\n".join(value) if isinstance(value,list) else value
        rows.append({
            "drug_id": "drug_test_1",
            "drug_name": name,
            "record_id": "record_test_1",
            "section_id": f"section_test_{idx}",
            "source_url": URL,
            "source_status": "third_party_unverified",
            "field": field,
            "section_text": section,
        })
    return rows


class FakeChroma:
    def __init__(self):
        self.docs={}
        self.calls=0
        self.fail_on_call=None

    def get(self, *, ids=None, include):
        ids = [ident for ident in (ids or []) if ident in self.docs]
        return {
            "ids": ids,
            "documents": [self.docs[ident][0] for ident in ids],
            "metadatas": [self.docs[ident][1] for ident in ids],
        }

    def add_texts(self, *, texts, metadatas, ids):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise ConnectionError("模拟 DashScope 服务不可用")
        for ident, text, metadata in zip(ids, texts, metadatas):
            if ident in self.docs:
                raise AssertionError("既有 Chroma ID 被重复写入")
            self.docs[ident]=(text, dict(metadata))
        return ids


def args(**overrides):
    values = dict(
        apply=True, verify_neo4j=False, retry_failed=False,
        fields=tuple(DEFAULT_FIELDS), limit_drugs=20, start_after="",
        drug=None, json=Path("drug_db_v2_cleaned.json"),
        max_new_chunks=100, max_embed_chars=100000,
        embedding_batch=8, pause_seconds=0.0, stop_on_write_error=True,
        allow_forced_cuts=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def state_identity(model="text-embedding-v3", fields=tuple(DEFAULT_FIELDS)):
    return mod.build_identity(
        path=Path("drug_db_v2_cleaned.json"), digest="snapshot-test",
        fields=fields, chunk_size=1000, overlap=200,
        collection="rag", persist_directory="./chroma_db",
        neo4j_uri="bolt://localhost:7687", database="neo4j",
        embedding_model=model,
    )


def run_silent(*a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.run_batch(*a, **kw)


class TestMedicalBulk(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state_path=Path(self.tmp.name)/"test_state.sqlite3"

    def state(self):
        obj=mod.Checkpoint(self.state_path,state_identity())
        self.addCleanup(obj.close)
        return obj

    def test_json_short_fields_preserved_and_aligned(self):
        records, report=mod.prepare_records(
            MEDICINE, DRUG, fields=tuple(DEFAULT_FIELDS),
            chunk_size=1000, overlap=200,
            rows=neo4j_rows(), json_filename="drug_db_v2_cleaned.json"
        )
        self.assertEqual(report["raw_chunks"], 6)
        self.assertEqual(report["matched"], 6)
        self.assertEqual(report["forced_cuts"],0)
        self.assertEqual(len(records),6)
        self.assertIn(DRUG["indication"],records[0]["text"])
        self.assertEqual(records[0]["metadata"]["section_field"],"indication")

    def test_full_batch_adds_six_and_checkpoints_done(self):
        db=self.state()
        fake=FakeChroma()
        report=run_silent(
            args(), drugs={MEDICINE:DRUG}, progress=db,
            fetch_rows=lambda *_:neo4j_rows(),chroma=fake,
            chunk_size=1000, overlap=200,collection="rag"
        )
        self.assertEqual(report["completed_drugs"],1)
        self.assertEqual(report["added_chunks"],6)
        self.assertEqual(len(fake.docs),6)
        self.assertEqual(db.get(MEDICINE),"done")

    def test_existing_six_reused_without_embedding(self):
        expected,_=mod.prepare_records(
            MEDICINE,DRUG,fields=tuple(DEFAULT_FIELDS),
            chunk_size=1000,overlap=200,
            rows=neo4j_rows(),json_filename="drug_db_v2_cleaned.json"
        )
        fake=FakeChroma()
        for record in expected:
            fake.docs[record["id"]] = (record["text"],record["metadata"])
        db=self.state()
        result=run_silent(
            args(),drugs={MEDICINE:DRUG},progress=db,
            fetch_rows=lambda *_:neo4j_rows(),chroma=fake,
            chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertEqual(result["added_chunks"],0)
        self.assertEqual(result["skipped_existing_chunks"],6)
        self.assertEqual(fake.calls,0)

    def test_second_batch_resumes_without_redo(self):
        db=self.state()
        chroma=FakeChroma()
        names={MEDICINE:DRUG}
        def fetch(*_):return neo4j_rows()
        first=run_silent(args(),drugs=names,progress=db,fetch_rows=fetch,
                         chroma=chroma,chunk_size=1000,overlap=200,collection="rag")
        second=run_silent(args(),drugs=names,progress=db,fetch_rows=fetch,
                          chroma=chroma,chunk_size=1000,overlap=200,collection="rag")
        self.assertEqual(first["added_chunks"],6)
        self.assertEqual(second["selected_drugs"],0)
        self.assertEqual(chroma.calls,1)

    def test_budget_guard_never_sends_unapproved_more(self):
        db=self.state()
        chroma=FakeChroma()
        result=run_silent(
            args(max_new_chunks=2),drugs={MEDICINE:DRUG},progress=db,
            fetch_rows=lambda *_:neo4j_rows(),chroma=chroma,
            chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertTrue(result["stopped_by_budget"])
        self.assertEqual(len(chroma.docs),0)
        self.assertIsNone(db.get(MEDICINE))

    def test_char_budget_guard(self):
        db=self.state()
        chroma=FakeChroma()
        result=run_silent(
            args(max_embed_chars=100),drugs={MEDICINE:DRUG},progress=db,
            fetch_rows=lambda *_:neo4j_rows(),chroma=chroma,
            chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertTrue(result["stopped_by_budget"])
        self.assertEqual(len(chroma.docs),0)

    def test_wrong_source_url_fails_closed(self):
        db=self.state()
        chroma=FakeChroma()
        bad=neo4j_rows()
        bad[0]["source_url"]="https://wrong.example/source"
        result=run_silent(
            args(),drugs={MEDICINE:DRUG},progress=db,
            fetch_rows=lambda *_:bad,chroma=chroma,
            chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertEqual(result["failed_drugs"],1)
        self.assertEqual(db.get(MEDICINE),"alignment_failed")
        self.assertEqual(chroma.calls,0)

    def test_ambiguous_source_section_fails_closed(self):
        db=self.state()
        chroma=FakeChroma()
        rows=neo4j_rows()
        rows.append(dict(rows[0], section_id="duplicate_section"))
        result=run_silent(
            args(),drugs={MEDICINE:DRUG},progress=db,
            fetch_rows=lambda *_:rows,chroma=chroma,
            chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertEqual(result["failed_drugs"],1)
        self.assertEqual(chroma.calls,0)

    def test_modified_parent_metadata_never_reused(self):
        db=self.state()
        expected,_=mod.prepare_records(
            MEDICINE,DRUG,fields=tuple(DEFAULT_FIELDS),
            chunk_size=1000,overlap=200,
            rows=neo4j_rows(),json_filename="drug_db_v2_cleaned.json"
        )
        chroma=FakeChroma()
        for r in expected:
            chroma.docs[r["id"]]=(r["text"],dict(r["metadata"]))
        changed=dict(expected[0]["metadata"],section_sha256="wrong_parent_sha")
        chroma.docs[expected[0]["id"]]=(expected[0]["text"],changed)
        result=run_silent(
            args(),drugs={MEDICINE:DRUG},progress=db,
            fetch_rows=lambda *_:neo4j_rows(),chroma=chroma,
            chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertEqual(result["failed_drugs"],1)
        self.assertEqual(db.get(MEDICINE),"chroma_conflict")
        self.assertEqual(chroma.calls,0)

    def test_partial_failure_resume_skips_prior_success(self):
        db=self.state()
        chroma=FakeChroma()
        chroma.fail_on_call=2
        first=run_silent(
            args(embedding_batch=2),drugs={MEDICINE:DRUG},progress=db,
            fetch_rows=lambda *_:neo4j_rows(),chroma=chroma,
            chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertEqual(first["failed_drugs"],1)
        self.assertEqual(len(chroma.docs),2)
        self.assertEqual(db.get(MEDICINE),"write_failed")
        chroma.fail_on_call=None
        next_result=run_silent(
            args(embedding_batch=2,retry_failed=True),
            drugs={MEDICINE:DRUG},progress=db,
            fetch_rows=lambda *_:neo4j_rows(),chroma=chroma,
            chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertEqual(next_result["added_chunks"],4)
        self.assertEqual(next_result["skipped_existing_chunks"],2)
        self.assertEqual(len(chroma.docs),6)
        self.assertEqual(db.get(MEDICINE),"done")

    def test_checkpoint_identity_includes_model_config(self):
        db=self.state()
        with self.assertRaises(RuntimeError):
            mod.Checkpoint(self.state_path,state_identity(model="another-model"))

    def test_empty_drug_no_write_and_logged(self):
        db=self.state()
        chroma=FakeChroma()
        item=dict(DRUG, indication="",contraindications=[],
                  adverse_reactions="",warnings="",drug_interactions="")
        result=run_silent(
            args(),drugs={MEDICINE:item},progress=db,
            fetch_rows=lambda *_:None,chroma=chroma,
            chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertEqual(result["no_chunks"],1)
        self.assertEqual(db.get(MEDICINE),"no_chunks")
        self.assertEqual(chroma.calls,0)

    def test_nonsemantic_forced_cut_is_blocked_before_neo4j_or_embedding(self):
        db=self.state()
        fake=FakeChroma()
        changed=dict(DRUG, warnings="连续长文本" * 200)
        def do_not_fetch(*_):
            raise AssertionError("禁止强制切分数据触发 Neo4j 或 Embedding")
        result=run_silent(
            args(),drugs={MEDICINE:changed},progress=db,
            fetch_rows=do_not_fetch,chroma=fake,
            chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertEqual(result["failed_drugs"],1)
        self.assertEqual(db.get(MEDICINE),"data_error")
        self.assertEqual(fake.calls,0)

    def test_preview_does_not_read_neo4j_or_write(self):
        chroma=FakeChroma()
        def cannot_call(*args):
            raise AssertionError("dry-run must not use graph")
        result=run_silent(
            args(apply=False),drugs={MEDICINE:DRUG},progress=None,
            fetch_rows=cannot_call,chroma=None,
            chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertEqual(result["completed_drugs"],1)
        self.assertEqual(result["planned_chunks"],6)
        self.assertFalse(self.state_path.exists())
        self.assertEqual(chroma.calls,0)

    def test_verify_neo4j_readonly_does_not_write_state(self):
        calls=[]
        def fetch(*arg):
            calls.append(arg)
            return neo4j_rows()
        report=run_silent(
            args(apply=False,verify_neo4j=True),
            drugs={MEDICINE:DRUG},progress=None,fetch_rows=fetch,
            chroma=None,chunk_size=1000,overlap=200,collection="rag"
        )
        self.assertEqual(report["completed_drugs"],1)
        self.assertEqual(len(calls),1)
        self.assertFalse(self.state_path.exists())

    def test_failed_drug_skipped_until_explicit_retry(self):
        db=self.state()
        db.set(MEDICINE,"alignment_failed",detail="mock")
        selected=mod.select_drug_names(
            {MEDICINE:DRUG},limit=3,start_after="",
            names=None,progress=db,retry_failed=False
        )
        self.assertEqual(selected,[])
        selected_again=mod.select_drug_names(
            {MEDICINE:DRUG},limit=3,start_after="",
            names=None,progress=db,retry_failed=True
        )
        self.assertEqual(selected_again,[MEDICINE])

    def test_named_subset_and_invalid_name(self):
        self.assertEqual(mod.select_drug_names(
            {MEDICINE:DRUG, "左氧氟沙星":DRUG},limit=4,
            start_after="",names=[MEDICINE],progress=None,
            retry_failed=False
        ),[MEDICINE])
        with self.assertRaises(ValueError):
            mod.select_drug_names(
                {MEDICINE:DRUG},limit=2,start_after="",
                names=["不存在的药品"],progress=None,retry_failed=False
            )

    def test_cli_requires_permission(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mod._args(["--json","test.json","--apply"])
        # --status is intentionally independent of JSON.
        status=mod._args(["--status", "--state", str(self.state_path)])
        self.assertTrue(status.status)

    def test_status_without_db_does_not_create_it(self):
        with contextlib.redirect_stdout(io.StringIO()):
            mod.checkpoint_summary(self.state_path)
        self.assertFalse(self.state_path.exists())


if __name__=="__main__":
    unittest.main()
