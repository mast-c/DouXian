# -*- coding: utf-8 -*-
"""P3-04.2 offline fake Chroma / Embedding tests; never touch real user DB or cloud."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from medical_p3041_medical_json_prepare_v117 import prepare
from medical_p3041_verify_staging_v117 import verify
from medical_p3042_disease_chroma_index_v117 import (
    ImportBlocked, ImportFileLock, DiseaseCheckpoint, _cli, _identity, _as_vectors,
    _get_rows, _list_collection_names, apply_job, audit_complete_job,
    check_stored_chunk, disease_metadata, get_or_create_disease_collection,
    main, partition_existing, rag_fingerprint, read_checkpoint_status,
    read_next_rows, validate_inputs, validate_staging_row,
    DISEASE_COLLECTION
)


def sample_record(oid, name):
    return {
        "_id":{"$oid":oid},
        "name":name,
        "desc":name+"是模拟疾患。用于验证疾病原文来源，不是医疗建议。",
        "cause":"模拟病因一。模拟病因二。不能代表真实疾病临床结论。",
        "prevent":"模拟预防内容。非正式预防指南。",
        "symptom":["测试症状一","测试症状二","不完整的症状..."],
        "get_prob":"100%",
        "cured_prob":"100%",
        "recommand_drug":["模拟药物甲（不得入库为疗效）"],
        "common_drug":["模拟药物乙"],
    }


class FakeCollection:
    def __init__(self,name,metadata=None,records=None):
        self.name=name
        self.metadata=metadata or {}
        self.records=dict(records or {})
        self.add_calls=0
        self.fail_next_partial=False

    def get(self,ids=None,include=None,limit=None,offset=None):
        if ids is None:
            sorted_ids=sorted(self.records)
            selected=sorted_ids[(offset or 0):((offset or 0)+(limit or len(sorted_ids)))]
        else:
            selected=[v for v in ids if v in self.records]
        return {
            "ids":selected,
            "documents":[self.records[x][0] for x in selected],
            "metadatas":[self.records[x][1] for x in selected],
        }

    def count(self):
        return len(self.records)

    def add(self,*,ids,documents,metadatas,embeddings):
        self.add_calls+=1
        if self.fail_next_partial:
            self.fail_next_partial=False
            self.records[ids[0]]=(documents[0],dict(metadatas[0]))
            raise RuntimeError("injected simulated process crash after partial add")
        for id_,doc,meta,vec in zip(ids,documents,metadatas,embeddings):
            if id_ in self.records:
                raise ValueError("Existing ID should have been reused")
            if not vec:
                raise ValueError("Embedding missing")
            self.records[id_]=(doc,dict(meta))


class FakeClient:
    def __init__(self):
        self.objects={
            "rag":FakeCollection(
                "rag",metadata={"old":"drug"},
                records={
                    "drug-old-1":("drug text 1",{"origin":"drug","dose":"x"}),
                    "drug-old-2":("drug text 2",{"origin":"drug","dose":"y"}),
                }
            )
        }
        self.created=[]

    def list_collections(self):
        return list(self.objects.keys())

    def get_collection(self,*,name):
        return self.objects[name]

    def create_collection(self,*,name,metadata):
        if name in self.objects:
            raise ValueError("collection exists")
        obj=FakeCollection(name,metadata=metadata)
        self.objects[name]=obj
        self.created.append(name)
        return obj


class FakeEmbedding:
    def __init__(self):
        self.calls=0
        self.texts=[]

    def embed_documents(self,texts):
        self.calls+=1
        self.texts.extend(texts)
        return [[float(len(text)%101),0.25,0.125] for text in texts]


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.source=self.root/"medical.json"
        self.stage=self.root/"chunks.jsonl"
        self.prep=self.root/"prepare.json"
        self.audit=self.root/"audit.json"
        self.state=self.root/"disease-state.sqlite3"
        self.chroma_path=self.root/"chroma_db"
        self.chroma_path.mkdir()
        values=[
            sample_record("5bb578b6831b973a137e3ee6","模拟疾病甲"),
            sample_record("5bb578b6831b973a137e3ee7","模拟疾病乙"),
        ]
        self.source.write_text(
            "".join(json.dumps(value,ensure_ascii=False)+"\n"
                    for value in values),
            encoding="utf-8"
        )
        self.prepared=prepare(self.source,output=self.stage,
                              report_path=self.prep,chunk_size=850)
        self.audit.write_text(json.dumps(
            verify(self.source,self.stage,self.prep),ensure_ascii=False
        ),encoding="utf-8")
        self.count=self.prepared["index_ready_records"]
        self.client=FakeClient()
        self.embedding=FakeEmbedding()

    def tearDown(self):
        self.tmp.cleanup()

    def job(self,**changes):
        params=dict(
            source=self.source,chunks=self.stage,
            prepare_report=self.prep,audit_report=self.audit,
            chroma_path=self.chroma_path,state_path=self.state,
            embedding_model="synthetic-embedding-for-test",
            client=self.client,embedder=self.embedding,
            expect_chunks=self.count,expect_rag_count=2,
            max_new_chunks=3,embedding_batch=2,
            max_embed_chars=100000,max_passes=1,
            continuous=False,pause_seconds=0,
        )
        params.update(changes)
        return apply_job(**params)

    def finish(self):
        return self.job(max_new_chunks=10, max_passes=10,
                        continuous=True)

    def test_dry_run_no_checkpoint_or_chroma_writes(self):
        ret=main([
            "--source",str(self.source),"--chunks",str(self.stage),
            "--prepare-report",str(self.prep),"--audit-report",str(self.audit),
            "--expect-chunks",str(self.count),
            "--chroma-path",str(self.chroma_path),
            "--state",str(self.state),
        ])
        self.assertEqual(ret,0)
        self.assertFalse(self.state.exists())
        self.assertNotIn(DISEASE_COLLECTION,self.client.objects)

    def test_preconditions_and_status(self):
        data=validate_inputs(
            self.source,self.stage,self.prep,self.audit,
            expect_count=self.count)
        self.assertEqual(data["count"],self.count)
        self.assertEqual(read_checkpoint_status(self.state)["status"],
                         "NO_CHECKPOINT")

    def test_first_pass_only_disease_collection_added(self):
        before=rag_fingerprint(self.client.get_collection(name="rag"))
        result=self.job()
        self.assertEqual(result["status"],"PASS_COMPLETED_RESUMABLE")
        self.assertEqual(result["this_run_new_chunks"],3)
        self.assertEqual(result["checkpoint_status"]["done"],3)
        self.assertEqual(self.client.objects[DISEASE_COLLECTION].count(),3)
        self.assertEqual(rag_fingerprint(
            self.client.get_collection(name="rag")),before)
        self.assertEqual(self.client.created,[DISEASE_COLLECTION])
        self.assertEqual(read_checkpoint_status(self.state)["counts"]["done"],3)

    def test_continuous_completes_with_exact_full_audit(self):
        original=rag_fingerprint(self.client.objects["rag"])
        output=self.job(
            max_new_chunks=5,max_passes=10,continuous=True
        )
        self.assertEqual(output["status"],"ALL_IMPORTED_NEEDS_FULL_AUDIT")
        self.assertEqual(output["checkpoint_status"]["done"],self.count)
        self.assertEqual(self.client.objects[DISEASE_COLLECTION].count(),
                         self.count)
        self.assertEqual(rag_fingerprint(self.client.objects["rag"]),original)
        audit=audit_complete_job(
            source=self.source,chunks=self.stage,
            prepare_report=self.prep,audit_report=self.audit,
            chroma_path=self.chroma_path,state_path=self.state,
            client=self.client,expect_chunks=self.count,expect_rag_count=2
        )
        self.assertEqual(
            audit["status"],"PASS_FULL_DISEASE_CHROMA_PROVENANCE"
        )
        self.assertEqual(audit["verified_chunks"],self.count)
        self.assertFalse(audit["medical_accuracy_reviewed"])

    def test_resume_from_pass_does_not_reembed(self):
        first=self.job()
        self.assertEqual(first["this_run_new_chunks"],3)
        self.assertEqual(len(self.embedding.texts),3)
        second=self.job(max_new_chunks=10)
        self.assertEqual(second["this_run_new_chunks"],self.count-3)
        self.assertEqual(second["this_run_reused_chunks"],0)
        self.assertEqual(len(self.embedding.texts),self.count)

    def test_partial_chroma_write_crash_recovers_without_overwrite(self):
        self.job(max_new_chunks=2)
        disease=self.client.objects[DISEASE_COLLECTION]
        disease.fail_next_partial=True
        with self.assertRaisesRegex(RuntimeError,"simulated process crash"):
            self.job(max_new_chunks=4,embedding_batch=2)
        # 两条已完成 + 一条异常批次中先写入的 Chunk；checkpoint 未 done。
        self.assertEqual(disease.count(),3)
        self.assertGreaterEqual(
            read_checkpoint_status(self.state)["counts"].get("in_progress",0),2)
        again=self.finish()
        self.assertEqual(again["status"],"ALL_IMPORTED_NEEDS_FULL_AUDIT")
        self.assertEqual(disease.count(),self.count)
        self.assertEqual(again["this_run_reused_chunks"],1)

    def test_conflicting_existing_chunk_blocks_without_new_add(self):
        self.job(max_new_chunks=2)
        # 选择一个未来会被扫描到的 Chunk，在 Chroma 中事先伪造相同 ID。
        all_rows=[json.loads(x) for x in self.stage.read_text(
            encoding="utf-8").splitlines()]
        third=all_rows[2]
        disease=self.client.objects[DISEASE_COLLECTION]
        disease.records[third["id"]]=("tampered document",third["metadata"])
        calls_before=self.embedding.calls
        with self.assertRaisesRegex(ImportBlocked,"正文冲突"):
            self.job(max_new_chunks=4,embedding_batch=2)
        self.assertEqual(self.embedding.calls,calls_before)

    def test_wrong_existing_disease_metadata_blocks(self):
        self.job(max_new_chunks=1)
        disease=self.client.objects[DISEASE_COLLECTION]
        disease.metadata["source_sha256"]="invalid"
        with self.assertRaisesRegex(ImportBlocked,"模型/数据来源"):
            self.job()

    def test_external_rag_change_blocks_resume(self):
        self.job()
        self.client.objects["rag"].records["drug-old-1"]=(
            "tampered",{"origin":"drug","dose":"x"})
        before=self.client.objects[DISEASE_COLLECTION].count()
        with self.assertRaisesRegex(ImportBlocked,"rag 基线"):
            self.job()
        self.assertEqual(
            self.client.objects[DISEASE_COLLECTION].count(),before
        )

    def test_wrong_rag_count_blocks_before_state_creation(self):
        with self.assertRaisesRegex(ImportBlocked,"与预期"):
            self.job(expect_rag_count=52104)
        self.assertFalse(self.state.exists())
        self.assertNotIn(DISEASE_COLLECTION,self.client.objects)

    def test_wrong_source_sha_blocks_and_preserves_chroma(self):
        self.source.write_bytes(self.source.read_bytes()+b" ")
        with self.assertRaisesRegex(ImportBlocked,"medical.json SHA256"):
            self.job()
        self.assertFalse(self.state.exists())
        self.assertNotIn(DISEASE_COLLECTION,self.client.objects)

    def test_staging_file_tampered_blocks(self):
        self.stage.write_bytes(self.stage.read_bytes()+b" ")
        with self.assertRaisesRegex(ImportBlocked,"staging JSONL SHA256"):
            self.job()
        self.assertFalse(self.state.exists())

    def test_an_unreviewed_recommended_drug_never_enters_embeddings(self):
        self.finish()
        combined="".join(self.embedding.texts)
        self.assertNotIn("模拟药物甲",combined)
        self.assertNotIn("模拟药物乙",combined)
        self.assertNotIn("100%",combined)
        self.assertNotIn("不完整的症状...",combined)

    def test_exact_existing_chunk_is_reused_after_inprogress(self):
        self.job(max_new_chunks=2)
        all_rows=[json.loads(x) for x in self.stage.read_text(
            encoding="utf-8").splitlines()]
        third=all_rows[2]
        disease=self.client.objects[DISEASE_COLLECTION]
        disease.records[third["id"]]=(third["text"],third["metadata"])
        calls_before=self.embedding.calls
        result=self.job(max_new_chunks=1,embedding_batch=1)
        self.assertEqual(result["this_run_reused_chunks"],1)
        self.assertEqual(result["this_run_new_chunks"],0)
        self.assertEqual(self.embedding.calls,calls_before)

    def test_incomplete_collection_fails_full_audit(self):
        self.job(max_new_chunks=2)
        with self.assertRaisesRegex(ImportBlocked,"数量"):
            audit_complete_job(
                source=self.source,chunks=self.stage,
                prepare_report=self.prep,audit_report=self.audit,
                chroma_path=self.chroma_path,state_path=self.state,
                client=self.client,expect_chunks=self.count,expect_rag_count=2
            )

    def test_complete_but_corrupted_doc_fails_full_audit(self):
        self.finish()
        disease=self.client.objects[DISEASE_COLLECTION]
        cid=next(iter(disease.records))
        _,m=disease.records[cid]
        disease.records[cid]=("wrong",m)
        with self.assertRaisesRegex(ImportBlocked,"正文冲突"):
            audit_complete_job(
                source=self.source,chunks=self.stage,
                prepare_report=self.prep,audit_report=self.audit,
                chroma_path=self.chroma_path,state_path=self.state,
                client=self.client,expect_chunks=self.count,expect_rag_count=2
            )

    def test_checkpoint_identity_cannot_rebind_changed_model(self):
        self.job()
        with self.assertRaisesRegex(ImportBlocked,"检查点身份"):
            self.job(embedding_model="new-bad-model")

    def test_explicit_consent_required_for_apply(self):
        with self.assertRaisesRegex(ImportBlocked,"--allow-cloud-embedding"):
            _cli(["--apply","--source",str(self.source),
                  "--chunks",str(self.stage),
                  "--prepare-report",str(self.prep),
                  "--audit-report",str(self.audit)])

    def test_status_only_does_not_require_files(self):
        ret=main(["--status","--state",str(self.state)])
        self.assertEqual(ret,0)
        self.assertFalse(self.state.exists())

    def test_not_create_rag(self):
        del self.client.objects["rag"]
        with self.assertRaisesRegex(ImportBlocked,"缺少原有 rag"):
            self.job()
        self.assertFalse(self.state.exists())

    def test_invalid_embedding_dimensions_and_values_rejected(self):
        with self.assertRaises(ImportBlocked):
            _as_vectors([[0.1],[0.1,0.2]],2)
        with self.assertRaises(ImportBlocked):
            _as_vectors([[float("nan")]],1)
        with self.assertRaises(ImportBlocked):
            _as_vectors([],1)

    def test_wrong_state_path_does_not_overwrite_existing_state(self):
        self.job()
        before=self.state.read_bytes()
        wrong_source=self.source.with_name("other.json")
        wrong_source.write_bytes(self.source.read_bytes())
        with self.assertRaisesRegex(ImportBlocked,"原始文件名"):
            self.job(source=wrong_source)
        self.assertEqual(self.state.read_bytes(),before)

    def test_repeated_same_stage_import_is_idempotent(self):
        self.finish()
        prior=len(self.embedding.texts)
        results=self.finish()
        self.assertEqual(results["this_run_new_chunks"],0)
        self.assertEqual(len(self.embedding.texts),prior)
        self.assertEqual(results["status"],"ALL_IMPORTED_NEEDS_FULL_AUDIT")

    def test_second_concurrent_writer_is_rejected(self):
        with ImportFileLock(self.state):
            with self.assertRaisesRegex(ImportBlocked, "已有另一个"):
                self.job()
        self.assertFalse(self.state.exists())

    def test_forged_success_audit_with_tampered_original_offsets_rejected(self):
        # An attacker who updates both staging/plan SHA fields cannot fake a
        # PASS if the independently re-run provenance check finds wrong offsets.
        rows = [json.loads(x) for x in self.stage.read_text(
            encoding="utf-8").splitlines()]
        rows[0]["metadata"]["field_start"] += 1
        self.stage.write_text(
            "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),
            encoding="utf-8"
        )
        sha=hashlib.sha256(self.stage.read_bytes()).hexdigest()
        prep=json.loads(self.prep.read_text(encoding="utf-8"))
        prep["output_chunks_sha256"]=sha
        self.prep.write_text(json.dumps(prep,ensure_ascii=False),
                             encoding="utf-8")
        audit=json.loads(self.audit.read_text(encoding="utf-8"))
        audit["staging_sha256"]=sha
        self.audit.write_text(json.dumps(audit,ensure_ascii=False),
                              encoding="utf-8")
        with self.assertRaisesRegex(ImportBlocked,"独立原文回溯失败"):
            self.job()
        self.assertFalse(self.state.exists())

    def test_rag_document_metadata_hash_order_independent(self):
        rag=self.client.objects["rag"]
        before=rag_fingerprint(rag)
        rag.records={
            "drug-old-2":rag.records["drug-old-2"],
            "drug-old-1":rag.records["drug-old-1"]
        }
        self.assertEqual(rag_fingerprint(rag),before)


if __name__=="__main__":
    unittest.main()
