# -*- coding: utf-8 -*-
"""P3-04.4A: synthetic sources, fake Neo4j; every test is read-only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from medical_p3043_disease_neo4j_align_v117 import align_names
from medical_p3044_disease_graph_link_plan_v117 import (
    PlanBlocked, build_plan, collect_source, live_recheck,
    _save_exclusive, _get_alignment, _validate_disease_chroma_audit, main,
    READ_DOCUMENT_CONFLICTS, LINK_TYPE, DOC_LABEL, SOURCE_STATUS,
)


def fake_disease(oid:str,name:str) -> dict:
    return {
        "_id":{"$oid":oid},
        "name":name,
        "desc":f"虚构疾病{name}的工程测试简介。",
        "symptom":["虚构症状甲"],
        "cause":"工程测试内容，不是临床指导。",
        "prevent":"模拟数据，请不要用于医疗决策。",
    }


class LinkPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.src=self.root/"medical.json"
        self.rep=self.root/"p3043_alignment.json"
        self.links=self.root/"p3043_alignment_links.jsonl"
        self.chroma=self.root/"p3042_final_audit_20261102.json"
        self.oids=[
            "5bb578b6831b973a137e3ee1",
            "5bb578b6831b973a137e3ee2",
            "5bb578b6831b973a137e3ee3",
            "5bb578b6831b973a137e3ee4",
        ]
        self.source_rows=[
            fake_disease(self.oids[0],"高血压"),
            fake_disease(self.oids[1],"糖尿病"),
            fake_disease(self.oids[2],"重复疾病"),
            fake_disease(self.oids[3],"重复疾病"),
        ]
        self.src.write_text(
            "".join(json.dumps(x,ensure_ascii=False)+"\n"
                    for x in self.source_rows),encoding="utf-8"
        )
        self.hash=hashlib.sha256(self.src.read_bytes()).hexdigest()
        self.names={
            "高血压":[self.oids[0]],
            "糖尿病":[self.oids[1]],
            "重复疾病":[self.oids[2],self.oids[3]],
        }
        self.term_records=[
            {"term_name":"高血压","term_id":"term_hbp"},
            {"term_name":"重复疾病","term_id":"term_duplicate"},
        ]
        self.term_total=2
        self.emit_alignment()
        self.chroma.write_text(
            json.dumps({
                "stage":"P3-04.2",
                "mode":"read_only_full_chroma_source_audit",
                "status":"PASS_FULL_DISEASE_CHROMA_PROVENANCE",
                "disease_collection":"disease_knowledge",
                "verified_chunks":39443,
                "source_sha256":self.hash,
                "source_status":"third_party_unverified",
                "medical_accuracy_reviewed":False,
                "staging_sha256":"c"*64,
            },ensure_ascii=False),encoding="utf-8"
        )

    def tearDown(self):
        self.temp.cleanup()

    def fetch(self,batch):
        return [x for x in self.term_records if x["term_name"] in batch]

    def emit_alignment(self, term_records=None, names=None):
        terms=self.term_records if term_records is None else term_records
        mapping=self.names if names is None else names
        report,lines=align_names(
            mapping,lambda batch:[x for x in terms if x["term_name"] in batch],
            graph_total_terms=self.term_total
        )
        out="".join(
            json.dumps(line,ensure_ascii=False,sort_keys=True)+"\n"
            for line in lines
        )
        self.links.write_text(out,encoding="utf-8")
        report["source_file_sha256"]=self.hash
        report["links_sha256"]=hashlib.sha256(out.encode("utf-8")).hexdigest()
        report["links_count"]=len(lines)
        self.rep.write_text(json.dumps(report,ensure_ascii=False),encoding="utf-8")

    def build(self,*,offline=False,fetch_docs=None,
              expected_records=4,expected_names=3):
        return build_plan(
            source_file=self.src,
            align_report_file=self.rep,
            align_links_file=self.links,
            chroma_audit_file=self.chroma,
            fetch_terms=None if offline else self.fetch,
            count_terms=None if offline else lambda:self.term_total,
            fetch_existing_docs=fetch_docs,
            expected_records=expected_records,
            expected_names=expected_names,
        )

    def test_strict_complete_current_graph_readiness(self):
        report,plan,review=self.build(fetch_docs=lambda ids:[])
        self.assertEqual(report["readiness"],"READY_FOR_HUMAN_REVIEW_NO_WRITES")
        self.assertTrue(report["graph_recheck"]["live_verified"])
        self.assertEqual(report["original_disease_records"],4)
        self.assertEqual(report["unique_disease_names"],3)
        self.assertEqual(report["aligned_unique_names"],2)
        self.assertEqual(len(plan),3)
        self.assertEqual(len(review),4)
        self.assertEqual(report["new_neo4j_relationships"],0)
        self.assertEqual(report["drug_treatment_relationships_created"],0)

    def test_duplicate_name_stays_two_separate_documents(self):
        _,plan,_=self.build(fetch_docs=lambda ids:[])
        duplicate=[x for x in plan if x["disease_name"]=="重复疾病"]
        self.assertEqual(len(duplicate),2)
        self.assertNotEqual(duplicate[0]["document_id"],duplicate[1]["document_id"])
        self.assertEqual(
            duplicate[0]["disease_term_id"],
            duplicate[1]["disease_term_id"]
        )

    def test_proposed_relation_cannot_be_treatment(self):
        _,plan,_=self.build(fetch_docs=lambda ids:[])
        self.assertEqual(LINK_TYPE,
                         "EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA")
        self.assertEqual(DOC_LABEL,"DiseaseDocument")
        for x in plan:
            self.assertEqual(x["proposed_relationship"],LINK_TYPE)
            self.assertEqual(x["review_status"],"PENDING_HUMAN_REVIEW")
            self.assertFalse(x["treatment_relation"])
            self.assertFalse(x["clinical_verified"])
            self.assertFalse(x["write_to_neo4j"])
            self.assertEqual(x["source_status"],SOURCE_STATUS)
            self.assertEqual(x["source_url"],"")
            self.assertEqual(x["source_publisher"],"")

    def test_offline_never_claims_live_verified(self):
        report,plan,_=self.build(offline=True)
        self.assertEqual(
            report["readiness"],"OFFLINE_SNAPSHOT_ONLY_NOT_GRAPH_VERIFIED"
        )
        self.assertFalse(report["graph_recheck"]["live_verified"])
        self.assertEqual(len(plan),3)

    def test_no_document_conflict_check_is_not_ready(self):
        report,_,_=self.build()
        self.assertEqual(
            report["readiness"],"GRAPH_VERIFIED_BUT_DOCUMENT_COLLISIONS_NOT_CHECKED"
        )

    def test_existing_graph_document_ids_block_readiness(self):
        def existing(ids):
            return [{"document_id":i} for i in ids
                    if i.endswith(self.oids[0])]
        report,plan,_=self.build(fetch_docs=existing)
        self.assertEqual(
            report["readiness"],"EXISTING_DOCUMENT_IDS_REQUIRE_REVIEW"
        )
        self.assertEqual(
            report["graph_recheck"]["document_id_conflict_count"],1
        )
        self.assertEqual(len(plan),3)

    def test_duplicate_existing_document_id_rejected(self):
        def twice(ids):
            if any(i.endswith(self.oids[0]) for i in ids):
                id_="medical_json:"+self.oids[0]
                return [{"document_id":id_},{"document_id":id_}]
            return []
        with self.assertRaisesRegex(PlanBlocked,"多个 DiseaseDocument"):
            self.build(fetch_docs=twice)

    def test_wrong_graph_term_id_fails_closed(self):
        old=self.term_records
        self.term_records=[
            {"term_name":"高血压","term_id":"other_term"},
            {"term_name":"重复疾病","term_id":"term_duplicate"},
        ]
        with self.assertRaisesRegex(PlanBlocked,"映射与 P3-04.3"):
            self.build(fetch_docs=lambda ids:[])
        self.term_records=old

    def test_missing_term_fails_closed(self):
        self.term_records=[self.term_records[0]]
        with self.assertRaisesRegex(PlanBlocked,"映射与 P3-04.3"):
            self.build(fetch_docs=lambda ids:[])

    def test_added_other_graph_term_does_not_pass_changed_count(self):
        self.term_total+=1
        with self.assertRaisesRegex(PlanBlocked,"总数已变化"):
            self.build(fetch_docs=lambda ids:[])

    def _set_windows_crlf_alignment(self):
        """Simulate Path.write_text translating LF to CRLF on Windows."""
        source=self.links.read_bytes().replace(b"\r\n",b"\n")
        self.assertIn(b"\n",source)
        self.links.write_bytes(source.replace(b"\n",b"\r\n"))

    def test_windows_crlf_legacy_p3043_mapping_is_accepted(self):
        # P3-04.3 computed SHA from LF text, Windows wrote CRLF bytes.
        expected=json.loads(self.rep.read_text(encoding="utf-8"))[
            "links_sha256"
        ]
        self._set_windows_crlf_alignment()
        self.assertNotEqual(
            hashlib.sha256(self.links.read_bytes()).hexdigest(),expected
        )
        report,plan,review=self.build(fetch_docs=lambda ids:[])
        self.assertEqual(
            report["alignment_link_hash_mode"],"WINDOWS_CRLF_EQUIVALENT"
        )
        self.assertEqual(report["readiness"],"READY_FOR_HUMAN_REVIEW_NO_WRITES")
        self.assertEqual(len(plan),3)
        self.assertEqual(len(review),4)

    def test_windows_crlf_with_tampered_value_is_still_rejected(self):
        self._set_windows_crlf_alignment()
        modified=self.links.read_bytes().replace(
            b"term_hbp",b"term_forged",1
        )
        self.links.write_bytes(modified)
        with self.assertRaisesRegex(PlanBlocked,"映射文件 SHA256"):
            self.build(offline=True)

    def test_mixed_windows_and_unix_mapping_line_endings_are_rejected(self):
        normalized=self.links.read_bytes().replace(b"\r\n",b"\n")
        lines=normalized.splitlines(keepends=True)
        self.assertGreaterEqual(len(lines),2)
        mixed=lines[0].replace(b"\n",b"\r\n")+b"".join(lines[1:])
        self.links.write_bytes(mixed)
        with self.assertRaisesRegex(PlanBlocked,"映射文件 SHA256"):
            self.build(offline=True)

    def test_windows_crlf_source_report_stays_byte_identical(self):
        self._set_windows_crlf_alignment()
        previous=self.links.read_bytes()
        prev_report=self.rep.read_bytes()
        self.build(fetch_docs=lambda ids:[])
        self.assertEqual(self.links.read_bytes(),previous)
        self.assertEqual(self.rep.read_bytes(),prev_report)

    def test_staged_mapping_hash_tamper_rejected(self):
        self.links.write_bytes(self.links.read_bytes()+b"X")
        with self.assertRaisesRegex(PlanBlocked,"映射文件 SHA256"):
            self.build(offline=True)

    def test_source_drift_rejected(self):
        self.src.write_bytes(self.src.read_bytes()+b"\n")
        with self.assertRaisesRegex(PlanBlocked,"SHA256"):
            self.build(offline=True)

    def test_source_baseline_counts_rejected(self):
        with self.assertRaisesRegex(PlanBlocked,"预期"):
            self.build(offline=True,expected_records=8807,expected_names=8806)

    def test_missing_valid_p3042_chroma_audit_rejected(self):
        data=json.loads(self.chroma.read_text(encoding="utf-8"))
        data["status"]="PARTIAL"
        self.chroma.write_text(json.dumps(data),encoding="utf-8")
        with self.assertRaisesRegex(PlanBlocked,"完整 Chroma"):
            self.build(offline=True)

    def test_medical_status_promotion_rejected(self):
        rows=self.links.read_text(encoding="utf-8").splitlines()
        first=json.loads(rows[0])
        first["is_treatment_relation"]=True
        rows[0]=json.dumps(first,ensure_ascii=False,sort_keys=True)
        output="\n".join(rows)+"\n"
        self.links.write_text(output,encoding="utf-8")
        rep=json.loads(self.rep.read_text(encoding="utf-8"))
        rep["links_sha256"]=hashlib.sha256(output.encode()).hexdigest()
        self.rep.write_text(json.dumps(rep),encoding="utf-8")
        with self.assertRaisesRegex(PlanBlocked,"冒充医学"):
            self.build(offline=True)

    def test_bad_term_sharing_two_disease_names_rejected(self):
        rep=json.loads(self.rep.read_text(encoding="utf-8"))
        lines=self.links.read_text(encoding="utf-8").splitlines()
        links=[json.loads(line) for line in lines]
        for entry in links:
            entry["disease_term_id"]="term_same"
        text="".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n"
                     for x in links)
        self.links.write_text(text,encoding="utf-8")
        rep["links_sha256"]=hashlib.sha256(text.encode()).hexdigest()
        self.rep.write_text(json.dumps(rep),encoding="utf-8")
        with self.assertRaisesRegex(PlanBlocked,"共用 Term ID"):
            self.build(offline=True)

    def test_plan_outputs_all_new_and_no_overwrites(self):
        report,plan,review=self.build(fetch_docs=lambda ids:[])
        dest=self.root/"reports"
        report_path=dest/"plan_report.json"
        plan_path=dest/"relations.jsonl"
        review_path=dest/"review.csv"
        _save_exclusive(
            report_path,plan_path,review_path,report,plan,review
        )
        self.assertEqual(len(plan_path.read_text(
            encoding="utf-8").splitlines()),3)
        with review_path.open(encoding="utf-8-sig") as stream:
            import csv
            recs=list(csv.DictReader(stream))
        self.assertEqual(len(recs),4)
        saved=json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["planned_document_links"],3)
        self.assertEqual(
            saved["plan_sha256"],
            hashlib.sha256(plan_path.read_bytes()).hexdigest()
        )
        prior=plan_path.read_bytes()
        with self.assertRaisesRegex(PlanBlocked,"目标文件已存在"):
            _save_exclusive(
                report_path,plan_path,review_path,report,plan,review
            )
        self.assertEqual(plan_path.read_bytes(),prior)

    def test_no_graph_mutations_in_all_cypher(self):
        text=READ_DOCUMENT_CONFLICTS.upper()
        self.assertIn("MATCH",text)
        for forbidden in ("MERGE","CREATE","DELETE","DETACH","SET ",
                          "REMOVE ","DROP"):
            self.assertNotIn(forbidden,text)

    def test_cli_offline_no_db_connection_and_report(self):
        folder=self.root/"out"
        status=main([
            "--source",str(self.src),
            "--align-report",str(self.rep),
            "--align-links",str(self.links),
            "--chroma-audit",str(self.chroma),
            "--expect-records","4",
            "--expect-names","3",
            "--offline-preview",
            "--report",str(folder/"summary.json"),
            "--plan",str(folder/"plan.jsonl"),
            "--review-csv",str(folder/"review.csv"),
        ])
        self.assertEqual(status,0)
        report=json.loads(
            (folder/"summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            report["readiness"],"OFFLINE_SNAPSHOT_ONLY_NOT_GRAPH_VERIFIED"
        )
        self.assertEqual(report["new_neo4j_nodes"],0)
        self.assertEqual(report["new_neo4j_relationships"],0)
        self.assertEqual(report["planned_document_links"],3)

    def test_chroma_audit_wrong_clinical_status_fails(self):
        data=json.loads(self.chroma.read_text(encoding="utf-8"))
        data["medical_accuracy_reviewed"]=True
        self.chroma.write_text(json.dumps(data),encoding="utf-8")
        with self.assertRaisesRegex(PlanBlocked,"医学状态"):
            self.build(offline=True)

    def test_read_only_plan_does_not_modify_any_input(self):
        originals={
            path:hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (self.src,self.rep,self.links,self.chroma)
        }
        report,plan,review=self.build(fetch_docs=lambda ids:[])
        self.assertEqual(len(plan),3)
        self.assertEqual(
            originals,{
                path:hashlib.sha256(path.read_bytes()).hexdigest()
                for path in originals
            }
        )

    def test_ambiguous_graph_names_remain_unresolved(self):
        self.term_total=3
        self.term_records.append(
            {"term_name":"高血压","term_id":"term_hbp_2"}
        )
        self.emit_alignment()
        report,plan,review=self.build(fetch_docs=lambda ids:[])
        self.assertEqual(report["aligned_unique_names"],1)
        self.assertEqual(len(plan),2)
        self.assertEqual(report["prior_alignment_counts"][
            "ambiguous_or_missing_term_id"],1)
        row=next(x for x in review if x["disease_name"]=="高血压")
        self.assertEqual(
            row["technical_match_status"],
            "AMBIGUOUS_TERM_REQUIRES_REVIEW"
        )


if __name__=="__main__":
    unittest.main()
