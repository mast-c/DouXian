"""Offline tests for cleaned DrugKG Neo4j candidate import.

No Neo4j server needed; no live LLM. Use source mock fixtures.
"""
from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from kg import neo4j_drug_graph as m


class ImportSafetyTests(unittest.TestCase):
    def test_hash_id_deterministic_and_source_ids(self):
        first=m.hash_id("DiseaseCandidate",
                        "section_1277b916e2c0b4316802175a",
                        "动脉粥样硬化",142,148)
        self.assertEqual(first,"diseasecandidate_4d3ecf31ac9e4d6ce3086184")

    def test_import_is_not_approved_treats_relation(self):
        joined=" ".join(m.QUERIES.values())
        self.assertNotIn(":TREATS",joined)
        self.assertNotIn(":INDICATED_FOR",joined)
        self.assertNotIn("[:TREATS]",joined)
        self.assertNotIn("[:INDICATED_FOR]",joined)
        self.assertIn(":DiseaseCandidate",joined)
        self.assertIn(":ModelAssessment",joined)

    def test_existing_clinical_review_not_reset(self):
        self.assertIn("ON CREATE SET c.candidate_status='pending_human_review'",
                      m.QUERIES["disease_candidates"])
        self.assertIn("ON CREATE SET a.review_status=",
                      m.QUERIES["model_assessments"])
        self.assertNotIn("SET c.clinical_status='not_validated'",
                         m.QUERIES["disease_candidates"])

    def test_batches(self):
        self.assertEqual(list(m.batched(iter([1,2,3,4,5]),2)),
                         [[1,2],[3,4],[5]])

    def test_existing_wrong_graph_aborts(self):
        db=Mock()
        db.run.return_value.single.return_value={"old_drugs":12043}
        with self.assertRaisesRegex(RuntimeError,"outside cleaned V2"):
            m.ensure_isolated_dataset(db,"a"*64)
        db.run.assert_called_once()

    def test_isolated_graph_allows(self):
        db=Mock()
        db.run.return_value.single.return_value={"old_drugs":0}
        m.ensure_isolated_dataset(db,"a"*64)
        db.run.assert_called_once()

    def test_missing_preflight_report_failfast(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                m.preflight(td,Path(td)/"nothere.csv")

    def test_candidate_payload_preserves_spans(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            with (folder/"source_records.jsonl").open("w",encoding="utf-8") as f:
                f.write(json.dumps({"id":"source_record_1",
                                    "drug_id":"drug_1"},ensure_ascii=False)+"\n")
            cs=folder/"candidates.csv"
            quote="用于Ⅱ型糖尿病"
            fields=list(m.REQUIRED_CANDIDATE_COLUMNS)
            row={
                "candidate_id":"diseasecandidate_fake",
                "drug_name":"example_drug",
                "disease_name":"Ⅱ型糖尿病",
                "source_record_id":"source_record_1",
                "section_id":"section_fake",
                "source_url":"https://example.invalid/data",
                "evidence_quote":quote,
                "quote_start":0,
                "quote_end":len(quote),
                "disease_start":2,
                "disease_end":2+len("Ⅱ型糖尿病"),
                "context_flag":"manual_semantic_review",
                "review_status":"pending","relation_type":""
            }
            with cs.open("w",encoding="utf-8-sig",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields)
                w.writeheader()
                w.writerow(row)
            payload=list(m.each_payload(
                "disease_candidates",folder,cs,None,"x"*64
            ))
            self.assertEqual(len(payload),1)
            self.assertEqual(payload[0]["disease_start"],2)
            self.assertEqual(payload[0]["disease_name"],"Ⅱ型糖尿病")
            self.assertEqual(payload[0]["drug_id"],"drug_1")

    def test_uses_merged_60_as_model_assessment_only(self):
        test_records={
            "disease_candidates": m.QUERIES["disease_candidates"],
            "model_assessments": m.QUERIES["model_assessments"]
        }
        self.assertNotIn("MERGE (d)-[:INDICATED_FOR]",str(test_records))
        self.assertIn("ASSESSES",m.QUERIES["model_assessments"])
        self.assertIn("candidate_status='pending_human_review'",
                      m.QUERIES["disease_candidates"])

    def test_correct_alias_cypher(self):
        self.assertIn("ALIAS_CANDIDATE_OF",m.QUERIES["drug_aliases"])
        self.assertIn("ON CREATE SET a.review_status='candidate'",
                      m.QUERIES["drug_aliases"])

if __name__=="__main__":
    unittest.main()
