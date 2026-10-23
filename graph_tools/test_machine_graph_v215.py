# -*- coding: utf-8 -*-
"""V2.15 reproducible evidence-gate and cross-book lookup tests (offline)."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from .verify_machine_graph_v215 import (
    HERE,SRC,SNAPSHOT,OLD_GRAPH,OLD_LINKS,OLD_ALIASES,SPECS,
    GRAPH_REL,LINK_REL,INDEX_REL,REPORT_REL,STATUS_NEW,
    create_artifacts,verify_only,sha_file,
)
from .probe_machine_multisource_v215 import run_query

class MachineExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=create_artifacts()
        cls.report=cls.data[REPORT_REL]
        cls.links=cls.data[LINK_REL]
        cls.graph=cls.data[GRAPH_REL]
        cls.index=cls.data[INDEX_REL]

    def test_existing_old_baseline_unchanged(self):
        self.assertEqual(self.report["baseline_verified_claims"],29)
        self.assertEqual(self.report["snapshot_total_chunk_count"],1953)
        self.assertEqual(self.report["new_source_snapshot_chunk_count"],895)
        self.assertEqual(sha_file(OLD_GRAPH),self.report["baseline_graph_sha256"])
        self.assertEqual(sha_file(OLD_LINKS),self.report["baseline_links_sha256"])

    def test_exact_gate_and_count(self):
        self.assertEqual(self.report["candidate_count"],55)
        self.assertEqual(self.report["passed_exact_quote_section_unique_chunk"],54)
        self.assertEqual(self.report["rejected_count"],1)
        self.assertEqual(self.report["new_evidence_distinct_chunk_ids"],23)
        self.assertEqual(self.report["merged_claim_count_for_offline_analysis"],83)
        self.assertEqual(self.report["merged_distinct_bound_chunk_ids"],41)

    def test_dont_choose_arbitrary_overlapping_chunks(self):
        rejected=self.report["rejected_candidates"]
        self.assertEqual(len(rejected),1)
        self.assertEqual(rejected[0]["subject"],"高斯核")
        self.assertEqual(rejected[0]["object"],"RBF核")
        self.assertEqual(len(rejected[0]["matching_native_chunk_ids"]),2)
        self.assertFalse(any(c["subject"]=="高斯核"
                             for c in self.links["claims"]))

    def test_no_unverified_claim_online(self):
        self.assertFalse(self.graph["metadata"]["online_eligible"])
        self.assertTrue(all(c["status"]==STATUS_NEW and not c["online_eligible"]
                            for c in self.links["claims"]))
        self.assertTrue(all(not c["online_eligible_in_new_v215"]
                            for c in self.index["claims"]))

    def test_provenance_and_qualifiers(self):
        ml=[c for c in self.links["claims"] if c["subject"]=="吉布斯采样"
            and c["predicate"]=="SPECIAL_CASE_OF"]
        self.assertEqual(len(ml),1)
        self.assertEqual(ml[0]["qualifier"],"有时被视为")
        crf=[c for c in self.links["claims"] if
             c["predicate"]=="CONDITIONAL_FORM_OF"]
        self.assertEqual(len(crf),1)
        self.assertEqual(crf[0]["qualifier"],"可看作")
        self.assertTrue(all(len(c["chunk_ids"])==1 and
                        c["binding_checks"]["exact_contiguous_quote_in_snapshot_chunk"]
                        for c in self.links["claims"]))

    def test_canonical_graph_link_checksums(self):
        from .verify_machine_graph_v215 import json_bytes,sha_bytes
        self.assertEqual(self.links["metadata"]["graph_sha256"],
                         sha_bytes(json_bytes(self.graph)))
        self.assertEqual(self.index["metadata"]["new_links_sha256"],
                         sha_bytes(json_bytes(self.links)))
        self.assertEqual(self.index["metadata"]["new_graph_sha256"],
                         sha_bytes(json_bytes(self.graph)))

    def test_full_release_is_reproducible_byte_for_byte(self):
        result=verify_only()
        self.assertEqual(result["passed_exact_quote_section_unique_chunk"],54)

    def test_cross_source_label_candidates_not_claimed_synonyms(self):
        self.assertEqual(self.report["cross_source_exact_label_candidate_count"],8)
        names=set(self.report["cross_source_exact_labels"])
        self.assertIn("MCMC",names)
        self.assertIn("马尔可夫随机场",names)
        self.assertNotIn("贝叶斯网",names)
        for bridge in self.index["cross_source_exact_label_candidates"]:
            self.assertEqual(bridge["status"],
                             "exact_label_lookup_only_not_claimed_equivalent")
            self.assertEqual(len(bridge["source_files"]),2)

    def test_no_legacy_source_changes_on_build(self):
        paths=[SRC,SNAPSHOT,OLD_GRAPH,OLD_LINKS,OLD_ALIASES]
        before={str(p):sha_file(p) for p in paths}
        _=create_artifacts()
        after={str(p):sha_file(p) for p in paths}
        self.assertEqual(before,after)

    def test_pilot_matches_both_sources(self):
        result=run_query(self.index,"吉布斯采样和MCMC有什么关系？")
        self.assertGreater(result["sources"].get("机器学习.md",0),0)
        self.assertGreater(result["sources"].get("神经网络与深度学习.md",0),0)
        self.assertTrue(result["no_online_promotion"])

    def test_longest_query_label_prevents_generic_substring_noise(self):
        # Mentioning 吉布斯采样 should not activate the generic 采样 term
        # solely because the latter is a substring of the former.
        result=run_query(self.index,"吉布斯采样和MCMC有什么关系？")
        self.assertIn("吉布斯采样",result["matched_labels"])
        self.assertIn("MCMC",result["matched_labels"])
        self.assertNotIn("采样",result["matched_labels"])

    def test_changed_textbook_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            altered=Path(d)/"机器学习.md"
            altered.write_bytes(SRC.read_bytes()+b"\nchanged")
            with self.assertRaises(ValueError):
                create_artifacts(source_path=altered)

if __name__=="__main__":
    unittest.main()
