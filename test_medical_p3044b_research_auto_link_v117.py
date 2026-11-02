# -*- coding: utf-8 -*-
"""P3-04.4B fully offline tests: fake Neo4j, synthetic sources and reports.

No actual Neo4j/Chroma/SQLite/medical provider/network calls.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import medical_p3044b_research_auto_link_v117 as impl


def sample(oid:str,name:str):
    return {
        "_id":{"$oid":oid},
        "name":name,
        "desc":"纯工程测试，不能作为疾病医学事实",
        "cause":"示例来源",
        "symptom":["虚构症状"],
        "prevent":"示例",
    }


class Result(list):
    def single(self):
        return self[0] if self else None


class FakeTx:
    def __init__(self,graph):
        self.graph=graph

    def run(self,query,**args):
        records=[]
        if query==impl.COUNT_TERMS:
            return Result([{"n":len(self.graph.terms)}])
        if query==impl.READ_TERMS:
            for term in args["ids"]:
                for name in self.graph.terms.get(term,[]):
                    records.append({"term_id":term,"disease_name":name})
        elif query==impl.READ_DOCS:
            for doc in args["ids"]:
                for meta in self.graph.docs.get(doc,[]):
                    records.append({
                        "document_id":doc, "properties":deepcopy(meta)
                    })
        elif query==impl.READ_LINKS:
            for doc in args["ids"]:
                for term,props in self.graph.links.get(doc,[]):
                    records.append({
                        "document_id":doc,
                        "term_id":term,
                        "properties":deepcopy(props)
                    })
        elif query==impl.CREATE_LINK:
            did=args["doc_id"]
            term=args["term_id"]
            name=args["name"]
            if self.graph.block_create:
                raise RuntimeError("simulated database write failure")
            if len(self.graph.terms.get(term,[]))!=1:
                return Result()
            if self.graph.terms[term][0]!=name or self.graph.docs.get(did):
                return Result()
            self.graph.docs[did]=[deepcopy(args["document_properties"])]
            self.graph.links[did]=[(
                term,deepcopy(args["relationship_properties"])
            )]
            self.graph.created+=1
            records.append({"created_document_id":did})
        else:
            raise AssertionError("Unexpected query / possible database mutation")
        return Result(records)


class FakeSession:
    def __init__(self,graph):
        self.graph=graph
    def __enter__(self):
        return self
    def __exit__(self,*args):
        return False
    def execute_read(self,callback):
        return callback(FakeTx(self.graph))
    def execute_write(self,callback):
        # Simulates Neo4j transaction atomicity / failure rollback.
        copy=deepcopy(self.graph)
        result=callback(FakeTx(copy))
        self.graph.docs=copy.docs
        self.graph.links=copy.links
        self.graph.created=copy.created
        return result


class FakeGraph:
    def __init__(self,terms):
        self.terms=deepcopy(terms)
        self.docs={}
        self.links={}
        self.created=0
        self.block_create=False
    def session(self,*,database):
        assert database=="neo4j"
        return FakeSession(self)
    def close(self):
        pass


class ResearchLinkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.home=Path(self.temp.name)
        self.src=self.home/"medical.json"
        self.plan=self.home/"candidates.jsonl"
        self.report=self.home/"graph_plan.json"
        self.audit=self.home/"p3042_final_audit.json"
        self.records=[
            sample("5bb578b6831b973a137e3ee1","虚构疾病甲"),
            sample("5bb578b6831b973a137e3ee2","虚构疾病乙"),
            sample("5bb578b6831b973a137e3ee3","虚构疾病丙"),
            sample("5bb578b6831b973a137e3ee4","未匹配疾病丁"),
        ]
        self.src.write_text(
            "".join(json.dumps(row,ensure_ascii=False)+"\n"
                    for row in self.records),encoding="utf-8"
        )
        self.src_sha=hashlib.sha256(self.src.read_bytes()).hexdigest()
        raw_lines=self.src.read_bytes().splitlines()
        self.candidates=[]
        self.terms={}
        for line_no,item in enumerate(self.records[:3],1):
            oid=item["_id"]["$oid"]
            term="diseaseterm_"+oid
            self.terms[term]=[item["name"]]
            self.candidates.append({
                "chroma_collection":"disease_knowledge",
                "clinical_verified":False,
                "disease_name":item["name"],
                "disease_term_id":term,
                "document_id":"medical_json:"+oid,
                "graph_document_label":"DiseaseDocument",
                "medical_authority_verified":False,
                "name_alignment":"exact_name_only",
                "proposed_relationship":impl.RELATIONSHIP,
                "review_status":"PENDING_HUMAN_REVIEW",
                "source_file":"medical.json",
                "source_file_sha256":self.src_sha,
                "source_line_number":line_no,
                "source_line_sha256":hashlib.sha256(
                    raw_lines[line_no-1]).hexdigest(),
                "source_publisher":"",
                "source_record_oid":oid,
                "source_status":"third_party_unverified",
                "source_url":"",
                "treatment_relation":False,
                "write_to_neo4j":False,
            })
        self.plan.write_text(
            "".join(json.dumps(row,ensure_ascii=False,sort_keys=True,
                               separators=(",",":"))+"\n"
                    for row in self.candidates),encoding="utf-8"
        )
        self.plan_sha=hashlib.sha256(self.plan.read_bytes()).hexdigest()
        self.audit.write_text(json.dumps({
            "stage":"P3-04.2",
            "mode":"read_only_full_chroma_source_audit",
            "status":"PASS_FULL_DISEASE_CHROMA_PROVENANCE",
            "verified_chunks":39443,
            "disease_collection":"disease_knowledge",
            "source_status":"third_party_unverified",
            "source_sha256":self.src_sha,
            "staging_sha256":"3"*64,
            "medical_accuracy_reviewed":False,
        },ensure_ascii=False),encoding="utf-8")
        self.metadata={
            "stage":"P3-04.4A",
            "mode":"strict_read_only_graph_link_plan",
            "readiness":"READY_FOR_HUMAN_REVIEW_NO_WRITES",
            "graph_recheck":{
                "live_verified":True,
                "document_id_conflict_count":0,
                "neo4j_term_count":3,
            },
            "aligned_records_in_review_plan":3,
            "planned_document_links":3,
            "aligned_unique_names":3,
            "source_status":"third_party_unverified",
            "clinical_verified":False,
            "source_rights_verified":False,
            "human_review_done":False,
            "new_neo4j_nodes":0,
            "new_neo4j_relationships":0,
            "drug_treatment_relationships_created":0,
            "source_file_sha256":self.src_sha,
            "plan_sha256":self.plan_sha,
            "chroma_audited_chunk_count":39443,
            "chroma_audit_staging_sha256":"3"*64,
        }
        self.save_report()
        self.graph=FakeGraph(self.terms)
        self.config=patch.object(impl,"EXPECTED_SOURCE_SHA",self.src_sha)
        self.config.start()

    def tearDown(self):
        self.config.stop()
        self.temp.cleanup()

    def save_report(self):
        self.report.write_text(
            json.dumps(self.metadata,ensure_ascii=False),encoding="utf-8"
        )

    def validate(self,**kwargs):
        return impl.validate_plan(
            source=self.src,plan=self.plan,
            report=self.report,chroma_audit=self.audit,
            expected_candidates=kwargs.get("expected_candidates",3),
            expected_source_records=kwargs.get("expected_source_records",4),
            expected_graph_terms=kwargs.get("expected_graph_terms",3),
        )

    def graph_job(self,**kwargs):
        return impl.run_graph(
            self.graph,"neo4j",self.validate(),
            batch_size=kwargs.get("batch_size",2),
            apply=kwargs.get("apply",False),
            max_new=kwargs.get("max_new",20)
        )

    def test_plan_valid_unreviewed_metadata_is_accepted_in_research_mode(self):
        data=self.validate()
        self.assertEqual(len(data["rows"]),3)
        self.assertEqual(data["plan_sha256"],self.plan_sha)
        self.assertEqual(data["source_sha256"],self.src_sha)

    def test_graph_dry_run_does_not_modify(self):
        result=self.graph_job()
        self.assertEqual(result["status"],"DRY_RUN_GRAPH_CHECK_OK")
        self.assertEqual(result["before"]["MISSING"],3)
        self.assertEqual(self.graph.created,0)
        self.assertEqual(self.graph.docs,{})

    def test_pilot_creates_at_most_two(self):
        res=self.graph_job(apply=True,max_new=2)
        self.assertEqual(res["new_nodes_this_run"],2)
        self.assertEqual(res["new_relationships_this_run"],2)
        self.assertEqual(res["after"]["EXISTING_EXACT"],2)
        self.assertEqual(res["after"]["MISSING"],1)
        self.assertEqual(res["status"],"PARTIAL_RESEARCH_GRAPH_LINKS")

    def test_resume_completes_remaining_without_overwrite(self):
        self.graph_job(apply=True,max_new=2)
        snapshot=deepcopy(self.graph.docs)
        result=self.graph_job(apply=True,max_new=3)
        self.assertEqual(result["status"],"FULL_GRAPH_LINK_AUDIT_OK")
        self.assertEqual(result["new_nodes_this_run"],1)
        self.assertEqual(len(self.graph.docs),3)
        for key in snapshot:
            self.assertEqual(self.graph.docs[key],snapshot[key])

    def test_idempotent_replay_creates_no_new_objects(self):
        self.graph_job(apply=True)
        before=self.graph.created
        result=self.graph_job(apply=True)
        self.assertEqual(result["status"],"FULL_GRAPH_LINK_AUDIT_OK")
        self.assertEqual(result["new_nodes_this_run"],0)
        self.assertEqual(result["new_relationships_this_run"],0)
        self.assertEqual(self.graph.created,before)

    def test_treatment_relations_never_created(self):
        self.graph_job(apply=True)
        for links in self.graph.links.values():
            for _,props in links:
                self.assertEqual(props["is_treatment_relation"],False)
                self.assertEqual(props["research_only"],True)
                self.assertEqual(props["clinical_verified"],False)
        self.assertNotIn("TREATS",impl.RELATIONSHIP)

    def test_source_licensing_and_medical_status_preserved(self):
        self.graph_job(apply=True)
        for arr in self.graph.docs.values():
            props=arr[0]
            self.assertEqual(props["source_status"],"third_party_unverified")
            self.assertFalse(props["medical_authority_verified"])
            self.assertFalse(props["clinical_verified"])
            self.assertFalse(props["source_rights_verified"])
            self.assertEqual(props["review_status"],"NOT_CLINICALLY_REVIEWED")
            self.assertTrue(props["research_only"])
            self.assertEqual(props["source_url"],"")

    def test_only_metadata_and_stable_source_ids_are_written(self):
        self.graph_job(apply=True)
        for arr in self.graph.docs.values():
            self.assertEqual(set(arr[0]),impl.ALLOWED_WRITE_FIELDS)
            self.assertNotIn("desc",arr[0])
            self.assertNotIn("symptom",arr[0])
            self.assertTrue(arr[0]["id"].startswith("medical_json:"))

    def test_mutated_existing_document_id_blocks_before_writes(self):
        first=self.candidates[0]
        self.graph.docs[first["document_id"]]=[
            {"id":first["document_id"],"source_status":"trusted"}]
        self.graph.links[first["document_id"]]=[
            (first["disease_term_id"],{"clinical_verified":True})]
        with self.assertRaisesRegex(impl.LinkBlocked,"冲突"):
            self.graph_job(apply=True)
        self.assertEqual(self.graph.created,0)

    def test_orphan_preexisting_document_blocks(self):
        row=self.candidates[0]
        doc,link=impl.expected_props(row,self.plan_sha)
        self.graph.docs[row["document_id"]]=[doc]
        with self.assertRaisesRegex(impl.LinkBlocked,"孤立"):
            self.graph_job(apply=True)
        self.assertEqual(self.graph.created,0)

    def test_existing_wrong_disease_term_blocks(self):
        row=self.candidates[0]
        doc,link=impl.expected_props(row,self.plan_sha)
        self.graph.docs[row["document_id"]]=[doc]
        self.graph.links[row["document_id"]]=[("another_term",link)]
        with self.assertRaisesRegex(impl.LinkBlocked,"冲突"):
            self.graph_job(apply=True)
        self.assertEqual(self.graph.created,0)

    def test_disease_term_name_modified_blocks(self):
        first=next(iter(self.graph.terms))
        self.graph.terms[first]=["different disease"]
        with self.assertRaisesRegex(impl.LinkBlocked,"名称不符"):
            self.graph_job(apply=True)
        self.assertEqual(self.graph.created,0)

    def test_disease_term_duplicated_blocks(self):
        first=next(iter(self.graph.terms))
        self.graph.terms[first]=[self.graph.terms[first][0]]*2
        with self.assertRaisesRegex(impl.LinkBlocked,"重复"):
            self.graph_job(apply=True)
        self.assertEqual(self.graph.created,0)

    def test_disease_term_count_changed_blocks(self):
        self.graph.terms["diseaseterm_"+"9"*24]=["extra"]
        with self.assertRaisesRegex(impl.LinkBlocked,"数量为"):
            self.graph_job(apply=True)
        self.assertEqual(self.graph.created,0)

    def test_atomic_batch_rolls_back_when_create_fails(self):
        self.graph.block_create=True
        with self.assertRaisesRegex(RuntimeError,"simulated database"):
            self.graph_job(apply=True)
        self.assertEqual(self.graph.created,0)
        self.assertEqual(self.graph.docs,{})

    def test_tampered_plan_hash_blocks(self):
        self.plan.write_bytes(self.plan.read_bytes()+b"X")
        with self.assertRaisesRegex(impl.LinkBlocked,"SHA256"):
            self.validate()

    def test_source_drift_blocks(self):
        self.src.write_bytes(self.src.read_bytes()+b" ")
        with self.assertRaisesRegex(impl.LinkBlocked,"SHA256"):
            self.validate()

    def test_line_number_tamper_blocks_even_with_updated_hash(self):
        rows=self.plan.read_text(encoding="utf-8").splitlines()
        item=json.loads(rows[0])
        item["source_line_number"]=999
        rows[0]=json.dumps(item,ensure_ascii=False)
        self.plan.write_text("\n".join(rows)+"\n",encoding="utf-8")
        self.metadata["plan_sha256"]=hashlib.sha256(
            self.plan.read_bytes()).hexdigest()
        self.save_report()
        with self.assertRaisesRegex(impl.LinkBlocked,"行号/哈希"):
            self.validate()

    def test_cannot_claim_human_reviewed_via_input_report(self):
        self.metadata["human_review_done"]=True
        self.save_report()
        with self.assertRaisesRegex(impl.LinkBlocked,"人工审核"):
            self.validate()

    def test_cannot_claim_clinically_verified_via_plan(self):
        rows=self.plan.read_text(encoding="utf-8").splitlines()
        obj=json.loads(rows[0])
        obj["clinical_verified"]=True
        rows[0]=json.dumps(obj,ensure_ascii=False)
        self.plan.write_text("\n".join(rows)+"\n",encoding="utf-8")
        self.metadata["plan_sha256"]=hashlib.sha256(
            self.plan.read_bytes()).hexdigest()
        self.save_report()
        with self.assertRaisesRegex(impl.LinkBlocked,"审核"):
            self.validate()

    def test_missing_p3042_audit_blocks(self):
        meta=json.loads(self.audit.read_text(encoding="utf-8"))
        meta["status"]="INCOMPLETE"
        self.audit.write_text(json.dumps(meta),encoding="utf-8")
        with self.assertRaisesRegex(impl.LinkBlocked,"Chroma"):
            self.validate()

    def test_previous_doc_conflicts_report_blocks(self):
        self.metadata["graph_recheck"]["document_id_conflict_count"]=1
        self.save_report()
        with self.assertRaisesRegex(impl.LinkBlocked,"冲突"):
            self.validate()

    def test_cli_needs_explicit_confirmation_for_writes(self):
        with self.assertRaisesRegex(impl.LinkBlocked,"科研实验写图"):
            impl.cli(["--source","raw","--plan","plan",
                      "--plan-report","report","--chroma-audit","audit",
                      "--apply"])

    def test_cli_offline_does_not_require_confirmation(self):
        args=impl.cli(["--source","raw","--plan","plan",
                       "--plan-report","report","--chroma-audit","audit",
                       "--offline-check"])
        self.assertFalse(args.apply)
        self.assertTrue(args.offline_check)

    def test_output_report_never_overwritten(self):
        target=self.home/"import_report.json"
        impl._save_report(target,{"status":"PASS"})
        before=target.read_bytes()
        with self.assertRaisesRegex(impl.LinkBlocked,"已存在"):
            impl._save_report(target,{"status":"SECOND"})
        self.assertEqual(target.read_bytes(),before)

    def test_lock_prevents_two_concurrent_jobs(self):
        with impl.ExclusiveResearchLock(self.plan):
            with self.assertRaisesRegex(impl.LinkBlocked,"另一个"):
                with impl.ExclusiveResearchLock(self.plan):
                    pass

    def test_no_unverified_source_content_sent_to_graph(self):
        self.graph_job(apply=True)
        written=json.dumps(self.graph.docs,ensure_ascii=False)
        self.assertNotIn("纯工程测试，不能作为疾病医学事实",written)

    def test_empty_or_duplicate_source_names_block(self):
        self.candidates[1]["disease_name"]=self.candidates[0]["disease_name"]
        self.plan.write_text(
            "".join(json.dumps(x,ensure_ascii=False)+"\n"
                    for x in self.candidates),encoding="utf-8"
        )
        self.metadata["plan_sha256"]=hashlib.sha256(
            self.plan.read_bytes()).hexdigest()
        self.save_report()
        with self.assertRaisesRegex(impl.LinkBlocked,"源疾病名称/行号/哈希"):
            self.validate()


if __name__=="__main__":
    unittest.main()
