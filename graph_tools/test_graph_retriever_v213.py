# -*- coding: utf-8 -*-
"""DouXian V2.13 offline regression tests; no Chroma server/DashScope requests."""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import graph_retriever_v213 as v
    from . import graph_chunk_linker_v211 as binder
except ImportError:
    import graph_retriever_v213 as v
    import graph_chunk_linker_v211 as binder


def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


@dataclass
class FakeDoc:
    page_content: str
    metadata: dict


class FakeChroma:
    def __init__(self, rows):
        self.rows={x["chunk_id"]:deepcopy(x) for x in rows}
        self.calls=0
        self.drift=False

    def get(self,ids,include):
        self.calls+=1
        docs=[self.rows[i] for i in reversed(ids) if i in self.rows]
        contents=[d["content"] for d in docs]
        if self.drift and contents:
            contents[0]+=" CORRUPTED"
        return {
            "ids":[d["chunk_id"] for d in docs],
            "documents":contents,
            "metadatas":[{
                "source":d["source"],
                "section_path":d["section_path"],
                "chunk_id":d["chunk_id"],
            } for d in docs],
        }


class FakeKB:
    def __init__(self):
        self.hybrid_result=[
            FakeDoc("Unrelated retrieved chunk",{
                "source":"AnotherBook.md",
                "section_path":"第1章 Introduction",
                "chunk_id":"HYBRID-0001",
            })
        ]
        self.chroma=FakeChroma([])
        self.rerank_calls=0

    def search_hybrid(self,query,k=5):
        return list(self.hybrid_result[:k])

    def search_with_rerank(self,query,recall_k=10,top_n=3):
        # This is the exact polymorphic interception point being tested.
        candidates=self.search_hybrid(query,k=recall_k)
        self.rerank_calls+=1
        return candidates[:top_n]


class V213Tests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)
        self.source=self.root/"TestBook.md"
        self.source.write_text(
            "# TestBook\n"
            "## 第1章 Test\n"
            "### 1.1 说明\n"
            "卷积层使用卷积核抽取图像特征\n"
            "汇聚层也叫子采样层\n",
            encoding="utf-8",
        )
        self.sec="第1章 Test / 1.1 说明"
        self.nodes=[
            {"id":"concept:cnv","label":"卷积层","type":"Concept",
             "aliases_for_search":["Convolutional Layer"]},
            {"id":"concept:knl","label":"卷积核","type":"Concept",
             "aliases_for_search":[]},
            {"id":"concept:pool","label":"汇聚层","type":"Concept",
             "aliases_for_search":["池化层"]},
            {"id":"concept:sub","label":"子采样层","type":"Concept",
             "aliases_for_search":[]},
        ]
        self.graph={
            "metadata":{
                "schema":"douxian-kg-2.7-offline-1",
                "source_file":self.source.name,
                "source_sha256":v._sha_file(self.source),
                "v212":{"phase":"promoted","schema":"douxian-kg-expansion-2.12-1"},
            },
            "nodes":self.nodes,
            "edges":[
                {
                    "id":"edge:conv",
                    "source":"concept:cnv",
                    "relation":"USES",
                    "target":"concept:knl",
                    "evidence":{
                        "source_file":self.source.name,
                        "section_path":self.sec,
                        "line":4,"text":"卷积层使用卷积核抽取图像特征",
                        "extraction":"exact_source_phrase",
                    },
                },
                {
                    "id":"edge:alias",
                    "source":"concept:pool",
                    "relation":"ALSO_CALLED",
                    "target":"concept:sub",
                    "evidence":{
                        "source_file":self.source.name,
                        "section_path":self.sec,
                        "line":5,"text":"汇聚层也叫子采样层",
                        "extraction":"exact_source_phrase",
                    },
                }
            ]
        }
        self.rows=[
            {
                "chunk_id":"CHUNK-CONV-01",
                "source":self.source.name,
                "section_path":"《TestBook》 / "+self.sec,
                "content":"章节："+self.sec+"\n卷积层使用卷积核抽取图像特征",
            },
            {
                "chunk_id":"CHUNK-POOL-02",
                "source":self.source.name,
                "section_path":"《TestBook》 / "+self.sec,
                "content":"章节："+self.sec+"\n汇聚层也叫子采样层",
            },
        ]
        self.graph_file=self.root/"graph.json"
        self.links_file=self.root/"links.json"
        self.snapshot=self.root/"snapshot.jsonl"
        self.alias_file=self.root/"alias.json"
        self.gate_file=self.root/"gate.json"
        write_json(self.graph_file,self.graph)
        self.snapshot.write_text(
            "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in self.rows),
            encoding="utf-8",
        )
        links=binder.build_link_report(self.graph,self.source,None,self.snapshot)
        self.assertEqual(links["metadata"]["status_counts"],{"verified":2})
        links["metadata"]["graph_sha256"]=v._sha_file(self.graph_file)
        write_json(self.links_file,links)
        self.alias={
            "schema":"douxian-verified-aliases-2.12-1",
            "graph_sha256":v._sha_file(self.graph_file),
            "links_sha256":v._sha_file(self.links_file),
            "verified_synonym_groups":[
                {"canonical_id":"concept:pool",
                 "member_ids":["concept:pool","concept:sub"],
                 "canonical_label":"汇聚层",
                 "verified_labels":["汇聚层","子采样层"],
                 "verified_claim_ids":["edge:alias"]}
            ],
            "search_only_hints":[
                {"concept_id":"concept:pool","term":"池化层",
                 "status":"search_hint_only_not_proven_equivalent"}
            ],
            "ambiguous_terms":[],
        }
        write_json(self.alias_file,self.alias)
        self.gate={
            "schema":"douxian-kg-expansion-gate-2.12-1",
            "source_sha256":v._sha_file(self.source),
            "snapshot_sha256":v._sha_file(self.snapshot),
            "final_graph_sha256":v._sha_file(self.graph_file),
            "final_links_sha256":v._sha_file(self.links_file),
            "alias_registry_sha256":v._sha_file(self.alias_file),
            "final_verified":2,
        }
        write_json(self.gate_file,self.gate)
        self.files=v.GraphFiles(
            graph=self.graph_file,links=self.links_file,
            aliases=self.alias_file,chunks=self.snapshot,
            source=self.source,gate=self.gate_file,
        )
        v._cached_bundle.cache_clear()

    def service(self):
        T=v.create_graph_augmented_service(FakeKB,self.files)
        obj=T()
        obj.chroma=FakeChroma(self.rows)
        return obj

    def test_bundle_strictly_loads_all_verified_claims(self):
        bundle=v.load_verified_bundle(self.files)
        self.assertEqual(bundle.total_verified,2)
        self.assertEqual(len(bundle.chunk_rows),2)
        self.assertEqual(bundle.linked_ids["edge:conv"],"CHUNK-CONV-01")

    def test_graph_query_returns_real_verified_chunk_ids(self):
        bundle=v.load_verified_bundle(self.files)
        ids,meta=v.candidate_chunk_ids("卷积层怎样使用卷积核？",bundle)
        self.assertEqual(ids,["CHUNK-CONV-01"])
        self.assertEqual(meta["matched_edges"],1)

    def test_query_alias_hint_is_retrieval_only(self):
        ids,meta=v.candidate_chunk_ids(
            "池化层与子采样层的关系是什么？",v.load_verified_bundle(self.files)
        )
        self.assertEqual(ids,["CHUNK-POOL-02"])
        self.assertIn("汇聚层",meta["matched_labels"])

    def test_english_alias_retains_word_boundary_and_spaces(self):
        bundle=v.load_verified_bundle(self.files)
        ids,details=v.candidate_chunk_ids(
            "How does the Convolutional Layer use a filter?",bundle
        )
        self.assertEqual(ids,["CHUNK-CONV-01"])
        ids,_=v.candidate_chunk_ids("Unrelated PreConvolutional Layered",bundle)
        self.assertEqual(ids,[])

    def test_unrelated_question_adds_nothing(self):
        svc=self.service()
        got=svc.search_with_rerank("今天午饭吃什么？")
        self.assertEqual(len(got),1)
        self.assertEqual(svc.chroma.calls,0)
        self.assertEqual(svc.graph_v213_last_diagnostics["status"],"no_match")
        self.assertEqual(svc.rerank_calls,1)

    def test_graph_injects_before_single_original_reranker(self):
        svc=self.service()
        results=svc.search_with_rerank("卷积层怎样使用卷积核？")
        self.assertEqual(len(results),2)
        self.assertEqual(svc.rerank_calls,1)
        self.assertEqual(svc.chroma.calls,1)
        self.assertEqual(results[1].metadata["chunk_id"],"CHUNK-CONV-01")
        self.assertEqual(svc.graph_v213_last_diagnostics["added_chunks"],1)
        self.assertEqual(svc.graph_v213_last_diagnostics["verified_relations_in_graph"],2)

    def test_verified_alias_real_chroma_hydration(self):
        svc=self.service()
        docs=svc.search_with_rerank("池化层和子采样层的关系是什么？")
        self.assertEqual(docs[-1].metadata["chunk_id"],"CHUNK-POOL-02")

    def test_live_chroma_changed_document_fallback(self):
        svc=self.service()
        svc.chroma.drift=True
        docs=svc.search_with_rerank("卷积层使用卷积核")
        self.assertEqual(len(docs),1)
        self.assertEqual(svc.graph_v213_last_diagnostics["status"],"fallback_error")
        self.assertEqual(svc.rerank_calls,1)

    def test_snapshot_sha_change_fallback_no_chroma_access(self):
        svc=self.service()
        self.snapshot.write_text(
            self.snapshot.read_text(encoding="utf-8")+"CHANGED\n",
            encoding="utf-8"
        )
        docs=svc.search_with_rerank("卷积层使用卷积核")
        self.assertEqual(len(docs),1)
        self.assertEqual(svc.chroma.calls,0)
        self.assertEqual(svc.graph_v213_last_diagnostics["status"],"fallback_error")

    def test_alias_sha_changed_denied(self):
        self.alias["search_only_hints"].append(
            {"concept_id":"concept:cnv","term":"MadeUp"}
        )
        write_json(self.alias_file,self.alias)
        with self.assertRaises(v.GraphV213Error):
            v.load_verified_bundle(self.files)

    def test_modified_graph_sha_is_rejected(self):
        self.graph["nodes"][0]["label"]="CHANGED"
        write_json(self.graph_file,self.graph)
        with self.assertRaises(v.GraphV213Error):
            v.load_verified_bundle(self.files)

    def test_if_graph_file_missing_hybrid_remains_unmodified(self):
        svc=self.service()
        self.graph_file.unlink()
        got=svc.search_with_rerank("卷积层使用卷积核")
        self.assertEqual(len(got),1)
        self.assertEqual(got[0].metadata["chunk_id"],"HYBRID-0001")
        self.assertEqual(svc.graph_v213_last_diagnostics["status"],"fallback_error")

    def test_duplicates_in_hybrid_not_reinjected(self):
        svc=self.service()
        svc.hybrid_result=[
            FakeDoc(self.rows[0]["content"],{
                "source":self.rows[0]["source"],
                "section_path":self.rows[0]["section_path"],
                "chunk_id":self.rows[0]["chunk_id"],
            })
        ]
        docs=svc.search_with_rerank("卷积层使用卷积核")
        self.assertEqual(len(docs),1)
        self.assertEqual(svc.graph_v213_last_diagnostics["status"],"already_in_hybrid")
        self.assertEqual(svc.rerank_calls,1)

    def test_non_verified_edge_rejected(self):
        links=_read_json(self.links_file)
        links["claims"][0]["status"]="ambiguous"
        links["claims"][0]["chunk_ids"]=[]
        write_json(self.links_file,links)
        # Fix dependent gate hash, still must not be trusted as fact.
        self.gate["final_links_sha256"]=v._sha_file(self.links_file)
        write_json(self.gate_file,self.gate)
        with self.assertRaises(Exception):
            v.load_verified_bundle(self.files)

    def test_capped_candidates_and_no_imagined_edges(self):
        bundle=v.load_verified_bundle(self.files)
        ids,_=v.candidate_chunk_ids("卷积层和汇聚层的关系？",
                                    bundle,max_candidates=1)
        self.assertEqual(len(ids),1)
        self.assertTrue(set(ids).issubset(set(bundle.chunk_rows)))

    def test_missing_service_contract_rejected(self):
        class Bad:
            def search_with_rerank(self,query):
                return []
        with self.assertRaises(v.GraphV213Error):
            v.create_graph_augmented_service(Bad,self.files)

    def test_graph_off_requires_no_file_loading(self):
        with patch.object(v,"load_verified_bundle",
                          side_effect=AssertionError("should not be called")):
            svc=FakeKB()
            docs=svc.search_with_rerank("卷积层使用卷积核")
            self.assertEqual(len(docs),1)
            self.assertEqual(svc.rerank_calls,1)

    def test_never_uses_alias_hint_to_create_fact(self):
        bundle=v.load_verified_bundle(self.files)
        ids,_=v.candidate_chunk_ids("池化层",bundle)
        self.assertEqual(ids,["CHUNK-POOL-02"])
        self.assertEqual(bundle.total_verified,2)

    def test_binder_tamper_detected_even_if_hashes_rewritten(self):
        links=_read_json(self.links_file)
        links["claims"][0]["chunk_ids"]=["CHUNK-POOL-02"]
        write_json(self.links_file,links)
        self.alias["links_sha256"]=v._sha_file(self.links_file)
        write_json(self.alias_file,self.alias)
        self.gate["final_links_sha256"]=v._sha_file(self.links_file)
        self.gate["alias_registry_sha256"]=v._sha_file(self.alias_file)
        write_json(self.gate_file,self.gate)
        with self.assertRaises(Exception):
            v.load_verified_bundle(self.files)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__=="__main__":
    unittest.main(verbosity=2)
