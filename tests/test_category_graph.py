"""Source-separated category graph and formulation recognition regressions."""
import contextlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medical_category_graph import (
    RAW, ensure_category_graph, search_category_graph, verify_category_edge, category_only, extend_category_entities,
)
from medical_query_understanding import understand_question
from medical_semantic_graph import build_graph, GraphBlocked
from medical_semantic_retrieval import search_semantic_graph
from medical_semantic_graph_view import semantic_graph_html


class CategoryGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.nodes={'d_a':{'id':'d_a','kind':'Drug','name':'甲试胶囊'},
                    'd_b':{'id':'d_b','kind':'Drug','name':'乙试片'}}
        export=self.root/'kg_output_clean_v2'
        export.mkdir()
        (export/'drugs.jsonl').write_text(''.join(json.dumps(n,ensure_ascii=False)+'\n' for n in self.nodes.values()),encoding='utf-8')
        self.raw=self.root/RAW
        self.raw.mkdir(parents=True)
        self.category=self.raw/'drug_categories_western.json'
        self.category.write_text(json.dumps({'categories':{'测试一级类':{'测试二级类':['甲试胶囊','未收录药片']}}},ensure_ascii=False,indent=2),encoding='utf-8')
        (self.raw/'drug_categories_chinese.json').write_text(json.dumps({'categories':{'中药用途':{'测试科室':['乙试片']}}},ensure_ascii=False),encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def understand(self,q):
        return understand_question(q,root=self.root,nodes=self.nodes)

    def search(self,q='甲试胶囊属于什么类药物'):
        return search_category_graph(q,self.root,self.nodes,self.understand(q))

    def test_full_catalog_names_are_recognized_before_intent(self):
        for tail in ['属于什么类药物','主要用于什么','通常什么时候吃','到底算什么药','具体怎么分类','为什么被归到这一类']:
            with self.subTest(tail=tail):
                medicines=self.understand('甲试胶囊'+tail)['medicines']
                self.assertEqual(medicines[0]['surface'],'甲试胶囊')
                self.assertTrue(medicines[0]['actual_patient_product_known'])

    def test_common_classification_wordings_share_one_intent(self):
        for tail in ['属于什么类药物','是什么类别','算不算抗生素','是不是抗生素','是中药还是西药']:
            with self.subTest(tail=tail):
                understood=self.understand('甲试胶囊'+tail)
                self.assertTrue(understood['intent']['classification_question'])
                self.assertTrue(category_only('甲试胶囊'+tail,understood))

    def test_longer_unknown_chemical_and_formulations_still_rejected(self):
        for q in ['未知甲试胶囊属于什么类','甲试胶囊钠属于什么类','甲试胶囊加强版怎么吃']:
            with self.subTest(q=q):
                self.assertFalse(self.understand(q)['medicines'])

    def test_entities_and_edges_are_built_but_do_not_become_pharmacological_classes(self):
        path,manifest=ensure_category_graph(self.root,self.nodes)
        self.assertEqual(manifest['matched_memberships'],2)
        self.assertEqual(manifest['unmatched_memberships'],1)
        conn=sqlite3.connect(path)
        try:
            kinds={r[0] for r in conn.execute('SELECT DISTINCT kind FROM entities')}
            relations={r[0] for r in conn.execute('SELECT DISTINCT relation FROM edges')}
            self.assertEqual(kinds,{'Drug','DatasetCategory','CatalogDrug'})
            self.assertEqual(relations,{'DATASET_CATEGORY_MEMBERSHIP','DATASET_CATEGORY_PARENT'})
            self.assertNotIn('CLASS_MEMBERSHIP_ASSERTION',relations)
        finally:
            conn.close()

    def test_two_hop_path_is_verified_with_exact_json_pointer(self):
        excerpts,report=self.search()
        self.assertEqual(report['status'],'ok')
        path=excerpts[0]['graph_path']
        self.assertEqual([n['name'] for n in path['nodes']],['甲试胶囊','测试二级类','测试一级类'])
        self.assertEqual(path['hops'],2)
        self.assertTrue(path['verified_against_original'])
        self.assertEqual(path['edges'][0]['json_pointer'],'/categories/测试一级类/测试二级类/0')
        self.assertEqual(excerpts[0]['evidence_type'],'dataset_category_graph_path')
        self.assertFalse(path['clinical_validation'])
        html=semantic_graph_html({'semantic_graph':report})
        self.assertIn('JSON路径',html)
        self.assertIn('数据集分类归属',html)

    def test_changed_source_rebuilds_index_and_never_reuses_old_category(self):
        _,first=self.search()
        self.category.write_text(json.dumps({'categories':{'新一级':{'新二级':['甲试胶囊']}}},ensure_ascii=False),encoding='utf-8')
        excerpts,second=self.search()
        self.assertNotEqual(first['category_index']['signature'],second['category_index']['signature'])
        self.assertEqual(excerpts[0]['graph_path']['nodes'][-1]['name'],'新一级')

    def test_false_json_pointer_edge_is_rejected(self):
        excerpts,_=self.search()
        edge=excerpts[0]['graph_path']['edges'][0]
        row={**edge,'json_pointer':'/categories/测试一级类/测试二级类/1'}
        with self.assertRaises(GraphBlocked):
            verify_category_edge(self.root,row,edge['src'],edge['dst'])

    def test_category_only_catalog_drug_can_be_queried_without_medical_product_equivalence(self):
        q='未收录药片属于什么类别'
        nodes,understanding=extend_category_entities(q,self.root,self.nodes,self.understand(q))
        medicine,=understanding['medicines']
        self.assertTrue(medicine['catalog_entry_only'])
        self.assertFalse(medicine['actual_patient_product_known'])
        excerpts,report=search_category_graph(q,self.root,nodes,understanding)
        self.assertEqual(report['status'],'ok')
        self.assertEqual(excerpts[0]['graph_path']['nodes'][0]['kind'],'CatalogDrug')
        q='未收录药片属于什么类别，能和乙试片一起吃吗'
        _,understanding=extend_category_entities(q,self.root,self.nodes,self.understand(q))
        self.assertFalse(any(m.get('catalog_entry_only') for m in understanding['medicines']))

    def test_classification_route_cannot_answer_interactions_or_other_clinical_questions(self):
        for q in ['甲试胶囊属于什么类药，能和乙试片一起用吗','甲试胶囊属于什么类，副作用有哪些','甲试胶囊分类和剂量是什么']:
            with self.subTest(q=q):
                self.assertFalse(category_only(q,self.understand(q)))

    def test_missing_taxonomy_files_returns_explicit_status(self):
        self.category.unlink()
        (self.raw/'drug_categories_chinese.json').unlink()
        excerpts,report=self.search()
        self.assertEqual(excerpts,[])
        self.assertEqual(report['status'],'category_sources_missing')

    def test_backend_displays_source_category_and_graph_without_cloud_calls(self):
        records=[{'id':'r'+identity,'drug_id':identity,'source_url':'https://example.invalid/'+identity} for identity in self.nodes]
        sections=[{'id':'s'+r['id'],'record_id':r['id'],'field':'composition','text':'测试素。'} for r in records]
        export=self.root/'kg_output_clean_v2'
        for name,rows in [('source_records',records),('sections',sections)]:
            (export/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        (export/'disease_candidates_full.csv').write_text('section_id\n',encoding='utf-8')
        (self.raw/'medical.json').write_text(json.dumps({'name':'试验病','symptom':['皮疹']},ensure_ascii=False)+'\n',encoding='utf-8')
        build_graph(self.root)
        q='甲试胶囊属于什么类药物'
        excerpts,report=search_semantic_graph(q,project_root=self.root)
        self.assertEqual(report['answer_route'],'dataset_category_source_display')
        stubs={}
        for name,symbol in [('knowledge_base','KnowledgeBaseService'),('vector_stores','VectorStoreService')]:
            module=ModuleType(name)
            setattr(module,symbol,type(symbol,(),{}))
            stubs[name]=module
        with patch.dict(sys.modules,stubs):
            import rag
        service=rag.RagService.__new__(rag.RagService)
        service._call_tool=lambda *a,**kw:self.fail('category lookup needs no network tools')
        trace={}
        with patch.object(rag,'search_semantic_graph',return_value=(excerpts,report)),patch.object(rag,'get_history'),contextlib.redirect_stdout(io.StringIO()):
            answer=''.join(service._stream_agent({'input':q,'mode':'知识库','trace':trace,'history':[]},'fixture'))
        self.assertIn('测试一级类 → 测试二级类',answer)
        self.assertIn('尚未核验为权威药理分类',answer)
        self.assertEqual(trace['medical_answer_mode'],'dataset_category_source_display')
        self.assertTrue(trace['semantic_graph']['evidence_bindings'][0]['cited_in_answer'])


if __name__=='__main__':
    unittest.main()
