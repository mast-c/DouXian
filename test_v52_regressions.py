# -*- coding: utf-8 -*-
"""GraphRAG v5.2 offline regression tests for explicitly uploaded source files.

Run beside the patched files: python -m unittest -v test_v52_regressions
No network, real Neo4j, or user credentials needed. All graph source examples
below are synthetic and must never be confused with clinical evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


THIS = Path(__file__).resolve().parent
UPLOAD_DIR = THIS.parent
sys.path.insert(0, str(THIS))
# The unmodified peer modules in the user's project are already on sys.path.
# For standalone offline verification in this sandbox only, permit the
# explicitly uploaded peer modules in the parent directory.
if (UPLOAD_DIR / 'medical_query_plan.py').is_file():
    sys.path.insert(1, str(UPLOAD_DIR))

# Missing graph-building modules are replaced ONLY for these isolated tests.
if 'medical_semantic_graph' not in sys.modules:
    try:
        import medical_semantic_graph  # noqa: F401
    except ImportError:
        graph = types.ModuleType('medical_semantic_graph')
        class GraphBlocked(Exception): pass
        class Lexicon:
            def __init__(self, values=None): self.values = values or {}
            def find(self, text):
                return [(name, text.index(name), text.index(name)+len(name))
                        for name in self.values if name in text]
        graph.GraphBlocked = GraphBlocked
        graph.Lexicon = Lexicon
        graph.RELATIONS = {
            'DISEASE_CATEGORY_MENTION':'疾病资料分类提及',
            'CLASS_MEMBERSHIP_ASSERTION':'药品资料药理类别',
            'ADVERSE_REACTION_MENTION':'药物不良反应',
            'INTERACTION_MENTION':'相互作用',
            'CAUTION_MENTION':'注意事项',
            'HAS_SYMPTOM':'症状',
        }
        graph.FIELD_RELATIONS = {
            'indication':'INDICATION_MENTION','warnings':'CAUTION_MENTION',
            'adverse_reactions':'ADVERSE_REACTION_MENTION',
            'drug_interactions':'INTERACTION_MENTION',
        }
        graph.MECHANISM_RELATIONS = set()
        graph.MECHANISM_STATEMENT_RELATIONS = set()
        graph.digest = lambda text: 'abc1234567890'
        graph.fingerprint = lambda p: {'size':0,'sha256':'test'}
        graph.entity_id = lambda *args: 'id'
        graph.class_assertion_spans = lambda *args: []
        graph.index_path = lambda *args: Path('not-loaded-in-test.sqlite')
        graph.ingredient_spans = lambda *args: []
        graph.literal_entity_boundary = lambda *args, **kw: True
        graph.open_graph = lambda *args: None
        graph.qualifiers = lambda text: {}
        graph.verify_source = lambda root, source: True
        sys.modules['medical_semantic_graph'] = graph

if 'medical_patient_context' not in sys.modules:
    try:
        import medical_patient_context  # noqa: F401
    except ImportError:
        context = types.ModuleType('medical_patient_context')
        context.context_assertion = lambda q, i: 'mentioned'
        sys.modules['medical_patient_context'] = context

if 'medical_query_understanding' not in sys.modules:
    try:
        import medical_query_understanding  # noqa: F401
    except ImportError:
        query = types.ModuleType('medical_query_understanding')
        query.understand_question = lambda *args, **kwargs: {}
        query.active_anchors = lambda groups,*args: groups
        query.catalog_aliases_for_question = lambda *args, **kwargs: {}
        sys.modules['medical_query_understanding'] = query

from evidence_guard import validate_judge_output
from medical_llm_query_understanding import enhance_understanding
from medical_category_discovery import category_discovery_plan
from medical_semantic_query_planner import build_semantic_task_plan, refine_semantic_tasks_with_model
from medical_semantic_retrieval import _run_semantic_task_plan


DRUG = {'surface': '复方氨酚烷胺片', 'node_ids': ['drug1'],
        'kinds': ['Drug'], 'ambiguous': False}


class FakeModel:
    def __init__(self, payload): self.payload, self.calls = payload, 0
    def bind(self, **opts): return self
    def invoke(self, messages):
        self.calls += 1
        return {'content': json.dumps(self.payload, ensure_ascii=False)}


class CoreRegressionTests(unittest.TestCase):
    def test_category_reference_does_not_block_known_drug(self):
        question = '复方氨酚烷胺片所属的大类是什么？该大类下具体哪些药物与之存在相互作用？'
        model = FakeModel({
            'intents': ['classification', 'interaction'],
            'entities': [
                {'type':'drug','surface':'复方氨酚烷胺片','role':'mentioned'},
                {'type':'drug','surface':'该大类','role':'mentioned'},
                {'type':'drug_class','surface':'所属的大类','role':'mentioned'},
            ],
            'unknown_products': [
                {'kind':'drug_class','surface':'该大类','contains':[]}
            ],
            'sub_queries':['复方氨酚烷胺片的分类大类是什么？'],
            'relations':[],
        })
        result, report = enhance_understanding(
            question, {}, model,
            allowed_entity_names={'Drug':{'复方氨酚烷胺片'}}
        )
        self.assertEqual(report['status'], 'ok', report)
        self.assertEqual([m['surface'] for m in result['llm_medicine_mentions']],
                         ['复方氨酚烷胺片'])
        self.assertEqual(result['llm_unlinked_medication_mentions'], [])
        self.assertEqual(result['vague_medicines'], [])
        self.assertFalse(result['needs_product_details'])
        self.assertTrue(result['intent']['classification_question'])
        self.assertTrue(result['intent']['interaction_question'])

    def test_genuinely_unknown_named_product_still_requires_verification(self):
        question = '这款未收录的复方甲片属于什么大类？'
        model = FakeModel({'intents':['classification'],
                           'entities':[{'type':'drug','surface':'复方甲片'}],
                           'unknown_products':[], 'relations':[], 'sub_queries':[]})
        result, report = enhance_understanding(question, {}, model,
                                             allowed_entity_names={'Drug':set()})
        self.assertEqual(report['status'], 'ok')
        self.assertEqual([e['surface'] for e in result['llm_unlinked_medication_mentions']], ['复方甲片'])
        self.assertTrue(result['needs_product_details'])

    def test_dependent_category_discovery_reaches_original_source_path(self):
        question = '复方氨酚烷胺片所属的大类是什么？该大类下具体哪些药与之存在相互作用？'
        understanding = {
            'medicines':[{'surface':'复方氨酚烷胺片', 'candidate_drug_ids':['drug1']}],
            'intent':{'classification_question':True,'interaction_question':True},
        }
        plan = category_discovery_plan(question, understanding)
        self.assertIsNotNone(plan)
        self.assertEqual(plan['level'], 'top')
        self.assertEqual(plan['seed_ids'], ['drug1'])
        self.assertEqual(plan['operation'], 'interaction')
        self.assertEqual(plan['fields'], ['drug_interactions'])

    def test_plain_classification_is_not_mistaken_for_discovery(self):
        parsed = {'medicines':[{'surface':'复方氨酚烷胺片','candidate_drug_ids':['drug1']}],
                  'intent':{'classification_question':True}}
        self.assertIsNone(category_discovery_plan('复方氨酚烷胺片所属的大类是什么？', parsed))
        self.assertIsNone(category_discovery_plan('复方氨酚烷胺片的药理类别是什么？', parsed))

    def test_dataset_category_outranks_drug_label_class_assertion(self):
        parsed = {'llm_query_understanding':{'intents':['classification']}, 'intent':{}}
        plan = build_semantic_task_plan('复方氨酚烷胺片属于什么大类', parsed, [DRUG],
            available_relations={'CLASS_MEMBERSHIP_ASSERTION','DATASET_CATEGORY_MEMBERSHIP','DATASET_CATEGORY_PARENT'})
        task = plan['tasks'][0]
        self.assertEqual(task['graph_strategy'],'dataset_category')
        self.assertEqual(task['graph_relations'],['DATASET_CATEGORY_MEMBERSHIP','DATASET_CATEGORY_PARENT'])
        self.assertEqual(task['kind'], 'drug_classification')
        self.assertFalse(task['text_retrieval_required'])

    def test_disease_classification_must_not_use_dataset_drug_category(self):
        d = {'surface':'荨麻疹','node_ids':['dis1'],'kinds':['Disease'],'ambiguous':False}
        parsed = {'llm_query_understanding':{'intents':['classification']},'intent':{}}
        task=build_semantic_task_plan('荨麻疹属于什么类型的疾病',parsed,[d],
            available_relations={'DATASET_CATEGORY_MEMBERSHIP','DATASET_CATEGORY_PARENT','DISEASE_CATEGORY_MENTION'})['tasks'][0]
        self.assertEqual(task['kind'],'disease_classification')
        self.assertEqual(task['graph_relations'],['DISEASE_CATEGORY_MENTION'])

    def test_short_exact_disease_category_quote_accepted_only_when_verified(self):
        edge = {'relation':'DISEASE_CATEGORY_MENTION', 'section_field':'category',
                'src':{'kind':'Disease','name':'荨麻疹'},
                'dst':{'kind':'DiseaseCategory','name':'皮肤科'},'quote':'皮肤科'}
        path={'verified_against_original':True,'edges':[edge],'nodes':[edge['src'],edge['dst']]}
        excerpt={'id':1,'evidence_type':'evidence_backed_semantic_graph_path',
                 'graph_path':path,'content':'原文：皮肤科'}
        raw={'aspects':[{'question':'荨麻疹属于哪类科别？','status':'supported',
                         'evidence':[{'id':1,'quote':'皮肤科'}],'reason':'资料原文字段'}]}
        review = validate_judge_output(raw, [excerpt])
        self.assertEqual(review['decision'],'answer',review)
        self.assertEqual(review['evidence_ids'],[1])
        for malicious in (
            {**excerpt,'graph_path':{**path,'verified_against_original':False}},
            {**excerpt,'graph_path':{**path,'edges':[{**edge,'section_field':'warnings'}]}},
            {**excerpt,'graph_path':{**path,'edges':[{**edge,'dst':{'kind':'DrugClass','name':'皮肤科'}}]}},
            {**excerpt,'graph_path':{**path,'edges':[{**edge,'quote':'皮肤病科别'}]}},
        ):
            verdict = validate_judge_output(raw,[malicious])
            self.assertEqual(verdict['decision'],'refuse',verdict)

    def test_short_quote_not_allowed_for_arbitrary_graph_edge(self):
        edge = {'relation':'INTERACTION_MENTION','section_field':'drug_interactions',
                'src':{'kind':'Drug','name':'甲药'},'dst':{'kind':'Drug','name':'乙药'},
                'quote':'不要同用'}
        excerpt={'id':1,'evidence_type':'evidence_backed_semantic_graph_path',
                 'graph_path':{'verified_against_original':True,'edges':[edge]},
                 'content':'原文：不要同用'}
        raw={'aspects':[{'question':'甲乙药能同用吗','status':'supported',
                         'evidence':[{'id':1,'quote':'不要同用'}]}]}
        self.assertEqual(validate_judge_output(raw,[excerpt])['decision'],'refuse')

    def test_llm_second_pass_rechecks_single_clause_misclassified_as_symptoms(self):
        q='荨麻疹属于什么疾病类型？'
        d = {'surface':'荨麻疹','node_ids':['dis1'],'kinds':['Disease'],'ambiguous':False}
        initial = {'intent':{'symptom_question':True},
                   'llm_query_understanding':{'intents':['disease_symptoms']},
                   'patient_context':[{'surface':'荨麻疹','kind':'disease_mention'}]}
        model = FakeModel({'intents':['classification'],
                           'sub_tasks':[{'intent':'classification','query':'荨麻疹的科别是什么？',
                                         'focus_entities':['荨麻疹']}]})
        with patch.dict(os.environ, {'MEDICAL_SEMANTIC_TASK_SECOND_PASS':'auto'}):
            refined, report = refine_semantic_tasks_with_model(q,initial,[d],model)
        self.assertEqual(report['status'],'ok')
        self.assertIn('classification',refined['llm_query_understanding']['intents'])
        self.assertEqual(model.calls,1)

    def test_dataset_route_calls_original_source_verified_retriever(self):
        import sqlite3
        import medical_semantic_retrieval as retriever
        from medical_semantic_query_planner import build_semantic_task_plan
        understanding={'intent':{'classification_question':True},
                       'medicines':[{'surface':'复方氨酚烷胺片','candidate_drug_ids':['drug1']}],
                       'llm_query_understanding':{'intents':['classification']}}
        task_plan=build_semantic_task_plan('复方氨酚烷胺片所属大类是什么', understanding, [DRUG],
            available_relations={'DATASET_CATEGORY_MEMBERSHIP','DATASET_CATEGORY_PARENT'})
        nodes={'drug1':{'id':'drug1','kind':'Drug','name':'复方氨酚烷胺片'}}
        source_proof={'id':'dataset_path_1','hops':2,
                      'relations':['DATASET_CATEGORY_MEMBERSHIP','DATASET_CATEGORY_PARENT'],
                      'verified_against_original':True,
                      'nodes':[nodes['drug1'],{'id':'leaf','kind':'DatasetCategory','name':'缓解感冒复方OTC'},
                               {'id':'top','kind':'DatasetCategory','name':'镇痛解热'}],
                      'edges':[{'id':'1','relation':'DATASET_CATEGORY_MEMBERSHIP'},
                               {'id':'2','relation':'DATASET_CATEGORY_PARENT'}]}
        fake_item={'graph_verified':True,'evidence_type':'dataset_category_graph_path',
                   'graph_path':source_proof,'content':'本地数据集分类原文和位置'}
        def fake_category(question,root,nodes,understanding,max_paths=8):
            self.assertEqual(understanding['medicines'][0]['candidate_drug_ids'],['drug1'])
            return [fake_item],{'status':'ok','paths':[source_proof]}
        conn=sqlite3.connect(':memory:')
        try:
            with patch.object(retriever,'search_category_graph',fake_category):
                items, report = retriever._run_semantic_task_plan(
                    conn,Path('.'),{'version':'test'},nodes,understanding,task_plan,
                    max_paths=4,max_expansions=100,max_neighbors=20,max_seconds=2,
                    report={'context_entities':[DRUG],'semantic_sub_queries':[]})
        finally: conn.close()
        self.assertEqual(len(items),1)
        self.assertEqual(report['status'],'ok')
        self.assertEqual(report['multi_relation_paths'],1)
        self.assertEqual(report['answer_route'],'dataset_category_source_display')
        self.assertEqual(report['task_coverage'][0]['status'],'graph_evidence_found')
        self.assertTrue(items[0]['graph_verified'])


    def test_full_search_prefers_dependent_category_discovery_over_independent_plan(self):
        """A two-intent question should not get swallowed by independent-task execution."""
        import sqlite3
        import tempfile
        import medical_semantic_retrieval as retriever

        question='复方氨酚烷胺片所属的大类是什么？该大类下具体哪些药与之存在相互作用？'
        patient={
            'intent': {'classification_question':True,'interaction_question':True,
                       'fields':['drug_interactions']},
            'medicines':[{'surface':'复方氨酚烷胺片','candidate_drug_ids':['drug1'],
                          'candidate_drug_names':['复方氨酚烷胺片'],
                          'actual_patient_product_known':True}],
            'llm_query_understanding': {'status':'ok','intents':['classification','interaction']},
            'vague_medicines':[], 'unresolved_medication_mentions':[],
            'alternative_medication_mentions':[], 'semantic_sub_queries':[],
            'patient_context':[], 'vague_medicine_details':[],
        }
        nodes={'drug1':{'id':'drug1','kind':'Drug','name':'复方氨酚烷胺片'}}
        tmpdir=Path(tempfile.mkdtemp(prefix='graphrag_v52_synthetic_'))
        target=tmpdir/'index.sqlite'; target.write_bytes(b'fake index for synthetic test')
        conn=sqlite3.connect(':memory:'); conn.row_factory=sqlite3.Row
        proof={'id':'seedpath','nodes':[nodes['drug1']],'edges':[],
               'relations':['DATASET_CATEGORY_MEMBERSHIP','DATASET_CATEGORY_PARENT'],
               'verified_against_original':True}
        item={'graph_path':proof,'source':'synthetic verified dataset category source',
              'content':'Category source data from synthetic fixture', 'id':1}
        def fake_graph(*args, **kwargs): return conn,{'version':'test','created_at':'test'}
        def fake_catalog(*args, **kwargs): return nodes,{},None
        def fake_link(*args, **kwargs): return [DRUG],patient,[{'surface':DRUG['surface']}]
        def fake_llm(*args, **kwargs): return patient,{'status':'ok','model_calls':1}
        def fake_refine(*args,**kwargs): return patient,{'status':'not_needed','model_calls':0}
        def fake_discovery(*args, **kwargs):
            return [item],{'status':'ok','paths':[proof], 'truncated':False, 'seconds':0.01}
        # Everything besides category discovery is intentionally made to fail
        # loudly to prove the high-priority routing is selected.
        with patch.object(retriever,'open_graph',fake_graph), \
             patch.object(retriever,'index_path',lambda *args: target), \
             patch.object(retriever,'_catalog',fake_catalog), \
             patch.object(retriever,'select_medical_query_model',lambda x: (object(),'injected_app_model')), \
             patch.object(retriever,'enhance_understanding',fake_llm), \
             patch.object(retriever,'extend_category_entities',lambda q,r,n,u: (n,u)), \
             patch.object(retriever,'_semantic_catalog_anchors',fake_link), \
             patch.object(retriever,'refine_semantic_tasks_with_model',fake_refine), \
             patch.object(retriever,'search_category_discovery',fake_discovery), \
             patch.object(retriever,'_run_semantic_task_plan',side_effect=AssertionError('wrong branch')):
            evidence, trace = retriever.search_semantic_graph(
                question,project_root=tmpdir,query_understanding_model=object())
        self.assertEqual(len(evidence),1)
        self.assertEqual(trace['status'],'ok')
        self.assertFalse(trace['independent_task_route'])
        self.assertEqual(trace['answer_route'],'category_discovery_source_display')
        self.assertEqual(trace['retrieval_plan']['category_discovery']['operation'],'interaction')


if __name__ == '__main__':
    unittest.main(verbosity=2)
