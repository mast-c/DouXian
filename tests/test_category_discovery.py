"""Dependent category discovery with actual temporary source-backed graphs."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medical_category_graph import RAW
from medical_category_discovery import category_discovery_plan, category_discovery_answer, search_category_discovery
from medical_semantic_graph import build_graph, open_graph, GraphBlocked
from medical_semantic_retrieval import search_semantic_graph, _catalog
from medical_semantic_graph_view import semantic_graph_html
from medical_evidence_coverage import build_answer_targets, evidence_bindings


class CategoryDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        export = self.root/'kg_output_clean_v2'
        export.mkdir()
        self.drugs = [dict(id=i, name=n, aliases=[]) for i,n in [
            ('a','甲试胶囊'), ('b','乙试片'), ('c','丙试片'), ('d','丁试片'),
            ('e','戊试滴眼液'), ('f','己试片'), ('g','庚试片')]]
        records = [dict(id='r'+d['id'], drug_id=d['id'], source_url='https://example.invalid/'+d['id']) for d in self.drugs]
        self.sections = []
        def section(drug, field, text):
            self.sections.append(dict(id=drug+'_'+field, record_id='r'+drug, field=field, text=text))
        for i, ingredient in [('a','阿莫西林'),('b','布洛芬'),('c','丙磺舒'),('d','丁试素'),('e','氯霉素'),('f','氯霉素'),('g','阿司匹林')]:
            section(i,'composition',ingredient+'。')
        # Only B/C supports a forward cross-category interaction; A is excluded.
        section('a','drug_interactions','本品与丁试片合用应谨慎。')
        section('b','drug_interactions','本品与丙磺舒合用可能增加血药浓度，临床意义不明。')
        # C is the owner of a reverse mention of D.
        section('c','drug_interactions','本品与丁试片合用应监测。')
        # Shared third drug does not prove a B/D interaction.
        section('d','drug_interactions','本品与氯霉素合用尚未见不良相互作用。')
        section('b','contraindications','胃溃疡患者禁用。')
        section('e','dosage','外用，涂于患处。')
        for name, rows in [('drugs',self.drugs),('source_records',records),('sections',self.sections)]:
            (export/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        (export/'disease_candidates_full.csv').write_text('section_id\n',encoding='utf-8')
        raw = self.root/RAW
        raw.mkdir(parents=True)
        (raw/'medical.json').write_text(json.dumps({'name':'胃溃疡','symptom':['腹痛']},ensure_ascii=False)+'\n',encoding='utf-8')
        self.category = raw/'drug_categories_western.json'
        self.category.write_text(json.dumps({'categories': {
            '第一大类': {'甲子类':['甲试胶囊','乙试片','丁试片','庚试片'], '另一子类':['戊试滴眼液']},
            '第二大类': {'乙子类':['丙试片','己试片']}}},ensure_ascii=False),encoding='utf-8')
        build_graph(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def search(self, q='跟甲试胶囊同一大类的其他药与其他大类药物有没有相互作用？', **kw):
        return search_semantic_graph(q, project_root=self.root, **kw)

    def examples(self, e):
        return [x for x in e if x['discovery_role'] == 'interaction_example']

    def test_dependent_plan_and_top_category_expand_other_products(self):
        e,r = self.search()
        self.assertEqual(r['answer_route'],'category_discovery_source_display')
        tasks = r['retrieval_plan']['tasks']
        self.assertEqual(tasks[1]['depends_on'],'category_candidates')
        self.assertEqual(tasks[2]['depends_on'],'candidate_sources')
        self.assertEqual(r['category_discovery']['plan']['level'],'top')
        pairs = {(i['product'],i['partner']) for i in self.examples(e)}
        self.assertIn(('乙试片','丙试片'),pairs)
        self.assertIn(('丁试片','丙试片'),pairs)  # Reverse search.
        self.assertFalse(any('甲试胶囊' in p for p in pairs))
        self.assertEqual(r['execution']['model_calls'],0)
        answer = category_discovery_answer(e,r['category_discovery'])
        self.assertIn('临床意义不明',answer)
        self.assertIn('尚未见不良相互作用',answer)
        self.assertIn('尚未核实该具体制剂组合',answer)

    def test_proofs_connect_every_edge_and_frontend_draws_mixed_sources(self):
        e,r = self.search()
        for item in e:
            p = item['graph_path']
            self.assertEqual(len(p['nodes']),len(p['edges'])+1)
            for n,edge in enumerate(p['edges']):
                self.assertEqual({edge['src']['id'],edge['dst']['id']},{p['nodes'][n]['id'],p['nodes'][n+1]['id']})
            self.assertTrue(p['verified_against_original'])
            self.assertFalse(p['clinical_validation'])
        html = semantic_graph_html({'semantic_graph':r})
        self.assertIn('JSON路径',html)
        self.assertIn('相互作用提及',html)
        self.assertIn('一级大类',html)

    def test_same_subclass_and_member_listing_reuse_executor(self):
        e,r = self.search('甲试胶囊同一子类的其他药有哪些？')
        self.assertEqual(r['category_discovery']['plan']['level'],'leaf')
        members = {i['product'] for i in e if i['discovery_role']=='member'}
        self.assertEqual(members,{'乙试片','丁试片','庚试片'})
        e,r = self.search('甲试胶囊同类药有哪些成分？')
        self.assertEqual(r['category_discovery']['plan']['operation'],'source_fields')
        self.assertTrue(any(i['discovery_role']=='source_field' for i in e))
        self.assertIn('二级子类理解',category_discovery_answer(e,r['category_discovery']))
        e,r = self.search('甲试胶囊同一子类的其他药与其他大类药物有没有相互作用？')
        self.assertEqual(r['category_discovery']['plan']['level'],'leaf')
        self.assertEqual(r['category_discovery']['plan']['other_level'],'top')
        self.assertEqual(r['category_discovery']['candidate_drugs_available'],3)

    def test_exclude_named_member_and_seed(self):
        e,r = self.search('跟甲试胶囊同一大类的药，排除乙试片，与其他大类药物有没有相互作用？')
        products = {i.get('product') for i in e}
        self.assertNotIn('乙试片',products)
        self.assertNotIn('甲试胶囊',products)
        self.assertTrue(r['category_discovery']['plan']['excluded_ids'])
        e,r = self.search('甲试胶囊同类其他药，排除乙试片和丁试片，与其他大类药物有没有相互作用？')
        self.assertEqual(set(r['category_discovery']['plan']['excluded_ids']),{'b','d'})

    def test_source_condition_filter_and_different_ingredient_comparison(self):
        e,r = self.search('甲试胶囊同类药中哪些有胃溃疡禁忌？')
        examples = [i for i in e if i['discovery_role']=='source_field']
        self.assertEqual({i['product'] for i in examples},{'乙试片'})
        self.assertIn('胃溃疡',r['category_discovery']['source_target_filters'])
        e,r = self.search('甲试胶囊同一子类药中哪些成分不同？')
        self.assertTrue(any(i['discovery_role']=='seed_composition' for i in e))
        self.assertTrue(any(i['discovery_role']=='source_field' for i in e))
        self.assertIn('仅比较本轮原文明确提取的成分',category_discovery_answer(e,r['category_discovery']))
        e,r = self.search('甲试胶囊同类药有哪些不同成分？')
        self.assertTrue(any(i['discovery_role']=='source_field' for i in e))

    def test_negative_risk_or_complete_composition_not_inferred_from_missing_edges(self):
        e,r = self.search('甲试胶囊同类药中哪些没有副作用？')
        self.assertFalse(self.examples(e))
        self.assertIn('absence_or_complete_identity_not_provable_by_mention_edges',r['category_discovery']['missing_links'])

    def test_path_cap_includes_anchor_evidence(self):
        e,r = self.search(max_paths=2)
        self.assertLessEqual(len(e),2)
        self.assertTrue(r['truncated'])

    def test_same_record_route_text_overrides_ambiguous_form_name(self):
        from medical_category_discovery import _source_route
        conn,_ = open_graph(self.root)
        try:
            edge = dict(conn.execute("SELECT * FROM edges WHERE src='e' AND relation='HAS_INGREDIENT'").fetchone())
            nodes = {d['id']:{**d,'kind':'Drug'} for d in self.drugs}
            nodes['e']['name']='戊试溶液'
            self.assertEqual(_source_route(conn,edge,'e',nodes),'cutaneous')
        finally:
            conn.close()

    def test_explicit_other_drug_constraint_is_preserved(self):
        e,r = self.search('甲试胶囊同类的其他药与丙试片有没有相互作用？')
        self.assertTrue(r['category_discovery']['plan']['partner_ids'])
        self.assertTrue(self.examples(e))
        self.assertTrue(all(i['partner']=='丙试片' for i in self.examples(e)))

    def test_shared_third_drug_never_proves_pair_and_route_conflict_rejected(self):
        e,r = self.search()
        pairs = {(i['product'],i['partner']) for i in self.examples(e)}
        self.assertNotIn(('乙试片','丁试片'),pairs)
        self.assertFalse(any('戊试滴眼液' in p for p in pairs))
        self.assertIn('inferred_identity_join_route_conflict',r['category_discovery']['missing_links'])

    def test_dataset_class_label_cannot_substitute_for_source_class_assertion(self):
        for section in self.sections:
            if section['id']=='b_drug_interactions':
                section['text']='本品与抗凝血药合用应谨慎。'
        path = self.root/'kg_output_clean_v2/sections.jsonl'
        def save():
            path.write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in self.sections),encoding='utf-8')
            build_graph(self.root)
        save()
        q='甲试胶囊同类其他药与丙试片有没有相互作用？'
        e,r = self.search(q)
        self.assertFalse(any(i['product']=='乙试片' for i in self.examples(e)))
        self.sections.append(dict(id='c_warnings',record_id='rc',field='warnings',text='本品为抗凝血药。'))
        save()
        e,r = self.search(q)
        found = [i for i in self.examples(e) if i['product']=='乙试片']
        self.assertTrue(found)
        self.assertIn('CLASS_MEMBERSHIP_ASSERTION',found[0]['graph_path']['relations'])

    def test_overlapping_partner_category_is_not_exclusive_other_class(self):
        data = json.loads(self.category.read_text(encoding='utf-8'))
        data['categories']['第一大类']['另类']=['丙试片','己试片']
        self.category.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
        e,r = self.search()
        self.assertFalse(self.examples(e))
        self.assertEqual(r['status'],'partial')
        self.assertIn('本轮没有取得符合全部条件',category_discovery_answer(e,r['category_discovery']))

    def test_missing_category_reports_actual_step_failure(self):
        self.category.unlink()
        e,r = self.search()
        self.assertEqual(e,[])
        self.assertIn('category_sources_missing',r['category_discovery']['missing_links'])

    def test_budget_keeps_verified_partial_classification(self):
        e,r = self.search(max_seconds=0.001)
        self.assertTrue(r['truncated'])
        self.assertLess(r['seconds'],3)
        self.assertIn('time_budget_exhausted',r['category_discovery']['missing_links'])

    def test_not_triggered_by_different_class_or_ordinary_classification(self):
        e,r = self.search('甲试胶囊属于什么类药物？')
        self.assertEqual(r['answer_route'],'dataset_category_source_display')
        e,r = self.search('甲试胶囊与丙试片不同类别，相互作用是什么？')
        self.assertNotIn('category_discovery',r)

    def test_individual_advice_not_answered_by_category_examples(self):
        e,r = self.search('甲试胶囊同类的其他药我能不能吃？')
        self.assertFalse(self.examples(e))
        self.assertIn('patient_specific_advice_requires_specific_products',r['category_discovery']['missing_links'])

    def test_coverage_targets_reference_discovered_products_not_seed_relations(self):
        e,r = self.search()
        targets = build_answer_targets('',r['query_understanding'],r['retrieval_plan'])
        self.assertEqual(len(targets),2)
        for item in self.examples(e):
            self.assertEqual(evidence_bindings(item,targets)[0]['target_id'],'target_discovered_sources')
        self.assertEqual(evidence_bindings(e[0],targets)[0]['target_id'],'target_category')

    def test_sources_are_reverified_before_admission(self):
        conn, _ = open_graph(self.root)
        target = self.root/'semantic_graph/medical_graph.sqlite'
        stat = target.stat()
        nodes,_,_ = _catalog(str(target),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        _,r = self.search()
        path = self.root/'kg_output_clean_v2/sections.jsonl'
        path.write_text(path.read_text(encoding='utf-8').replace('临床意义不明','完全安全可用'),encoding='utf-8')
        try:
            with self.assertRaises(GraphBlocked):
                search_category_discovery(conn,self.root,nodes,r['category_discovery']['plan'])
        finally:
            conn.close()

    def test_mixed_category_discovery_does_not_use_pure_category_shortcut(self):
        # Pure source taxonomy remains local-only. Mixed clinical discovery
        # retains category paths but continues into the established RAG lanes.
        source = (Path(__file__).resolve().parents[1]/'rag.py').read_text(encoding='utf-8')
        shortcut = source.split('if _semantic_report.get("answer_route") == "dataset_category_source_display":',1)[1].split('_retrieval_question = question',1)[0]
        self.assertIn('return',shortcut)
        mixed = source.split('if _semantic_report.get("answer_route") == "category_discovery_source_display":',1)[1].split('_retrieval_question = question',1)[0]
        self.assertNotIn('return',mixed)
        self.assertIn('category_discovery_with_standard_retrieval',mixed)
        self.assertIn('_category_candidates',source)

    def test_frontend_filters_only_structured_paths_not_original_source_cards(self):
        frontend = (Path(__file__).resolve().parents[1]/'豆馅.py').read_text(encoding='utf-8')
        block = frontend.split('# 图谱路径在下方独立区域展示',1)[1].split('index = item.get("id")',1)[0]
        self.assertIn('"category_discovery_source_path"',block)
        self.assertIn('"evidence_backed_semantic_graph_path"',block)
        self.assertNotIn('"json_neo4j_text_verified"',block)
        self.assertNotIn('"unverified_drug_source_section"',block)

    def test_discovered_candidate_sections_can_cover_separate_review_target(self):
        e,r = self.search()
        targets = build_answer_targets('',r['query_understanding'],r['retrieval_plan'])
        target = next(t for t in targets if t['id']=='target_discovered_sources')
        target['candidate_names'] = ['乙试片']
        item = {'evidence_type':'json_neo4j_text_verified','drug_name':'乙试片','section_field':'drug_interactions',
                'parent_verified':True,'content':'原文：与丙试片合用需谨慎。'}
        self.assertEqual(evidence_bindings(item,targets)[0]['target_id'],'target_discovered_sources')


if __name__ == '__main__':
    unittest.main()
