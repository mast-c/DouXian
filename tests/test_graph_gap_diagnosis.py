"""Meaningful source-vs-index diagnostic cases using small real graph fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medical_semantic_graph import build_graph, fingerprint, index_path
from medical_graph_gap_diagnosis import diagnose_questions, classify_case, render_markdown


def fixture(root):
    export=root/'kg_output_clean_v2'
    export.mkdir()
    drugs=[{'id':'a','name':'甲试片','aliases':[]},{'id':'b','name':'乙试片','aliases':[]}]
    records=[{'id':'r'+d['id'],'drug_id':d['id'],'source_url':'https://example.invalid/'+d['id']} for d in drugs]
    # Both mention ketoconazole; it must never be a bridge for A/B interaction.
    sections=[{'id':'a_c','record_id':'ra','field':'composition','text':'甲试素。'},
              {'id':'a_i','record_id':'ra','field':'drug_interactions','text':'本品与酮康唑合用时应谨慎。'},
              {'id':'b_i','record_id':'rb','field':'drug_interactions','text':'应避免与酮康唑合用。'},
              {'id':'b_w','record_id':'rb','field':'warnings','text':'儿童应在医师指导下使用。'}]
    for name,rows in [('drugs',drugs),('source_records',records),('sections',sections)]:
        (export/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (export/'disease_candidates_full.csv').write_text('section_id\n',encoding='utf-8')
    disease=root/'data/raw/drug_db/MedGraphRAG-Drug-DB/medical.json'
    disease.parent.mkdir(parents=True)
    disease.write_text(json.dumps({'name':'试验病','symptom':['皮疹']},ensure_ascii=False)+'\n',encoding='utf-8')
    # Add a third product so ketoconazole is a graph entity, without linking A/B.
    with (export/'drugs.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'id':'c','name':'酮康唑片','aliases':[]},ensure_ascii=False)+'\n')
    with (export/'source_records.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'id':'rc','drug_id':'c','source_url':'https://example.invalid/c'},ensure_ascii=False)+'\n')
    with (export/'sections.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'id':'c_c','record_id':'rc','field':'composition','text':'酮康唑。'},ensure_ascii=False)+'\n')
    build_graph(root)


class GapDiagnosisTests(unittest.TestCase):
    def test_missing_source_is_distinct_from_missing_graph_edge_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixture(root)
            before=fingerprint(index_path(root))
            report=diagnose_questions(['甲试片与乙试片的相互作用'],root)
            self.assertEqual(report['edges_created'],0)
            self.assertEqual(fingerprint(index_path(root)),before)
            case=report['cases'][0]
            self.assertEqual(case['category'],'no_pair_link_in_checked_sources')
            product=next(p for p in case['products'] if p['drug_name']=='乙试片')
            composition=next(c for c in product['columns'] if c['field']=='composition')
            self.assertEqual(composition['status'],'source_missing_in_local_snapshot')
            shared=case['pair_diagnosis']['shared_third_partners']
            self.assertEqual(shared[0]['name'],'酮康唑')
            self.assertTrue(all(not p['usable_for_pair_conclusion'] for p in shared))
            self.assertEqual(case['pair_diagnosis']['source_identity_joins'],[])
            self.assertIn('不能作为',render_markdown(report))

    def test_sources_retain_exact_record_coordinates_and_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixture(root)
            report=diagnose_questions(['甲试片与乙试片的相互作用'],root)
            source=next(c for c in report['cases'][0]['products'][0]['columns'] if c['field']=='composition')['sources'][0]
            with (root/source['source_file']).open('rb') as stream:
                stream.seek(source['byte_start'])
                original=json.loads(stream.read(source['byte_length']).decode('utf-8'))
            self.assertEqual(source['quote_preview'],original['text'][source['quote_start']:source['quote_end']])
            self.assertEqual(source['record_id'],original['record_id'])

    def test_existing_paths_are_not_diagnosed_as_failure_just_because_extra_paths_truncated(self):
        category,_=classify_case({'status':'ok','returned_paths':6,'truncated':True},[],{})
        self.assertEqual(category,'graph_paths_available')

    def test_unlinked_class_warning_does_not_assign_other_drug_to_class(self):
        category,_=classify_case({'status':'no_connected_evidence_path'},[],
                                {'unlinked_class_mentions':[{'class_name':'抗凝药'}]})
        self.assertEqual(category,'no_pair_link_in_checked_sources')

    def test_real_literal_extraction_lead_prioritizes_parser_review(self):
        category,_=classify_case({'status':'no_connected_evidence_path'},[],
                                {'raw_mentions_without_edges':[{'reason':'unextracted_literal_mention_candidate'}]})
        self.assertEqual(category,'interaction_extraction_gap_candidate')

    def test_existing_identity_join_without_path_prioritizes_retrieval(self):
        category,_=classify_case({'status':'no_connected_evidence_path'},[],
                                {'source_identity_joins':[{'edge_ids':['e1','e2']}]})
        self.assertEqual(category,'existing_join_not_returned_candidate')

    def test_ambiguity_is_not_diagnosed_as_missing_medical_relation(self):
        category,_=classify_case({'status':'needs_medication_clarification'},[],{})
        self.assertEqual(category,'entity_resolution_or_clarification')


if __name__=='__main__':
    unittest.main()
