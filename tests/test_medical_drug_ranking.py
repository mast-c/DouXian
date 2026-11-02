"""离线回归：检索/模型均使用替身，不调用外部服务或修改数据库。"""
import ast
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medical_drug_ranking import rank_drug_candidates, requested_drug_count
from medical_drug_ranking import narrow_candidates, load_guideline_cache
from medical_overview_answer import build_multi_drug_source_answer, scope_graph_discovered_drugs
from drug_graph_retriever import DrugGraphRetriever, format_candidate, compact_same_quote_candidates
from drug_graph_retriever import fields_for_question, is_disease_drug_list_question
from drug_graph_retriever import format_section


def candidate(i, name=None):
    return format_candidate({
        "drug_name": name or f"示例成分{i:02d}片", "term_name": "示例疾病",
        "evidence_quote": f"本品用于治疗示例疾病，适用于符合条件的成人。编号{i}。",
        "candidate_id": f"c{i}", "record_id": f"r{i}", "section_id": f"s{i}",
        "source_url": f"https://example.org/{i}",
    })


def proposal(row, **changes):
    return {"candidate_id": row["candidate_id"], "support_quote": row["evidence_quote"],
            "group": row["drug_name"].removesuffix("片"),
            "basis": "model_common_use", "guideline_quote": "", **changes}


class Ask:
    def __init__(self, rows, selected=None, indices=None):
        self.rows = rows
        self.selected = selected
        self.indices = indices
        self.calls = []

    def __call__(self, system, payload):
        self.calls.append(payload)
        if "drugs" in payload:
            return {"indices": self.indices if self.indices is not None
                    else list(reversed(range(len(payload["drugs"]))))}
        return {"selected": self.selected if self.selected is not None
                else [proposal(r) for r in reversed(self.rows)]}


class RankingTests(unittest.TestCase):
    def test_count_chinese_arabic_and_bound(self):
        for q, n in [("不少于10个", 10), ("至少十二种药", 12), ("列20种", 20),
                     ("列100个", 30), ("治疗示例疾病的药物", 5), ("两种药", 2)]:
            self.assertEqual(requested_drug_count(q), n)

    def test_late_alphabet_drugs_can_rank_first_and_ten_survive_render(self):
        rows = [candidate(i) for i in range(16)]
        ask = Ask(rows)
        selected, report = rank_drug_candidates("列10种治疗药物", rows, ask=ask)
        self.assertEqual([r["candidate_id"] for r in selected], [f"c{i}" for i in range(15, 5, -1)])
        self.assertEqual(len(ask.calls[0]["candidates"]), 16)
        self.assertEqual(len(ask.calls), 1)
        evidence = [{**r, "id": i+1} for i, r in enumerate(selected)]
        names = [r["drug_name"] for r in selected]
        kept, scope = scope_graph_discovered_drugs(evidence, names)
        answer, coverage = build_multi_drug_source_answer("列10种", kept, names, [], ranking_report=report)
        self.assertEqual(len(scope["retrieved_drugs"]), 10)
        self.assertEqual(len(coverage["listed_drugs"]), 10)
        self.assertEqual(answer.count("### "), 10)
        self.assertNotIn("排序方式", answer)
        self.assertNotIn("排序依据", answer)
        self.assertNotIn("[11]", answer)

    def test_invented_id_and_changed_quote_rejected(self):
        rows = [candidate(1)]
        ask = Ask(rows, [proposal(rows[0], candidate_id="invented"),
                         proposal(rows[0], support_quote="自行编造治疗示例疾病的文字")])
        selected, report = rank_drug_candidates("列10种", rows, ask=ask)
        self.assertEqual(selected, [])
        self.assertEqual(report["rejected_proposals"], 2)
        self.assertEqual(report["shortfall"], 10)

    def test_complete_short_indication_is_not_rejected_by_character_count(self):
        row = candidate(1)
        row['evidence_quote'] = '示例疾病。'
        chosen, report = rank_drug_candidates('列10种', [row], ask=Ask([row]))
        self.assertEqual(len(chosen), 1)
        self.assertEqual(report['rejected_proposals'], 0)

    def test_short_ids_recover_full_original_quotes_and_default_five(self):
        rows = [candidate(i) for i in range(8)]
        def ask(system, payload):
            self.assertEqual(payload['requested_count'], 5)
            self.assertEqual(payload['proposal_count'], 7)
            self.assertTrue(all(r['candidate_id'].startswith('pick_') for r in payload['candidates']))
            return {'selected':[{'candidate_id':r['candidate_id']} for r in payload['candidates']]}
        selected, report = rank_drug_candidates('治疗示例疾病的药物', rows, ask=ask)
        self.assertEqual(len(selected), 5)
        for row in selected:
            self.assertTrue(row['candidate_id'].startswith('c'))
            self.assertEqual(row['ranking_support_quote'], row['evidence_quote'])
        self.assertEqual(report['model_calls'], 1)

    def test_list_only_skip_preserves_retrieval_for_detail_questions_or_failures(self):
        tree = ast.parse((ROOT/'rag.py').read_text(encoding='utf-8-sig'))
        branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                      and 'skipped_list_only' in ast.unparse(n)
                      and 'tool_name ==' in ast.unparse(n.test))
        code = compile(ast.Expression(branch.test), '<skip>', 'eval')
        env = dict(tool_name='search_knowledge_base', overview_question=True,
            graph_result={'selected_drugs':['示例药'], 'supplement_fields':[]},
            fields_for_question=fields_for_question, re=__import__('re'))
        for q, expected in [('治疗示例疾病的药物', True),
                            ('治疗示例疾病的药物及用法用量', False),
                            ('治疗示例疾病的药物及作用机制', False),
                            ('治疗示例疾病的药物及禁忌', False)]:
            self.assertEqual(eval(code, {**env, 'question':q}), expected)
        env['graph_result'] = {'selected_drugs':[]}
        self.assertFalse(eval(code, {**env, 'question':'治疗示例疾病的药物'}))

    def test_guideline_web_search_uses_short_request_without_changing_regular_search(self):
        from unittest.mock import patch
        import search_tool
        with patch.object(search_tool, '_run_web_search', return_value='{}') as run:
            search_tool.search_guideline_summary('疾病指南')
            run.assert_called_once_with('疾病指南', timeout=20.0, max_retries=0, max_output_tokens=1800)
            run.reset_mock()
            search_tool.web_search.invoke({'query':'普通联网查询'})
            run.assert_called_once_with('普通联网查询')

    def test_allergic_rhinitis_dosage_does_not_request_contraindications(self):
        for phrase in ['用量用法', '用法用量', '用量', '用法']:
            question = '过敏性鼻炎治疗药物及' + phrase
            self.assertTrue(is_disease_drug_list_question(question))
            self.assertEqual(fields_for_question(question), ['indication', 'dosage'])
        for question in ['过敏性鼻炎治疗药物及禁忌', '对氯雷他定过敏能用吗', '有药物过敏史能用吗']:
            self.assertIn('contraindications', fields_for_question(question))

    def test_dosage_display_retains_conditions_and_explicit_missing_notice(self):
        rows = [candidate(i) for i in range(3)]
        evidence = [{**row, 'id':i+1, 'ranking_support_quote':row['evidence_quote']}
                    for i, row in enumerate(rows)]
        text = '成人用法原文。' * 30 + '特殊人群：应由医师根据年龄及肝肾功能调整，不能照搬成人剂量。'
        section = format_section({'drug_name':rows[0]['drug_name'], 'section_field':'dosage',
            'section_text':text, 'record_id':'r1', 'section_id':'dose1'})
        unknown = format_section({'drug_name':rows[1]['drug_name'], 'section_field':'dosage',
            'section_text':'尚不明确', 'record_id':'r2', 'section_id':'dose2'})
        evidence += [{**section, 'id':4}, {**unknown, 'id':5}]
        body, coverage = build_multi_drug_source_answer('药物及用量用法', evidence,
            [r['drug_name'] for r in rows], ['dosage'], ranking_report={'requested_count':3})
        self.assertIn(text + ' [4]', body)
        self.assertIn('不能据此推断剂量或服用方法。[5]', body)
        self.assertIn('本轮未获得该药的用法用量原文', body)
        self.assertEqual(coverage['supported_cells'], 2)
        self.assertEqual(coverage['missing_cells'], 1)

    def test_guideline_claim_requires_actual_search_and_exact_summary(self):
        row = candidate(1, "示例成分片")
        quote = "示例指南检索摘要列出了示例成分作为一种治疗选择。"
        search = lambda q: {"ok": True, "search_calls": 1, "answer": quote,
                            "sources": [{"url": "https://example.org/guide", "title": "示例指南"}]}
        ask = Ask([row], [proposal(row, basis="guideline_search", guideline_quote=quote)])
        chosen, report = rank_drug_candidates("列10种", [row], ask=ask, search=search)
        self.assertEqual(chosen[0]["ranking_basis"], "guideline_search")
        self.assertFalse(report["clinical_validation"])
        fake = Ask([row], [proposal(row, basis="guideline_search", guideline_quote="不存在的指南引用")])
        chosen, _ = rank_drug_candidates("列10种", [row], ask=fake, search=search)
        self.assertEqual(chosen[0]["ranking_basis"], "model_common_use")

    def test_bounded_pool_prioritizes_late_guideline_drug_and_direct_indication(self):
        rows = [candidate(i) for i in range(200)]
        rows.append(candidate(201, "常用成分片"))
        special = candidate(202, "常用成分片")
        special['evidence_quote'] = '合并示例疾病的其他疾病的治疗。'
        rows.insert(0, special)
        pool, report = narrow_candidates(rows, ['示例疾病'], '指南提及常用成分。', 10)
        self.assertEqual(pool[0]['candidate_id'], 'c201')
        self.assertLessEqual(report['model_candidate_drugs'], 30)
        self.assertLessEqual(report['model_candidate_nodes'], 60)
        self.assertLessEqual(report['model_evidence_characters'], 24000)

    def test_recent_cache_avoids_live_search_and_expired_cache_is_rejected(self):
        from tempfile import TemporaryDirectory
        from datetime import date
        with TemporaryDirectory() as temp:
            path = Path(temp)/'cache.json'
            path.write_text(json.dumps({'entries':[{'diseases':['示例疾病'],
                'retrieved_date':date.today().isoformat(), 'summary':'示例成分01可供参考。',
                'sources':[{'url':'https://example.org/guide'}]}]}), encoding='utf-8')
            def search(q):
                self.fail('缓存命中不应重新联网')
            row = candidate(1)
            _, report = rank_drug_candidates('列10种', [row], ask=Ask([row]), search=search, cache_path=path)
            self.assertEqual(report['guideline_status'], 'cached_search_summary')
            self.assertIsNone(load_guideline_cache(['另一疾病'], path=path))
            self.assertIsNone(load_guideline_cache(['示例疾病'], path=path, today=date(2099,1,1)))

    def test_five_drug_scope_preserves_diversity_and_ignores_dosage_words(self):
        rows = [candidate(i, f'无关成分{i}缓释片') for i in range(30)]
        rows += [candidate(40+i, '常用成分'+suffix)
                 for i, suffix in enumerate(['片', '胶囊', '缓释片', '分散片'])]
        rows += [candidate(50+i, f'优选成分{i}片') for i in range(12)]
        summary = '指南：常用成分，以及' + '、'.join(f'优选成分{i}' for i in range(12)) + '；缓释片需区分。'
        pool, report = narrow_candidates(rows, ['示例疾病'], summary, 5)
        names = {r['drug_name'] for r in pool}
        self.assertEqual(report['model_candidate_drugs'], 15)
        self.assertIn('优选成分11片', names)
        self.assertIn('常用成分片', names)
        self.assertTrue(all('无关成分' not in r['drug_name'] for r in pool[:13]))

    def test_rhinitis_shortlist_keeps_nasal_sprays_without_guideline_cache(self):
        rows = [candidate(i, f'短药{i}片') for i in range(40)]
        for row in rows:
            row['evidence_quote'] = '用于过敏性鼻炎。'
        sprays = ['布地奈德鼻喷雾剂', '丙酸氟替卡松鼻喷雾剂', '丙酸倍氯米松鼻喷雾剂']
        for i, name in enumerate(sprays, 100):
            rows.append({**candidate(i, name), 'evidence_quote':'预防和治疗常年性及季节性过敏性鼻炎。'})
        for i, name in enumerate(['眼药滴眼液','皮肤软膏','其他注射液'], 200):
            rows.append({**candidate(i, name), 'evidence_quote':'用于过敏性鼻炎。'})
        pool, report = narrow_candidates(rows, ['过敏性鼻炎'], '', 5,
            question='过敏性鼻炎治疗药物及用量用法')
        names = {r['drug_name'] for r in pool}
        self.assertTrue(set(sprays) <= names)
        self.assertEqual(report['model_candidate_drugs'], 15)
        self.assertEqual(report['model_candidate_routes']['nasal'], 3)
        self.assertFalse({'眼药滴眼液','皮肤软膏','其他注射液'} & names)

    def test_nasal_form_suffix_does_not_hide_guideline_ingredient_match(self):
        rows = [candidate(i) for i in range(40)]
        rows.append(candidate(100, '布地奈德鼻喷雾剂'))
        pool, _ = narrow_candidates(rows, ['示例疾病'], '示例摘要提及布地奈德。', 5)
        self.assertEqual(pool[0]['drug_name'], '布地奈德鼻喷雾剂')

    def test_duplicate_ingredient_and_inadequate_count(self):
        rows = [candidate(1, "示例成分片"), candidate(2, "示例成分胶囊")]
        chosen, report = rank_drug_candidates("列10种", rows,
            ask=Ask(rows, [proposal(r, group="示例成分") for r in rows]))
        self.assertEqual(len(chosen), 1)
        body, _ = build_multi_drug_source_answer("列10种", [{**chosen[0], "id": 1}],
            [chosen[0]["drug_name"]], [], ranking_report=report)
        self.assertIn("本轮仅筛选并保留 1 种", body)

    def test_model_cannot_merge_salt_forms_using_similar_substring(self):
        rows = [candidate(1, "盐甲示例成分片"), candidate(2, "盐乙示例成分片")]
        chosen, _ = rank_drug_candidates("列10种", rows,
            ask=Ask(rows, [proposal(r, group="示例成分") for r in rows]))
        self.assertEqual(len(chosen), 2)

    def test_expanded_linkage_cap_keeps_ten_drugs_and_detail_sections(self):
        from medical_p303b_disease_runtime_v117 import merge_disease_evidence
        rows = [{**candidate(i), "id": i+1} for i in range(10)]
        kept, report = merge_disease_evidence(rows, [], cap=38)
        self.assertEqual(len(kept), 10)
        self.assertEqual(report['trimmed_items'], 0)

    def test_model_failure_does_not_fallback_to_fake_popular_list(self):
        def fail(*args):
            raise TimeoutError()
        selected, report = rank_drug_candidates("列10种", [candidate(1)], ask=fail)
        self.assertEqual(selected, [])
        self.assertEqual(report["status"], "ranking_failed")
        answer, _ = build_multi_drug_source_answer("列10种", [], [], [], ranking_report=report)
        self.assertIn("筛选未完成", answer)

    def test_search_failure_is_explicit_but_common_use_ranker_can_continue(self):
        row = candidate(1)
        def fail(q):
            self.assertNotIn("个人姓名", q)
            raise ConnectionError()
        chosen, report = rank_drug_candidates("个人姓名：列10种", [row], ask=Ask([row]), search=fail)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(report["guideline_status"], "unavailable")

    def test_retrieval_pool_not_old_ten_row_clamp_and_preserves_explicit_drug(self):
        retriever = DrugGraphRetriever(driver=object())
        calls = []
        def run(query, **args):
            calls.append((query, args))
            if "RETURN d.id AS id" in query:
                return [{"id": "drug1", "name": "指定示例药"}]
            if "RETURN t.id AS id" in query:
                return [{"id": "term1", "name": "示例疾病"}]
            return [{"drug_name": f"示例药{i}", "term_name": "示例疾病", "candidate_id": f"c{i}",
                     "evidence_quote": "用于治疗示例疾病的原文。"} for i in range(30)]
        retriever._run = run
        rows, report = retriever.search_disease_drug_pool("指定示例药治疗示例疾病")
        self.assertEqual(len(rows), 30)
        self.assertEqual(calls[-1][1]["drug_ids"], ["drug1"])
        self.assertEqual(calls[-1][1]["limit"], 5000)
        self.assertIn("d.id IN $drug_ids", calls[-1][0])
        self.assertFalse(report["truncated"])

    def test_actual_graph_tool_routes_ranking_and_ten_drugs(self):
        # 执行 rag.py 中的真实工具函数，避免初始化模型和在线数据库。
        tree = ast.parse((ROOT/"rag.py").read_text(encoding="utf-8-sig"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "search_drug_graph")
        fn.decorator_list = []
        rows = [candidate(i) for i in range(15)]
        ask = Ask(rows)
        class Model:
            model_name = 'qwen-test'
            root_client = SimpleNamespace(with_options=lambda **kw: (
                client_options.update(kw) or SimpleNamespace(chat=SimpleNamespace(completions=object()))))
            def model_copy(self, **kwargs):
                model_options.update(kwargs['update'])
                return self
            def invoke(self, messages):
                return SimpleNamespace(content=json.dumps(ask(messages[0].content,
                    json.loads(messages[1].content)), ensure_ascii=False))
        class Retriever:
            def search_disease_drug_pool(self, q):
                return rows, {"truncated": False}
            def search_sections(self, q, limit):
                return []
            def search_fields_for_drugs(self, names, fields, **kwargs):
                self.names = names
                self.fields = fields
                self.limits = kwargs
                return [format_section({'drug_name':name, 'section_field':field,
                    'section_text':'示例用法原文，仅供回归。', 'record_id':name,
                    'section_id':name+field}) for name in names for field in fields]
        class Message:
            def __init__(self, content): self.content = content
        client_options, model_options = {}, {}
        env = dict(json=json, re=__import__('re'), os=__import__('os'),
            self=SimpleNamespace(graph_retriever=Retriever(), chat_model=Model()),
            is_disease_drug_list_question=is_disease_drug_list_question,
            requested_drug_count=requested_drug_count, rank_drug_candidates=rank_drug_candidates,
            compact_same_quote_candidates=compact_same_quote_candidates,
            fields_for_question=fields_for_question, SystemMessage=Message, HumanMessage=Message,
            _plain_content=lambda x:x,
            web_search=SimpleNamespace(invoke=lambda x: {"ok": False}))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ROOT/"rag.py"), "exec"), env)
        result = json.loads(env["search_drug_graph"]("治疗示例疾病的药物，列出来不少于10个"))
        self.assertEqual(len(result["selected_drugs"]), 10)
        self.assertEqual(len(result["excerpts"]), 10)
        self.assertEqual(result["drug_ranking"]["status"], "ranked")
        self.assertFalse(hasattr(env['self'].graph_retriever, 'limits'))
        self.assertEqual(client_options, {'timeout': 40, 'max_retries': 0})
        self.assertEqual(model_options['extra_body']['enable_thinking'], False)
        result = json.loads(env['search_drug_graph']('治疗示例疾病的药物及用量用法'))
        self.assertEqual(result['supplement_fields'], ['dosage'])
        self.assertEqual(result['supplemented_sections'], 5)
        self.assertEqual(env['self'].graph_retriever.limits['max_sections'], 5)
        detail_assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'requested_details' for t in n.targets))
        requested_details = eval(compile(ast.Expression(detail_assignment.value), '<details>', 'eval'),
            {'graph_report':result})
        body, coverage = build_multi_drug_source_answer('治疗示例疾病的药物及用量用法',
            result['excerpts'], result['selected_drugs'], requested_details,
            ranking_report=result['drug_ranking'])
        self.assertEqual(body.count('**用法用量（原文节选）**'), 5)
        self.assertEqual(coverage['supported_cells'], 5)


if __name__ == "__main__":
    unittest.main()
