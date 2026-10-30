from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drug_graph_retriever import DrugGraphRetriever, format_candidate


SAMPLE = {
    "drug_name": "阿托伐他汀钙片", "term_name": "动脉粥样硬化",
    "candidate_id": "diseasecandidate_123", "evidence_quote": "高胆固醇血症并有动脉粥样硬化危险的病人。",
    "candidate_status": "pending_human_review", "clinical_status": "not_validated",
    "source_status": "third_party_unverified", "context_flag": "manual_semantic_review",
    "source_url": "https://www.yaopinnet.com/huayao/xyz.htm",
    "record_id": "record_123", "section_id": "section_123",
}

class Row:
    def __init__(self, d): self.d = d
    def data(self): return self.d

class TX:
    def __init__(self, history): self.history=history
    def run(self, query, **params):
        self.history.append((query, params))
        if "MATCH (d:Drug)" in query:
            return [Row({"id":"drug_1","name":"阿托伐他汀钙片"})] if "阿托伐他汀钙片" in params.get("question", "") else []
        if "MATCH (t:DiseaseTerm)" in query:
            return [Row({"id":"term_1","name":"动脉粥样硬化"})] if "动脉粥样硬化" in params.get("question", "") else []
        if "MATCH (d:Drug {id:" in query and "term_id" in params:
            return [Row(SAMPLE)]
        if "MATCH (d:Drug {id:" in query:
            return [Row(SAMPLE)]
        raise AssertionError(query)

class Session:
    def __init__(self, history): self.history=history
    def __enter__(self): return self
    def __exit__(self, typ, v, tb): pass
    def execute_read(self, callback): return callback(TX(self.history))

class Driver:
    def __init__(self): self.history=[]
    def session(self, database): return Session(self.history)


def test_graph_evidence_preserves_candidate_status():
    d=Driver()
    results=DrugGraphRetriever(driver=d).search("阿托伐他汀钙片治疗动脉粥样硬化吗？")
    assert len(results)==1
    text=results[0]["content"]
    assert "not_validated" in text
    assert "third_party_unverified" in text
    assert "不代表该药治疗/预防" in text
    assert "危险的病人" in text
    assert results[0]["source_url"].startswith("https://")
    assert all('limit' in params or 'question' in params for _, params in d.history)
    assert all('MERGE ' not in query and 'DELETE ' not in query for query,_ in d.history)


def test_empty_question_and_no_match():
    d=Driver(); g=DrugGraphRetriever(driver=d)
    assert g.search("") == []
    assert not d.history
    assert g.search("你好") == []


def test_bad_source_and_no_evidence_skipped():
    a=format_candidate({**SAMPLE, "source_url": "file:///secrets"})
    assert a["source_url"] == ""
    assert not a["source"].startswith("file:")
    assert format_candidate({**SAMPLE, "evidence_quote": ""}) is None


def test_pair_query_is_narrowed_and_does_not_mix_other_diseases():
    d=Driver()
    r=DrugGraphRetriever(driver=d).search("阿托伐他汀钙片与动脉粥样硬化的关系")
    assert len(r)==1
    assert any('term_id' in params for query,params in d.history)
    assert not any('ORDER BY d.name' in query for query,_ in d.history)
