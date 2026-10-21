# -*- coding: utf-8 -*-
"""V2.11 离线测试：来源核验、精确绑定、歧义/冲突、安全与回滚。"""
from pathlib import Path
import json
import tempfile
import sys
from zipfile import ZipFile
import graph_chunk_linker_v211 as linker

ROOT=Path(__file__).resolve().parent

def get_fixture():
    # V2.11 ZIP 内包含 V2.7 图谱与教材快照。
    graph=json.loads((ROOT/'knowledge_graph_v27.json').read_text(encoding='utf-8'))
    source=ROOT/'graph_sources'/'神经网络与深度学习.md'
    if not source.is_file():
        source=ROOT/'神经网络与深度学习.md'
    assert source.is_file()
    sem=[e for e in graph['edges'] if e['relation'] in linker.SEMANTIC]
    assert len(sem)==12
    return graph,source,sem

def write_snapshot(path,rows):
    path.write_text(
        ''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),
        encoding='utf-8'
    )
def mk(e,cid='chroma-native-id-001',source=None,section=None,content=None):
    ev=e['evidence']
    return {
        'chunk_id':cid,
        'source':source if source is not None else ev['source_file'],
        'section_path':section if section is not None else ev['section_path'],
        'content':content if content is not None else ('章节：'+ev['section_path']+'\n'+linker.quote_for_matching(e)),
    }
def statuses(report):
    return report['metadata']['status_counts']

def test_real_graph_original_book_provenance():
    graph,source,sem=get_fixture()
    assert linker.check_graph_source(graph,source)
    assert all(linker.quote_for_matching(e) for e in sem)

def test_unique_matched_chunk_required():
    graph,source,edges=get_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        s=Path(tmp)/'snap.jsonl'
        write_snapshot(s,[mk(edges[0])])
        report=linker.build_link_report(graph,source,None,s)
        assert report['metadata']['semantic_claims']==12
        assert statuses(report).get('verified')==3
        assert statuses(report).get('unbound')==9
        assert all(row['chunk_ids']==['chroma-native-id-001'] for row in report['claims'] if row['status']=='verified')

def test_false_match_wrong_source_rejected():
    graph,source,edges=get_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        s=Path(tmp)/'snap.jsonl'
        write_snapshot(s,[mk(edges[0],source='别的书.md')])
        report=linker.build_link_report(graph,source,None,s)
        assert statuses(report).get('verified',0)==0

def test_wrong_section_yields_conflict_not_verified():
    graph,source,edges=get_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        s=Path(tmp)/'snap.jsonl'
        write_snapshot(s,[mk(edges[0],section='第99章 不相关章节')])
        report=linker.build_link_report(graph,source,None,s)
        assert statuses(report).get('verified',0)==0
        assert statuses(report).get('section_conflict')==3
        assert all(not r['chunk_ids'] for r in report['claims'])

def test_ambiguous_chunks_do_not_become_verified():
    graph,source,edges=get_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        s=Path(tmp)/'snap.jsonl'
        write_snapshot(s,[mk(edges[0],cid='a'),mk(edges[0],cid='b')])
        report=linker.build_link_report(graph,source,None,s)
        assert statuses(report).get('ambiguous')==3
        assert statuses(report).get('verified',0)==0
        assert all(not r['chunk_ids'] for r in report['claims'])

def test_section_alone_never_verified():
    graph,source,edges=get_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        s=Path(tmp)/'snap.jsonl'
        write_snapshot(s,[mk(edges[0],content='这一 Chunk 只碰巧有相同章节，根本没有证明关系的原文。')])
        report=linker.build_link_report(graph,source,None,s)
        assert statuses(report).get('verified',0)==0

def test_duplicate_ids_are_rejected():
    graph,source,edges=get_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        s=Path(tmp)/'snap.jsonl'
        write_snapshot(s,[mk(edges[0]),mk(edges[0])])
        try:linker.read_snapshot(s)
        except ValueError as err: assert '重复' in str(err)
        else:raise AssertionError('duplicate ids accepted')

def test_source_change_or_tampering_detected():
    graph,source,edges=get_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp)
        s=d/'snap.jsonl'
        write_snapshot(s,[mk(edges[0])])
        links=d/'links.json'
        output=linker.build_link_report(graph,source,None,s)
        links.write_text(json.dumps(output,ensure_ascii=False),encoding='utf8')
        assert linker.verify_links(graph,source,s,links)==12
        result=json.loads(links.read_text(encoding='utf8'))
        result['claims'][0]['status']='verified'
        result['claims'][0]['chunk_ids']=['invented-id']
        links.write_text(json.dumps(result,ensure_ascii=False),encoding='utf8')
        try:linker.verify_links(graph,source,s,links)
        except ValueError as err:assert '不一致' in str(err)
        else:raise AssertionError('tamper accepted')

def test_changed_snapshot_hash_detected():
    graph,source,edges=get_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp); s=d/'snap.jsonl'
        write_snapshot(s,[mk(edges[0])])
        report=linker.build_link_report(graph,source,None,s)
        path=d/'links.json'
        path.write_text(json.dumps(report,ensure_ascii=False),encoding='utf8')
        write_snapshot(s,[mk(edges[0],content='changed content')])
        try:linker.verify_links(graph,source,s,path)
        except ValueError as err:assert '快照' in str(err)
        else:raise AssertionError('changed source not detected')

def test_cli_link_verify_end_to_end_and_no_overwrite():
    import subprocess,sys
    graph,source,edges=get_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp); snapshot=d/'snap.jsonl'
        write_snapshot(snapshot,[mk(edges[0])])
        graphfile=ROOT/'knowledge_graph_v27.json'
        links=d/'links.json'; report=d/'integrity.json'
        cli=[sys.executable,'-S',str(ROOT/'graph_chunk_linker_v211.py')]
        args=['--graph',str(graphfile),'--source',str(source),
              '--chunks',str(snapshot),'--links',str(links)]
        build=subprocess.run(cli+['link']+args+['--report',str(report)],
                             text=True,capture_output=True,timeout=20)
        assert build.returncode==0,build.stderr
        assert links.exists() and report.exists()
        check=subprocess.run(cli+['verify']+args,
                             text=True,capture_output=True,timeout=20)
        assert check.returncode==0,check.stderr
        repeat=subprocess.run(cli+['link']+args+['--report',str(report)],
                              text=True,capture_output=True,timeout=20)
        assert repeat.returncode!=0

def test_export_is_explicit_and_no_network_dependencies():
    import ast
    p=ROOT/'graph_chunk_linker_v211.py'
    tree=ast.parse(p.read_text(encoding='utf8'))
    forbidden={'dashscope','requests','urllib3','openai','langchain'}
    modules=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            modules.update(n.name.split('.')[0] for n in node.names)
        if isinstance(node,ast.ImportFrom) and node.module:
            modules.add(node.module.split('.')[0])
    assert not (forbidden & modules)
    assert 'chromadb' in modules
    # Chroma 未安装的机器上 link/verify 也能运行

if __name__=='__main__':
    functions=sorted((name,fn) for name,fn in globals().items() if name.startswith('test_') and callable(fn))
    for name,fn in functions:
        fn()
        print('PASS:',name)
    print(f'V2.11: {len(functions)} offline tests passed (no Qwen, no Chroma writes).')
