"""Escaped, script-free arrow paths for the existing Streamlit trace panel."""
from __future__ import annotations

import html
from urllib.parse import urlsplit

CSS = """
.kb-sg{margin:18px 0 14px;color:#34465d;font:13px/1.65 system-ui,-apple-system,'Microsoft YaHei',sans-serif}
.kb-sg *{box-sizing:border-box}
.kb-sg-title{font-size:14px;font-weight:700;color:#294b6b;margin-bottom:5px}
.kb-sg-note{color:#7b8799;font-size:11.5px;margin:5px 0 12px}
.kb-sg-card{background:#fbfdff;border:1px solid #dfe7f0;border-radius:12px;margin:10px 0;overflow:hidden}
.kb-sg-head{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;padding:11px 14px;border-bottom:1px solid #edf1f6}
.kb-sg-name{font-weight:650;color:#365779}
.kb-sg-usage{font-size:11px;border-radius:20px;padding:3px 9px;background:#f0f3f7;color:#7a8697}
.kb-sg-usage-used{background:#e9f5ef;color:#2d7353}
.kb-sg-usage-context{background:#eaf1fc;color:#4c6d9e}
.kb-sg-scroll{overflow-x:auto;padding:21px 16px 19px;scrollbar-width:thin}
.kb-sg-chain{display:flex;align-items:center;width:max-content;min-width:100%;justify-content:center;gap:0}
.kb-sg-node{flex:0 0 140px;width:140px;min-height:76px;border:1px solid #d5e2ee;border-radius:11px;background:#f0f6fc;padding:10px 12px;text-align:center}
.kb-sg-node-drug{background:#edf5fc;border-color:#c5dcef}
.kb-sg-node-ingredient{background:#edf7f2;border-color:#cbe3d4}
.kb-sg-node-disease{background:#f5f0fc;border-color:#ded2ee}
.kb-sg-node-symptom{background:#fff7eb;border-color:#efdebe}
.kb-sg-node-population,.kb-sg-node-condition{background:#fff1ef;border-color:#ebd3ce}
.kb-sg-kind{font-size:10px;color:#7b8a9a;line-height:1.5;margin-bottom:4px}
.kb-sg-entity{font-size:13px;font-weight:650;line-height:1.55;overflow-wrap:anywhere;color:#304d67}
.kb-sg-edge{flex:0 0 116px;width:116px;display:flex;flex-direction:column;align-items:center;padding:0 7px;text-align:center}
.kb-sg-relation{font-size:10.5px;color:#6d819a;line-height:1.5;min-height:32px;display:flex;align-items:center;justify-content:center;max-width:110px}
.kb-sg-arrow{position:relative;width:100%;height:16px;margin:5px 0 16px}
.kb-sg-arrow::before{content:'';position:absolute;left:0;right:0;top:7px;border-top:1.5px solid #8da5bd}
.kb-sg-arrow-right::after{content:'';position:absolute;right:0;top:3px;width:8px;height:8px;border-right:1.5px solid #8da5bd;border-top:1.5px solid #8da5bd;transform:rotate(45deg)}
.kb-sg-arrow-left::after{content:'';position:absolute;left:0;top:3px;width:8px;height:8px;border-left:1.5px solid #8da5bd;border-bottom:1.5px solid #8da5bd;transform:rotate(45deg)}
.kb-sg-proof{border-top:1px solid #edf1f6;padding:0 14px}
.kb-sg-proof>summary{cursor:pointer;list-style:none;font-size:11.5px;color:#567492;padding:10px 0}
.kb-sg-proof>summary::-webkit-details-marker{display:none}
.kb-sg-proof>summary::before{content:'▸ ';color:#829ab2}
.kb-sg-proof[open]>summary::before{content:'▾ '}
.kb-sg-proof-edge{padding:10px 0;border-top:1px solid #edf1f6}
.kb-sg-proof-label{color:#4d6884;font-size:11px;font-weight:600;overflow-wrap:anywhere}
.kb-sg-quote{border-left:2px solid #c2d5e5;padding:7px 10px;margin:7px 0;color:#4c6076;background:#f4f8fc;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
.kb-sg-source{font-size:10.5px;color:#8795a6;overflow-wrap:anywhere}
.kb-sg-source a{color:#547a9c;text-decoration:none}
.kb-sg-warning{font-size:11px;color:#9b7646;margin-top:8px}
@media(max-width:780px){.kb-sg-node{width:120px;flex-basis:120px}.kb-sg-edge{width:96px;flex-basis:96px}.kb-sg-chain{justify-content:flex-start}}
"""

KINDS = {"Drug": "药品", "Ingredient": "成分", "Disease": "疾病", "Symptom": "症状", "Population": "人群", "Condition": "原文条件"}
SHORT_RELATIONS = {
    "HAS_INGREDIENT": "成分记载", "HAS_SYMPTOM": "症状记载",
    "INDICATION_MENTION": "用途章节提及", "CONTRAINDICATION_MENTION": "禁忌章节提及",
    "CAUTION_MENTION": "注意事项提及", "INTERACTION_MENTION": "相互作用提及",
    "ADVERSE_REACTION_MENTION": "不良反应提及",
}
STATUS_LABELS = {
    "ok": "已逐边核对原文", "no_connected_evidence_path": "未取得连通的证据路径",
    "no_exact_entity": "未识别到实体全名", "ambiguous_or_too_many_entities": "实体存在歧义或数量过多",
    "unresolved_named_entity": "部分制剂名称未收录或未解析", "question_too_long": "问题过长，请拆分查询",
    "source_verification_blocked": "来源或索引校验失败，请重建索引", "index_missing": "尚未构建关系图索引",
    "disabled": "关系图检索已关闭",
}


def safe(value) -> str:
    return html.escape(str(value or ""), quote=True)


def _node(node):
    kind = str(node.get("kind") or "")
    css_kind = kind.lower() if kind in KINDS else "unknown"
    return (
        '<div class="kb-sg-node kb-sg-node-' + css_kind + '">'
        '<div class="kb-sg-kind">' + safe(KINDS.get(kind, "实体")) + '</div>'
        '<div class="kb-sg-entity">' + safe(node.get("name")) + '</div></div>'
    )


def _source_link(url):
    try:
        parsed = urlsplit(str(url or ""))
        if len(str(url or "")) <= 2048 and parsed.scheme in ("http", "https") and parsed.netloc:
            return '<a target="_blank" rel="noopener noreferrer" href="' + safe(url) + '">查看来源 ↗</a>'
    except ValueError:
        pass
    return "本地来源记录"


def semantic_graph_html(trace) -> str:
    report = trace.get("semantic_graph") if isinstance(trace, dict) else None
    if not isinstance(report, dict):
        return ""
    paths = [p for p in report.get("paths", []) if isinstance(p, dict)]
    pieces = ['<style>' + CSS + '</style><section class="kb-sg" aria-label="多关系图可视化路径">',
              '<div class="kb-sg-title">多关系图路径 · ' + str(len(paths)) + ' 条</div>',
              '<div class="kb-sg-note">' + safe(STATUS_LABELS.get(report.get("status"), "未执行"))
              + '。箭头表示原文关系方向；路径连接资料陈述，不代表临床因果。横向滑动可查看完整路径。</div>']
    bindings = {v["path_id"]: v for v in report.get("evidence_bindings", []) if isinstance(v, dict) and v.get("path_id")}
    for number, path in enumerate(paths, 1):
        edges, nodes = path.get("edges", []), path.get("nodes", [])
        if len(nodes) != len(edges) + 1 or not edges:
            continue
        # Fail closed instead of drawing an invented connection in a historical trace.
        if any({e.get("src", {}).get("id"), e.get("dst", {}).get("id")} != {nodes[i].get("id"), nodes[i+1].get("id")} for i, e in enumerate(edges)):
            continue
        binding = bindings.get(path.get("id"), {})
        usage, style = ("回答已引用", "used") if binding.get("cited_in_answer") else ("已进入生成上下文", "context") if binding.get("accepted_for_generation") else ("未被回答采用", "retrieved")
        cite = f" · 证据 [{binding['evidence_id']}]" if isinstance(binding.get("evidence_id"), int) else ""
        pieces.append('<div class="kb-sg-card"><div class="kb-sg-head"><div class="kb-sg-name">'
                      + safe(f"路径 {number} · {len(edges)} 跳 · {len(set(e['relation'] for e in edges))} 类关系" + cite)
                      + '</div><div class="kb-sg-usage kb-sg-usage-' + style + '">' + usage + '</div></div>')
        pieces.append('<div class="kb-sg-scroll"><div class="kb-sg-chain">' + _node(nodes[0]))
        for i, edge in enumerate(edges):
            forward = edge["src"]["id"] == nodes[i]["id"]
            direction, arrow = ("right", "→") if forward else ("left", "←")
            relation = SHORT_RELATIONS.get(edge.get("relation"), "资料关系")
            pieces.append('<div class="kb-sg-edge" aria-label="' + safe(relation + ' ' + arrow) + '">'
                          '<div class="kb-sg-relation">' + safe(relation) + '</div>'
                          '<div class="kb-sg-arrow kb-sg-arrow-' + direction + '" aria-hidden="true"></div></div>'
                          + _node(nodes[i+1]))
        pieces.append('</div></div><details class="kb-sg-proof"><summary>逐边原文与来源（' + str(len(edges)) + ' 条）</summary>')
        for i, edge in enumerate(edges, 1):
            label = f"{i}. {edge['src']['name']} → {SHORT_RELATIONS.get(edge.get('relation'), '资料关系')} → {edge['dst']['name']}"
            pieces.append('<div class="kb-sg-proof-edge"><div class="kb-sg-proof-label">' + safe(label) + '</div>'
                          '<div class="kb-sg-quote">' + safe(edge.get("quote")) + '</div>'
                          '<div class="kb-sg-source">' + _source_link(edge.get("source_url"))
                          + ' · 原文位置 ' + safe(f"{edge.get('quote_start')}:{edge.get('quote_end')}")
                          + ' · 记录 ' + safe(edge.get("record_id")) + '</div></div>')
        pieces.append('</details></div>')
    if report.get("truncated"):
        pieces.append('<div class="kb-sg-warning">本次达到检索预算，仅展示部分路径。</div>')
    if report.get("missing_requested_relations"):
        missing = '、'.join(SHORT_RELATIONS.get(r, r) for r in report["missing_requested_relations"])
        pieces.append('<div class="kb-sg-warning">未取得的关系：' + safe(missing) + '。未命中不表示无风险。</div>')
    pieces.append('</section>')
    return ''.join(pieces)
