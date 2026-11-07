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
.kb-sg-node-drugclass{background:#f8f3e9;border-color:#e9dbbd}
.kb-sg-node-disease{background:#f5f0fc;border-color:#ded2ee}
.kb-sg-node-diseasecategory{background:#edf6f1;border-color:#cde3d6}
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
.kb-sg-source-list{padding:4px 0 12px;border-top:1px solid #edf1f6}
.kb-sg-source-item{padding:9px 0;color:#60758d;font-size:11.5px}
.kb-sg-source-name{font-weight:600;color:#4d6884;margin-right:10px}
.kb-sg-technical{margin-top:5px;color:#8795a6;font-size:10.5px;overflow-wrap:anywhere}
.kb-sg-technical summary{cursor:pointer;color:#8795a6}
.kb-sg-technical code{white-space:pre-wrap;overflow-wrap:anywhere;font:10.5px/1.7 ui-monospace,monospace}
.kb-sg-warning{font-size:11px;color:#9b7646;margin-top:8px}
@media(max-width:780px){.kb-sg-node{width:120px;flex-basis:120px}.kb-sg-edge{width:96px;flex-basis:96px}.kb-sg-chain{justify-content:flex-start}}
"""

KINDS = {"Drug": "药品", "Ingredient": "成分", "DrugClass": "药物类别", "DatasetCategory": "数据集分类", "CatalogDrug": "分类目录药名", "Disease": "疾病", "DiseaseCategory": "疾病分类/科别", "Symptom": "症状", "Population": "人群", "Condition": "原文条件"}
SHORT_RELATIONS = {
    "DATASET_CATEGORY_MEMBERSHIP": "数据集分类归属",
    "DATASET_CATEGORY_PARENT": "数据集上级分类",
    "CLASS_MEMBERSHIP_ASSERTION": "原文类别归属",
    "DISEASE_CATEGORY_MENTION": "疾病资料列出的分类/科别",
    "HAS_INGREDIENT": "成分记载", "HAS_SYMPTOM": "症状记载",
    "INDICATION_MENTION": "用途章节提及", "CONTRAINDICATION_MENTION": "禁忌章节提及",
    "CAUTION_MENTION": "注意事项提及", "INTERACTION_MENTION": "相互作用提及",
    "ADVERSE_REACTION_MENTION": "不良反应提及",
    "MECHANISM_MENTION": "机制原文提及",
    "MECHANISM_STATEMENT_MENTION": "机制陈述原文记载",
    "MECHANISM_PARTNER_MENTION": "机制陈述提及相关药物",
    "MECHANISM_ENTITY_MENTION": "机制陈述提及机制概念",
    "STATEMENT_ENZYME_MENTION": "机制陈述涉及酶或转运体",
    "STATEMENT_PROCESS_MENTION": "机制陈述涉及药代过程",
    "STATEMENT_EXPOSURE_MENTION": "机制陈述涉及暴露指标",
    "STATEMENT_OUTCOME_MENTION": "机制陈述涉及作用结果",
    "METABOLISM_MENTION": "代谢关系原文提及",
    "ENZYME_MODULATION_MENTION": "酶调节关系原文提及",
    "EXPOSURE_CHANGE_MENTION": "药物暴露变化原文提及",
    "TRANSPORT_MENTION": "转运或排泄关系原文提及",
    "MECHANISM_OUTCOME_MENTION": "机制相关结果原文提及",
}


def relation_label(relation: str) -> str:
    """Keep unknown future relation IDs legible without implying a clinical claim."""
    if relation in SHORT_RELATIONS:
        return SHORT_RELATIONS[relation]
    return relation.replace("_", " ").lower() + "（原文关系）"
STATUS_LABELS = {
    "partial": "已取得部分来源，后续连接仍有缺口", "no_discovery_evidence": "未取得符合分类约束的来源示例",
    "ok": "已逐边核对原文", "no_connected_evidence_path": "未取得连通的证据路径",
    "no_exact_entity": "未识别到实体全名", "ambiguous_or_too_many_entities": "实体存在歧义或数量过多",
    "unresolved_named_entity": "部分制剂名称未收录或未解析", "question_too_long": "问题过长，请拆分查询",
    "source_verification_blocked": "来源或索引校验失败，请重建索引", "index_missing": "尚未构建关系图索引",
    "disabled": "关系图检索已关闭",
    "needs_medication_clarification": "请补充药盒名称或成分，以便核对具体产品",
    "not_graph_relation_question": "本问题使用药品原文章节检索",
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


def _source_key(edge):
    # Group clinical sections by record; JSON membership/parent by file.
    return (edge.get("source_file"), "" if edge.get("json_pointer") else edge.get("record_id"), edge.get("source_url"))


def _source_name(edge):
    file = str(edge.get("source_file") or "").replace("\\", "/").rsplit("/", 1)[-1]
    if file == "drug_categories_western.json":
        return "本地西药分类"
    if file == "drug_categories_chinese.json":
        return "本地中药分类"
    if edge.get("src", {}).get("kind") == "Drug":
        return str(edge["src"].get("name") or "药品资料")+" · 药品资料"
    return "本地来源资料"


def semantic_graph_html(trace) -> str:
    report = trace.get("semantic_graph") if isinstance(trace, dict) else None
    if not isinstance(report, dict):
        return ""
    paths = [p for p in report.get("paths", []) if isinstance(p, dict)]
    pieces = ['<style>' + CSS + '</style><section class="kb-sg" aria-label="多关系图可视化路径">',
              '<div class="kb-sg-title">多关系图路径 · ' + str(len(paths)) + ' 条</div>',
              '<div class="kb-sg-note">' + safe(STATUS_LABELS.get(report.get("status"), "未执行"))
              + '。箭头表示原文关系方向；路径连接资料陈述，不代表临床因果。横向滑动可查看完整路径。</div>']
    patient = report.get("query_understanding", {})
    labels = [g["surface"] + ' → ' + ('已点名制剂' if g["actual_patient_product_known"] else f"{g['candidate_count']} 个检索候选，实际产品待核对") for g in patient.get("medicines", [])]
    if labels:
        pieces.append('<div class="kb-sg-note">' + safe('；'.join(labels)) + '。候选名称仅用于查资料。</div>')
    bindings = {v["path_id"]: v for v in report.get("evidence_bindings", []) if isinstance(v, dict) and v.get("path_id")}
    discovery = report.get("category_discovery")
    if discovery:
        from medical_category_discovery import MISSING_LABELS
        plan = discovery["plan"]
        pieces.append('<div class="kb-sg-note">' + safe(
            f"分类展开：{plan['level_label']}；同类目录候选 {discovery['candidate_drugs_available']} 种，"
            f"本轮检索 {discovery['candidate_drugs_searched']} 种。分类仅筛选候选，医疗关系逐条核对具体药物原文。") + '</div>')
        if discovery.get("missing_links"):
            pieces.append('<details class="kb-sg-proof"><summary>检索范围与未连接的候选</summary><div class="kb-sg-note">' + safe('；'.join(
                MISSING_LABELS.get(k, k) for k in discovery["missing_links"])) + '</div></details>')
    cohort = report.get("plan", {}).get("disease_pair_search", {})
    if cohort:
        labels = [f"{s['disease']}：检索 {s['candidate_drugs_searched']} 种候选制剂，排除 {s.get('excluded_ambiguous_or_background_records', 0)} 条背景或含糊记录"
                  for s in cohort.get("cohorts", [])]
        pieces.append('<div class="kb-sg-note">' + safe('；'.join(labels))
                      + '。候选来自用途章节提及，不是完整治疗用药清单。</div>')
        missing_classes = sorted({v["class_name"] for v in cohort.get("missing_links", []) if v.get("class_name")})
        if missing_classes:
            pieces.append('<details class="kb-sg-proof"><summary>尚缺证据的类别连接</summary><div class="kb-sg-note">'
                          + safe('、'.join(missing_classes))
                          + '：相互作用资料有类别提及，但未取得另一组具体制剂的同来源类别归属。未将其补成关系边。</div></details>')
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
            relation = relation_label(edge.get("relation", ""))
            pieces.append('<div class="kb-sg-edge" aria-label="' + safe(relation + ' ' + arrow) + '">'
                          '<div class="kb-sg-relation">' + safe(relation) + '</div>'
                          '<div class="kb-sg-arrow kb-sg-arrow-' + direction + '" aria-hidden="true"></div></div>'
                          + _node(nodes[i+1]))
        sources = {}
        for edge in edges:
            sources.setdefault(_source_key(edge), []).append(edge)
        source_numbers = {key: n for n,key in enumerate(sources, 1)}
        pieces.append('</div></div><details class="kb-sg-proof"><summary>查看逐边原文与来源 · ' + str(len(sources)) + ' 个来源</summary>')
        for i, edge in enumerate(edges, 1):
            label = f"{i}. {edge['src']['name']} → {relation_label(edge.get('relation', ''))} → {edge['dst']['name']}"
            pieces.append('<div class="kb-sg-proof-edge"><div class="kb-sg-proof-label">' + safe(label) + '</div>'
                          '<div class="kb-sg-quote">' + safe(edge.get("quote")) + '</div>'
                          '<div class="kb-sg-source">来源 ' + str(source_numbers[_source_key(edge)]) + ' · '
                          + safe(_source_name(edge)) + '</div></div>')
        pieces.append('<div class="kb-sg-source-list">')
        for key, group in sources.items():
            edge = group[0]
            fields = {"composition":"成分", "drug_interactions":"药物相互作用", "warnings":"注意事项",
                      "contraindications":"禁忌", "indication":"适应症", "adverse_reactions":"不良反应"}
            sections = list(dict.fromkeys(fields.get(e.get('section_field'), '') for e in group))
            pieces.append('<div class="kb-sg-source-item"><span class="kb-sg-source-name">'
                          + safe(f"来源 {source_numbers[key]} · {_source_name(edge)}") + '</span>'
                          + _source_link(edge.get('source_url'))
                          + (' · ' + safe(' / '.join(s for s in sections if s)) if any(sections) else ''))
            # Preserve full provenance once per source, hidden behind a technical fold.
            locations = list(dict.fromkeys(
                'JSON路径：'+str(e['json_pointer']) if e.get('json_pointer') else
                '原文位置：'+str(e.get('quote_start'))+':'+str(e.get('quote_end')) for e in group))
            metadata = ['文件：'+str(edge.get('source_file') or ''), *locations]
            if not edge.get('json_pointer'):
                metadata.append('记录：'+str(edge.get('record_id') or ''))
            metadata.extend('SHA256：'+str(sha) for sha in dict.fromkeys(e.get('source_sha256') for e in group) if sha)
            pieces.append('<details class="kb-sg-technical"><summary>溯源详情</summary><code>'
                          + safe('\n'.join(metadata)) + '</code></details></div>')
        pieces.append('</div>')
        pieces.append('</details></div>')
    if report.get("truncated"):
        pieces.append('<div class="kb-sg-warning">本次达到检索预算，仅展示部分路径。</div>')
    if report.get("missing_requested_relations"):
        missing = '、'.join(relation_label(r) for r in report["missing_requested_relations"])
        pieces.append('<div class="kb-sg-warning">未取得的关系：' + safe(missing) + '。未命中不表示无风险。</div>')
    pieces.append('</section>')
    return ''.join(pieces)
