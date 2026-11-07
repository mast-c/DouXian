"""Evidence-backed typed property graph derived from local source snapshots.

SQLite is the graph storage, not a text similarity index. Nodes and typed edges
are independently traversable. Original Neo4j/Chroma and source files are read
only. Source mentions are deliberately NOT promoted to clinical assertions.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from urllib.parse import quote

VERSION = "semantic-graph-6"
DEFAULT_INDEX = "semantic_graph/medical_graph.sqlite"
SOURCE_DIR = "kg_output_clean_v2"
DISEASE_FILE = "data/raw/drug_db/MedGraphRAG-Drug-DB/medical.json"
RELATIONS = {
    "CLASS_MEMBERSHIP_ASSERTION": "原文明确的药品类别记载（未临床核验）",
    "HAS_SYMPTOM": "资料列出的症状",
    "DISEASE_CATEGORY_MENTION": "疾病资料列出的分类/科别",
    "HAS_INGREDIENT": "成分原文记载",
    "INDICATION_MENTION": "用途章节提及（非确证治疗）",
    "CONTRAINDICATION_MENTION": "禁忌章节提及（保留原文条件）",
    "CAUTION_MENTION": "注意事项提及",
    "INTERACTION_MENTION": "相互作用章节提及（非自动确认风险）",
    "ADVERSE_REACTION_MENTION": "不良反应章节提及",
}
FIELD_RELATIONS = {
    "contraindications": "CONTRAINDICATION_MENTION",
    "warnings": "CAUTION_MENTION",
    "drug_interactions": "INTERACTION_MENTION",
    "adverse_reactions": "ADVERSE_REACTION_MENTION",
}
MECHANISM_RELATIONS = {
    "MECHANISM_MENTION": "机制原文提及",
    "MECHANISM_STATEMENT_MENTION": "机制陈述原文记载",
    "MECHANISM_PARTNER_MENTION": "机制陈述提及相关药物",
    "MECHANISM_ENTITY_MENTION": "机制陈述提及机制概念",
    "METABOLISM_MENTION": "代谢关系原文提及",
    "ENZYME_MODULATION_MENTION": "酶抑制或诱导原文提及",
    "EXPOSURE_CHANGE_MENTION": "药物暴露变化原文提及",
    "TRANSPORT_MENTION": "转运或排泄关系原文提及",
    "MECHANISM_OUTCOME_MENTION": "机制相关结果原文提及",
}
MECHANISM_STATEMENT_RELATIONS = {
    "STATEMENT_ENZYME_MENTION": "机制陈述涉及酶或转运体",
    "STATEMENT_PROCESS_MENTION": "机制陈述涉及药代过程",
    "STATEMENT_EXPOSURE_MENTION": "机制陈述涉及暴露指标",
    "STATEMENT_OUTCOME_MENTION": "机制陈述涉及作用结果",
}
RELATIONS.update(MECHANISM_RELATIONS)
RELATIONS.update(MECHANISM_STATEMENT_RELATIONS)
# Exact terms and broad pharmacology concepts collected from source language.
# These are searchable concepts, not a pre-built pharmacological ontology.
MECHANISM_TERMS = (
    "细胞色素P450", "细胞色素 P450", "CYP3A4", "CYP2C9", "CYP2C19", "CYP2D6", "CYP1A2",
    "P-糖蛋白", "P-gp", "有机阴离子转运体", "有机阳离子转运体", "肾小管分泌", "肾脏排泄",
    "肝脏代谢", "肠肝循环", "首过效应", "首过代谢", "药物代谢酶", "代谢酶", "酶活性",
    "药理机制", "作用机制", "相互作用机制",
    "血药浓度", "血浆浓度", "血清浓度", "药物暴露", "生物利用度", "清除率", "半衰期",
    "吸收", "分布", "代谢", "排泄", "转运", "蛋白结合", "药效", "毒性", "肌病", "横纹肌溶解",
    "肾毒性", "肝毒性", "耳毒性", "出血风险", "肌病风险", "心律失常", "QT间期",
)
MECHANISM_CLAUSE = re.compile(
    r"代谢|酶|底物|抑制|诱导|浓度|暴露|生物利用度|清除率|半衰期|吸收|分布|排泄|转运|"
    r"蛋白结合|首过|毒性|肌病|横纹肌溶解|药效|相互作用机制|肾功能|肝功能|QT间期",
    re.I,
)
MECHANISM_RELATION_RULES = (
    ("ENZYME_MODULATION_MENTION", re.compile(r"酶|CYP|P-gp|P-糖蛋白|转运体|抑制|诱导", re.I)),
    ("METABOLISM_MENTION", re.compile(r"代谢|底物|首过", re.I)),
    ("EXPOSURE_CHANGE_MENTION", re.compile(r"血药浓度|血浆浓度|血清浓度|暴露|生物利用度|清除率|半衰期", re.I)),
    ("TRANSPORT_MENTION", re.compile(r"吸收|分布|排泄|转运|肾小管分泌", re.I)),
    ("MECHANISM_OUTCOME_MENTION", re.compile(r"毒性|肌病|横纹肌溶解|出血风险|心律失常|QT间期|药效", re.I)),
)
MECHANISM_TARGETS = {
    "ENZYME_MODULATION_MENTION": re.compile(r"CYP|细胞色素|酶$|P-gp|P-糖蛋白|转运体|抑制剂$|诱导剂$", re.I),
    "METABOLISM_MENTION": re.compile(r"代谢|底物|首过|CYP|细胞色素", re.I),
    "EXPOSURE_CHANGE_MENTION": re.compile(r"浓度|暴露|生物利用度|清除率|半衰期"),
    "TRANSPORT_MENTION": re.compile(r"吸收|分布|排泄|转运|肾小管分泌"),
    "MECHANISM_OUTCOME_MENTION": re.compile(r"毒性|肌病|横纹肌溶解|出血风险|心律失常|QT间期|药效"),
}
MECHANISM_STATEMENT_ROLE_RULES = (
    ("STATEMENT_ENZYME_MENTION", re.compile(r"CYP|细胞色素|药物代谢酶|代谢酶|酶活性|P-gp|P-糖蛋白|转运体", re.I)),
    ("STATEMENT_PROCESS_MENTION", re.compile(r"吸收|分布|代谢|排泄|转运|肾小管分泌|首过|蛋白结合", re.I)),
    ("STATEMENT_EXPOSURE_MENTION", re.compile(r"血药浓度|血浆浓度|血清浓度|药物暴露|生物利用度|清除率|半衰期", re.I)),
    ("STATEMENT_OUTCOME_MENTION", re.compile(r"药效|毒性|肌病|横纹肌溶解|肾毒性|肝毒性|耳毒性|出血风险|心律失常|QT间期", re.I)),
)
POPULATIONS = ("孕妇", "妊娠期", "哺乳期妇女", "哺乳期", "儿童", "新生儿", "老年人")
# Literal vocabulary, NOT a drug-to-class lookup or synonym map. Membership
# requires an explicit self-classification sentence in the same product record.
DRUG_CLASS_TERMS = (
    "糖皮质类固醇", "糖皮质激素", "皮质类固醇", "皮质激素", "非甾体抗炎药",
    "非甾体类抗炎药", "抗组胺药", "抗凝血药", "抗凝药", "抗血小板药",
    "噻嗪类利尿剂", "β受体阻滞剂", "β-受体阻滞剂", "钙通道阻滞剂",
    "大环内酯类抗生素", "四环素类药物", "磺胺类药物", "口服降糖药",
)
GENERIC_TERMS = {"其他", "疾病", "患者", "症状", "疼痛", "不适", "药物", "本品", "无", "尚不明确"}
MAX_TEXT = 24_000


class GraphBlocked(ValueError):
    """Snapshot integrity, source integrity or graph contract failed."""


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def entity_id(kind: str, name: str) -> str:
    # No salt stripping, formulation removal, synonym or brand equivalence.
    return kind.lower() + "_" + digest(kind + "\0" + name)[:24]


def index_path(root: Path | str, path: Path | str | None = None) -> Path:
    root = Path(root).resolve()
    target = Path(path or os.getenv("MEDICAL_SEMANTIC_GRAPH_INDEX", DEFAULT_INDEX))
    return (target if target.is_absolute() else root / target).resolve()


@lru_cache(maxsize=48)
def _hash_at_stat(path: str, size: int, mtime: int, ctime: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "sha256": _hash_at_stat(
        str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
    )}


def json_rows(path: Path):
    """Byte locations let retrieval re-open the actual source row, not trust DB text."""
    with path.open("rb") as stream:
        while True:
            at = stream.tell()
            raw = stream.readline()
            if not raw:
                return
            if raw.strip():
                yield json.loads(raw.decode("utf-8-sig")), at, len(raw), digest(raw)


class Lexicon:
    """Literal longest matches; linear trie scan with original character offsets."""
    def __init__(self, terms):
        self.trie = {}
        for name in terms:
            if len(name) < 2 or name in GENERIC_TERMS:
                continue
            node = self.trie
            for char in name:
                node = node.setdefault(char, {})
            node[None] = name

    def find(self, text: str):
        pos = 0
        while pos < len(text):
            node, cursor, match = self.trie, pos, None
            while cursor < len(text) and text[cursor] in node:
                node = node[text[cursor]]
                cursor += 1
                if None in node:
                    term = node[None]
                    # Do not match English fragments inside a different word.
                    if not (term[0].isascii() and term[0].isalpha() and pos and text[pos-1].isascii() and text[pos-1].isalnum()) and not (
                        term[-1].isascii() and term[-1].isalpha() and cursor < len(text) and text[cursor].isascii() and text[cursor].isalnum()
                    ):
                        match = (term, pos, cursor)
            if match:
                yield match
                pos = match[2]
            else:
                pos += 1


def ingredient_spans(text: str) -> list[tuple[str, int, int]]:
    """Conservative lists/atomic composition only; complex prose is omitted.

    Preserve salts and processing names. Never infer composition from drug name.
    Auxiliary ingredients are excluded; unparsed material remains in the source.
    """
    end = re.search(r"辅料|化学名|分子式|分子量|化学结构", text)
    body = text[:end.start()] if end else text
    prefix = re.match(r"\s*(?:本品)?(?:主要|活性)?成[分份](?:为|是)?\s*[:：]?\s*", body)
    start = prefix.end() if prefix else 0
    result = []
    for match in re.finditer(r"[^、，,；;。\n和及与]+", body[start:]):
        segment = match.group()
        token = segment.strip()
        # A literal ingredient followed by source-origin and formulation prose:
        # "胰岛素（猪或牛）的灭菌水溶液" states insulin explicitly. Preserve the
        # entire original composition as the edge quote, including origin.
        formulation = re.fullmatch(r"([\u4e00-\u9fffA-Za-zα-ωΑ-Ω·\-]+)(?:[（(][^（）()]{1,16}[）)])?的灭菌(?:水溶液|混悬液)", token)
        if formulation:
            token = formulation.group(1)
        # An explicit quantity can follow a literal ingredient in a simple list.
        token = re.sub(r"\s*\d+(?:\.\d+)?\s*(?:mg|g|ml|毫克|克|毫升|%|％).*$", "", token, flags=re.I).strip()
        if not 2 <= len(token) <= 28 or not re.fullmatch(r"[\u4e00-\u9fffA-Za-zα-ωΑ-Ω·\-（）() ]+", token):
            continue
        if re.search(r"本品|每|含有|含量|制剂|组成|组分|配制|制成|化学|相当|适量|等\d*味|应为|由|可|用于|含|为|是|系|不能|不得|不应|不详|不明|从|名称|推断|药物|作用|主要|成分|成份|建议|规定|详见", token):
            continue
        offset = start + match.start() + segment.index(token)
        result.append((token, offset, offset + len(token)))
    return result


def qualifiers(text: str) -> dict:
    # Directional cues are literal features, not inferred pharmacological facts.
    increase = bool(re.search(r"升高|增高|增加|增强|促进|延长", text))
    decrease = bool(re.search(r"降低|下降|减少|减弱|缩短", text))
    if increase and decrease:
        change_direction = "mixed_or_multiple"
    elif increase:
        change_direction = "increase_like"
    elif decrease:
        change_direction = "decrease_like"
    else:
        change_direction = "not_stated"
    inhibit = bool(re.search(r"抑制|阻断", text))
    induce = bool(re.search(r"诱导|促进表达", text))
    return {
        "negative_or_absent": bool(re.search(r"尚未见|未发现|未(?:能|曾)?(?:抑制|诱导|增加|升高|降低)|无明显|无不良|无相互作用|无(?:抑制|诱导)作用|不发生|没有(?:抑制|诱导)作用|无显著.*相互作用", text)),
        "uncertain": bool(re.search(r"尚不明确|尚不清楚|临床意义不明|相关性尚不明确|可能|据报", text)),
        "prohibition": bool(re.search(r"禁用|禁忌|禁止|不得|不应合用", text)),
        "caution": bool(re.search(r"慎用|谨慎|监测|注意", text)),
        "conditional": bool(re.search(r"如果|若|当|患者|妊娠|孕妇|剂量|超过|大于|≥|≤", text)),
        "change_direction": change_direction,
        "modulation": "mixed_or_multiple" if inhibit and induce else "inhibit" if inhibit else "induce" if induce else "not_stated",
    }


def clause_span(text: str, start: int, end: int) -> tuple[int, int]:
    # Keep comma-separated conditions and group risks together; do not drop negation.
    left = max(text.rfind("。", 0, start), text.rfind("\n", 0, start), text.rfind("；", 0, start)) + 1
    stops = [i for token in ("。", "\n", "；") if (i := text.find(token, end)) >= 0]
    right = min(stops) + 1 if stops else len(text)
    # A following anaphoric sentence can qualify the interaction just mentioned.
    # Keep it with the claim, especially "该相互作用的临床相关性尚不明确".
    for _ in range(3):
        remaining = text[right:]
        if not re.match(r"\s*(?:该相互作用|这种相互作用|此相互作用|其临床|但|然而|上述|这些|有关)", remaining):
            break
        stops = [i for token in ("。", "\n", "；") if (i := text.find(token, right)) >= 0]
        right = min(stops) + 1 if stops else len(text)
    return left, right


def literal_entity_boundary(text: str, start: int, end: int, *, query: bool = False) -> bool:
    """Conservatively reject substrings of unknown longer clinical/chemical names."""
    han = lambda c: bool(c) and "\u4e00" <= c <= "\u9fff"
    before = r"(?:对|有|与|及|或|和|如|为|于|伴|出现|导致|引起|包括|发生|增加|减少|降低|升高|提高|患|的|用|含|本品|严重|急性|慢性|活动性|经|由|后|使)"
    after = r"(?:患者|病人|者|时|及|或|和|与|的|等|伴|可|合|同|有|会|能|应|禁|慎|药物|类药|类|史|倾向|风险|发生|降低|增加|升高|增高|下降|血|被|不|代谢|排泄|抑制|诱导|清除|浓度|水平)"
    if query:
        before = r"(?:" + before + r"|请问|介绍|解释|分析|关于|使用|服用|比较|知道|一下|查询|查查|查找|列出|查|问|跟|以及|治|治疗|用于治疗|治疗了)"
        after = r"(?:" + after + r"|是|是否|有何|什么|怎么|如何|相关|之间|用的药|用药|常用药|所用药|治疗药|药跟|药和)"
    if start and han(text[start-1]) and not re.search(before + "$", text[:start]):
        return False
    if end < len(text) and han(text[end]) and not re.match(after, text[end:]):
        return False
    return True


def mechanism_entity_boundary(text: str, start: int, end: int, medicine_spans=()) -> bool:
    """Allow an exposure concept directly suffixed to a literal medicine name.

    Source text commonly omits a separator in phrases such as
    ``阿托伐他汀血药浓度升高``. Keep this exception limited to exposure
    concepts and exact medicine spans; other mechanism and clinical entities
    retain the conservative boundary rule.
    """
    if literal_entity_boundary(text, start, end):
        return True
    name = text[start:end]
    if re.fullmatch(r"CYP\d[A-Z0-9]*|细胞色素\s*P450|P-gp|P-糖蛋白|有机阴/阳离子转运体", name, re.I):
        if re.search(r"(?:抑制|诱导|作用于|经|由|代谢)$", text[:start]):
            return True
    if not re.fullmatch(r"血药浓度|血浆浓度|血清浓度|药物暴露|生物利用度|清除率|半衰期", name):
        return False
    return any(medicine_end == start for _, _, medicine_end in medicine_spans)


def class_assertion_spans(text: str):
    """Only explicit self-classifications; mentions/comparisons are not membership."""
    lexicon = Lexicon(DRUG_CLASS_TERMS)
    for name, start, end in lexicon.find(text):
        qs, qe = clause_span(text, start, end)
        before = text[qs:start]
        # Anchored to the current product and affirmative identity grammar.
        # Reject "本品与...", "本品不是...", "本品可能为...", other drugs,
        # withdrawn treatment and adverse-effect descriptions.
        if re.fullmatch(r"\s*(?:\d+[.、．]\s*)?本品(?:是|为|属于)(?:一种|一类)?(?:吸入用|吸入型|鼻用|局部用|口服|合成|长效|短效|选择性|非镇静性|强效|中效|弱效|的)*", before):
            yield name, qs, qe, start, end


def _schema(conn):
    conn.executescript("""
    PRAGMA foreign_keys=ON;
    CREATE TABLE entities(id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
                          UNIQUE(kind,name));
    CREATE INDEX entity_name ON entities(name);
    CREATE TABLE sources(id TEXT PRIMARY KEY, drug_id TEXT, record_id TEXT NOT NULL,
        field TEXT NOT NULL, url TEXT NOT NULL, authority TEXT NOT NULL,
        text TEXT NOT NULL, text_sha256 TEXT NOT NULL, file TEXT NOT NULL,
        byte_start INTEGER NOT NULL, byte_length INTEGER NOT NULL,
        row_sha256 TEXT NOT NULL, selector TEXT NOT NULL);
    CREATE TABLE edges(id TEXT PRIMARY KEY, src TEXT NOT NULL REFERENCES entities(id),
        dst TEXT NOT NULL REFERENCES entities(id), relation TEXT NOT NULL,
        source_id TEXT NOT NULL REFERENCES sources(id), quote_start INTEGER NOT NULL,
        quote_end INTEGER NOT NULL, mention_start INTEGER NOT NULL, mention_end INTEGER NOT NULL,
        qualifier_json TEXT NOT NULL, status TEXT NOT NULL);
    CREATE INDEX edge_src ON edges(src,relation);
    CREATE INDEX edge_dst ON edges(dst,relation);
    CREATE INDEX edge_source ON edges(source_id);
    CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    """)


def build_graph(root: Path | str, path: Path | str | None = None) -> dict:
    root = Path(root).resolve()
    target = index_path(root, path)
    if target.suffix.lower() != ".sqlite":
        raise GraphBlocked("Derived graph target must be a .sqlite file")
    if target.exists():
        manifest_path = target.with_suffix(".manifest.json")
        if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")).get("storage") != "local_sqlite_property_graph_from_source_exports":
            raise GraphBlocked("Refusing to overwrite a file that is not this derived graph")
    names = [f"{SOURCE_DIR}/{v}" for v in (
        "drugs.jsonl", "source_records.jsonl", "sections.jsonl", "disease_candidates_full.csv",
    )] + [DISEASE_FILE]
    before = {name: fingerprint(root / name) for name in names}
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="medical_graph_build_", suffix=".sqlite", dir=target.parent)
    os.close(handle)
    conn = sqlite3.connect(temporary)
    stats = Counter()
    entities, sources = {}, {}
    try:
        _schema(conn)

        def entity(kind, name, identity=None):
            name = str(name).strip()
            key = (kind, name)
            if key not in entities:
                identity = identity or entity_id(kind, name)
                conn.execute("INSERT INTO entities VALUES(?,?,?)", (identity, kind, name))
                entities[key] = identity
                stats["entities_" + kind] += 1
            return entities[key]

        def source(identity, drug, record, field, url, authority, text, file, loc, selector):
            if identity in sources:
                if sources[identity] != (drug, record, text):
                    raise GraphBlocked("Conflicting source identity: " + identity)
                return
            conn.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                identity, drug, record, field, url, authority, text, digest(text), file,
                *loc, selector,
            ))
            sources[identity] = (drug, record, text)

        def edge(src, dst, relation, source_id, start, end, mention_start=None, mention_end=None):
            if src == dst:
                return
            text = sources[source_id][2]
            ms = start if mention_start is None else mention_start
            me = end if mention_end is None else mention_end
            if not (0 <= start <= ms < me <= end <= len(text)):
                raise GraphBlocked("Invalid evidence coordinates")
            identity = "edge_" + digest(json.dumps([src, dst, relation, source_id, start, end, ms, me], ensure_ascii=False))[:32]
            inserted = conn.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                identity, src, dst, relation, source_id, start, end, ms, me,
                json.dumps(qualifiers(text[start:end]), ensure_ascii=False, sort_keys=True),
                "source_assertion_unverified" if relation in ("HAS_INGREDIENT", "HAS_SYMPTOM", "DISEASE_CATEGORY_MENTION", "CLASS_MEMBERSHIP_ASSERTION") else "literal_mention_not_clinically_validated",
            )).rowcount
            stats[relation] += inserted

        drug_by_id, drug_by_name = {}, {}
        for row, *_ in json_rows(root / names[0]):
            name = str(row.get("name") or "").strip()
            if len(name) >= 2 and not name.isdigit():
                identity = entity("Drug", name, row["id"])
                drug_by_id[row["id"]] = identity
                drug_by_name[name] = identity
        records = {v["id"]: v for v, *_ in json_rows(root / names[1]) if v.get("drug_id") in drug_by_id}
        disease_names, symptom_names = set(), set()
        for row, at, length, sha in json_rows(root / DISEASE_FILE):
            name = str(row.get("name") or "").strip()
            symptoms = row.get("symptom")
            if not name or not isinstance(symptoms, list):
                continue
            symptoms = [s for s in symptoms if isinstance(s, str) and s.strip() and len(s) >= 2]
            disease_names.add(name)
            disease = entity("Disease", name)
            text = "\n".join(symptoms)
            oid = str(row.get("_id", {}).get("$oid") or sha)
            sid = "disease_symptoms_" + digest(oid)[:24]
            source(sid, None, oid, "symptom", "", "third_party_unverified", text,
                   DISEASE_FILE, (at, length, sha), "symptom_list")
            cursor = 0
            for symptom in symptoms:
                symptom_names.add(symptom)
                edge(disease, entity("Symptom", symptom), "HAS_SYMPTOM", sid, cursor, cursor + len(symptom))
                cursor += len(symptom) + 1
            categories = [c.strip() for c in row.get("category", [])
                          if isinstance(c, str) and c.strip()]
            if categories:
                category_text = "\n".join(categories)
                category_sid = "disease_categories_" + digest(oid)[:24]
                source(category_sid, None, oid, "category", "", "third_party_unverified",
                       category_text, DISEASE_FILE, (at, length, sha), "disease_category_list")
                cursor = 0
                for category in categories:
                    edge(disease, entity("DiseaseCategory", category),
                         "DISEASE_CATEGORY_MENTION", category_sid,
                         cursor, cursor + len(category))
                    cursor += len(category) + 1
        sections = {}
        ingredient_names = set()
        for row, at, length, sha in json_rows(root / names[2]):
            record = records.get(row.get("record_id"))
            text = row.get("text")
            if not record or not isinstance(text, str) or not text.strip() or row.get("placeholder"):
                continue
            if len(text) > MAX_TEXT:
                stats["oversize_sections_omitted"] += 1
                continue
            sid, drug = row["id"], drug_by_id[record["drug_id"]]
            source(sid, drug, record["id"], row["field"], record.get("source_url", ""),
                   record.get("authority_status", "third_party_unverified"), text,
                   names[2], (at, length, sha), "text")
            sections[sid] = (drug, row["field"], text)
            if row["field"] == "composition":
                spans = ingredient_spans(text)
                if not spans:
                    stats["composition_sections_unparsed"] += 1
                for token, start, end in spans:
                    ingredient_names.add(token)
                    # Entire composition remains accessible alongside exact token coordinates.
                    edge(drug, entity("Ingredient", token), "HAS_INGREDIENT", sid, 0, len(text), start, end)
        # Existing candidate CSV is only admitted after exact section/record/drug/offset checks.
        with (root / names[3]).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    sid = row["section_id"]
                    drug, field, text = sections[sid]
                    start, end = int(row["quote_start"]), int(row["quote_end"])
                    ms, me = int(row["disease_start"]), int(row["disease_end"])
                    name = row["disease_name"].strip()
                    if (field != "indication" or drug_by_name.get(row["drug_name"]) != drug
                            or row["source_record_id"] != sources[sid][1]
                            or text[start:end] != row["evidence_quote"] or text[ms:me] != name
                            or not 0 <= start <= ms < me <= end <= len(text)):
                        raise ValueError("Candidate source mismatch")
                    disease_names.add(name)
                    edge(drug, entity("Disease", name), "INDICATION_MENTION", sid, start, end, ms, me)
                except (KeyError, ValueError, TypeError):
                    stats["invalid_indication_candidates_omitted"] += 1
        for population in POPULATIONS:
            entity("Population", population)
        for sid, (drug, field, text) in sections.items():
            if field not in ("indication", "warnings", "composition"):
                continue
            for name, start, end, ms, me in class_assertion_spans(text):
                edge(drug, entity("DrugClass", name), "CLASS_MEMBERSHIP_ASSERTION", sid, start, end, ms, me)
        clinical = {name: [] for name in disease_names | symptom_names | set(POPULATIONS)}
        for (kind, name), identity in entities.items():
            if kind in ("Disease", "Symptom", "Population") and name in clinical:
                clinical[name].append(identity)
        medicines = {}
        for (kind, name), identity in entities.items():
            if kind in ("Drug", "Ingredient", "DrugClass"):
                medicines.setdefault(name, []).append(identity)
        # Also retain unlinked class mentions as explicit missing-link leads.
        # Creating a class node does not assert membership for any drug.
        for name in DRUG_CLASS_TERMS:
            identity = entity("DrugClass", name)
            if identity not in medicines.setdefault(name, []):
                medicines[name].append(identity)
        clinical_lexicon, medicine_lexicon = Lexicon(clinical), Lexicon(medicines)
        mechanism_lexicon = Lexicon(MECHANISM_TERMS)
        mechanism_partner_ids = {identity for (kind, _), identity in entities.items()
                                 if kind in ("Drug", "Ingredient")}
        own_ingredients = {}
        for drug_id, record_id, ingredient in conn.execute("""SELECT e.src,s.record_id,e.dst FROM edges e
                JOIN sources s ON s.id=e.source_id WHERE e.relation='HAS_INGREDIENT'"""):
            own_ingredients.setdefault((drug_id, record_id), set()).add(ingredient)
        for sid, (drug, field, text) in sections.items():
            # Index explicit mechanism language across all source sections. Each edge
            # points to a literal concept and retains the whole source clause.
            for clause in re.finditer(r"[^。；;\n]+[。；;]?", text):
                raw = clause.group()
                if not MECHANISM_CLAUSE.search(raw):
                    continue
                qs = clause.start() + len(raw) - len(raw.lstrip())
                qe = clause.start() + len(raw.rstrip())
                if qe <= qs:
                    continue
                statement_text = text[qs:qe]
                statement = entity("MechanismStatement", statement_text)
                edge(drug, statement, "MECHANISM_STATEMENT_MENTION", sid, qs, qe, qs, qe)
                medicine_mentions = [(name, a, b) for name, a, b in medicine_lexicon.find(raw)
                                     if literal_entity_boundary(raw, a, b)]
                concepts = list(mechanism_lexicon.find(raw))
                concepts = [(name, a, b) for name, a, b in concepts
                            if mechanism_entity_boundary(raw, a, b, medicine_mentions)]
                if not concepts:
                    generic = MECHANISM_CLAUSE.search(raw)
                    concepts = [(generic.group(), generic.start(), generic.end())] if generic and len(generic.group()) >= 2 else []
                    base_relation = "MECHANISM_MENTION"
                else:
                    base_relation = None
                for name, a, b in concepts:
                    ms, me = clause.start() + a, clause.start() + b
                    context = raw[max(0, a - 32):min(len(raw), b + 32)]
                    statement_roles = [relation for relation, pattern in MECHANISM_STATEMENT_ROLE_RULES
                                       if pattern.search(name)]
                    for role in statement_roles:
                        edge(statement, entity("Mechanism", name), role, sid, qs, qe, ms, me)
                    matched_relations = [rel for rel, pattern in MECHANISM_RELATION_RULES
                                         if pattern.search(context) and MECHANISM_TARGETS[rel].search(name)]
                    if not matched_relations:
                        matched_relations = [base_relation or "MECHANISM_MENTION"]
                    for relation in matched_relations:
                        edge(statement, entity("Mechanism", name), "MECHANISM_ENTITY_MENTION", sid, qs, qe, ms, me)
                        edge(drug, entity("Mechanism", name), relation, sid, qs, qe, ms, me)
                for name, ms, me in medicine_mentions:
                    for destination in medicines[name]:
                        if destination not in mechanism_partner_ids:
                            continue
                        if destination == drug or destination in own_ingredients.get((drug, sources[sid][1]), set()):
                            continue
                        edge(statement, destination, "MECHANISM_PARTNER_MENTION", sid, qs, qe,
                             clause.start() + ms, clause.start() + me)
            relation = FIELD_RELATIONS.get(field)
            if not relation:
                continue
            lexicon = medicine_lexicon if field == "drug_interactions" else clinical_lexicon
            targets = medicines if field == "drug_interactions" else clinical
            hits = list(lexicon.find(text))
            hits = [(name, ms, me) for name, ms, me in hits if literal_entity_boundary(text, ms, me)]
            for name, ms, me in hits:
                start, end = clause_span(text, ms, me)
                for destination in targets[name]:
                    if (field == "drug_interactions" and text[me:].lstrip().startswith("类")
                            and destination != entities.get(("DrugClass", name))):
                        # "四环素类" is a class reference, not exact tetracycline.
                        # No unreviewed taxonomy may silently bridge that gap.
                        stats["class_substring_interactions_omitted"] += 1
                        continue
                    if field == "drug_interactions" and destination in own_ingredients.get((drug, sources[sid][1]), set()):
                        # A self ingredient appearing as an outcome (e.g. its
                        # concentration increases) is not an interacting partner.
                        stats["self_ingredient_interaction_mentions_omitted"] += 1
                        continue
                    edge(drug, destination, relation, sid, start, end, ms, me)
            if field == "contraindications" and not hits:
                # Preserve otherwise unrecognized contraindication conditions as exact clause nodes.
                for match in re.finditer(r"[^。；\n]+[。；]?", text):
                    token = match.group().strip()
                    if 2 <= len(token) <= 180 and re.search(r"禁用|慎用|禁忌", token):
                        ms = match.start() + match.group().index(token)
                        edge(drug, entity("Condition", token), relation, sid, ms, ms + len(token))
        after = {name: fingerprint(root / name) for name in names}
        if before != after:
            raise GraphBlocked("Source snapshot changed during build")
        report = {
            "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "builder_sha256": fingerprint(Path(__file__))["sha256"],
            "inputs": before, "counts": dict(sorted(stats.items())),
            "nodes": conn.execute("SELECT count(*) FROM entities").fetchone()[0],
            "edges": conn.execute("SELECT count(*) FROM edges").fetchone()[0],
            "sources": len(sources), "clinical_validation": False,
            "storage": "local_sqlite_property_graph_from_source_exports",
        }
        conn.execute("INSERT INTO metadata VALUES('version',?)", (VERSION,))
        conn.commit()
        if conn.execute("PRAGMA foreign_key_check").fetchone():
            raise GraphBlocked("Graph endpoints missing")
        conn.close()
        report["index"] = fingerprint(Path(temporary))
        # Both outputs are disposable derived artifacts. Mismatched pair fails closed.
        os.replace(temporary, target)
        manifest = target.with_suffix(".manifest.json")
        temp_manifest = manifest.with_suffix(".json.tmp")
        temp_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_manifest, manifest)
        return report
    finally:
        conn.close()
        if Path(temporary).exists():
            Path(temporary).unlink()


def open_graph(root: Path | str, path: Path | str | None = None):
    root = Path(root).resolve()
    target = index_path(root, path)
    manifest = json.loads(target.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("version") != VERSION
            or manifest.get("builder_sha256") != fingerprint(Path(__file__))["sha256"]
            or fingerprint(target) != manifest.get("index")):
        raise GraphBlocked("Derived graph index changed; rebuild required")
    for name, expected in manifest["inputs"].items():
        source_path = (root / name).resolve()
        if not source_path.is_relative_to(root) or fingerprint(source_path) != expected:
            raise GraphBlocked("Source snapshot changed; rebuild required: " + name)
    conn = sqlite3.connect("file:" + quote(target.as_posix(), safe="/:") + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if conn.execute("SELECT value FROM metadata WHERE key='version'").fetchone()[0] != VERSION:
            raise GraphBlocked("Unsupported graph schema")
    except Exception:
        conn.close()
        raise
    return conn, manifest


def verify_source(root: Path, source: dict) -> None:
    path = (root / source["file"]).resolve()
    if not path.is_relative_to(root):
        raise GraphBlocked("Source outside project")
    with path.open("rb") as stream:
        stream.seek(source["byte_start"])
        raw = stream.read(source["byte_length"])
    if digest(raw) != source["row_sha256"]:
        raise GraphBlocked("Source row hash mismatch")
    row = json.loads(raw.decode("utf-8-sig"))
    if source["selector"] == "text":
        text = row["text"]
        if row["id"] != source["id"] or row["record_id"] != source["record_id"] or row["field"] != source["field"]:
            raise GraphBlocked("Source identity mismatch")
    elif source["selector"] == "symptom_list":
        text = "\n".join(s for s in row["symptom"] if isinstance(s, str) and s.strip() and len(s) >= 2)
    elif source["selector"] == "disease_category_list":
        text = "\n".join(s.strip() for s in row["category"] if isinstance(s, str) and s.strip())
    else:
        raise GraphBlocked("Unknown source selector")
    if text != source["text"] or digest(text) != source["text_sha256"]:
        raise GraphBlocked("Source text mismatch")
