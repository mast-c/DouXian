"""Patient-language understanding. Query candidates never become medical edges.

Formulation/salt morphology expands SEARCH scope only. Brand aliases must be
literal entries in the hashed local drug export. No guessed brand equivalence,
fuzzy automatic correction, or assumption about the patient's actual product.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

from medical_semantic_graph import Lexicon, literal_entity_boundary, SOURCE_DIR
from medical_patient_context import (
    DURATION, EFFICACY, MEDICATION_ACTION, context_assertion,
    mention_role, symptom_mentions, timing_mentions,
)

FORMS = tuple(sorted((
    "缓释胶囊", "肠溶胶囊", "软胶囊", "分散片", "缓释片", "肠溶片", "咀嚼片", "控释片",
    "口崩片", "泡腾片", "口含片", "缓释颗粒", "鼻喷雾剂", "鼻气雾剂", "口服溶液", "口服混悬液",
    "干混悬剂", "混悬滴剂", "混悬液", "注射液", "注射用", "滴眼液", "滴耳液", "滴鼻液",
    "滴剂", "气雾剂", "喷雾剂", "胶囊剂", "胶囊", "片剂", "颗粒剂", "颗粒", "口服液", "糖浆",
    "软膏", "乳膏", "凝胶", "粉针剂", "粉剂", "栓剂", "丸剂", "溶液", "片", "丸", "散", "栓", "酊",
), key=len, reverse=True))
SALT_PREFIXES = ("盐酸", "硫酸", "氢溴酸", "醋酸", "枸橼酸", "马来酸", "酒石酸", "富马酸", "磷酸", "甲磺酸", "苯磺酸")
VAGUE_MEDICINES = ("降糖药", "胃药", "消炎药", "感冒药", "降压药", "止痛药", "抗过敏药", "鼻炎药", "头孢", "降脂药", "安眠药")
DOSE_FORM_WORDS = {"片", "胶囊", "口服液", "颗粒", "软膏", "药", "复方", "缓释", "注射液", "盐酸", "肠溶", "缓释片"}
# Retrieval vocabulary, not diagnoses or entity equivalence assertions.
PATIENT_TERMS = (
    (r"怀孕|有孕|备孕", "pregnancy", ["妊娠", "孕妇"]),
    (r"喂奶|母乳|哺乳", "breastfeeding", ["哺乳期"]),
    (r"宝宝|小孩|孩子|儿童|婴儿", "child", ["儿童用药"]),
    (r"肾不好|肾功能不好", "renal_question", ["肾功能", "肾脏"]),
    (r"肝不好|肝功能不好", "hepatic_question", ["肝功能", "肝脏"]),
    (r"拉肚子", "symptom_word", ["腹泻"]),
    (r"心慌", "symptom_word", ["心悸"]),
)
RISK_PATTERN = r"会(?:不会)?有(?:什么)?问题|有没有事|会不会(?:出事|有事|冲突|影响|中毒)|有事吗|要紧吗|有影响吗|有危险吗|有害吗|能不能|可不可以|能一起|可以一起|能吃吗|能用吗|行不行|能混着|能搭配|有没有影响|有冲突吗|还能吃|还可以吃|能再吃|可以再吃|有问题没"
CLASSIFICATION_PATTERN = r"属于|属不属于|归属|分类|类别|哪(?:一)?类|什么类|算不算|中药还是西药|西药还是中药|是中药|是西药|是不是.{0,12}(?:类|抗生素)"


def patient_intent(question: str, medicine_count: int = 0) -> dict:
    q = str(question or "")
    drug_discovery = bool(re.search(
        r"(?:吃|服|用|治疗|缓解).{0,8}(?:什么|哪种|哪些|哪类)(?:药|药物|制剂)"
        r"|(?:什么|哪种|哪些|哪类)(?:药|药物|制剂).{0,8}(?:治疗|缓解|适合)", q
    ))
    explicit = bool(re.search(r"相互作用|合用|同用|同服|联用|配伍|(?:一起|一块|同时)(?:吃|用|服|喝)|同吃|混着|搭配|冲突|相克|犯冲", q))
    risk = bool(re.search(RISK_PATTERN + r"|安全吗|安全性|风险", q))
    medication = bool(re.search(r"吃|服|用药|打针|喷|涂|药", q))
    interaction = explicit or (medicine_count >= 2 and medication and risk)
    fields = []
    if interaction:
        fields += ["drug_interactions", "warnings"]
    mechanism = bool(re.search(r"机制|原理|为什么(?:会|会有|会发生)|背后(?:的)?原因|药理(?:作用|原理)", q))
    if mechanism:
        fields += ["drug_interactions", "warnings"]
    if risk and medication:
        fields += ["contraindications", "warnings", "adverse_reactions"]
    if re.search(r"怎么吃|怎么用|怎么服|一天几次|吃多少|一次几片|饭前|饭后|剂量|用量", q):
        fields += ["dosage"]
    adverse = bool(re.search(r"副作用|不良反应|吃(?:了|完|药后).{0,16}(?:恶心|头晕|难受|拉肚子|起疹|不舒服|心慌|吐)|(?:越吃|吃完).{0,8}(?:疼|痛|痒)", q))
    reported_symptoms = [item for item in symptom_mentions(q) if item["assertion"] != "negated"]
    # Persistent pre-existing symptoms alone are not new drug reactions.
    new_symptoms = [item for item in reported_symptoms if not (
        re.search(r"(?:还是|仍然|仍|一直)$", q[max(0,item["start"]-4):item["start"]])
        or re.match(r"(?:还是|仍然|仍|一直)(?:好不了|没好|未改善|没改善|未缓解|没缓解)", q[item["end"]:])
    )]
    if medicine_count and new_symptoms and (
        re.search(r"(?:吃|服|用|喷|涂|擦|滴|喝|打)(?:了|完|后)|服用后|用药后|之后|以后", q)
        or re.search(r"出现|开始|今天|昨天|最近|这两天|后来", q) and re.search(MEDICATION_ACTION, q)
    ):
        adverse = True
    if symptom_mentions(q) and not reported_symptoms and not re.search(r"副作用|不良反应", q):
        adverse = False
    efficacy = bool(medication and re.search(EFFICACY, q))
    if adverse:
        fields += ["adverse_reactions", "warnings"]
    if re.search(r"干什么的|干嘛的|治什么|管什么|有什么用", q):
        fields += ["indication"]
    if efficacy:
        fields += ["indication", "dosage", "warnings"]
    return {"interaction_question": interaction, "mechanism_question": mechanism, "risk_question": risk,
            "drug_discovery_question": drug_discovery,
            "classification_question": bool(re.search(CLASSIFICATION_PATTERN, q)),
            "adverse_event_question": adverse, "efficacy_question": efficacy,
            "drug_causality_asserted": False,
            "medication_context": medication, "fields": list(dict.fromkeys(fields)),
            "basis": "patient_language_rules_no_clinical_inference"}


def formulation_stem(name: str) -> str:
    name = re.sub(r"[（(][^()（）]*[)）]$", "", name).strip()
    if name.startswith("注射用") and len(name) > 5:
        name = name[3:]
    for suffix in FORMS:
        if name.endswith(suffix) and len(name) >= len(suffix) + 2:
            return name[:-len(suffix)]
    return name


def core_candidate(stem: str) -> str:
    for salt in SALT_PREFIXES:
        if stem.startswith(salt) and len(stem) >= len(salt) + 3:
            return stem[len(salt):]
    return stem


def patient_boundary(text: str, start: int, end: int, *, exact_product=False) -> bool:
    """Do not require professional grammar around a known drug name."""
    han = lambda c: bool(c) and "\u4e00" <= c <= "\u9fff"
    before_ok = not start or not han(text[start-1]) or bool(re.search(
        r"(?:吃|吃了|吃着|吃过|服|服了|服用|用了|用着|使用|喝|喝了|打|打了|喷|喷了|涂|涂了|加|加了|加上|加用|换用|换成|换了|停了|是|叫|叫做|买了|买的|药是|包括|和|与|跟|及|还有|以及|配|配了|搭配|一起|然后|再|同时|拿了|开了|比较|对比|介绍|请问|咨询|关于|查一下|想问|在用|用的|吃的|用|我|我的|或|或者|还是)$", text[:start]))
    after_ok = end == len(text) or not han(text[end]) or bool(re.match(
        r"(?:的|是|和|与|跟|及|还|一块|同时|一起|能|可以|可不可以|会|有|没|是否|要|行|不|又|再|以后|后|之前|前|怎么|用|吃|服|吗|多久|一天|一日|每次|干什么|干嘛|治什么|管什么|安全吗|相克)", text[end:]))
    if re.search(r"(?:想了解|想知道|了解一下|了解|查询|介绍一下)$", text[:start]):
        before_ok = True
    # Set operators introduce catalog entities, not medication actions.
    if re.search(r"(?:排除|不包括|除了|除去|不要|包含)$", text[:start]):
        before_ok = True
    if re.match(r"(?:已经|大概|约)?" + DURATION + r"(?:了|后|以后|以来|，|,|。|$)", text[end:]):
        after_ok = True
    if re.match(r"(?:没效果|不管用|不见效|效果|好像|似乎|到底|是不是|怎么样|如何)", text[end:]):
        after_ok = True
    if re.match(r"(?:属于|属不属于|归属|算不算|算是|算哪|分类|类别|哪(?:一)?类)", text[end:]):
        after_ok = True
    # Known catalog names can be topics followed by a patient/action clause,
    # or objects of a clinician's prescription. Do not accept arbitrary Han
    # suffixes: "甲药片加强版" must not silently resolve to "甲药片".
    if re.search(r"(?:医生|医师|大夫|医院|门诊)(?:给我|给他|给她|给患者)?(?:开|配)(?:了|的)$", text[:start]):
        before_ok = True
    if re.match(r"(?:我|我们|他|她|患者|病人)(?:(?:已经|正在|一直|之前|刚刚|刚|现在|还在|早就|也|还|已|没|没有|从未|不再)){0,3}(?:吃|服用|服|使用|用|喷|涂|停用|停服|停了)", text[end:]):
        after_ok = True
    if exact_product and before_ok:
        # A complete catalog formulation is an entity before its intent is
        # known. Free-form narrative after it must not need a phrase whitelist.
        # Retain explicit checks for longer chemical/formulation names.
        extension = r"(?:钠|钙|钾|镁|锌|盐|复方|缓释|肠溶|控释|胶囊|颗粒|注射液|加强版|增强版|改良版)"
        return not bool(re.match(extension, text[end:]))
    return before_ok and after_ok


@lru_cache(maxsize=4)
def _name_catalog(root: str, export_size: int, export_mtime: int, node_signature: tuple):
    # Source file verified by open_graph before entry. Cache invalidates with
    # source stats and graph catalog contents; no mutable medical equivalence.
    nodes = {identity: {"id": identity, "kind": kind, "name": name} for identity, kind, name in node_signature}
    terms = {}
    products = {v["name"]: v["id"] for v in nodes.values() if v["kind"] == "Drug"}
    def put(token, identity, basis):
        token = unicodedata.normalize("NFKC", token).strip()
        if 2 <= len(token) <= 70 and token not in DOSE_FORM_WORDS and not token.isdigit():
            terms.setdefault(token, {})[(identity, basis)] = None
    for node in nodes.values():
        if node["kind"] == "Drug":
            put(node["name"], node["id"], "exact_product")
            unsalted = core_candidate(node["name"])
            if unsalted != node["name"]:
                put(unsalted, node["id"], "salt_name_search_candidate")
            # Parenthetical catalog synonyms do not make the shorter label
            # an exact patient formulation, but may be searched explicitly.
            short = re.sub(r"[（(][^()（）]*[)）]$", "", node["name"]).strip()
            if short != node["name"]:
                put(short, node["id"], "source_catalog_alias")
            stem = formulation_stem(node["name"])
            if stem != node["name"] and not stem.startswith("复方"):
                put(stem, node["id"], "formulation_search_candidate")
                if node["name"].endswith("鼻喷雾剂"):
                    # A shortened formulation is a search candidate, never an
                    # assertion that nasal and other spray routes are equivalent.
                    for form in ("喷雾剂", "喷雾", "喷剂", "鼻喷剂", "鼻喷"):
                        put(stem + form, node["id"], "partial_form_search_candidate")
                core = core_candidate(stem)
                if core != stem:
                    put(core, node["id"], "salt_name_search_candidate")
        elif node["kind"] == "Ingredient":
            put(node["name"], node["id"], "exact_ingredient")
    source = Path(root) / SOURCE_DIR / "drugs.jsonl"
    with source.open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            identity = products.get(row.get("name"))
            if identity:
                for alias in row.get("aliases", []):
                    if isinstance(alias, str):
                        put(alias, identity, "source_catalog_alias")
    return nodes, terms, Lexicon(terms)


def understand_question(question: str, *, root: Path, nodes: dict) -> dict:
    source = root / SOURCE_DIR / "drugs.jsonl"
    stat = source.stat()
    catalog_nodes, terms, lexicon = _name_catalog(str(root), stat.st_size, stat.st_mtime_ns,
        tuple((v["id"], v["kind"], v["name"]) for v in nodes.values() if v["kind"] in ("Drug", "Ingredient")))
    # Query offsets only; Unicode normalization is never applied to source text.
    q = unicodedata.normalize("NFKC", str(question))
    groups = []
    priority = {"exact_product": 0, "source_catalog_alias": 1, "exact_ingredient": 2, "partial_form_search_candidate": 3,
                "formulation_search_candidate": 3, "salt_name_search_candidate": 4}
    for token, start, end in lexicon.find(q[:600]):
        is_exact = any(basis == "exact_product" for _, basis in terms[token])
        if not patient_boundary(q, start, end, exact_product=is_exact):
            continue
        entries = sorted(terms[token], key=lambda v: (priority[v[1]], catalog_nodes[v[0]]["name"]))
        exact = [v for v in entries if v[1] == "exact_product"]
        aliases = [v for v in entries if v[1] == "source_catalog_alias"]
        ingredient_ids = [v[0] for v in entries if v[1] == "exact_ingredient"]
        candidates = exact or aliases or [v for v in entries if catalog_nodes[v[0]]["kind"] == "Drug"]
        route = ""
        prefix = q[max(0,start-16):start]
        if re.search(r"(?:吃|服用|服|喝)(?:了|着|过|的)?$", prefix):
            route = "oral"
        elif re.search(r"(?:打|注射)(?:了|的)?$", prefix):
            route = "injection"
        elif re.search(r"(?:涂|擦)(?:了|的)?$", prefix):
            route = "topical"
        elif re.search(r"(?:喷|喷了|喷着)$", prefix):
            route = "spray"
        elif re.search(r"一起吃|一起服|口服", q):
            route = "oral"
        excluded_routes = []
        if route and not exact:
            selected = []
            for entry in candidates:
                name = catalog_nodes[entry[0]]["name"]
                injection = "注射" in name or "粉针" in name
                topical = any(s in name for s in ("滴眼", "滴耳", "滴鼻", "软膏", "乳膏", "凝胶", "喷雾", "气雾", "栓", "外用"))
                spray = any(s in name for s in ("喷雾", "气雾"))
                if (route == "oral" and (injection or topical)) or (route == "injection" and not injection) or (route == "topical" and not topical) or (route == "spray" and not spray):
                    excluded_routes.append(name)
                else:
                    selected.append(entry)
            candidates = selected
        candidate_ids = list(dict.fromkeys(v[0] for v in candidates))
        existing = next((g for g in groups if g["surface"] == token), None)
        mention = {"surface": token, "start": start, "end": end, **mention_role(q, start, end)}
        if existing:
            existing["mentions"].append(mention)
            continue
        groups.append({
            "surface": token, "start": start, "end": end, "ingredient_ids": ingredient_ids,
            "candidate_drug_ids": candidate_ids[:40], "candidate_drug_names": [catalog_nodes[v]["name"] for v in candidate_ids[:40]],
            "candidate_count": len(candidate_ids), "candidates_truncated": len(candidate_ids) > 40,
            "basis": "exact_product" if exact else "source_catalog_alias" if aliases else "generic_search_scope",
            "match_bases": sorted({v[1] for v in entries}),
            "actual_patient_product_known": bool(exact),
            "clinical_equivalence_asserted": False,
            "route_hint": route, "route_excluded_candidates": excluded_routes,
            **mention_role(q, start, end),
            "mentions": [mention],
        })
    intent = patient_intent(q, len(groups))
    for previous, current in zip(groups, groups[1:]):
        if (current["role"] == "mentioned"
                and previous["role"] in ("current", "taken", "proposed", "not_taken", "stopped")
                and re.fullmatch(r"(?:和|与|跟|及|以及|、)", q[previous["end"]:current["start"]])):
            for key in ("role", "assertion"):
                current[key] = previous[key]
                current["mentions"][0][key] = previous[key]
            current["role_basis"] = "shared_medication_action"
            current["mentions"][0]["role_basis"] = "shared_medication_action"
    context = [{"surface": match.group(), "kind": kind, "retrieval_terms": synonyms,
                "assertion": context_assertion(q, match.start()),
                "diagnosis_asserted": False}
               for pattern, kind, synonyms in PATIENT_TERMS for match in re.finditer(pattern, q)]
    context.extend(symptom_mentions(q))
    clinical_names = tuple(sorted({node["name"] for node in nodes.values() if node["kind"] == "Disease"}))
    for name, start, end in _clinical_lexicon(clinical_names).find(q[:600]):
        if any(m["start"] < end and start < m["end"] for m in groups):
            continue
        context.append({"surface": name, "start": start, "end": end, "kind": "disease_mention",
                        "retrieval_terms": [name], "assertion": context_assertion(q, start),
                        "diagnosis_asserted": False})
    context = list({(c["surface"], c["kind"], c.get("start")): c for c in context}.values())
    if groups and intent["risk_question"] and any(c["kind"] != "symptom_word" and c.get("assertion") != "negated" for c in context):
        intent["fields"] = list(dict.fromkeys(intent["fields"] + ["warnings", "contraindications"]))
    occupied = [(g["start"], g["end"]) for g in groups]
    vague = [term for term in VAGUE_MEDICINES if any(
        not any(a <= match.start() and match.end() <= b for a, b in occupied)
        for match in re.finditer(re.escape(term), q)
    )]
    unresolved = []
    # Only inspect explicit medication slots. Suggestions require confirmation;
    # they are never injected into the graph or silently corrected.
    for match in re.finditer(r"(?:吃了|吃着|吃|服用|服了|用了|用着|喝了|药叫|药是|打了|喷了|加上|加用|换成|换用|[(（](?:或者|或))([\u4e00-\u9fffA-Za-z0-9·-]{2,24})", q):
        token = re.split(r"又|然后|以后|之后|会不|会有|能不|还能|还可|可以|每天|一天|每次|一次|已经|因为|但是|有点|感觉|好几天|几天|后|和|跟|与", match.group(1))[0]
        begin, end = match.start(1), match.start(1) + len(token)
        if len(token) < 2 or any(a < end and begin < b for a,b in occupied) or any(v in token for v in vague):
            continue
        question_placeholder = bool(re.fullmatch(r"(?:什么|哪种|哪些|哪类)(?:药|药物|制剂)?", token))
        if (token in {"药", "什么", "多少", "几次", "几片", "完", "饭", "东西", "多了", "错了", "药后", "药了", "这个", "那个", "行不行", "好不好", "没事吧", "安全吗"}
                or question_placeholder and intent["drug_discovery_question"]
                or re.match(r"[一二三四五六七八九十两半\d]+(?:片|粒|颗|次|毫|天)",token)
                or re.search(r"不舒服|难受|头晕|拉肚子|过敏|恶心|药后|药会|完药|药能", token)):
            continue
        suggestions = sorted(t for t in terms if len(t) == len(token) and len(t) >= 3 and sum(a != b for a,b in zip(t,token)) == 1)[:5]
        if not (groups or intent["risk_question"] or intent["interaction_question"] or "药" in q
                or any(token.endswith(form) for form in FORMS) or suggestions):
            continue
        unresolved.append({"surface": token, "start": begin, "end": end,
                           "suggestions": suggestions, "automatic_correction": False,
                           **mention_role(q, begin, end)})
    alias_ambiguities = [g["surface"] for g in groups if g["basis"] == "source_catalog_alias" and len({core_candidate(formulation_stem(catalog_nodes[v]["name"])) for v in g["candidate_drug_ids"]}) > 1]
    route_conflicts = [g["surface"] for g in groups if g["route_hint"] and not g["candidate_drug_ids"] and g["route_excluded_candidates"]]
    return {"medicines": groups, "intent": intent, "vague_medicines": vague,
            "schema_version": 2, "normalized_question": q,
            "offset_basis": "normalized_question",
            "timing": timing_mentions(q, groups),
            "alternative_medication_mentions": [g["surface"] for g in groups if g["role"] == "alternative"],
            "patient_context": context,
            "unresolved_medication_mentions": unresolved, "ambiguous_aliases": alias_ambiguities,
            "route_conflicts": route_conflicts,
            "needs_product_details": any(not g["actual_patient_product_known"] for g in groups),
            "requested_details": ["药盒上的名称或成分", "剂型（普通片、缓释片、胶囊、针剂等）"],
            "policy": "query_expansion_only_not_patient_product_identification"}


@lru_cache(maxsize=4)
def _clinical_lexicon(names: tuple):
    return Lexicon(names)


def active_anchors(exact_groups: list[dict], understanding: dict, nodes: dict) -> list[dict]:
    medicines = understanding["medicines"]
    if not medicines:
        return exact_groups
    result = []
    for group in medicines:
        ids = group["candidate_drug_ids"] or group["ingredient_ids"]
        if ids:
            result.append({"surface": group["surface"], "node_ids": ids,
                           "kinds": sorted({nodes[v]["kind"] for v in ids}),
                           "start": group["start"], "end": group["end"],
                           "ambiguous": False, "search_candidates": not group["actual_patient_product_known"]})
    if len(medicines) == 1:
        result.extend(g for g in exact_groups if any(k in ("Disease", "Symptom", "Population") for k in g["kinds"]))
    # Diseases/symptoms stay in the trace as patient context. They must not
    # force the drug-pair path through unrelated disease/symptom edges.
    return result
