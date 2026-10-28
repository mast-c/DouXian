"""Conservative, repeatable name repair; source medical text is never rewritten.

Full names come from the *same record's* approval-page heading, not fuzzy
matching. Different records for one name remain separate source records.
"""
from collections import Counter, defaultdict
import copy
import json
import re

VERSION = "1.0"
TITLE_SUFFIX = "批准文号及生产厂家"
BAD_CHAR = re.compile(r"[\ue000-\uf8ff\ufffd\x00-\x1f]")
FORM_ONLY = {"片", "丸", "散", "胶囊", "软膏", "乳膏", "合剂", "结片", "结丸"}
# Context-specific substitutions only: private-use code points are not a font map.
NAME_SUBSTITUTIONS = {
    "复方磺胺甲\ue001唑片": "复方磺胺甲噁唑片",
    "盐酸地拉\ue000片": "盐酸地拉卓片",
    "九制\ue23f莶草药酒": "九制豨莶草药酒",
    "苯二氮\ue01c": "苯二氮䓬",
    "盐酸地尔硫(艹卓)": "盐酸地尔硫卓",
}
# These two pages contain concatenated unrelated names and no clinical text.
CORRUPT_PAGES = {"hy70174.htm", "hy7025j.htm"}


def normalize_spelling(name):
    for old, new in NAME_SUBSTITUTIONS.items():
        name = name.replace(old, new)
    # Whitespace between characters is a crawler artifact; retain Latin spacing.
    name = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", name)
    return name.strip()


def balanced(name):
    stack = []
    for ch in name.translate(str.maketrans("（）［］【】", "()[][]")):
        if ch in "([":
            stack.append(ch)
        elif ch in ")]":
            if not stack or stack.pop() != {')': '(', ']': '['}[ch]:
                return False
    return not stack


def name_errors(name):
    errors = []
    if not isinstance(name, str) or not name.strip():
        return ["empty_name"]
    if name != name.strip():
        errors.append("edge_whitespace")
    if not balanced(name):
        errors.append("unbalanced_brackets")
    if name[0] in "（()）/、，;；":
        errors.append("name_fragment")
    if name in FORM_ONLY:
        errors.append("dosage_form_only")
    if BAD_CHAR.search(name):
        errors.append("invalid_character")
    if re.search(r"别名\s*[:：]|商品名称\s*[:：]|通用名称\s*[:：]", name):
        errors.append("metadata_in_name")
    return errors


def heading(record):
    rows = record.get("approval_numbers") or []
    if isinstance(rows, str):
        rows = [rows]
    titles = {x[:-len(TITLE_SUFFIX)].strip() for x in rows
              if isinstance(x, str) and x.endswith(TITLE_SUFFIX)}
    return next(iter(titles)) if len(titles) == 1 else ""


def canonical_name(original, record):
    title = heading(record)
    candidate = title or original
    aliases = []
    if candidate.startswith(("位置：", "【执行标准】", "通用名称")):
        # A crawler may put an empty metadata label in its page heading.
        # Keep a valid existing name; the heading is not a replacement name.
        candidate = original
    # A title can include composition, a website label, or a toxicity marker.
    candidate = re.split(r"[;；]", candidate, maxsplit=1)[0]
    candidate = re.sub(r"说明$", "", candidate).rstrip("。． ")
    candidate = re.sub(r"\s*(?:\[剧\]|［剧］|〔剧〕|[（(]剧[）)])$", "", candidate)
    explicit_alias = re.search(r"[（(]别名\s*[:：]([^）)]+)[）)]", candidate)
    if explicit_alias:
        aliases.append(normalize_spelling(explicit_alias[1]))
        candidate = candidate[:explicit_alias.start()] + candidate[explicit_alias.end():]
    # Known defective heading; the existing source also has 西沙必利片.
    if candidate == "西沙必利片片":
        candidate = "西沙必利片"
    if candidate == "当归补血胶囊)":
        candidate = "当归补血胶囊"
    candidate = normalize_spelling(candidate)
    if title and ("**" in title or len(candidate) > 75):
        return "", aliases, "contaminated_heading"
    if name_errors(candidate):
        return "", aliases, ",".join(name_errors(candidate))
    # A corrupted heading must not turn an already valid name into web metadata.
    if candidate.startswith(("位置：", "【执行标准】", "通用名称")):
        return "", aliases, "non_name_heading"
    return candidate, aliases, "same_record_heading" if title else "source_name"


def fingerprint(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def match_key(name):
    """Formatting equivalence only; retain every formulation/ratio character."""
    name = normalize_spelling(name).translate(str.maketrans("（）：", "():"))
    for roman, letters in (("Ⅰ", "I"), ("Ⅱ", "II"), ("Ⅲ", "III"), ("Ⅳ", "IV"), ("Ⅴ", "V")):
        name = name.replace(roman, letters)
    return re.sub(r"[()\s]", "", name)


def clean_dataset(raw):
    """Return cleaned source and audit. Safe to call on its own output again."""
    if raw.get("name_cleaning_version") == VERSION:
        for name in raw["drugs"]:
            if name_errors(name):
                raise ValueError(f"已清洗数据含非法药名: {name!r}")
        return copy.deepcopy(raw), {"already_cleaned": True, "changes": [], "quarantine": []}

    groups, originals, aliases = defaultdict(list), defaultdict(set), defaultdict(set)
    redirects, changes, quarantine = {}, [], []
    for old_name, record in raw["drugs"].items():
        if not isinstance(record, dict):
            quarantine.append({"original_name": old_name, "record": record,
                               "reason": "non_object_record"})
            continue
        source = str(record.get("source", ""))
        if any(source.endswith(page) for page in CORRUPT_PAGES):
            new_name, extra_aliases, reason = "", [], "concatenated_unrelated_names_empty_record"
        else:
            new_name, extra_aliases, reason = canonical_name(old_name, record)
        if not new_name:
            quarantine.append({"original_name": old_name, "record": record, "reason": reason})
            continue
        groups[new_name].append((old_name, record))
        originals[new_name].add(old_name)
        aliases[new_name].update(extra_aliases)
        # Broken fragments remain in provenance, never in alias-based resolution.
        if old_name != new_name:
            changes.append({"old_name": old_name, "new_name": new_name,
                            "source_url": source, "heading": heading(record), "reason": reason})
            redirects[old_name] = new_name
            if not name_errors(old_name) and len(old_name) > 2:
                aliases[new_name].add(old_name)

    cleaned, variants, duplicates = {}, {}, 0
    for name, rows in groups.items():
        # Prefer an existing correctly named record. Do not merge field strings.
        rows.sort(key=lambda pair: (pair[0] != name, pair[0]))
        distinct, seen = [], set()
        for old, record in rows:
            key = fingerprint(record)
            if key not in seen:
                distinct.append(record)
                seen.add(key)
            else:
                duplicates += 1
        cleaned[name] = distinct[0]
        if len(distinct) > 1:
            variants[name] = distinct[1:]

    equivalent = defaultdict(set)
    for name, rows in groups.items():
        for alias in [name] + list(aliases[name]):
            equivalent[match_key(alias)].add(name)
        for _, record in rows:
            # An explicit 别名 label can occur in interaction endpoints too.
            title = heading(record)
            if title and title != name:
                equivalent[match_key(title)].add(name)

    interactions, seen_interactions, interaction_changes = [], set(), []
    duplicate_interactions = []
    invalid_interactions = []
    for index, row in enumerate(raw.get("interactions", [])):
        pair = row.get("pair") if isinstance(row, dict) else None
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2
                and all(isinstance(x, str) and x.strip() for x in pair)):
            invalid_interactions.append({"index": index, "record": row, "reason": "invalid_pair"})
            continue
        new_pair = []
        for position, endpoint in enumerate(pair):
            name = normalize_spelling(endpoint)
            # Trade-name metadata is not part of the generic entity name.
            metadata = re.search(r"(?:商品名称|通用名称)\s*[:：]", name)
            if metadata and name[:metadata.start()].strip():
                name = name[:metadata.start()].strip()
            # A target category such as 胶囊 must not acquire a specific drug ID.
            if name in redirects and (position == 0 or not name_errors(name)):
                name = redirects[name]
            matches = equivalent.get(match_key(name), set())
            if len(matches) == 1 and name not in FORM_ONLY:
                name = next(iter(matches))
            if name.endswith("说明书") and name[:-3] in cleaned:
                name = name[:-3]
            new_pair.append(name)
        malformed = [(name, [err for err in name_errors(name) if err != "dosage_form_only"])
                     for name in new_pair]
        if any(errors for name, errors in malformed):
            invalid_interactions.append({"index": index, "record": row,
                                         "normalized_pair": new_pair,
                                         "reason": "malformed_endpoint", "details": malformed})
            continue
        new_row = copy.deepcopy(row)
        new_row["pair"] = new_pair
        if list(pair) != new_pair:
            new_row.setdefault("original_pair", list(pair))
            interaction_changes.append({"index": index, "old_pair": list(pair), "new_pair": new_pair})
        # original_pair is provenance, not candidate identity.
        key = fingerprint({k: v for k, v in new_row.items() if k != "original_pair"})
        if key not in seen_interactions:
            interactions.append(new_row)
            seen_interactions.add(key)
        else:
            duplicate_interactions.append({"index": index, "record": row,
                                           "normalized_pair": new_pair})

    result = copy.deepcopy(raw)
    result.update({"drugs": cleaned, "interactions": interactions,
                   "name_cleaning_version": VERSION,
                   "additional_source_records": variants,
                   "drug_original_names": {n: sorted(v) for n, v in originals.items() if v != {n}},
                   "drug_aliases": {n: sorted(v - {n}) for n, v in aliases.items() if v - {n}},
                   "drug_name_redirects": redirects})
    audit = {"changes": changes, "quarantine": quarantine,
             "interaction_changes": interaction_changes, "invalid_interactions": invalid_interactions,
             "duplicate_interactions": duplicate_interactions,
             "before_drugs": len(raw["drugs"]), "after_drugs": len(cleaned),
             "duplicate_source_records_removed": duplicates,
             "additional_source_records": sum(map(len, variants.values())),
             "before_interactions": len(raw.get("interactions", [])),
             "after_interactions": len(interactions)}
    return result, audit


def mention_status(name, drug_id=""):
    """Describe the extraction result, never promote a category to Drug."""
    if drug_id:
        return "resolved_dataset_drug"
    if name in FORM_ONLY:
        return "dosage_form_mention_not_drug"
    if name_errors(name):
        return "malformed_mention_needs_review"
    if re.fullmatch(r"(?:上述|上列|下列|下述|以下|其他|其它|某些)?药物(?:的)?", name):
        return "non_specific_reference"
    if re.search(r"[、﹑]|及|以及|或|同时|相互作用|应慎重|会出现", name):
        return "unresolved_compound_or_text"
    if re.search(r"(?:类(?:药物|药|抗生素|抗菌药|抗抑郁药|抗心律失常药)?|抑制剂|阻滞剂|拮抗剂|激动剂)$", name):
        return "category_mention_not_drug"
    return "unresolved_name_or_category"
