"""Extract patient statements for retrieval, without diagnosing or inferring causality."""
from __future__ import annotations

import re


DURATION = r"(?:好几|几|数|[一二三四五六七八九十两半\d]+)(?:个)?(?:小时|分钟|星期|个月|天|周|月|年)(?:多|左右)?"
MEDICATION_ACTION = r"(?:吃|服用|服|使用|用|喷|涂|擦|滴|喝|打针|注射)(?:了|着|过|完)?"
EFFICACY = r"不管用|不见效|没(?:有)?效果|没(?:有)?好转|(?:仍|还|一直).{0,5}(?:堵|疼|痛|痒|难受|好不了|没好)|(?:症状|鼻塞).{0,8}(?:未|没)(?:改善|缓解)"


def local_prefix(text: str, start: int) -> str:
    return re.split(r"[，,。；;！？!?\n]", text[:start])[-1][-48:]


def mention_role(text: str, start: int, end: int) -> dict:
    before = local_prefix(text, start)
    after = text[end:end + 20]
    # A postposed subject belongs to this medicine only, stopping at punctuation.
    post = re.split(r"[，,。；;！？!?\n]", text[end:])[0][:24]
    post_subject = re.match(r"(?:我|我们|他|她|患者|病人)", post)
    if post_subject:
        action = post[post_subject.end():]
        if re.match(r"(?:已经|之前|现在|也|还|已)?(?:没有|从未|没)(?:吃|服用|服|使用|用|喷|涂)", action):
            return {"role": "not_taken", "assertion": "negated", "role_basis": "postposed_medication_action"}
        if re.match(r"(?:已经|之前|现在|也|已)?(?:停用|停服|停了|不再吃|不再用)", action):
            return {"role": "stopped", "assertion": "user_reported", "role_basis": "postposed_medication_action"}
        if re.match(r"(?:正在|一直|还在)(?:吃|服用|服|使用|用|喷|涂)", action):
            return {"role": "current", "assertion": "user_reported", "role_basis": "postposed_medication_action"}
        if re.match(r"(?:已经|刚刚|之前|现在|早就|也|还|已|刚){0,3}" + MEDICATION_ACTION, action):
            return {"role": "taken", "assertion": "user_reported", "role_basis": "postposed_medication_action"}
    role, assertion = "mentioned", "unspecified"
    if re.search(r"(?:或(?:者)?|还是)$", before):
        role, assertion = "alternative", "alternative_not_confirmed"
    elif re.search(r"(?:没有|没|不曾|从未|未)(?:吃|服用|服|使用|用|喷|涂|打)(?:过|了)?$", before):
        role, assertion = "not_taken", "negated"
    elif re.search(r"(?:停了|停用|停服|不再吃|不再用)(?:了)?$", before):
        role, assertion = "stopped", "user_reported"
    elif re.search(r"(?:想(?!问|了解|知道)|打算|准备|考虑|能否|能不能|可不可以|可以|要不要).{0,12}$", before) or re.search(r"(?:加上|加用|换成|换用|试试|再加|加|(?:能|可否|要)(?:再)?(?:吃|用|喷|服用|服|涂|滴))$", before):
        role, assertion = "proposed", "hypothetical"
    elif re.search(r"(?:正在|一直|长期|在)(?:吃|服用|服|使用|用|喷|涂)(?:着)?$", before):
        role, assertion = "current", "user_reported"
    elif re.search(MEDICATION_ACTION + r"$", before):
        role, assertion = "taken", "user_reported"
    elif re.match(r"(?:(?:吃|服用|用|喷)(?:了)?)?" + DURATION, after):
        role, assertion = "taken", "user_reported"
    elif re.match(r"(?:能|可不可以|可以|能否|可否).{0,8}(?:吃|用|喷|服)", after):
        role, assertion = "proposed", "hypothetical"
    return {"role": role, "assertion": assertion, "role_basis": "explicit_patient_wording"}


def timing_mentions(text: str, medicines: list[dict]) -> list[dict]:
    result = []
    mentions = sorted(
        [mention for medicine in medicines for mention in medicine.get("mentions", [medicine])],
        key=lambda item: item["start"],
    )
    for match in re.finditer(r"今天|昨天|前天|刚才|最近|" + DURATION, text):
        # Attach a duration only inside the same medication clause. A later
        # symptom or another action prevents guessing which drug it belongs to.
        previous = [m for m in mentions if m["end"] <= match.start()]
        linked = None
        if previous and re.fullmatch(r"(?:已经|大概|约|用了|吃了|喷了|有)?", text[previous[-1]["end"]:match.start()]):
            linked = previous[-1]["surface"]
        if linked is None:
            following = [m for m in mentions if m["start"] >= match.end()]
            if following and re.fullmatch(r"(?:才|刚)?(?:吃|服用|用|喷|涂|擦|滴|喝|打)(?:了|过)?", text[match.end():following[0]["start"]]):
                linked = following[0]["surface"]
        result.append({"surface": match.group(), "start": match.start(), "end": match.end(),
                       "medicine_surface": linked,
                       "exact_duration_known": bool(re.fullmatch(DURATION, match.group()))
                       and not bool(re.search(r"好几|几|数|多|左右", match.group()))})
    return result


def symptom_mentions(text: str) -> list[dict]:
    patterns = (
        (r"发烧|发热|低热|高热", ["发烧", "发热", "退热"]),
        (r"鼻子(?:还是|仍然|一直|还)?(?:很|特别|有点)?堵|鼻塞", ["鼻塞"]),
        (r"鼻子(?:里|里面|内)?[^，,。；;！？!?]{0,10}?(?:疼|痛)|鼻腔(?:疼痛|痛|疼)", ["鼻腔疼痛"]),
        (r"拉肚子|腹泻", ["腹泻"]),
        (r"心慌|心悸", ["心悸"]),
        (r"胃(?:部|里)?(?:有点|有些|很|特别)?(?:疼痛|疼|痛)", ["胃痛"]),
        (r"胃(?:部)?(?:有点|有些|很)?不舒服", ["胃部不适"]),
        (r"头(?:有点|很)?(?:疼|痛)", ["头痛"]),
        (r"头晕|恶心|呕吐|起疹子|起疹|皮疹|瘙痒|鼻出血", None),
    )
    result = []
    for pattern, terms in patterns:
        for match in re.finditer(pattern, text):
            prefix = local_prefix(text, match.start())
            assertion = "negated" if re.search(r"没有|并无|未出现|不再|没", prefix[-5:]) else "user_reported"
            if assertion != "negated" and re.search(r"会不会|是否|会|可能|如果", prefix[-8:]):
                assertion = "questioned"
            result.append({"surface": match.group(), "start": match.start(), "end": match.end(),
                           "kind": "symptom_word", "retrieval_terms": terms or [match.group()],
                           "assertion": assertion, "diagnosis_asserted": False,
                           "drug_causality_asserted": False})
    return sorted(result, key=lambda item: item["start"])


def context_assertion(text: str, start: int) -> str:
    before = local_prefix(text, start)
    if re.search(r"(?:没有|没患|无|不是|否认)(?:严重的|严重|患有|得过|得了)?$", before):
        return "negated"
    if re.search(r"妈妈|爸爸|母亲|父亲|家人|家族", before):
        return "other_person"
    if re.search(r"如果|假如|会不会|是不是|是否|可能", before):
        return "questioned"
    return "user_reported"


def patient_context_summary(understanding: dict) -> dict:
    """Compact user statements for generation; never part of the evidence pool."""
    return {
        "medicines": [{key: medicine.get(key) for key in (
            "surface", "role", "assertion", "mentions", "route_hint", "actual_patient_product_known",
        )} for medicine in understanding.get("medicines", [])[:8]],
        "patient_statements": understanding.get("patient_context", [])[:20],
        "timing": understanding.get("timing", [])[:12],
        "intent": understanding.get("intent", {}),
        "meaning": "用户表述解析，仅用于理解问题；不是医学证据，不证明诊断、药物因果或合用安全。未服用、拟加用、备选、他人病史不能当作当前实际用药或患者确诊。",
    }
