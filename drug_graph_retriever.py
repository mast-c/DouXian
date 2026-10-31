"""仅查询现有 Neo4j 药品—疾病候选图谱，返回可追溯、明确标记为未核验的证据。

这个模块绝不写数据库，不创建 :TREATS/:INDICATED_FOR 关系，也不把
DiseaseCandidate 自动解释成获批适应症。需要安装 neo4j Python Driver。
"""
from __future__ import annotations

import os
import re
from urllib.parse import urlsplit
from typing import Any

_DRUG_IDENTIFY = """
MATCH (d:Drug)
WHERE d.name IS NOT NULL AND size(d.name) >= 3
  AND $question CONTAINS d.name
RETURN d.id AS id, d.name AS name
ORDER BY size(d.name) DESC
LIMIT 2
"""

_TERM_IDENTIFY = """
MATCH (t:DiseaseTerm)
WHERE t.name IS NOT NULL AND size(t.name) >= 3
  AND $question CONTAINS t.name
RETURN t.id AS id, t.name AS name
ORDER BY size(t.name) DESC
LIMIT 2
"""

_DRUG_CANDIDATES = """
MATCH (d:Drug {id: $id})-[:HAS_CANDIDATE]->(c:DiseaseCandidate)
      -[:MENTIONS_TERM]->(t:DiseaseTerm)
OPTIONAL MATCH (c)-[:FROM_RECORD]->(r:SourceRecord)
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(s:Section)
WITH d, c, t, head(collect(DISTINCT r)) AS r,
     head(collect(DISTINCT s)) AS s
RETURN d.name AS drug_name, t.name AS term_name,
       c.id AS candidate_id, c.evidence_quote AS evidence_quote,
       c.candidate_status AS candidate_status,
       c.clinical_status AS clinical_status,
       c.source_status AS source_status,
       c.context_flag AS context_flag,
       r.source_url AS source_url, r.id AS record_id,
       s.id AS section_id
ORDER BY c.id
LIMIT $limit
"""

# 用户同时明确药品和疾病时先查二者的交集，避免返回该药的其他疾病候选。
_PAIR_CANDIDATES = """
MATCH (d:Drug {id: $id})-[:HAS_CANDIDATE]->(c:DiseaseCandidate)
      -[:MENTIONS_TERM]->(t:DiseaseTerm {id: $term_id})
OPTIONAL MATCH (c)-[:FROM_RECORD]->(r:SourceRecord)
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(s:Section)
WITH d, c, t, head(collect(DISTINCT r)) AS r,
     head(collect(DISTINCT s)) AS s
RETURN d.name AS drug_name, t.name AS term_name,
       c.id AS candidate_id, c.evidence_quote AS evidence_quote,
       c.candidate_status AS candidate_status,
       c.clinical_status AS clinical_status,
       c.source_status AS source_status,
       c.context_flag AS context_flag,
       r.source_url AS source_url, r.id AS record_id,
       s.id AS section_id
ORDER BY c.id
LIMIT $limit
"""

_TERM_CANDIDATES = """
MATCH (t:DiseaseTerm {id: $id})<-[:MENTIONS_TERM]-(c:DiseaseCandidate)
      <-[:HAS_CANDIDATE]-(d:Drug)
OPTIONAL MATCH (c)-[:FROM_RECORD]->(r:SourceRecord)
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(s:Section)
WITH d, c, t, head(collect(DISTINCT r)) AS r,
     head(collect(DISTINCT s)) AS s
RETURN d.name AS drug_name, t.name AS term_name,
       c.id AS candidate_id, c.evidence_quote AS evidence_quote,
       c.candidate_status AS candidate_status,
       c.clinical_status AS clinical_status,
       c.source_status AS source_status,
       c.context_flag AS context_flag,
       r.source_url AS source_url, r.id AS record_id,
       s.id AS section_id
ORDER BY d.name, c.id
LIMIT $limit
"""


def _http_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value or len(value) > 2048:
        return ""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    return value if parsed.scheme in {"https", "http"} and parsed.netloc else ""


def format_candidate(record: dict[str, Any]) -> dict[str, Any] | None:
    """构造成现有 RagService 所需的 evidence；不进行医学语义升级。"""
    drug = str(record.get("drug_name") or "").strip()
    term = str(record.get("term_name") or "").strip()
    quote = str(record.get("evidence_quote") or "").strip()
    candidate_id = str(record.get("candidate_id") or "").strip()
    if not all((drug, term, quote, candidate_id)):
        return None

    source_url = _http_url(record.get("source_url"))
    candidate_status = str(record.get("candidate_status") or "unknown")
    clinical_status = str(record.get("clinical_status") or "unknown")
    source_status = str(record.get("source_status") or "unknown")
    context_flag = str(record.get("context_flag") or "unknown")
    source = source_url or "Neo4j（原始来源链接缺失）"

    content = (
        "【Neo4j 疾病提及候选｜未经医学验证】\n"
        f"药品：{drug}\n疾病词条：{term}\n"
        f"原文证据：{quote}\n"
        f"候选 ID：{candidate_id}\n"
        f"候选审核状态：{candidate_status}\n"
        f"临床验证状态：{clinical_status}\n"
        f"来源核验状态：{source_status}\n"
        f"语义复核标记：{context_flag}\n"
        f"来源链接：{source_url or '未提供'}\n"
        "重要：这仅证明文档中提到了该疾病词条，不代表该药治疗/预防该疾病，"
        "更不代表已获监管机构批准。须核对全文上下文及权威说明书。"
    )
    return {
        "source": source,
        "section_path": "Neo4j / 适应症语料 / 疾病提及候选（未审核）",
        "content": content,
        "candidate_id": candidate_id,
        "drug_name": drug,
        "disease_term": term,
        "evidence_quote": quote,
        "candidate_status": candidate_status,
        "clinical_status": clinical_status,
        "source_status": source_status,
        "evidence_type": "unverified_drug_disease_candidate",
        "source_url": source_url,
        "record_id": str(record.get("record_id") or ""),
        "section_id": str(record.get("section_id") or ""),
    }


# 现有图谱已有 Drug -> SourceRecord -> Section 证据链。
# 医学证据检索只使用固定字段白名单、参数化查询，不允许用户构造 Cypher。
_SECTION_BY_DRUG = """
MATCH (d:Drug {id: $id})-[:HAS_RECORD]->(r:SourceRecord)
      -[:HAS_SECTION]->(s:Section)
WHERE toLower(coalesce(s.field, '')) IN $fields
  AND s.text IS NOT NULL AND trim(s.text) <> ''
RETURN d.name AS drug_name,
       r.id AS record_id, r.source_url AS source_url,
       r.authority_status AS authority_status,
       s.id AS section_id, s.field AS section_field, s.text AS section_text
ORDER BY r.id, s.id
LIMIT $limit
"""

_MEDICAL_FIELDS = {
    "indication": "适应症",
    "contraindications": "禁忌",
    "warnings": "注意事项",
    "adverse_reactions": "不良反应",
    "drug_interactions": "药物相互作用",
    "dosage": "用法用量",
    "composition": "成分",
}


def fields_for_question(question: str) -> list[str]:
    """按原问题覆盖的所有医学维度返回字段并集，而非命中第一个字段就停止。"""
    text = str(question or "")
    found: set[str] = set()
    patterns = (
        (r"治疗|哪些药|什么药|药物有哪些|有哪些药物|适应症|适应证|用于什么|预防", ("indication",)),
        (r"禁忌|禁用|不宜|不能用|过敏", ("contraindications",)),
        (r"不良反应|副作用|不适反应|毒性|安全性", ("adverse_reactions",)),
        (r"注意事项|警告|慎用|注意什么", ("warnings",)),
        (r"相互作用|药物联用|合用|一起吃|配伍", ("drug_interactions", "warnings")),
        (r"剂量|用法|用量|怎么服|服用次数|给药途径", ("dosage",)),
        (r"成分|组成|主要物质", ("composition",)),
    )
    for pattern, fields in patterns:
        if re.search(pattern, text):
            found.update(fields)

    # V1.17 阶段 2：儿童适用性/自行用药问题需同时核对说明书的安全栏目。
    # 只调整只读 Neo4j Section 检索范围，不据此批准自行用药或推荐剂量。
    pediatric = bool(re.search(r"儿童|小儿|婴幼儿|婴儿|新生儿|孩子", text))
    suitability = bool(re.search(
        r"能否|能不能|可不可以|可以|可否|是否适合|适合|"
        r"能给|能用|能吃|能服用|自行|自己|安全吗|安全性", text
    ))
    self_medication = bool(re.search(
        r"自行(?:服用|服药|用药|吃药)?|自己(?:服用|服药|用药|吃药)|"
        r"未经(?:医生|医师|医嘱)|不用看医生", text
    ))
    if pediatric and suitability:
        found.update(("indication", "warnings", "contraindications", "dosage"))
    elif self_medication:
        found.update(("indication", "warnings", "contraindications"))

    if not found:
        found.add("indication")
    priority = ("indication", "adverse_reactions", "warnings", "contraindications",
                "drug_interactions", "dosage", "composition")
    return [field for field in priority if field in found]


def format_section(record: dict[str, Any]) -> dict[str, Any] | None:
    drug = str(record.get("drug_name") or "").strip()
    field = str(record.get("section_field") or "").strip().lower()
    section_text = str(record.get("section_text") or "").strip()
    if not drug or field not in _MEDICAL_FIELDS or not section_text:
        return None

    source_url = _http_url(record.get("source_url"))
    status = str(record.get("authority_status") or "unknown")
    excerpt = section_text[:2600]
    section = _MEDICAL_FIELDS[field]
    content = (
        "【药品资料原文章节｜来源未完成医学权威核验】\n"
        f"药品：{drug}\n栏目：{section}\n原文：{excerpt}"
        + ("……（超过 2600 字，已截取）" if len(section_text) > 2600 else "")
        + "\n重要：这是数据集中保存的第三方资料文本；"
        "仅能描述资料如何陈述，不等同于某个上市产品的最新版官方核准说明书。"
    )
    return {
        "source": f"药品资料 · {drug}",
        "section_path": f"Neo4j / 药品资料 / {section}（来源待核验）",
        "content": content,
        "source_url": source_url,
        "evidence_type": "unverified_drug_source_section",
        "source_status": status,
        "drug_name": drug,
        "section_field": field,
        "record_id": str(record.get("record_id") or ""),
        "section_id": str(record.get("section_id") or ""),
    }



_SECTION_BY_DRUG_NAME_AND_FIELD = """
MATCH (d:Drug {name: $drug_name})-[:HAS_RECORD]->(r:SourceRecord)
      -[:HAS_SECTION]->(s:Section)
WHERE toLower(coalesce(s.field, '')) = $field
  AND s.text IS NOT NULL AND trim(s.text) <> ''
RETURN d.name AS drug_name,
       r.id AS record_id, r.source_url AS source_url,
       r.authority_status AS authority_status,
       s.id AS section_id, s.field AS section_field, s.text AS section_text
ORDER BY r.id, s.id
LIMIT $limit
"""


def is_disease_drug_list_question(question: str) -> bool:
    """识别要求列出疾病相关药品的查询，不据此推定任何治疗关系。"""
    text = str(question or "").strip()
    return bool(re.search(
        r"(?:药物|药品|药).{0,5}有哪些|有哪些.{0,5}(?:药物|药品|药)|"
        r"(?:什么|哪些)(?:药物|药品|药)|"
        r"治疗.{0,24}(?:药物|药品|什么药|哪些药)",
        text,
    ))



_PARENT_SECTION_BY_IDS = """
MATCH (d:Drug {id: $drug_id})-[:HAS_RECORD]->
      (r:SourceRecord {id: $record_id})-[:HAS_SECTION]->
      (s:Section {id: $section_id})
RETURN d.name AS drug_name, r.source_url AS source_url,
       s.field AS field, s.text AS full_section
LIMIT 2
"""


def compact_same_quote_candidates(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """同一药品同一 Section 的相同证据句只展示一遍，保留词条与候选 ID。"""
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for item in items:
        if item.get("evidence_type") != "unverified_drug_disease_candidate":
            ordered.append(dict(item))
            continue
        record_id = str(item.get("record_id") or "").strip()
        section_id = str(item.get("section_id") or "").strip()
        candidate_id = str(item.get("candidate_id") or "").strip()
        term = str(item.get("disease_term") or "").strip()
        quote = " ".join(str(item.get("evidence_quote") or "").split())
        # 只有能够确定属于同一原文 Section 的候选才合并。
        if not all((record_id, section_id, quote, candidate_id)):
            result = dict(item)
            result["candidate_ids"] = [candidate_id] if candidate_id else []
            result["disease_terms"] = [term] if term else []
            ordered.append(result)
            continue
        key = (
            str(item.get("drug_name") or ""), record_id, section_id,
            quote, str(item.get("source_url") or ""),
            str(item.get("candidate_status") or ""),
            str(item.get("clinical_status") or ""),
            str(item.get("source_status") or ""),
        )
        if key not in groups:
            result = dict(item)
            result["candidate_ids"] = [candidate_id]
            result["disease_terms"] = [term] if term else []
            groups[key] = result
            ordered.append(result)
        else:
            result = groups[key]
            if candidate_id not in result["candidate_ids"]:
                result["candidate_ids"].append(candidate_id)
            if term and term not in result["disease_terms"]:
                result["disease_terms"].append(term)

    for item in ordered:
        terms = item.get("disease_terms") or []
        ids = item.get("candidate_ids") or []
        if len(terms) <= 1:
            continue
        item["disease_term"] = "、".join(terms)
        original = item.get("content") or ""
        original_first_term = str(original.split("疾病词条：", 1)[-1].split("\n", 1)[0])
        if "疾病词条：" in original:
            item["content"] = original.replace(
                "疾病词条：" + original_first_term,
                "疾病词条（同句多个候选）： " + "、".join(terms), 1
            )
        oldid = str(item.get("candidate_id") or "")
        if oldid and "候选 ID：" in item["content"]:
            item["content"] = item["content"].replace(
                "候选 ID：" + oldid,
                "候选 ID（同句）： " + "、".join(ids), 1
            )
    return ordered


class DrugGraphRetriever:
    def __init__(self, driver: Any = None):
        self._owns_driver = driver is None
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        if driver is None:
            uri = os.getenv("NEO4J_URI", "").strip()
            username = (os.getenv("NEO4J_USER") or
                        os.getenv("NEO4J_USERNAME") or "neo4j").strip()
            password = os.getenv("NEO4J_PASSWORD", "")
            if not uri or not password:
                raise RuntimeError("需要在 .env 设置 NEO4J_URI 和 NEO4J_PASSWORD")
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                uri, auth=(username, password), connection_timeout=5.0,
            )
        self.driver = driver

    def close(self) -> None:
        if self._owns_driver and self.driver is not None:
            self.driver.close()

    def _run(self, cypher: str, **parameters: Any) -> list[dict[str, Any]]:
        # execute_read 在只读事务中运行；Cypher 均为静态 MATCH/RETURN 查询。
        with self.driver.session(database=self.database) as session:
            return session.execute_read(
                lambda tx: [row.data() for row in tx.run(cypher, **parameters)]
            )

    def search(self, question: str, limit: int = 5, *,
               explicit_disease_only: bool = False) -> list[dict[str, Any]]:
        """以用户完整问题匹配图谱中的明确药品/疾病名称，避免错误模糊关联。"""
        if not isinstance(question, str) or not question.strip():
            return []
        limit = max(1, min(int(limit), 10))
        normalized = " ".join(question.split())[:500]
        drugs = self._run(_DRUG_IDENTIFY, question=normalized)
        # 疾病名必须明确出现在用户提问中，而不是仅出现在药品名称的内部
        # 子串里（例如药名中的通用词恰好与某个疾病词匹配）。
        disease_part = normalized
        for drug in drugs:
            name = str(drug.get("name") or "").strip()
            if name:
                disease_part = disease_part.replace(name, " ")
        terms = self._run(_TERM_IDENTIFY, question=disease_part)
        # 对适应症比较与开放的治疗问题，未命中明确疾病名称时，不应该
        # 按候选 ID 枚举恰好排在前面的三条疾病词；优先返回说明书章节。
        if explicit_disease_only and not terms:
            return []
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        if drugs and terms:
            # 明确查询“药品 X 与疾病 Y 是否有关”时，只显示这个药品与词条的候选；
            # 不以该药的其他疾病候选代替查询目标。
            for drug in drugs:
                for term in terms:
                    if not drug.get("id") or not term.get("id"):
                        continue
                    # 两药比较同一疾病时，不能让第一种药占满全局候选上限。
                    per_pair_limit = max(
                        1, (limit + len(drugs) * len(terms) - 1)
                        // (len(drugs) * len(terms))
                    )
                    records = self._run(
                        _PAIR_CANDIDATES,
                        id=drug["id"], term_id=term["id"],
                        limit=per_pair_limit,
                    )
                    for record in records:
                        candidate = format_candidate(record)
                        if candidate and candidate["candidate_id"] not in seen:
                            seen.add(candidate["candidate_id"])
                            results.append(candidate)
                            if len(results) >= limit:
                                return results
            return results

        for cypher, entities in ((_DRUG_CANDIDATES, drugs),
                                 (_TERM_CANDIDATES, terms)):
            for entity in entities:
                eid = str(entity.get("id") or "").strip()
                if not eid:
                    continue
                # 没有疾病名的开放探索，尽可能均匀覆盖不同被点名药品。
                per_entity_limit = max(
                    1, (limit + len(entities) - 1) // len(entities)
                )
                for record in self._run(
                    cypher, id=eid, limit=per_entity_limit
                ):
                    candidate = format_candidate(record)
                    if candidate is None or candidate["candidate_id"] in seen:
                        continue
                    seen.add(candidate["candidate_id"])
                    results.append(candidate)
                    if len(results) >= limit:
                        return results
        return results

    def search_sections(self, question: str, limit: int = 3) -> list[dict[str, Any]]:
        """从现有 Neo4j 药品说明书 Section 中返回原文证据（只读）。"""
        if not isinstance(question, str) or not question.strip():
            return []
        text = " ".join(question.split())[:500]
        drugs = self._run(_DRUG_IDENTIFY, question=text)
        fields = fields_for_question(text)
        limit = max(1, min(int(limit), 4))
        results: list[dict[str, Any]] = []
        seen = set()
        valid_drugs = [d for d in drugs if d.get("id")]
        # 一药多 SourceRecord 时不能把所有名额占满：先为每种明确药物
        # 分配自己的 Section 配额，避免双药对比只显示第一种药。
        per_drug = max(1, limit // max(1, len(valid_drugs)))
        for drug in valid_drugs:
            remaining = min(per_drug, limit - len(results))
            if remaining <= 0:
                break
            rows = self._run(
                _SECTION_BY_DRUG,
                id=drug["id"], fields=fields, limit=remaining,
            )
            for row in rows:
                section = format_section(row)
                if section is None:
                    continue
                key = (section["record_id"], section["section_id"], section["section_field"])
                if key in seen:
                    continue
                seen.add(key)
                results.append(section)
        return results


    def search_fields_for_drugs(
        self, drug_names: list[str], fields: list[str],
        *, max_drugs: int = 4, max_sections: int = 8,
    ) -> list[dict[str, Any]]:
        """对疾病候选识别出的药品，按药名×栏目补查真实说明书 Section。

        只使用 Drug.name 精确匹配和固定字段白名单；每药每栏目最多一段原文。
        结果只是第三方数据集中的原文，不证明临床有效性。
        """
        if not (1 <= max_drugs <= 6 and 1 <= max_sections <= 16):
            raise ValueError("max_drugs/max_sections 超出受控范围")
        distinct = list(dict.fromkeys(
            name.strip() for name in drug_names
            if isinstance(name, str) and name.strip()
        ))[:max_drugs]
        allowed_fields = list(dict.fromkeys(
            field for field in fields if field in _MEDICAL_FIELDS
        ))
        if not distinct or not allowed_fields:
            return []

        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        # 先每药选第一个栏目，再选第二个栏目，保证跨药品公平性。
        for field in allowed_fields:
            for drug_name in distinct:
                if len(results) >= max_sections:
                    return results
                rows = self._run(
                    _SECTION_BY_DRUG_NAME_AND_FIELD,
                    drug_name=drug_name, field=field, limit=1,
                )
                for row in rows:
                    section = format_section(row)
                    if section is None:
                        continue
                    key = (section["record_id"], section["section_id"])
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(section)
                    break
        return results

    def fetch_parent_section(self, meta: dict[str, Any]) -> dict[str, Any] | None:
        """按 JSON Chunk 中的三个来源 ID 精确读取父级原文（只读）。"""
        if not isinstance(meta, dict):
            return None
        keys = ("drug_id", "record_id", "section_id")
        if not all(isinstance(meta.get(key), str) and meta.get(key) for key in keys):
            return None
        records = self._run(_PARENT_SECTION_BY_IDS, **{
            key: meta[key] for key in keys
        })
        # 0条或异常的多条均拒绝自动关联。
        return records[0] if len(records) == 1 else None
