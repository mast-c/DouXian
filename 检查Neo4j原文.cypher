// Neo4j Browser 只读检查：核实数据库的原始来源 URL、字段名和原文位置
MATCH (d:Drug {name: '阿托伐他汀钙片'})
      -[:HAS_RECORD]->(r:SourceRecord)
      -[:HAS_SECTION]->(s:Section)
WHERE s.field IN [
  'indication', 'contraindications',
  'adverse_reactions', 'warnings', 'drug_interactions'
]
RETURN d.id AS drug_id, d.name AS drug,
       r.id AS record_id, r.source_url AS source_url,
       s.id AS section_id, s.field AS field,
       left(s.text, 240) AS section_preview,
       size(s.text) AS length
ORDER BY field, record_id, section_id
LIMIT 50;
