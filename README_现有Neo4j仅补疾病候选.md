# DrugKG：复用 7687 现有 Neo4j，只增量导入疾病候选

## 已确认的现状

根据你的 Neo4j 查询，原有数据库节点数与清洗版 V2 预期一致：

| 既有图谱标签 | 数量 |
|---|---:|
| Drug | 11,953 |
| SourceRecord | 11,960 |
| Section | 106,445 |
| Chunk | 107,787 |
| DrugAlias | 343 |

旧图谱的 `Drug.kg_snapshot` 是空属性，这是此前全量导入程序
`--stages candidates` 无法直接继续的原因之一。

**本工具不会更改这些既有节点，也不会要求它们先拥有 `kg_snapshot`。**

## 安装

在 `E:\VsCode_Project\DockerTest_Agent\RAG_project\data`
解压补丁，只会新增：

```
kg/neo4j_existing_candidates.py
```

不要覆盖旧版：
`kg/neo4j_drug_graph.py` / `kg/build.py` / `kg/prepare_cleaned_v2.py`。

原数据不动：
- `kg_output_clean_v2_import_fixed`（已重建完整 JSONL）
- `kg_output_clean_v2/disease_candidates_full.csv`
- `kg_output_clean_v2/pilot_v31_merged60.csv`

请确认 `NEO4J_PASSWORD` 已在本机 `.env` 设置，且 `bolt://localhost:7687`
能正常连接。不要分享密码。

## 一条命令：先检查，再增量写入

**这条命令会修改 Neo4j，仅在你确认现有图谱可复用后执行。**

```powershell
python -u -m kg.neo4j_existing_candidates attach --in-dir "kg_output_clean_v2_import_fixed" --candidates "kg_output_clean_v2/disease_candidates_full.csv" --pilot "kg_output_clean_v2/pilot_v31_merged60.csv" --uri "bolt://localhost:7687" --batch-size 400 --apply --report "kg_output_clean_v2_import_fixed/attach_candidates_result.json"
```

写库之前，程序自动完成：
1. 完整本地 JSONL / 候选来源位置校验；
2. 分别提取既有 Neo4j `Drug/SourceRecord/Section/Chunk/DrugAlias`
   的所有节点 ID，与清洗版 JSONL 的完整 ID 集合逐个比较；
3. 核实 19,685 条候选的 Drug → SourceRecord → Section 真实连接路径；
4. 校验 Neo4j 原文章节的引用片段及疾病命中字符串；
5. 确认已有疾病候选、模型评估不属于其他快照。

这些检查有一项失败，会在**首次写库前**停止，不会强制修改老数据。
只凭“节点数量恰好一致”不足以判断节点对应一致。

核对成功后：
- 创建或复用 1,150 个 `DiseaseTerm`（未归一化术语）；
- 创建 19,685 个 `DiseaseCandidate` 候选；
- 导入 60 条 Qwen `ModelAssessment` 开发集评估；
- 以 HAS_CANDIDATE、MENTIONS_TERM、SUPPORTED_BY、
  FROM_RECORD、ASSESSES 建立证据链。

不会创建 Drug-[:TREATS]->Disease、
Drug-[:INDICATED_FOR]->Disease 等医学确认关系。

自动输出类似：
```
[BASE] ID 完全匹配：SourceRecord 11,960
[BASE] ID 完全匹配：Chunk 107,787
[EVIDENCE] 19,685 条候选原文验证通过
[WRITE] DiseaseCandidate: 已提交 19,685
[WRITE] ModelAssessment: 已提交 60
{
  "status": "VERIFIED",
  "disease_candidates": 19685,
  "model_assessments": 60,
  "invalid_links_or_evidence": 0,
  "verified_clinical_indications": 0
}
```
上面的示例只是预期；以本机实际运行结果为准。

## 如果只想先只读检查（可选）

```powershell
python -u -m kg.neo4j_existing_candidates inspect --in-dir "kg_output_clean_v2_import_fixed" --candidates "kg_output_clean_v2/disease_candidates_full.csv" --pilot "kg_output_clean_v2/pilot_v31_merged60.csv" --uri "bolt://localhost:7687"
```

输出 `SAFE_TO_ATTACH` 表示已完成数据库只读检查，但尚未写入。

## 导入后直接在 Neo4j Browser 检索

```cypher
MATCH (d:Drug)-[:HAS_CANDIDATE]->(c:DiseaseCandidate)
      -[:MENTIONS_TERM]->(t:DiseaseTerm)
MATCH (c)-[:FROM_RECORD]->(r:SourceRecord)
WHERE d.name CONTAINS '左氧氟沙星'
RETURN d.name AS drug,
       t.name AS candidate_disease,
       c.evidence_quote AS original_evidence,
       c.candidate_status AS candidate_status,
       r.source_url AS source_url
LIMIT 20;
```

如果需要在疾病反查药品：

```cypher
MATCH (t:DiseaseTerm {name:'心绞痛'})
      <-[:MENTIONS_TERM]-(c:DiseaseCandidate)
      <-[:HAS_CANDIDATE]-(d:Drug)
RETURN d.name AS drug,
       c.evidence_quote AS original_evidence,
       c.clinical_status AS clinical_status
LIMIT 30;
```

### 重要说明

- `DiseaseTerm` 仅是原文匹配候选词条，可能实际是症状、体征或其他语义。
- 原药品数据来自第三方汇编资料，未核验每份产品的最新监管批准说明书。
- `candidate_status=pending_human_review`，不能将疾病候选直接用于
  可信治疗建议或已确认的获批适应症。
- 程序采用批量 `MERGE`；如写入部分成功后出现中断，可以在查清原因后
  原命令续跑。旧基础节点保持不变。
- 当前环境未连接你 Windows 上的 Neo4j，因此已测试的范围是本地实际
  数据生成和离线单元测试；最终验收以你本机数据库结果为准。
