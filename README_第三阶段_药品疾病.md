# 第三阶段｜药品—疾病关系的保守抽取与 Neo4j 入库

此补丁用于你现有的 `RAG_project`。**不需要重新建库，不删除旧节点，不替换 ChromaDB**。

已完成阶段：
- `drug_db_v2_cleaned.json` 清洗与命名修复
- 药品/来源/章节/Chunk 入库
- 过期 InteractionCandidate 和旧 Drug 残留清理

现在增加 `kg/disease_stage3.py`，完成：
1. 从 `medical.json` 构建疾病候选词典；
2. 从清洗后 `Section(field='indication')` 原文生成待审查药品—疾病候选 CSV；
3. 人工审核治疗/预防/症状缓解语义；
4. 从原始 JSONL 逐字验证证据；
5. Neo4j **只增量写入 DiseaseTerm、IndicationClaim 及证据关系**，不重导上十万 Chunk。

## 文件放置

从本 ZIP 解压 `kg/disease_stage3.py` 到：

```text
RAG_project/
├── kg/
│   ├── build.py                  # 原有，保留
│   ├── neo4j_import.py          # 原有，保留
│   └── disease_stage3.py        # 新增
├── data/
│   ├── kg_output_clean_v2/      # 已清洗版中间 JSONL
│   └── raw/drug_db/MedGraphRAG-Drug-DB/
│       ├── drug_db_v2_cleaned.json
│       └── medical.json         # 先检查是否为疾病知识图谱数据
└── .env                          # 原有 Neo4j 连接配置
```

**路径以你 `sync_plan.json` 中的 `data\\raw\\drug_db\\...` 为准。**
请在 `RAG_project` 项目根目录打开终端运行。
如果你的 VS Code 工作目录比截图多了一层 `data/`，相应调整路径。

## 第一步｜检查疾病词表格式

务必先确认本地 `medical.json` 是之前 QASystemOnMedicalKG 的疾病记录，
而不是另一份药品数据库。使用：

```bash
python -m kg.disease_stage3 inspect --medical "data/raw/drug_db/MedGraphRAG-Drug-DB/medical.json"
```

重点观察 `sample_names` 和 `top_fields`：
- 应以 **疾病名** 为主，例如有 `name`、`symptom`、`common_drug` 等字段；
- 如果显示药品名称，不能直接拿来充当疾病词表；
- 它只是历史医疗知识数据集，并不等于现行医疗术语标准。

## 第二步｜先扫描 300 种药品

```bash
python -m kg.disease_stage3 extract --medical "data/raw/drug_db/MedGraphRAG-Drug-DB/medical.json" --in-dir "data/kg_output_clean_v2" --out "data/kg_output_clean_v2/disease_candidates.csv" --limit-drugs 300 --min-chars 3
```

- 随机种子固定为 42，重复运行可复现抽样。
- `--limit-drugs 0` 扫描全部 11,953 种药品。
- 默认疾病名最短 3 个字符，减少两字短词误匹配。
- 如果要覆盖「哮喘」「肺炎」等两字疾病，可尝试 `--min-chars 2`，
  但必须加强人工筛查。
- 如果药名有多份来源说明书，可能输出多个不同原文支持的候选。
- 词典匹配的关系 **100% 是待审核候选**，不会自动批准。

结果：

```
data/kg_output_clean_v2/
  disease_candidates.csv
  disease_candidates.report.json
```

## 第三步｜人工审核 CSV

可用 Excel / WPS 查看 CSV（UTF-8 BOM）。不要改变 `candidate_id`、
`section_id` 和任何证据偏移字段。

对每条候选，重点检查：
- 药品与疾病名称是否正确；
- 证据是否真的表述适应症，还是病例、禁忌、例外或否定；
- 是否为治疗、预防，还是仅缓解症状；
- 该历史第三方说明书是否提供足以支持研究标注的原文。

**仅对确实完成语义判断的行**填写：

| CSV 列 | 值 |
|---|---|
| `review_status` | `approved` |
| `relation_type` | `treatment`、`prevention`、`symptom_relief` 任选其一 |
| `reviewer` | 研究者标识，例如 `reviewer_01` |
| `review_note` | 审核依据/保留条件（可选） |

不通过的行写 `rejected` 或保持 `pending`。

`context_flag=negation_or_restriction_check` 的候选不能直接作为正向关系入库。
当前保守版会拒绝它们，需要以后单独设计否定/禁忌断言抽取。

**不要把整列候选统一标记 approved。**

## 第四步｜审查数据，生成 Claim

```bash
python -m kg.disease_stage3 review --csv "data/kg_output_clean_v2/disease_candidates.csv" --in-dir "data/kg_output_clean_v2" --out-dir "data/kg_output_clean_v2"
```

会生成：
- `disease_terms.jsonl`
- `indication_claims.jsonl`
- `disease_review_report.json`

审核时会核对引文、原文偏移、药品 ID、来源记录、所属章节以及疾病片段。
对于可疑否定句拒绝导出正向 Claim。

每条 Claim 保留 `clinical_status=not_validated` 和
`regulatory_status=source_version_unverified`。
这不是 FDA / NMPA 确认的药品获批适应症。

## 第五步｜仅导入新增疾病节点与 Claim

先不写数据库，预览：

```bash
python -m kg.disease_stage3 import --out-dir "data/kg_output_clean_v2"
```

审核结果合理后再执行：

```bash
python -m kg.disease_stage3 import --out-dir "data/kg_output_clean_v2" --apply
```

导入时会**先读取现有 Neo4j**，确认每条 Claim 能匹配：
`Drug → SourceRecord → Section` 和精确的证据字符串。
如果对不上，整批写入会在预检查阶段被阻止。

新增图谱：

```
(IndicationClaim)-[:SUBJECT]->(Drug)
(IndicationClaim)-[:OBJECT]->(DiseaseTerm)
(IndicationClaim)-[:SUPPORTED_BY]->(Section)
```

药品之间共享同一个 `DiseaseTerm` 时，自然可以沿图谱检索多种药物。
`INDICATION_TEXT_MENTIONS` 表示“经过审核的说明书原文提及及关联类别”，
**不等于该药在中国获批治疗该病**。

首次审核时建议只选择 20—50 条容易判断的候选作为试运行，
图谱结构和证据检索通过后再扩大至全部 300 种药品。

**注意：** 入库采用 `MERGE`，重新审核时把原来的批准改为拒绝，不会自动撤销已经写入 Neo4j 的旧 Claim。需要单独的撤回/版本更新流程，不要将重审 CSV 误认为数据库已经同步删除。

## 第六步｜Neo4j Browser 检查

检查新增关系：

```cypher
MATCH (claim:IndicationClaim)-[:SUBJECT]->(drug:Drug),
      (claim)-[:OBJECT]->(disease:DiseaseTerm),
      (claim)-[:SUPPORTED_BY]->(s:Section)
RETURN drug.name AS drug,
       disease.name AS disease,
       claim.relation_type AS relation,
       claim.evidence_quote AS evidence,
       claim.clinical_status AS status
LIMIT 20;
```

可视化共享疾病节点：

```cypher
MATCH p1=(a:Drug)<-[:SUBJECT]-(c1:IndicationClaim)-[:OBJECT]->(x:DiseaseTerm)
MATCH p2=(b:Drug)<-[:SUBJECT]-(c2:IndicationClaim)-[:OBJECT]->(x)
WHERE a.id < b.id
RETURN p1,p2 LIMIT 5;
```

第一条查询没有结果时，先确认 `disease_review_report.json` 中
`accepted_claims > 0`，以及导入确实使用了 `--apply`。

## 质量与论文记录

建议在报告中记录：候选覆盖率、实体精准率、人工确认比例、
负向/限制性表述错误率、证据匹配成功率、审核用时、模型/词表版本。

对比普通 RAG / GraphRAG 之前，先建立独立的标注集，
不要把同一份用于调试的候选也作为最终测试集。

**已知边界：**
- `medical.json` 未由用户上传到当前对话，完整词表结果需要在本机执行；
- 新脚本已在清洗后的真实 JSONL 上用小词表抽样验证运行；
- 尚未连接用户的本机 Neo4j，数据库写入只在本机 `--apply` 时发生；
- GitHub 第三方整理说明书的再利用许可与版本有效性仍需核查。
