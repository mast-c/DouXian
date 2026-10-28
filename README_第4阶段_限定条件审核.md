# 第四阶段：试标 60 条 → 50 条带限定条件的暂存 Claim

已核查用户的 `pilot_review_60.csv`：
- 60 条全部完成审核，批准 50，拒绝 10，待审核 0。
- 批准类别：treatment 44 / symptom_relief 5 / prevention 1。
- 其中 6 条来自 `negation_or_restriction_check`，旧版
  `kg.disease_stage3.review` 会将它们一律拦截。
- 50 条药品—疾病文本 Claim 对应 **44 个 DiseaseTerm**。
- 所有候选 ID、药品/来源/Section/原文引文及字符位置都已与清洗版图谱吻合。
- `reviewer` 为 `Codex-医学文本审核`。本补丁一律视为**未经过独立专家核验的
  AI 辅助文本标注**，不是临床事实，也不是独立专家金标准。

## 功能与安全设计

新增 `kg/review_context_v2.py`，**不替换**现有模块，不重建药品图谱。

它的 `export` 功能：
1. 逐条检查候选 ID 与原始 `Section` 的精确字符位置；
2. 只处理 CSV 中 `approved` 的关系，拒绝/未决不成为图边；
3. 保留 6 条上下文敏感的批准记录；
4. 在 Claim 上保留人工审核备注 `review_note`、
   `qualification_note`、`requires_condition_check`；
   注意 `requires_condition_check=false` **不表示无适用条件**，
   而是未被当前规则明确标记，仍需保留原文。
5. 全部写成 **`provisional_text_relation`**、
   **`expert_review_status=pending`**、
   **`clinical_status=not_validated`**；
6. `import` 时先对 Neo4j 再做逐条证据预检查，
   仅 MERGE 新 DiseaseTerm 与 IndicationClaim，绝不新建无条件
   `Drug-[:TREATS]->Disease`。

**不应将这些 AI 辅助记录当作已核验获批适应症用于临床问答。**

## 放置路径

将 ZIP 在现有 `RAG_project` 根目录解压：

```text
RAG_project/
├── kg/
│   └── review_context_v2.py
├── data/
│   └── kg_output_clean_v2/
│       └── pilot_review_v2/
│           ├── pilot_indication_claims.jsonl
│           ├── pilot_disease_terms.jsonl
│           └── pilot_review_report.json
├── prompts/
│   └── indication_semantic_review.md
└── README_第4阶段_限定条件审核.md
```

保留原有的 `kg/disease_stage3.py`、`data/kg_output_clean_v2/` 原始
JSONL 及 Neo4j 中全部源文档节点不变。

## 操作 A：重新用本机已标注 CSV 导出（推荐，可复现）

将标注后的 `pilot_review_60.csv` 放到项目：

`data/kg_output_clean_v2/pilot_review_60.csv`

在 VS Code 的 `RAG_project` 根目录运行：

```bash
python -m kg.review_context_v2 export --csv "data/kg_output_clean_v2/pilot_review_60.csv" --in-dir "data/kg_output_clean_v2" --out-dir "data/kg_output_clean_v2/pilot_review_v2"
```

预期输出：
- `accepted_claims`: 50
- `disease_terms`: 44
- `context_sensitive_approved`: 6

此外，ZIP 已包含用本次上传 CSV **实际验证导出的文件**，
因此如果只想验证图结构，也可以跳过操作 A。

## 操作 B：先进行只读导入预览

```bash
python -m kg.review_context_v2 import --out-dir "data/kg_output_clean_v2/pilot_review_v2"
```

输出预计：
```json
{
  "mode": "READ_ONLY",
  "database": "neo4j",
  "disease_term_count": 44,
  "provisional_claim_count": 50,
  "context_sensitive_claim_count": 6,
  "expert_validation": "pending"
}
```

这条命令**不连接 Neo4j，也不写数据库**。

## 操作 C：可选的研究图谱入库

确保 Neo4j 已正常运行且已做好备份。
只有在你接受「研究暂存 / 待专家核验」状态时再运行：

```bash
python -m kg.review_context_v2 import --out-dir "data/kg_output_clean_v2/pilot_review_v2" --apply
```

该命令会先在 Neo4j 校验全部 50 条的 Drug、SourceRecord、
Section 及原文位置，再创建节点与关系。
如果任何一条证据不一致，直接停止写入。
若已存在专家批准/拒绝的同 ID Claim，脚本也会阻止覆盖。

## 操作 D：Neo4j Browser 中核查

```cypher
MATCH (claim:IndicationClaim)-[:SUBJECT]->(drug:Drug),
      (claim)-[:OBJECT]->(disease:DiseaseTerm)
WHERE claim.schema_version = 'qualified_review_v2_2026_10'
RETURN count(DISTINCT claim) AS provisional_claims,
       count(DISTINCT disease) AS disease_terms,
       sum(CASE WHEN claim.requires_condition_check THEN 1 ELSE 0 END)
           AS flagged_for_conditions;
```

预期 50 / 44 / 6。

可视化：

```cypher
MATCH p1=(drug:Drug)<-[:SUBJECT]-
      (claim:IndicationClaim)-[:OBJECT]->(disease:DiseaseTerm)
WHERE claim.schema_version = 'qualified_review_v2_2026_10'
RETURN p1 LIMIT 20;
```

检查限定条件：

```cypher
MATCH (claim:IndicationClaim)-[:SUBJECT]->(drug:Drug),
      (claim)-[:OBJECT]->(disease:DiseaseTerm)
WHERE claim.schema_version = 'qualified_review_v2_2026_10'
  AND claim.requires_condition_check = true
RETURN drug.name AS drug,
       disease.name AS disease,
       claim.relation_type AS relation,
       claim.review_note AS reviewer_notes,
       claim.evidence_quote AS evidence
LIMIT 20;
```

## 之后如何提升 19,685 条全量候选的质量

当前 60 条开发标注最突出的错误是：
- 6 条**疾病名过宽/不完整**：例如「强直性脊椎炎」不可归为所有「脊椎炎」；
- 3 条**否定/限定条件**：不要整句因出现“无效”就误删肯定适应症；
- 1 条**对象角色混淆**：「伴有肾功能不全」中的疾病是使用条件，
  不一定是治疗目标。

推荐先在开发集上优化：
1. 加入疾病原文边界/更具体诊断名检测；
2. 为疾病提及分类 `indication_target`、
   `patient_condition`、`exclusion`、`background`；
3. 提取 `therapy_failure`、`disease_subtype`、`population`、
   `combination_therapy` 等限定条件；
4. LLM 判断只做候选提议，最后保留人审与字符级证据核对；
5. **冻结提示词和规则后**才开启 300 条独立评测集。

不要用这 60 条试标集的 83.3% 直接宣称最终模型准确率，
更不要用独立评测集进行调参。
