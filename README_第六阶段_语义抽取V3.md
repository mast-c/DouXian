# DrugKG 第六阶段：药品—疾病语义抽取 V3

**功能：疾病名称具体性检测 + 否定/限定条件解析 + 治疗目标角色识别 +
可选 LLM 语义分类 + 原文证据校验。**

本补丁由 `kg/indication_v3.py` 实现，只增量**新增一个 Python 文件**，
不替换现有 `kg.build`、`kg.disease_stage3`、`kg.review_context_v2`，
也不更改 Neo4j / ChromaDB。

## 数据范围与已有验证

输入可以是清洗后词表生成的：
`data/kg_output_clean_v2/disease_candidates_full_v2.csv`，
或者开发试标集：
`data/kg_output_clean_v2/pilot_review_60.csv`。

本版本已用现有真实清洗版药品文档图谱验证：
- 19,686 条候选的 ID、药品、SourceRecord、Section、引用位置全部校验通过；
- 1,439 条被标记为疾病名称具体性风险；
- 110 条被提示为患者条件、排除或结果角色风险；
- 1,789 条被提示为极性/使用条件风险（计数互相重叠）；
- 在既有 60 条 **AI 辅助试标开发集**中，10 条拒绝样本均命中至少一项
  针对性风险提示，规则没有直接拒绝任何已标注批准的样本；
- 规则版对开发集只给出 2 条明确拒绝建议，其余 58 条保持 `uncertain`；
- 这些数据不是最终算法准确率，**未运行独立的 300 条评测集**；
- 8 项单元测试通过，包括模拟 LLM 响应和避免「出现高血压危象」的误拒绝。

## 解压后目录

```
RAG_project/
├── kg/
│   ├── indication_v3.py             # 新增
│   ├── disease_stage3.py             # 旧文件，保留
│   └── review_context_v2.py          # 旧文件，保留
├── prompts/
│   └── indication_semantic_review_v3.md
└── data/
    └── kg_output_clean_v2/
        ├── drugs.jsonl
        ├── source_records.jsonl
        ├── sections.jsonl
        ├── pilot_review_60.csv
        └── disease_candidates_full_v2.csv
```

**在 `RAG_project` 根目录打开 VS Code 终端，逐条执行下列命令。**

## 第一步（无需 API）：对 60 条试标做规则筛查

```bash
python -m kg.indication_v3 run --input "data/kg_output_clean_v2/pilot_review_60.csv" --in-dir "data/kg_output_clean_v2" --out "data/kg_output_clean_v2/pilot_v3_rules.csv" --mode rules
```

命令输出 `pilot_v3_rules.v3_report.json`，
同时生成包含原有审核列 + V3 语义风险标记的 `pilot_v3_rules.csv`。

评测开发集：

```bash
python -m kg.indication_v3 evaluate --pred "data/kg_output_clean_v2/pilot_v3_rules.csv" --labels "data/kg_output_clean_v2/pilot_review_60.csv"
```

这是 **开发集诊断**，不能把结果写成 300 条独立测试集的研究结论。

## 第二步（可选）：LLM 语义分类试验

只有在你的 API Key、模型、授权、发送第三方文档到相应服务的合规性都确认后运行。
下列命令只请求 **最多 60 条**，不自动调用全量 19,686 条。

PowerShell 示例（请自行填入自己的安全凭据，不要放进源码/提交仓库）：

```powershell
$env:OPENAI_API_KEY="YOUR_PRIVATE_API_KEY"
```

先安装 Neo4j 关系抽取所需的旧模块及本补丁；
这个模块的 HTTP 调用使用 Python 标准库，不需要额外安装 OpenAI SDK。

```bash
python -m kg.indication_v3 run --input "data/kg_output_clean_v2/pilot_review_60.csv" --in-dir "data/kg_output_clean_v2" --out "data/kg_output_clean_v2/pilot_v3_llm.csv" --mode llm --model "YOUR_CHAT_MODEL" --max-llm 60 --limit 60 --prompt-file "prompts/indication_semantic_review_v3.md"
```

这条命令默认请求 `https://api.openai.com/v1/chat/completions`。
如果使用其他 OpenAI-compatible 服务或本地兼容接口，可通过
`--endpoint "https://YOUR_SERVICE/v1/chat/completions"` 自定义。
仅支持 HTTPS 或 `http://localhost` / `http://127.0.0.1`。
如果模型服务不支持 JSON mode，补充 `--no-json-mode`。

再在开发集上比对：

```bash
python -m kg.indication_v3 evaluate --pred "data/kg_output_clean_v2/pilot_v3_llm.csv" --labels "data/kg_output_clean_v2/pilot_review_60.csv"
```

**重要：** 模型预测 `accept` 并不修改 `review_status`，
也不会插入 Neo4j。
对于证据不完整、疾病过宽、角色冲突和负向语义，程序会回退 `uncertain`。
所有预测一律要求人审并保留原文可追溯信息。

### 控制调用成本

- `--mode rules`：只使用规则，不发送请求
- `--mode llm`：按顺序请求最多 `--max-llm N` 条
- `--mode hybrid`：先使用规则，只对标记存在语义风险的候选发请求，
  最多 `--max-llm N` 次；其余候选仍是人工待审核
- 默认 `--max-llm 60`，避免意外批量调用
- `--limit 60` 会限制本次输入为前 60 条，便于试运行
- `--prompt-file` 的内容哈希被记入运行报告，便于论文复现

若发生 HTTP、JSON、证据引用错误，单条输出记录会记下错误，
不会凭空生成通过的关系。

### 隐私及数据使用边界

第三方药品说明书不一定允许上传到云端模型服务；
请选择你有权使用的端点或经过适当部署的本地服务。
医疗研究结论在实际使用前仍需要权威来源与医学专业审核。

## 第三步：全量 19,686 条的离线筛查

```bash
python -m kg.indication_v3 run --input "data/kg_output_clean_v2/disease_candidates_full_v2.csv" --in-dir "data/kg_output_clean_v2" --out "data/kg_output_clean_v2/disease_candidates_v3_rules.csv" --mode rules
```

这一步不需要 LLM API，也不会删除历史候选 CSV。
输出约 19,686 条记录，带 `v3_risk_flags` 等附加字段。

全部 19,686 条候选都保留原始 `review_status`，
所以 **不要把 V3 建议的 `accept` 当作已审核 `approved`**。
真正的 Claim 导出仍需走 `kg.review_context_v2` 的审核流程。

## V3 输出字段

| 字段 | 说明 |
|---|---|
| `v3_proposed_decision` | accept / reject / uncertain（机器建议） |
| `v3_proposed_relation` | 仅模型明确建议接受时才有关系类型 |
| `v3_target_role` | indication_target / patient_condition / exclusion / adverse_effect_or_cause / background / uncertain |
| `v3_specificity` | exact / broader_than_evidence / narrower_than_evidence / ambiguous |
| `v3_polarity` | positive / negative / conditional / uncertain |
| `v3_qualifiers_json` | 适用条件的原文引用片段 |
| `v3_risk_flags` | 疾病过宽、患者条件等风险 |
| `v3_llm_status` | not_requested / validated_model_output / failed_needs_manual_review |
| `v3_needs_human_review` | 始终 true |
| `review_status` | 原始人工审核状态，绝不自动改写 |
| `candidate_id` / `section_id` | 与原始文档证据的稳定关联 |

## 质量评测和后续里程碑

1. **现在：** 规则版筛查 60 条开发集 + 全量候选数据。
2. **然后：** 在有权访问的 LLM 端点上，使用 60 条开发集验证三类错误，
   迭代提示词、角色判定与具体疾病名称识别。
3. 冻结 V3 模型 ID、提示词和预处理规则。
4. 最后运行此前保留的 300 条独立评测数据，
   执行人工/专家复核后计算分层加权候选级 Precision。
5. 从**未被词典匹配**的适应症原文建立另一个金标准集，
   才能计算完整召回率和 F1。
6. 只将证据支持且经合格审核的 Claim 逐步增量导入 Neo4j，
   然后再扩展疾病—症状—科室与 GraphRAG。

### 不要误用

- 此程序的 `reject` 仍是**机器拒绝建议**，不可直接删除旧 Claim。
- 由 Codex 辅助标注的 60 条试标不是独立医生审核金标准。
- 如果 `v3_proposed_decision=accept`，
  `review_status` 仍保持 `pending` 或原有状态。
- 不会自动生成已获批 `Drug-[:TREATS]->Disease` 关系。
