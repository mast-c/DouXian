# DrugKG · V3.1 适应症语义范围与限定条件改造

适用于现有 `RAG_project/data` 目录（终端位于 `...\RAG_project\data>`）。

## 1. 为什么要做 V3.1

上一轮 Qwen V3 的 60 条开发集：
- `approved|accept = 43`
- `approved|uncertain = 7`
- `rejected|accept = 1`
- `rejected|reject = 3`
- `rejected|uncertain = 6`
- LLM 58 条输出验证成功、2 条字段校验失败。

**当前并没有临床专家金标准。** 上述与试标一致率只能用于开发诊断。

14 条异常中：
- 11 条触发“疾病名过宽”规则，将模型 accept 调整为 uncertain；
- 2 条模型原文引用或疾病名对应关系校验失败；
- 1 条“注射用布美他尼—原发性高血压”涉及
  “不是首选，但其他药物疗效不佳等条件下适用”标注口径争议。

正确做法不是取消所有的安全规则，而是把：
**原文支持关系、完整疾病目标、适用限制、无条件候选判断**
拆分成不同字段，并让审查人员独立复裁。

## 2. 新增内容

- `kg/indication_v31.py`：新增，不覆盖 `kg.indication_v3`。
- `prompts/indication_semantic_review_v31.md`：Qwen 结构化判断提示词。
- `tests/test_indication_v31.py`：12 项本地单测，不产生 API 费用。
- `reports/v31_local_validation.json`：现有真实数据的静态验证汇总。

输出 CSV 主要字段：

| 新字段 | 含义 |
|---|---|
| `v31_first_model_raw_json` | 第一次模型原始响应（即使后续重试成功） |
| `v31_model_raw_json` | 最后一次模型响应 |
| `v31_attempts_json` | 各次调用的模型输出与校验结果（可追踪重试） |
| `v31_model_raw_decision` | 最后一次模型原始 accept / reject / uncertain |
| `v31_specific_target_surface` | LLM 原文中的完整疾病名（必须覆盖候选命中位置） |
| `v31_qualifiers_json` | 逐字引用的适用条件 |
| `v31_candidate_scope` | `specific_disease_or_subtype` / `conditional_only` 等 |
| `v31_text_relation_status` | 包括 `conditional_positive_proposal` / `specific_target_proposed_review` |
| `v31_candidate_decision` | 无条件候选 `accept / reject / uncertain` |
| `v31_override_reasons_json` | 原始判断被安全规则调整的原因 |
| `v31_evidence_validated` | 引用、疾病名、具体候选位置是否都通过核验 |
| `review_status` | **原始人工/AI辅助标签不变** |

特别注意：对于条件性关系，`v31_text_relation_status =
conditional_positive_proposal` 且 `v31_candidate_decision = uncertain`
是**正常、预期的结果**，因为系统不准把有条件关系直接改成无条件边。

### 针对两条原先校验失败的策略

- 模型若把候选“尿路感染”错误链接到同句的“肾盂肾炎”：
  直接判定证据不匹配；保留模型原始响应，最多重试一次。
- 模型若把肝硬化的 `supporting_quote` 改写，而不是逐字复制：
  提示重试；绝不为了通过校验偷偷把模型引用替换成原文。

重试不成功：候选仍是 `uncertain`，必须人工复核。

## 3. 解压方式

ZIP 解压到 **`RAG_project/data/`**，新文件应位于：

```text
RAG_project/
└── data/
    ├── kg/
    │   ├── indication_v3.py       # 旧版保留
    │   ├── disease_stage3.py      # 旧版保留
    │   └── indication_v31.py      # 新增
    ├── prompts/
    │   └── indication_semantic_review_v31.md
    └── kg_output_clean_v2/
        ├── drugs.jsonl
        ├── source_records.jsonl
        ├── sections.jsonl
        ├── pilot_review_60.csv
        └── pilot_v3_review_queue.csv
```

`pilot_v3_review_queue.csv` 为上一阶段你已经生成的 14 条异常记录文件。
它只用于读取 `candidate_id`，不作为 V3.1 的原文证据来源。

## 4. 第一步：免费检查 14 条异常的证据和范围风险

在现有 `RAG_project/data` PowerShell 终端执行：

```powershell
python -m kg.indication_v31 run --input "kg_output_clean_v2/pilot_review_60.csv" --in-dir "kg_output_clean_v2" --ids-from "kg_output_clean_v2/pilot_v3_review_queue.csv" --out "kg_output_clean_v2/pilot_v31_14_rules.csv" --mode rules
```

应返回 `selected_rows: 14`、`graph_writes: 0`。
`mode rules` 只做**证据完整性和安全风险筛查**，不会自动得出 14 条可靠临床结论。

若出现证据校验失败，先检查使用的是否仍是旧版
`drug_db_v2_cleaned.json` 生成的 `kg_output_clean_v2`。

## 5. 第二步：仅让 Qwen 重跑这 14 条（推荐）

你此前已成功配置百炼 `DASHSCOPE_API_KEY`。
继续沿用同一 PowerShell 终端，避免泄露 API Key。

```powershell
python -m kg.indication_v31 run --input "kg_output_clean_v2/pilot_review_60.csv" --in-dir "kg_output_clean_v2" --ids-from "kg_output_clean_v2/pilot_v3_review_queue.csv" --out "kg_output_clean_v2/pilot_v31_14_llm.csv" --mode llm --model "qwen3.7-flash" --endpoint "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" --api-key-env "DASHSCOPE_API_KEY" --max-llm 14 --max-retries 1 --prompt-file "prompts/indication_semantic_review_v31.md"
```

**实际可能发出最多 28 次请求**：14 个候选，若验证失败各重试一次。
这产生模型调用费用。模型若不支持 JSON mode，可尝试追加 `--no-json-mode`。

预期 `counts` 中：
- `llm_attempted = 14`
- `rows = 14`
- 理想情况下 `llm_valid = 14`，`llm_errors = 0`

每条还会保留第一次和最后一次模型输出，便于审查。

如果输出文件已存在，默认拒绝覆盖。可以换一个输出文件名，
或者显式添加 `--overwrite`（慎用）。

## 6. 第三步：与原先 60 条试标标签比较

```powershell
python -m kg.indication_v31 evaluate --pred "kg_output_clean_v2/pilot_v31_14_llm.csv" --labels "kg_output_clean_v2/pilot_review_60.csv"
```

这里的 `evaluated_rows` 应为 **14**，而非 60；
因为只对 14 条异常做了重跑。这一组被筛选出来的高风险样本
不能当作模型整体精确率的无偏估计。

注意查看：
- `v31_text_relation_status=conditional_positive_proposal`
  （有条件成立，但非无条件关系）
- `v31_text_relation_status=specific_target_proposed_review`
  （原文包含更具体疾病名）
- `v31_validation_error`（引用或位置校验失败）
- `v31_override_reasons_json`（模型 accept 被规则保留为 uncertain 的原因）

如果要查看高优先级样本，可以用 PowerShell：

```powershell
Import-Csv "kg_output_clean_v2/pilot_v31_14_llm.csv" |
  Select-Object drug_name,disease_name,v31_model_raw_decision,v31_candidate_decision,v31_text_relation_status,v31_specific_target_surface,v31_validation_error |
  Format-Table -AutoSize
```

## 7. 第四步（在修正 14 条后）：重新测试全部 60 条

注意不同版本的提示词可能改变判断，所以**不能**把
“旧版 46 条输出”和“新版 14 条输出”拼在一起称为一个完整评测。
应当对开发集全部 60 条重新调用同一版本。

```powershell
python -m kg.indication_v31 run --input "kg_output_clean_v2/pilot_review_60.csv" --in-dir "kg_output_clean_v2" --out "kg_output_clean_v2/pilot_v31_60_llm.csv" --mode llm --model "qwen3.7-flash" --endpoint "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" --api-key-env "DASHSCOPE_API_KEY" --max-llm 60 --max-retries 1 --prompt-file "prompts/indication_semantic_review_v31.md"
```

最多 120 次 API 请求（包括校验失败后最多一次重试）。
完成后冻结代码、提示词哈希与模型版本，再评价独立的 300 条数据；
300 条评测文件尚未作为本补丁的调参或测试依据。

## 8. 入图的边界

**本版本完全不修改 Neo4j。**

不能直接把所有：
- `v31_candidate_decision = accept`、
- `v31_text_relation_status = conditional_positive_proposal`、
- `v31_text_relation_status = specific_target_proposed_review`

转换为 `Drug-[:TREATS]->Disease`。
正式的 Claim 入库还必须经过来源权威及版本审查、
疾病本体链接、专业复核和证据/适用条件审核。

现有 50 条已经批准的 AI 辅助文本 Claim 也不自动升级为临床专家确认。
