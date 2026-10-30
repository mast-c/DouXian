# DrugKG · 第九阶段：人工审查 / 版本冻结 / 300 条独立评测

当前基础：`pilot_v31_merged60.csv`（60 条，58 条有效 Qwen 输出，
26 接受、34 不确定）；上一阶段所有样本仅是开发集，原始 `review_status`
为 AI 辅助试标，**不是医学专家金标准**。

此资料包只用来组织开发集审查、冻结工程代码和
在未来测试独立 300 条时避免泄漏标签；**不修改 Neo4j**。

## 解压位置

将 ZIP 内容解压到：

`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

应看到：

```text
data/
├── scripts/
│   └── v31_study_gate.py
├── review/
│   ├── reviewer_A_21_blind.csv
│   └── reviewer_B_21_blind.csv
├── protocol/
│   └── 人工盲审标注规范.md
├── tests/
│   └── test_v31_study_gate.py
└── README_第九阶段.md
```

本包不覆盖旧版 `kg` 代码与开发集结果。

---

## 第一步：21 条重点人工复核

分别发送给两位审核者：

- `review/reviewer_A_21_blind.csv`
- `review/reviewer_B_21_blind.csv`

每个 CSV 都只包含原文证据、候选疾病与来源，不含 Qwen 原始判断、
旧版 AI 标签或事先的高风险标记。
两名审核者独立填写，不要互相抄写结论。

审核时重点确认药品产品身份、病种范围、症状/疾病类型、
疾病是否治疗目标、治疗/预防/缓解关系及限制条件。
如果来源不充分，请诚实使用 `INSUFFICIENT_EVIDENCE`；
如果未核验正式药品说明书，`source_identity_verified=unknown`。

**审查完成后**运行：

```powershell
python scripts/v31_study_gate.py review-status --reviewer-a "review/reviewer_A_21_blind.csv" --reviewer-b "review/reviewer_B_21_blind.csv" --disagreements "review/reviewer_disagreements21.csv"
```

会报告已填多少条、多少条核心标签一致、多少条需人工仲裁。
它不会自动认定某人的裁决正确，也不会伪造医学专家验证。

### 需要时重新生成盲审表（请先备份已填写的版本）

```powershell
python scripts/v31_study_gate.py make-review-forms --priority "kg_output_clean_v2/v31_priority_review.csv" --out-dir "review/new_round"
```

注意：如果文件已经存在，程序会拒绝覆盖，避免删除人工审核结果。

---

## 第二步：正式冻结工程基线（不联网、不访问 300 条）

在 PowerShell 的 `RAG_project\data>` 目录运行：

```powershell
python scripts/v31_study_gate.py freeze --code "kg/indication_v31_checkpoint.py" --prompt "prompts/indication_semantic_review_v31_checkpoint.md" --dev-csv "kg_output_clean_v2/pilot_v31_merged60.csv" --dev-report "kg_output_clean_v2/pilot_v31_merged60.v31_report.json" --model "qwen3.7-flash" --timeout 90 --max-retries 0 --out "configs/freeze_v31_engineering.json"
```

输出 `configs/freeze_v31_engineering.json`，包含：
- Python 源码、提示词、开发集 CSV/报告的 SHA-256；
- 模型与 endpoint、timeout/retries；
- 完成状态和此前的开发集数据来源；
- `status=engineering_baseline_only`，明确不是临床审批。

请**只在完成开发集调试且不再修改代码/提示词时冻结**。

任何时候可检查：

```powershell
python scripts/v31_study_gate.py verify-freeze --freeze "configs/freeze_v31_engineering.json"
```

如果源码或提示词变更，检查会拒绝通过。
冻结时不会读取 `independent_evaluation_300.csv` 的内容。

---

## 第三步：冻结后才准备 300 条独立推理输入

已有的 `independent_evaluation_300.csv` 位置可能与你本地不同。
以下仅用参考路径，**实际文件不存在时请改成自己的真实路径**。
脚本要求存在候选原文、`section_id`、`source_record_id`、
`quote_start/end`、`disease_start/end` 等字段。

```powershell
python scripts/v31_study_gate.py prepare-holdout --freeze "configs/freeze_v31_engineering.json" --input "kg_output_clean_v2/independent_evaluation_300.csv" --out "kg_output_clean_v2/independent_300_inference_only.csv" --annotation-out "review/independent_300_blind_annotation.csv" --expect 300
```

执行时：
- 检查冻结源码、提示词没有变动；
- 检查 300 条候选 ID 与 60 条开发集是否重复；
- 另外统计相同来源记录、Section、证据原文重叠的数量；
- 清除任何旧 `review_status`、`relation_type`、gold 列；
- 只将**无标签原文**写入模型推理输入；
- 生成一个完全空白答案列的盲审模板。

注意：`candidate_id` 无重叠只表示候选层面的独立，
**不表示药品来源文档也不重叠**。如有源文件重叠，
应在评测报告中披露，不能称为严格文档级独立。

---

## 第四步：对冻结版本运行 300 条预测

使用现有的可断点恢复模块，模型、提示词、timeout 与冻结清单一致。
此阶段会调用外部 API 并产生费用，请先核实当前账户可用性和调用预算。

在 PowerShell 设置：

```powershell
$heldoutArgs = @(
  "run",
  "--input", "kg_output_clean_v2/independent_300_inference_only.csv",
  "--in-dir", "kg_output_clean_v2",
  "--out", "kg_output_clean_v2/independent_300_predictions.csv",
  "--mode", "llm",
  "--model", "qwen3.7-flash",
  "--endpoint", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
  "--api-key-env", "DASHSCOPE_API_KEY",
  "--max-llm", "300",
  "--max-retries", "0",
  "--timeout", "90",
  "--prompt-file", "prompts/indication_semantic_review_v31_checkpoint.md"
)
```

开始 10 条，主要用于检查接口是否正常保存、没有新格式问题；
**不要根据这 10 条的正确与否修改模型或规则**：

```powershell
python -u -m kg.indication_v31_checkpoint @heldoutArgs --stop-after 10
```

继续剩余的 290 条：

```powershell
python -u -m kg.indication_v31_checkpoint @heldoutArgs --resume
```

每条保存后实时显示进度。如果 `kg_output_clean_v2` 中存在同名预测文件，
务必检查是否属于同一次冻结配置；不要用 `--overwrite` 清空已有进度。

## 第五步：专家/标注人员独立生成最终 gold

复制 `review/independent_300_blind_annotation.csv` 给两位独立审核者，
他们不能看到预测的 `v31_*` 字段。
对分歧进行裁定，形成单独文件，例如：

`review/independent_300_gold_adjudicated.csv`

最重要的字段：
- `gold_classification`：SUPPORTED_UNQUALIFIED、SUPPORTED_QUALIFIED、
  NOT_SUPPORTED、NON_TARGET_CONTEXT、INSUFFICIENT_EVIDENCE；
- `gold_exact_target`：必须是原文子串；
- `gold_target_type`：Disease、Symptom、ClinicalFinding 等；
- `gold_relation_type`：treatment、prevention、symptom_relief；
- `gold_polarity`：positive、negative、conditional、uncertain；
- `gold_qualifiers_json`：适用条件 JSON 数组（必要时写 `[]`）；
- `source_identity_verified`：yes / no / unknown；
- `annotator_name`、`annotator_role`；
- 若 source identity yes，提供官方核实的来源和说明书版本。

没有人工标注，不应编造金标准来输出精度指标。

## 第六步：300 条独立评测

待 **300 条预测全部保存，且完整专家裁决 CSV 已准备**：

```powershell
python scripts/v31_study_gate.py score-holdout --freeze "configs/freeze_v31_engineering.json" --preflight "kg_output_clean_v2/independent_300_inference_only.holdout_preflight.json" --pred "kg_output_clean_v2/independent_300_predictions.csv" --pred-report "kg_output_clean_v2/independent_300_predictions.v31_report.json" --gold "review/independent_300_gold_adjudicated.csv" --out "reports/independent_300_eval_v31.json"
```

脚本校验 300 条 ID、冻结源码、提示词哈希、推理输入哈希、
模型报告 SHA、专家标注值和引文一致性。
输出 accepted 覆盖率、与无条件专家金标准吻合的比例、
错误/条件失真接受数、未核实来源的直接接受数和弃答数。

本脚本**不会**自动核查审查者职业资格或连接国家药监局。
输出不能替代临床验证与监管批准。

---

## 重要限制

- 用户此前上传的 `independent_evaluation_300.csv` 本次会话中
  **并未提供给脚本**，因此本包没有运行真实 300 条预测或金标准评测。
- 当前已有的 60 条全部来自第三方药品网站，仍需正规说明书核对。
- 300 条独立测试如果被拿来调参，再次报指标时就需要新的独立测试集。
- 本阶段代码仅使用 Python 标准库，所有测试均为合成数据，
  真实 60 条只用于冻结/生成 21 条盲审表，不涉及患者数据或诊疗决策。
