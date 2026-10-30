# V3.1 精确疾病命中补丁 · 重复实体定位

## 结果和问题
你本地运行的两条 V3.1 复测中：
- `llm_attempted=2`，`llm_valid=1`，`llm_errors=1`，`api_requests=3`
- `碘苷滴眼液 → 单纯疱疹` 的网络失败已恢复，具体的疾病仍需保持“单纯疱疹性角膜炎”限定。
- `注射用甲氨蝶呤 → 淋巴瘤` 的模型输出仍指错了“淋巴瘤”的出现位置。

原文：
`恶性淋巴瘤，非何杰金淋巴瘤和蕈样肉芽肿，多发性骨髓病；`

本条候选是第二处的“淋巴瘤”，而不是第一处“恶性淋巴瘤”。
即使模型回答“恶性淋巴瘤”，由于没有覆盖第二处候选的字符偏移，
程序仍必须拒绝该输出。

## 补丁修改
直接替换 `data/kg/indication_v31.py`，但不改 V3、不改数据库。
本补丁增强：
1. 所有同名命中的字符位置、候选出现序号、带目标标记原文（仅用于模型定位）。
2. 原文中用标点分隔出的局部文本提示：对这条样本得到
   `非何杰金淋巴瘤`，但此提示**并非已经验证的疾病实体**。
3. 校验仍要求模型返回的完整病名和原文引用覆盖指定的真实字符位置。
4. 只要完整病名比候选疾病名更具体，即使模型自称
   `specificity=exact`，仍保持 `uncertain` 等待审核。
5. 完整保留模型原始输出和重试记录，不将提示直接当作知识图谱事实。

## 解压路径

将 ZIP 解压到 `RAG_project/data`（不是项目根目录）。新增/更新：

```
data/
├── kg/
│   └── indication_v31.py                 # 替换旧 V3.1
├── prompts/
│   └── indication_semantic_review_v31_anchor.md
├── kg_output_clean_v2/
│   └── v31_retry_one_id.csv
├── tests/
│   └── test_indication_v31.py
└── README_第七阶段_V31精确定位补丁.md
```

保留其他旧文件，不用重新解析药品数据。

## 第一步：不调用 API 的定位检查

在终端 `E:\VsCode_Project\DockerTest_Agent\RAG_project\data>` 执行：

```powershell
python -m kg.indication_v31 run --input "kg_output_clean_v2/pilot_review_60.csv" --in-dir "kg_output_clean_v2" --ids-from "kg_output_clean_v2/v31_retry_one_id.csv" --out "kg_output_clean_v2/pilot_v31_anchor_rules.csv" --mode rules
```

打开生成的 CSV，确认：
- `v31_target_mention_ordinal` = `2`
- `v31_target_mention_count` = `2`
- `v31_targeted_source_segment_hint` = `非何杰金淋巴瘤`
- `v31_candidate_decision` = `uncertain`（规则模式是预期行为）

## 第二步：仅让 Qwen 重跑这 1 条

在同一 PowerShell 中保留已有 `DASHSCOPE_API_KEY`：

```powershell
python -m kg.indication_v31 run --input "kg_output_clean_v2/pilot_review_60.csv" --in-dir "kg_output_clean_v2" --ids-from "kg_output_clean_v2/v31_retry_one_id.csv" --out "kg_output_clean_v2/pilot_v31_anchor_one_llm.csv" --mode llm --model "qwen3.7-flash" --endpoint "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" --api-key-env "DASHSCOPE_API_KEY" --max-llm 1 --max-retries 0 --prompt-file "prompts/indication_semantic_review_v31_anchor.md"
```

`--max-retries 0` 是为了先花费**一次请求**验证新提示词。
如果失败，保留结果和模型输出，交给人工审查，不建议无休止重试。

目标：`llm_valid=1`，`llm_errors=0`，原始病名“非何杰金淋巴瘤”，
最终候选“淋巴瘤”由于过宽，仍应为 `uncertain` 而不是无条件 `accept`。

## 第三步：如果单条有效，再重新测试 60 条完整开发集

新提示词改变了模型输入，所以不能把旧 13 条 + 新 1 条混起来算一次准确率。

```powershell
python -m kg.indication_v31 run --input "kg_output_clean_v2/pilot_review_60.csv" --in-dir "kg_output_clean_v2" --out "kg_output_clean_v2/pilot_v31_anchor_60_llm.csv" --mode llm --model "qwen3.7-flash" --endpoint "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" --api-key-env "DASHSCOPE_API_KEY" --max-llm 60 --max-retries 1 --prompt-file "prompts/indication_semantic_review_v31_anchor.md"
```

最多可发生 120 次请求（每条 1 次，校验失败可最多重试 1 次）。
随后运行：

```powershell
python -m kg.indication_v31 evaluate --pred "kg_output_clean_v2/pilot_v31_anchor_60_llm.csv" --labels "kg_output_clean_v2/pilot_review_60.csv"
```

只对 AI 辅助试标做开发集比较；**不是医学专家金标准**。
确认输出质量和人工复裁口径后，冻结代码与提示词，再进入独立的
300 条评测。不要直接把待复核关系写成无条件 `Drug-[:TREATS]->Disease`。

## 校验与数据保护
- 14 项本地回归测试，使用 Mock LLM，不调用在线服务。
- 两次疾病提及的实际用户样本用于检查定位 hint 的正确性。
- 未操作 Neo4j，未修改原始 `review_status`。
