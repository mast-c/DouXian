# 药品—疾病候选关系科研标注与评测包

源文件：用户上传的 `disease_candidates_full.csv`，19,685 条候选记录。
匹配的数据版本与词表版本已写入 `sampling_manifest.json`。

## 文件

- `pilot_review_60.csv`：开发集，60 条，供人工试标、调整提示词和审核指南。
- `independent_evaluation_300.csv`：独立测试集，300 条，用于最终固定版本评测。
- `sampling_manifest.json`：随机种子、原 CSV 哈希、层大小和样本权重。
- `evaluate_holdout.py`：完成标注后，计算分层加权候选级 Precision 的脚本。

所有 CSV 候选初始状态都是 `review_status=pending`，没有人为填入任何医学结论。
Python 编码为 `utf-8-sig`，便于 Excel/WPS 阅读中文。

## 为什么不是直接对全量 19,685 条逐个标注？

先用 60 条开发集评估难点和标注一致性，然后用不重叠的 300 条独立集衡量实际质量。
按 `context_flag` 分层抽取：

| 层 | 全量 | 开发集 | 独立评测 |
|---|---:|---:|---:|
| 常规语义审核 | 12,275 | 30 | 150 |
| 短词歧义 | 7,183 | 20 | 100 |
| 否定/限制提示 | 227 | 10 | 50 |
| 合计 | 19,685 | 60 | 300 |

否定/限制样本被刻意超额采样。**不要直接用 300 条的未经加权正确率代表总体准确率**，
应使用 `sampling_manifest.json` 中各层权重。

## 人工审核规则（建议）

- `approved`：依据说明书适应症栏目和提供的引文，药品与疾病存在明确的正向关系；
  必须填写 `relation_type`（`treatment` / `prevention` / `symptom_relief`）和 `reviewer`。
- `rejected`：只出现疾病名称，但属于禁忌、否定、病例背景、无关描述、较短的疾病词误命中，
  或不能支持特定关系类别。
- `uncertain` / `pending`：缺少上下文、条文有歧义、资料难以判断；
  不要把不确定的关系硬标为正确。
- `review_note`：建议记录拒绝/不确定原因、剂型限制、适用人群以及需要的原文上下文。

务必保留 `candidate_id`、`section_id`、`source_record_id`、引文和字符位置不变。

注意：数据库来源仍是第三方历史说明书。即使候选在文本层通过审核，
也不等于该关系已获得国家药监局批准或具备真实临床使用的权威性。

## 评测脚本

在本文件夹运行：

```bash
python evaluate_holdout.py --csv independent_evaluation_300.csv --manifest sampling_manifest.json
```

若没有把 300 条样本全部明确标记为 `approved` 或 `rejected`，脚本只输出未完成数，
不会编造精确率。标注完成后，输出按三层总体占比加权的候选级 Precision，
以及简单近似的置信区间。

**重要方法学边界：** 这只能衡量当前**已提出的候选关系中，多少在文本证据层正确**。
从 `disease_candidates_full.csv` 无法衡量模型遗漏了多少正确疾病关系。
要算 Recall/F1，还需另外从**没有匹配到疾病**的适应症章节和原文片段随机抽样，
人工独立识别全部应抽取关系，再与程序输出比较。

## 下一步（关系图谱）

1. 先标注 `pilot_review_60.csv`，观察错误类型。
2. 修复短词/否定/条件处理，冻结抽取规则。
3. 最后标注 `independent_evaluation_300.csv`，计算独立 Precision。
4. 按人工审核结果把有效 Claim 输入 `kg.disease_stage3.review`。
5. 只把经证据审核的 Claim 和共享 DiseaseTerm 节点导入 Neo4j。
6. 比较 Text RAG、图检索、Hybrid GraphRAG 的证据召回与引用质量。

不要因为某疾病名匹配成功，就自动生成无条件 `Drug-[:TREATS]->Disease`。
