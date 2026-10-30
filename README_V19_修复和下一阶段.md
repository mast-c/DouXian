# 豆馅 V1.9：药品实体约束 + 可重复的检索基线

## 一、基于你提供的 V1.8 实际运行截图

### 已通过
- Neo4j 图谱疾病候选正常命中；
- JSON 感知切分产生的 Chroma Chunk 正常命中；
- Chroma Chunk 回溯 Neo4j `SourceRecord → Section` 原文，技术匹配正常；
- `pending_human_review` 被正确解释为待人工审核，不作为治疗事实；
- 相同 URL 的疾病候选与适应症原文合并显示为一个来源链接。

### 发现并修复的问题
原问题是“阿托伐他汀钙片与动脉粥样硬化有什么关联？”。
V1.8 的证据 3 却来自“10%、20%、30%脂肪乳注射液”。
**同一个疾病词被提及并不代表是同一种药，原文核验成功也不代表对当前问题相关。**

V1.9 在已有 Neo4j/Chroma 证据合并前后增加药品实体门控：
1. 如果问题精确包含了检索证据的药品名称，仅允许这些被明确点名的药品进入正式证据；
2. Chroma 重排后，在 Neo4j 父级 Section 核验前，先过滤已明确标明为其他药品的文档，减少占用有限的 top-k 配额；
3. Self-RAG 补检也使用同一组明确药品名，不允许补检重新引入其他药品；
4. 问题只含疾病、不含明确药品时，不做单药限制；用户明确比较两个药品时，同时保留这两种药；
5. 对没有 `drug_name` 元数据的老文档，必须在标题或正文直接包含所问药品名称，才能成为该药证据。

**限制**：当前为精确名称约束，不解决药品简称、别名、错别字与成分—制剂映射。Neo4j/Chroma 当前候选里都没有精确药品名时，不会猜测，也不会宣称已经成功执行药品归一。

### 另一个细节
严格模式的 Evidence Guard 可能额外生成“缺乏作用机制资料”的拒答说明，即使用户只问“有什么关联”。
V1.9 在流式结果尾部增加只针对用户实际询问维度的缺证据提醒；
用户确实问了机制、禁忌、剂量时，对应缺证据提醒仍保留。

## 二、安装

本包基于你刚刚实际运行的 **V1.8 rag.py** 生成。
请先备份当前运行的 `rag.py`、`medical_answer_quality.py`。

将以下文件复制到现有项目目录（与 `knowledge_base.py`、`豆馅.py` 同级）：
- `rag.py`：替换 V1.8；
- `medical_answer_quality.py`：替换 V1.8，增加请求维度过滤；
- `medical_entity_gate.py`：新增；
- `medical_retrieval_eval.py`：新增，仅评测；
- `retrieval_eval_seed.jsonl`：新增，仅评测题库，14 条均标记 `unreviewed`。

**不要替换、不重新导入** `knowledge_base.py`、`medical_evidence_bridge.py`、
`drug_graph_retriever.py`、`medical_json_bulk_index.py`、`config_data.py`、
`豆馅.py`、Chroma、Neo4j、SQLite 检查点及 `.env`。

若你已经自行修改过 V1.8 `rag.py`，请先用包内 `changes_v18_to_v19.patch`
合并差异，不要直接覆盖。

在 PowerShell 执行：

```powershell
python -m py_compile rag.py medical_entity_gate.py medical_answer_quality.py medical_retrieval_eval.py
streamlit run "豆馅.py"
```

**原有 Streamlit 进程需重启**，否则仍会使用缓存中的旧 `RagService`。

## 三、最重要的实际验证

在“知识库”模式问：

`阿托伐他汀钙片与动脉粥样硬化有什么关联？`

预期：
- 证据只应出现“阿托伐他汀钙片”的候选和原文章节；
- 不再出现“脂肪乳注射液”；
- 相同来源 URL 仍只展示一个合并的参考资料；
- 对“危险/风险条件”不能断言为直接治疗或已批准适应症。

对比问题：
`阿托伐他汀钙片与10%、20%、30%脂肪乳注射液的适应症有何不同？`
明确提到了两种药，应允许保留两种药的原文。

疾病-only 问题：
`有哪些药品资料提到动脉粥样硬化？`
不应因为没有明确药名而只留下阿托伐他汀钙片。

注意：两种药能否找到足够资料取决于实际入库覆盖范围，
不能仅根据“过滤规则通过”就认定医学结论正确。

## 四、下一阶段的评测基线（不需新 Embedding）

本包 `retrieval_eval_seed.jsonl` 只提供 14 个代表性检索问题：
有药品专名问题、栏目问题、两药比较和疾病-only 问题；
不包含经过医学人员审定的正确答案或 gold Section ID。
这只是**待标注的测试集合**，不能称为临床金标准。

在项目根目录运行（只使用你现有 BM25，不调用云端模型）：

```powershell
python medical_retrieval_eval.py --mode bm25 --top-k 5 --report medical_baseline_bm25.json
```

脚本将依次输出：
- 原始 Top 5 目标药品命中；
- 经过药品约束后的 Top 5 目标药品/栏目命中；
- 过滤前/后的“其他药品 Chunk 占比”；
- 每个问题检索耗时；
- 目标药品候选来源 Metadata（不导出原文正文）。

再尝试原来的 Hybrid+Reranker（**将文本发送给外部模型，可能产生费用**；确认许可再执行）：

```powershell
python medical_retrieval_eval.py --mode hybrid-rerank --allow-cloud-query --top-k 5 --report medical_baseline_hybrid.json
```

**准确解释指标：** 这是*药品/栏目匹配率*，不是医学答案准确率或标准 Recall@5。
如果某些条目实际未入库指定栏目，即使检索算法正常也可能为 0；需要人工标注
`gold_section_ids`、目标语义、是否有证据支持，才能评估真实检索召回。

本工具默认不写 Chroma/Neo4j，只在本地写报告 JSON。
Chroma 实际存在与 Collection 名称在调用 `KnowledgeBaseService` 前检查。

## 五、安全扩容：88 条 → 数百/数千条

你当前上传日志：Chroma `rag` **88 条**（旧 6 + 新 82），
SQLite 检查点记为 `done=18、no_chunks=2`。
本次只改检索，没有新增 Embedding 或图谱节点。

先预览选取上一轮尾部“一粒珠”之后的 50 种药品：

```powershell
python medical_json_bulk_index.py --json "data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --start-after "一粒珠" --limit-drugs 50 --verify-neo4j
```

如检查通过，确认第三方资料向云端模型发送的使用许可与费用，再明确受控写入：

```powershell
python medical_json_bulk_index.py --json "data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --start-after "一粒珠" --limit-drugs 50 --max-new-chunks 300 --max-embed-chars 160000 --embedding-batch 8 --pause-seconds 0.5 --apply --allow-cloud-embedding
```

命令使用之前 `medical_bulk_state.sqlite3` 检查点，不清除已有 88 条 Chunk；
具体新增多少要以运行日志为准。单次到达限制后会停止，属于预期保护行为。

最后检查：

```powershell
python medical_json_bulk_index.py --status
python diagnose_chroma.py
```

**建议每扩容几百条 Chunk，就重跑同一套评测题库对照**，
特别观察无关药品率与耗时是否随规模增加。
真正目标是有效证据覆盖率，不是盲目追求 Chunk 总数量。

## 六、后续里程碑

1. **近期开拓覆盖面：** 逐批扩展到约 300–500 种药品，保留快照、来源 ID、Section 偏移与受控成本；这只是建议阶段目标。
2. **正式构造评测集：** 由现有 14 条启发，再建立 50–100 条经人工标注的问答，明确 gold Section、语义类型、引用与拒答标准，评估真实 Recall@5 等。
3. **疾病语义可信层：** 从 ~19,685 条未审核 DiseaseCandidate 中抽样人工核查，区分治疗目标、预防、风险条件、合并症、不良反应、禁忌和背景提及。
4. **性能：** 当前一次 53 秒问答应记录分解、Reranker、Evidence Guard、Self-RAG、生成耗时；优先优化最慢环节，不要为了速度默默绕过医学证据核验。
5. **部署：** 数据与应用可复现、页面引用简洁、错误/超时可控、历史回答可回溯，完成端到端演示。

## 七、测试说明

```powershell
python -m unittest discover -s tests -p "test_medical_*.py" -v
```

离线测试包括你截图的“阿托伐他汀钙片 + 脂肪乳注射液”交叉污染模拟，
与 V1.8 原有流式回答、相同来源引用、候选医疗风险边界的回归。

**只在模拟环境完成了代码与集成路径测试。**
没有连接你本机的 Chroma、Neo4j、Qwen 或 Streamlit 实际生成回答。
