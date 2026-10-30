# 豆馅医疗 Hybrid GraphRAG V1.4 · 正式问答接入父级 Section 原文回溯

## 为什么需要这版

用户已完成真实本地试验：
- Chroma `rag` 现有 6 条 `medical_json_v2` Chunk；
- BM25 返回 3 条，Hybrid + Reranker 返回 3 条；
- 命中的三条新版 JSON Chunk 都能严格回溯到 Neo4j 的 `Section.text`；
- 但 Hybrid Top 3 中的第 2、3 条是禁忌和注意事项，与“阿托伐他汀钙片与动脉粥样硬化有什么关联”不直接相关。

因此现在不需要重新导入 JSON，也不需要修改 `knowledge_base.py` 的切分、BM25 / RRF 权重或 Qwen Reranker。

这版让已有 `rag.py` 的**知识库模式**直接使用已入库的医疗 JSON Chunk，并进行：
1. 用户意图 → 医疗章节字段筛选（多目标问题可同时保留多个字段）；
2. Neo4j `Drug → SourceRecord → Section` 精确回溯：
   `drug_id + record_id + section_id + quote_start + quote_end + section_sha256 + source_url`；
3. 只把已核验的 JSON 原文交给 Evidence Guard 和生成模型，不把父级 Section 中可能混杂的其他字段全文灌入；
4. 原始 Neo4j `DiseaseCandidate` 仍保留“未经医学核验候选”的独立意义；
5. 若 Chroma 命中已经核验的同一 Section，去掉重复的 Neo4j 原文章节展示；
6. Self-RAG 补检也走同样的字段选择和原文校验流程；
7. Streamlit 折叠详情展示：Neo4j 命中、Chroma 命中、精确核验数量、相关栏目过滤以及拒绝的原因。

**特别注意：** 原文字符和 Section 的一致性是技术校验，绝不等于医疗临床、说明书监管审批或独立来源核验。
即使候选和 JSON Chunk 有两个编号，如果 `source_url` 相同，也不是两份独立医学证据。

## 安装

本包基于之前发出的 **V1.3 统一医疗知识库版**。如你已经修改过对应文件，请先备份并按 `changes_v13_to_v14.patch` 合并，不要直接覆盖自定义代码。

1. 关闭 Streamlit，备份当前 `rag.py`、`drug_graph_retriever.py`、`豆馅.py`。
2. 将以下四份文件复制到 `E:\VsCode_Project\DockerTest_Agent\RAG_project\data`，与 `knowledge_base.py` 同级：
   - `rag.py`（覆盖之前 V1.3）
   - `drug_graph_retriever.py`（覆盖之前 V1.3）
   - `豆馅.py`（覆盖之前 V1.3）
   - `medical_evidence_bridge.py`（新增）
3. **保持原样**：`knowledge_base.py`、`config_data.py`、`graph_answer_view.py`、`chroma_db`、Neo4j、原 JSON 和 `.env`。
4. 从项目工作目录运行：

```powershell
cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"
python -m py_compile rag.py drug_graph_retriever.py medical_evidence_bridge.py "豆馅.py"
streamlit run "豆馅.py"
```

只更新代码，不需要再次执行 `medical_json_index.py --apply`，你现有的 6 条 Chroma Chunk 会直接复用。

## 问答演示

在唯一的“知识库”模式提问：

`阿托伐他汀钙片与动脉粥样硬化有什么关联？请给出原文证据。`

现在逻辑上会：
- Neo4j 检索候选关系及适应症 Section；
- Chroma 使用原来的 **向量 Top 20 + BM25 Top 20 + RRF 0.5/0.5 + Qwen Reranker**；
- 先从 Reranker 返回最多 8 个候选，再按原始问题保留相关医疗栏目（此问题优先 `indication`）；
- 对选择的 Chroma 医疗 Chunk 逐字校验 Neo4j 原文，不符则拒绝证据；
- 合并去重后送至 Evidence Guard 和 Qwen；结论必须遵循“风险条件≠确定治疗适应症”。

当前六条数据如果保持不变，期待在终端看到：

```text
[医疗证据门控] 最终=1 ... Neo4j 精确回溯通过=1 回溯不通过=0 ...
```

具体栏目过滤条数以模型返回的候选为准，不固定为 2 或 5。
在页面折叠的检索详情里应能看到“原文回溯：Neo4j Section 精确匹配 ✓（不代表临床审核）”。

## 实施细节与行为边界

- **不会**修改 Neo4j、Chroma 数据或调用新的 Embedding 入库。检索阶段的 Qwen Embedding / Rerank / LLM 仍按原架构使用，可能产生模型调用费用。
- 跨字段问答如“禁忌与不良反应是什么”同时选中对应两类字段；泛化问题若未明确指向任何栏目，则不进行栏目硬过滤。
- 栏目判断只是保守规则，依赖问题中的明显词语，不保证适合所有复杂医疗问题；证据充分性仍取决于 Evidence Guard、原文内容和最终医学审核。
- 对新 `data_origin=json_matched_to_neo4j_section` 的 Chunk，原文回溯失败时**绝不作为已核验证据使用**；其他旧版 Chroma 文档为了向后兼容仍可按旧机制检索，但不会被标注为“回溯已验证”。
- Neo4j `Section.text` 有可能比当前 JSON 字段更长，甚至包含相邻字段内容：为了避免“章节污染”，回答用的仍是精确定位的 JSON Chunk，不直接把整个父级原文当成当前证据。
- 自行修改了 `.env` 或数据库结构时，`fetch_parent_section` 使用的 ID 与关系路径可能需要适配，不能绕过验证。

## 离线验收

```powershell
python -m unittest discover -s tests -p "test_medical_parent_integration.py" -v
```

完成 13 项离线测试，覆盖字段路由、禁止伪关联、来源 URL、SHA256、字符坐标、图谱不可用时的安全拒绝、去重、旧库兼容及两轮检索接入。
另外用假数据驱动了 `rag.py` 的实际 `_stream_agent` 知识库分支，确认 1 条 DiseaseCandidate + 1 条准确来源的 Chroma 原文能够正确合并与引用。

**没有**真实连接你的本地 Streamlit、Neo4j、Chroma 和 Qwen 进行本版 V1.4 的端到端测试。用户上一次真实试验验证了旧的独立 Hybrid Probe；本次是将相同的验证规则加入正式 `rag.py`。

## 之后怎样扩展

下一步可以基于已经验证的 JSON 入库脚本开发可恢复、批量增量的医疗 Chroma 索引，保留精确原文回溯和来源使用限制。建议先在一个小批次里检查不同字段及有数组、长文本的药品，而不是立刻把 11,953 款药品全部向量化。
