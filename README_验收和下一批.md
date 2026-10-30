# 豆馅医疗知识库 V1.7：88 条 Chunk 只读验收与下一批建议

## 已确认的实际进展

根据你本地运行的 V1.5 批量索引输出：

- 选择药品 20 个，18 个药品成功入库，2 个没有有效原文；
- 计划与新增的 82 条 JSON Chunk 均通过 Neo4j 对照，写入失败 0；
- 82 条没有发生非自然边界截断，本次发送 Embedding 文本字符 11,416；
- Chroma `rag` 由 6 增至 **88 条**；
- 检查点 `medical_bulk_state.sqlite3` 记录 `done=18 / no_chunks=2`。

这些是导入和原文对齐成功的证据，**不自动意味着 Hybrid 检索已经按预期找到全部新药品，更不是医学来源权威审核**。

## 为什么这次不再给一份全新的 rag.py

已有的 V1.4 `rag.py` + `medical_evidence_bridge.py` + `drug_graph_retriever.py` 已经实现：

1. Chroma `data_origin=json_matched_to_neo4j_section` Chunk 识别；
2. 根据字段路由筛选适应症、禁忌、不良反应等相关栏目；
3. 利用 `drug_id + record_id + section_id` 只读回查 Neo4j；
4. 核对 `section_sha256`、`quote_start`、`quote_end` 和完整原文；
5. 证据复核和与 Neo4j 原文章节去重，避免误作独立来源。

目前无需重写 `knowledge_base.py` 或重新导入这 20 种药品。如果本地 `rag.py` 不是 V1.4，请先按旧版 V1.4 的安装说明升级；不要用 V1.3 覆盖 V1.4 的改动。

## 步骤 1：确认正式问答已有父级 Section 检索

在项目目录运行（只读）：

```powershell
Select-String -Path .\rag.py -Pattern "medical_evidence_bridge"
```

- 有匹配：说明已有 V1.4 的关键导入语句；但这本身不等于已经完成实际连接测试。
- 没有：说明尚未接入 V1.4，不能以为 Chroma 命中后已经做了父级原文核验。

## 步骤 2：检查全部 Chroma 元数据，并抽样核验 Neo4j

把本包 `medical_chroma_acceptance.py` 复制到 `rag.py`、`config_data.py` 同级目录。
不需要修改已有任何脚本或数据。确保 V1.4 的 `medical_evidence_bridge.py` 和 `drug_graph_retriever.py` 已经在同级目录。

停止正在运行的批量入库进程后执行：

```powershell
python medical_chroma_acceptance.py --expect-count 88
```

它只读 Chroma，扫描全部 88 条，输出：
- 新版 JSON Chunk / 旧版 Chunk 数量；
- `chunk_id`、正文 SHA256、父级 SHA256、来源 URL、原文坐标和字段元数据一致性；
- 不同药品和 Section 数量、各字段数量、强制截断计数。

然后使用现有 Neo4j 只读查源（默认不抽样，明确指定后才查询）：

```powershell
python medical_chroma_acceptance.py --expect-count 88 --sample-parent 12
```

这会对最多 12 条来自不同药品/Section 的 JSON Chunk 进行 Neo4j 精确回查；
输出 `[PARENT] 药品名 / 字段 => 精确匹配 (text_verified)`。

它不会发出 Embedding、Reranker 或 LLM 调用，不会修改图谱或 Chroma 内容；
但读取 Chroma 需要原项目已安装的 `chromadb`，抽检还需 `neo4j`、V1.4 模块和已有 `.env`。

**验收尺度：** 88 是数据库总数，不等于这 88 条都一定属于本次 JSON 管线；
此前 6 条若也由同一脚本写入，预期 `json_structured_chunks = 88`。若有旧 Markdown 文档，
允许出现 `legacy_or_other`，此时需要检查真实 Metadata，不能强求 88 条都是新版。

## 步骤 3：使用原有 BM25 / Hybrid 检索验证新入库的药品

先试非云端 BM25：

```powershell
python medical_hybrid_probe.py --query "5-氨基水杨酸肠溶片的适应症是什么？" --mode bm25 --top 5
```

再试实际 Hybrid + Qwen Reranker（调用云端 API、可能产生费用）：

```powershell
python medical_hybrid_probe.py --query "5-氨基水杨酸肠溶片的适应症是什么？" --mode hybrid-rerank --allow-cloud-query --top 5
```

应观察：
- 检索结果是否包含该药**适应症**；
- 命中的 `data_origin=json_matched_to_neo4j_section` 证据能否回溯；
- Top K 是否混入其他药品或禁忌栏目，必要时应由 V1.4 的医疗字段路由过滤。

参考问题也可以问“阿托伐他汀钙片与动脉粥样硬化有什么关联？”，该药原有 6 条 Chunk
用于检测旧数据是否受到本批新增 82 条数据的影响。

## 步骤 4：启动正式问答

关闭旧的 Streamlit 进程，再执行：

```powershell
streamlit run "豆馅.py"
```

选唯一的“知识库”模式，尝试：

- `5-氨基水杨酸肠溶片的适应症是什么？请给出原文依据。`
- `阿托伐他汀钙片与动脉粥样硬化有什么关联？请给出原文证据。`
- `5-氨基水杨酸肠溶片有哪些禁忌？`

检索详情应能显示 `Neo4j N 条 / Chroma M 条`；在 V1.4 的详细证据中还应能看到
“Neo4j Section 精确匹配”，并且未审核疾病候选不得当作直接治疗推荐。

## 步骤 5：通过验收后，继续下一批

V1.5 批量索引会根据 `medical_bulk_state.sqlite3` 的状态跳过 `done` 和 `no_chunks`，
**重复同一个 20 药品命令会选接下来的 20 个尚未处理药品，而不是重算先前 82 条**（前提是使用同一 JSON、
参数、checkpoint 和 Chroma Collection）。

先验证新一批：

```powershell
python medical_json_bulk_index.py --json "data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --limit-drugs 20 --verify-neo4j
```

注意单独 `--verify-neo4j` 不读取写入检查点；若想确保对新的尚未处理药品进行预览，
可使用 `--start-after` 指定上轮排序末尾药品 `"一粒珠"`，或者直接查看写入模式中的 `[PLAN]`。
不要把只是重复前 20 个已验证药品误认为下一批新药品。

**经确认的写入**（会产生 DashScope 费用，并把第三方来源资料送至外部 Embedding 模型）：

```powershell
python medical_json_bulk_index.py --json "data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --limit-drugs 20 --max-new-chunks 150 --max-embed-chars 150000 --apply --allow-cloud-embedding
```

若要预览“接下来的一批”而不触及检查点，可加 `--start-after "一粒珠"`，注意该选项依据药名字典排序；
下一批写入仍应使用相同完整路径、快照和原有检查点。

## SQLite BM25 (V1.6) 的时机

现在只有 88 条 Chunk，保留原内存 BM25 就足够了。若后续扩容到数千或数万条，
再考虑启用可选 SQLite FTS5 BM25 索引。新旧两种 BM25 有评分/排序差异，
不是“保持原算法完全不变”的替换；先比较 Top K，再决定是否启用。

## 医学安全边界

技术上的 JSON→Neo4j 原文精确匹配，**仅证明文本和来源位置一致**，不代表独立临床验证。
DiseaseCandidate 仍属于未经人审的疾病提及候选。多个证据编号如果都源自同一
`source_url`，也不能当作互相独立的医学来源。
