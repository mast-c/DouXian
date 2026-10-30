# 豆馅 Medical JSON-Aware Hybrid RAG · 试点版 V2

## 这版到底改变什么

你原始的数据是 `drug_db_v2_cleaned.json`，其中一个药品对象包含 `indication`（字符串）、`contraindications`（字符串列表）、`adverse_reactions`、`warnings` 等不同结构。

上一版脚本把 Neo4j Section 组装成 Markdown，再调用 `KnowledgeBaseService.upload_by_str()` 切分。这只是第一版的兼容方案。

本版：
1. **直接读 JSON**，字段就是一级语义单元，列表中的每个条目独立处理；短条目不做递归切碎。
2. 超过上限的长条目优先在完整句、换行和明显编号处拆分，窗口有重叠，且标记被迫切断的片段。医学语义无法凭标点完全保证，仍需原文复核。
3. 只读查询现有 Neo4j 的 `Drug → SourceRecord → Section`；要求药名、字段、来源 URL 和原文片段在恰好一个 Section 内**逐字符精确匹配**。
4. 只有匹配成功时，才把 `drug_id`、`record_id`、`section_id`、证据起止下标和原文 SHA256 写成 Chroma Metadata。
5. 写入时仅使用你原来的 `KnowledgeBaseService().chroma.add_texts()`，因此继续使用你配置的 DashScope Embedding、Chroma `rag` Collection。
6. **检索不改算法**：BM25 + 向量检索 + 0.5/0.5 RRF + `qwen3-vl-rerank` 还是你原有 `knowledge_base.py` 的函数。提供只读的回溯验证工具检验每条召回证据。

代码**不会改动**现有 `knowledge_base.py`、`config_data.py`、`rag.py`、`豆馅.py`、Neo4j 或 `.env`。试点完成后，再将父级原文验证逻辑接到你的 `rag.py` 回答生成链中；**这版只是数据层打通与独立验证**。

## 一、安装

把三个 Python 文件放进下面这个项目目录（与 `knowledge_base.py`、`config_data.py` 同级）：

`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

新文件：
- `medical_json_chunker.py`：纯 Python JSON 结构分块与原文精准定位
- `medical_json_index.py`：预览、Neo4j 只读核验、受控写入 Chroma
- `medical_hybrid_probe.py`：使用原来的 BM25/Hybrid/Rerank 检索后核验 Neo4j 原文章节

请勿覆盖已有的数据库文件或 `rag.py`。

确认使用你的原 Python 环境，例如：

```powershell
cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"
python -m pip install neo4j python-dotenv
```

项目现有 `knowledge_base.py` 必需的 LangChain、Chroma、DashScope 依赖仍按你原项目配置使用。

已上传的 `config_data.py` 是：

- `collection_name = "rag"`
- `persist_directory = "./chroma_db"`
- `chunk_size = 1000`
- `chunk_overlap = 200`
- `max_split_char_number = 120`

本版 **保留 1000/200 的单 Chunk 大小与重叠约束**；但不采用 120 字符的 Markdown 通用递归切分阈值，因为 JSON 字段/数组条目优先保持完整。`max_split_char_number=120` 仍对你其他通过 `upload_by_str()` 导入的普通文档生效，不必修改 `config_data.py`。

## 二、必须选对清洗版 JSON

使用 **11,953 个药品的清洗版** `drug_db_v2_cleaned.json`，不要混用未清洗的 12,043 名称原始快照或 `medical.json` 疾病词典。

根据你此前报告，文件可能位于：

`raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json`

请先确认文件实际存在；如果它放在其他目录，替换下面 `--json` 参数。注意 PowerShell 的当前目录需与 `config_data.py` 同级，以确保 `./chroma_db` 指向原库。

## 三、三步试点（阿托伐他汀钙片）

**第一步：仅检查 JSON 切分（完全无数据库写入）**

```powershell
python medical_json_index.py --json "raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --drug "阿托伐他汀钙片"
```

我已经在你之前提供的清洗版 JSON 上离线试过，默认五个字段得到：

| 字段 | Chunk 数 |
|---|---:|
| 适应症 `indication` | 1 |
| 禁忌 `contraindications` | 3 |
| 不良反应 `adverse_reactions` | 1 |
| 注意事项 `warnings` | 1 |
| 药物相互作用 `drug_interactions`（空） | 0 |
| **合计** | **6** |

这六个 Chunk 没有强制截断。**这个结果只是 JSON 切分预览，不代表实际 Neo4j 原文匹配或 Chroma 写入已完成。**

**第二步：Neo4j 精确校验（仍不写库）**

```powershell
python medical_json_index.py --json "raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --drug "阿托伐他汀钙片" --verify-neo4j
```

必须看见 `精确匹配：6/6` 才建议进入写入阶段。

如果结果是 0/6 或少于 6/6，**不应跳过校验强行导入**。常见原因有：
- Neo4j `SourceRecord.source_url` 和 JSON `source` 不一致；
- Neo4j `Section.text` 与当前 JSON 不是同一清洗快照；
- 旧图谱的 `Section.field` 名称不同；
- 某原文在多个来源记录重复出现，导致不能唯一关联。

可用 `检查Neo4j原文.cypher` 中的只读查询观察真实字段与 URL，必要时适配程序中的 Schema。错误输出通常只显示路径及匹配数量，不会让未匹配证据冒充已回溯证据。

**第三步：明确确认数据使用权限和 Embedding 费用后，才写入**

```powershell
python medical_json_index.py --json "raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --drug "阿托伐他汀钙片" --apply --allow-cloud-embedding
```

这一步会实际调用 DashScope 并写入 Chroma `rag` 集合，产生模型请求/可能产生费用。你应确认对第三方来源文本拥有相应处理和向外部模型提交的权限。没有 `--apply --allow-cloud-embedding` 就不会写入或调用云端 Embedding。

脚本有额外保护：
- 单次只处理一款指定药品，试点最多 100 Chunk；
- 全部证据需精确匹配，任意失败则整批拒绝写入；
- 固定 Chunk ID，重复运行会跳过相同内容，避免重复 Embedding；
- 发现同 ID 已有不同内容时拒绝覆盖；
- 绝不执行 Neo4j 写查询、Chroma delete、清库或重新生成已有药品图谱。

如果发生网络错误导致部分 Batch 已提交，可原参数再次执行，脚本会跳过已写入的相同内容；仍建议核对 Chroma 记录数。

## 四、检验你原来的 Hybrid RAG

第一步先单独测试 BM25（不会调用云端模型）：

```powershell
python medical_hybrid_probe.py --query "阿托伐他汀钙片与动脉粥样硬化有什么关联？" --mode bm25
```

第二步测试真正的向量 + BM25 + RRF + Qwen Reranker（会请求模型）：

```powershell
python medical_hybrid_probe.py --query "阿托伐他汀钙片与动脉粥样硬化有什么关联？" --mode hybrid-rerank --allow-cloud-query
```

这两个命令对命中的新版医疗 Chunk，都按其 `drug_id / record_id / section_id / quote_start / quote_end` 回查 Neo4j 父级原文，并验证 Section 全文 SHA256 和逐字片段一致性。

`验证通过` = **技术上的证据回溯成功**，并不意味着药品—疾病语义已获得临床核验。

然后停止并重启你当前的 Streamlit：

```powershell
python diagnose_chroma.py
streamlit run "豆馅.py"
```

如果之前诊断确实是 0，导入成功后 `rag` Collection 应当大于 0。已有 `rag.py` 的 Hybrid 路径会读取新 Chunk，但**当前问答链尚未自动使用 `section_id` 回读父级全文**；这个功能先由 `medical_hybrid_probe.py` 验证，待试点成功后再安全地接进真实问答流程。

## 五、已知限制

- 不用语言模型做“语义分段”，而是“JSON 字段边界优先 + 编号/句末边界 + 长文保守回退”。不能保证超长的自然语言条目从不跨句割裂，因此使用父级 Section 回溯。
- 如果 JSON 文本与 Neo4j Section 不是同一快照，严格校验会失败。这是数据质量保护，不应通过忽略匹配结果来处理。
- 原来的 `medical.json` 是疾病百科数据，**不是**此脚本的药品 JSON Schema，暂时不要拿它直接导入。
- 第三方来源只代表原文线索，不等于官方核准说明书，不会自动建立 `TREATS` 等已验证治疗关系。
- 原 Hybrid 检索仍按你 `knowledge_base.py` 的配置，未对 BM25/RRF 的融合权重做任何修改。
- 需要先确认网页来源的版权、使用许可和对外部 Embedding 服务的数据发送权限，不能仅凭网页公开访问就默认有商业使用权限。

## 六、离线测试

```powershell
python -m unittest discover -s tests -p "test_medical_json.py" -v
```

这次通过 10 项纯 Python 单元测试，并在你此前上传的 11,953 药品清洗 JSON 副本上跑通了 6 Chunk 只读预览。**没有**连接你本机 Neo4j、Chroma 或 DashScope 完成真实写入。

## 我希望你先回传的关键结果

请先运行**第二步**的 `--verify-neo4j`，将下面两项信息发来：

1. `图谱 Section 行数` 和 `精确匹配 X/6`；
2. 若不是 6/6，将 `不能自动入库` 的错误行发来。

在没有 6/6 的核验结果前，暂时不要执行 `--apply`。
