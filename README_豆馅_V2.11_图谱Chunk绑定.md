# 豆馅 V2.11｜本地图谱与 Chroma Chunk 精确绑定

## 项目阶段

本轮是 **独立离线准备阶段**，不是上线 GraphRAG。
不修改 `rag.py`、`knowledge_base.py`、前端、Qwen、Hybrid、Reranker、V2.10 的文本压缩策略。

本阶段输出 3 类文件：

- `graph_chroma_chunks_v211.jsonl`：从**你自己的真实 Chroma Collection** 导出的 Chunk 快照；
- `graph_claim_links_v211.json`：语义关系 -> 实际 `chunk_id` 绑定，以及不可绑定的原因；
- `graph_integrity_report_v211.json`：绑定率、失败情况汇总（不能把未绑定关系在线使用）。

**本 ZIP 不包含上述三个真实结果文件**，因为当前没有访问你 Windows 本机的 Chroma 数据库。
必须在你本机运行一次 `export -> link -> verify` 才能生成。不要把模拟数据误当生产数据。

## V2.11 的安全边界

- 图谱、教材、Chroma 输入原则上**只读**；新增文件只有导出的 JSONL、绑定 JSON、完整性报告。
- 绑定成功不仅需要 `source` 与 `section_path` 一致，而且需要**证据短句能完整、连续地出现在同一 Chunk 正文中**。
- 同一来源、同一章节、同一证据原文若出现在多条 Chunk 中，状态是 `ambiguous` 而非 `verified`。
- 证据在 Chunk 里但章节不同，状态是 `section_conflict`。
- 查不到完全一致的原文则是 `unbound`。包括 **跨 Chunk 的证据**，本版不冒险拼接。
- 目前仅绑定 V2.7 中的 **12 条明确语义关系**；736 条 `MENTIONED_IN` 和 510 条 `PART_OF` 不等于事实 Claim，暂不参与绑定。
- `verified` 是机器的**原文和出处匹配状态**，不代表关系语义经过专家人工审查。
- 源 Markdown 使用 SHA256 核验，Chroma 快照也有 SHA256，后续 `verify` 可以检测变动。
- 不能据此得出 GraphRAG 已改善回答；在线检索尚未改动。

## 包内文件

| 路径 | 作用 |
|---|---|
| `graph_chunk_linker_v211.py` | 离线 `export` / `link` / `verify` 命令行工具 |
| `test_graph_chunk_linker_v211.py` | 11 组不调用模型的离线测试 |
| `knowledge_graph_v27.json` | 现有离线 JSON 图谱（保持原版） |
| `graph_sources/神经网络与深度学习.md` | 和图谱 SHA256 完全匹配的教材快照；不会直接覆盖根目录同名教材 |
| `README_豆馅_V2.11_图谱Chunk绑定.md` | 本说明 |
| `graph_integrity_report_v211_示例.json` | 使用人工合成 Chroma 记录生成的**演示报告，不是你真实知识库统计** |

## 一、安装

先用 `Ctrl+C` 停止 Streamlit。进入你本机 `data` 目录，并激活你的环境：

```powershell
cd 'E:\VsCode项目\DockerTest_Agent\RAG项目\data'
conda activate rag

# ZIP 放到当前目录后解压（新增文件，不覆盖 rag.py）
Expand-Archive -LiteralPath .\豆馅_RAG_V2.11_图谱Chunk精确绑定.zip -DestinationPath . -Force

# 只跑本地测试，不收费
python .\test_graph_chunk_linker_v211.py
```

预期：

```text
V2.11: 11 offline tests passed (no Qwen, no Chroma writes).
```

## 二、确认现有 Chroma 持久化目录与 Collection 名

不同电脑实际路径不同；**不要使用猜测的路径**。

你的知识库服务使用 `config.persist_directory` 和 `config.collection_name`；
可以从项目的 `config_data.py` 或其它项目配置文件检查：

```powershell
Select-String -Path .\config_data.py -Pattern "persist_directory|collection_name"
```

如果没有对应文件，请搜索你项目中负责定义 `config` 的源文件。
下方将其分别写作 `<实际持久化目录>` 和 `<实际集合名>`。
如果你的 `persist_directory` 是相对路径，请先转换成实际文件夹路径。

## 三、导出真实 Chroma Chunk 快照

**建议先对 Chroma 数据目录做独立备份，并确保 Streamlit 已停止。**
工具只调用 `collection.count()`、`collection.get()`，不主动更新、增加或删除文档；
不过 `chromadb.PersistentClient` 启动数据库时存在其内部兼容/迁移的可能，
因此不能把该步骤承诺为文件系统级严格只读。

```powershell
python .\graph_chunk_linker_v211.py export `
  --persist-directory "<实际持久化目录>" `
  --collection "<实际集合名>" `
  --output .\graph_chroma_chunks_v211.jsonl
```

导出结果包括原始 `chunk_id`、`source`、`section_path` 和 `content`。
**不包含向量 Embedding、不调用 DashScope，不产生新的 Reranker Token。**

`graph_chroma_chunks_v211.jsonl` 是你的知识库原文副本，可能包含私人资料。
不要随意公开上传；分享给他人之前可以先检查是否有敏感内容。

## 四、生成图谱 Claim 与真实 Chunk 的绑定

```powershell
python .\graph_chunk_linker_v211.py link `
  --graph .\knowledge_graph_v27.json `
  --source .\graph_sources\神经网络与深度学习.md `
  --chunks .\graph_chroma_chunks_v211.jsonl `
  --links .\graph_claim_links_v211.json `
  --report .\graph_integrity_report_v211.json
```

**不会覆盖已有输出文件**，防止无意覆盖以前的绑定结果。
要重新执行，先把旧 JSON 重命名/移走，然后运行以上命令。

结果状态：

| 状态 | 含义 | 能否在未来上线作为已绑定 Claim |
|---|---|---|
| `verified` | 来源相同、完整证据短句命中、章节完全匹配、且唯一的 Chunk ID | 可以进入下一阶段**人工语义审核** |
| `ambiguous` | 同样证据对应多个 Chunk ID | 不可以，先处理去重或歧义 |
| `section_conflict` | 短句存在，但 Chroma 章节和图谱路径不一致 | 不可以，先核查切分和元数据 |
| `unbound` | 无对应证据 Chunk 或该证据跨 Chunk 被切断 | 不可以，需要人工查证 |

**特别注意**：有的教材 Chunk 可能截断到一句话的一部分，即使章节名称正确也会变成 `unbound`。
这恰好是需要暴露的知识库切分问题，不要自动填一个相邻 `chunk_id` 冒充验证通过。

## 五、验证生成的结果

```powershell
python .\graph_chunk_linker_v211.py verify `
  --graph .\knowledge_graph_v27.json `
  --source .\graph_sources\神经网络与深度学习.md `
  --chunks .\graph_chroma_chunks_v211.jsonl `
  --links .\graph_claim_links_v211.json
```

它会重新计算所有绑定，验证 JSON 内容和源文件、快照未发生变化。

先检查报告的 `status_counts`，再看每条非 `verified` 的 `reason`。

**给我后续分析最有用的是：** `graph_integrity_report_v211.json` 和控制台完整命令输出。
报告只含关系名称、状态和章节，不附原始大段 Chunk。
如果需要深入排查，再针对某个失败 Claim 提供对应少量原文。

## 六、对这份数据的下一步判断

V2.11 的优先任务是让真实 Claim 可追踪，不是强求绑定率达到 100%：
- 如果 `section_conflict` 多，先检查 Markdown 标题层级与 Chroma 元数据的差异；
- 如果 `unbound` 多，重点看证据短句是否跨 Chunk 被切断、原文 OCR/排版变化；
- 如果 `ambiguous` 多，检查重复 Chunk、重叠窗口、重传文件或真实一对多支持；
- 如果 `verified` 足够多且人工确认无误，再开始 V2.12 的语义关系扩展。

**什么时候引入 Neo4j？**
等 Claim 与 Chunk 绑定机制、实体规范化、关系审核、关系检索 A/B 都稳定后，
再迁移 `Claim / Concept / Chunk` 模型到 Neo4j。V2.11 不是 Neo4j 迁移。
