# 豆馅 Medical Hybrid GraphRAG V1.5
## 11,953 款药品 · 可恢复、限额、分批增量 Chroma 入库

这版承接已经通过的单药试点：

- `drug_db_v2_cleaned.json`：11,953 个清洗后药品；
- Neo4j：原药品节点、Section 和证据链已经存在，不需重建；
- Chroma `rag`：已成功写入“阿托伐他汀钙片”**6 条** JSON 结构 Chunk；
- 单药 BM25、向量 + BM25 + RRF + `qwen3-vl-rerank` 检索与 Neo4j 原文定位均已通过；
- 豆馅的 `rag.py` 已升级至前一轮 V1.4：可对已入库的新版 JSON Chunk 进行父级 Section 精确原文校验。

**V1.5 只新增数据层批量索引工具，不覆盖 `rag.py`、`豆馅.py`、`knowledge_base.py` 或已有 Chroma、Neo4j。**

### 基于你的真实完整 JSON 的离线规模统计

在五个默认字段 `indication / contraindications / adverse_reactions / warnings / drug_interactions` 上、按原 `chunk_size=1000` 和 `chunk_overlap=200` 统计：

| 项目 | 统计 |
|---|---:|
| 药品记录 | 11,953 |
| 计划 Chunk | **52,199** |
| 适应症 Chunk | 11,571 |
| 禁忌 Chunk | 13,550 |
| 不良反应 Chunk | 8,336 |
| 注意事项 Chunk | 11,198 |
| 药物相互作用 Chunk | 7,544 |
| 五个字段都没有有效文本的药品 | 356 |
| 非自然边界截断 Chunk | 1 |
| 字段类型错误 | 0 |

**这些是纯 JSON 分块预测，不是 Neo4j 全量关联通过数，也不是实际 Chroma 向量数。** 52,199 Chunk 总体大约涉及数百万字符，切勿直接全量发送云端。原始来源为第三方药品资料，正式批量提交前须核实数据处理及模型服务使用许可。

## 1. 安装（不会影响已有代码）

把 ZIP 中的两个脚本放到以下项目工作目录（与 `knowledge_base.py`、`config_data.py` 同级）：

`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

- `medical_json_bulk_index.py`：新的批量索引、断点续传、预算和失败记录入口。
- `medical_json_chunker.py`：**与已经验证通过的 JSON V2 试点完全相同**，已有版本可直接保留。
- `tests/test_medical_bulk.py`：不用真实 Neo4j/Chroma/DashScope 的离线测试。

先停止正在运行的 Streamlit，并备份 `chroma_db`。如果用的是 Windows PowerShell：

```powershell
cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"
python -m py_compile medical_json_chunker.py medical_json_bulk_index.py
```

`--json` 使用你上轮已验证过的路径：

`data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json`

若工作目录不同，需以该目录为基准调整相对路径。**不要修改** `config_data.py` 的 `collection_name="rag"`、Embedding 模型或已有 Chroma 切分设置。

## 2. 先做不花费模型调用的预览

**只看前 20 个尚未入库的药品：**

```powershell
python medical_json_bulk_index.py --json "data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --limit-drugs 20
```

不会连接 Neo4j 或 Chroma，也不会调用 DashScope；只显示 JSON 生成 Chunk 的计划数量。此步不创建检查点。

**再做 20 个药品的 Neo4j 精确回溯校验：**

```powershell
python medical_json_bulk_index.py --json "data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --limit-drugs 20 --verify-neo4j
```

这个命令只读查询 Neo4j；每条 Chunk 必须在指定字段、同源 URL 下**唯一匹配**到一个 Section，才能被未来写入。来源缺失、同段重复、快照不一致均会报告并跳过。

建议先检查输出中 `failed_drugs` 和 `[SKIP]`，尤其是数据不匹配的药品，不要忽略校验结果直接强制写入。

## 3. 先确认已有六条可复用（可选，但推荐）

先确认原先已导入的六条阿托伐他汀钙片数据仍在 Chroma，并确认沿用**同一个** `DASHSCOPE_EMBEDDING_MODEL`。

当且仅当你确认有权将涉及的第三方资料提交给 DashScope 并接受相应模型调用费用时，运行：

```powershell
python medical_json_bulk_index.py --json "data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --drug "阿托伐他汀钙片" --apply --allow-cloud-embedding
```

本脚本使用**与前一轮 JSON V2 完全一致的 Chunk ID 生成规则**。如果原始内容、来源 ID、快照、元数据、Embedding 配置保持一致，预计：

- `新增 Chunk=0`
- `复用 Chunk=6`
- 当前 Chroma 记录数仍为 6
- 不对这六条重复执行 Embedding。

如果旧记录元数据已改变，则会标记 `chroma_conflict`，**不会强制覆盖**。

## 4. 正式运行小批量（需要用户明确授权）

推荐先以 20 种药品为一批，同时限制单次实际新向量数量与字符量：

```powershell
python medical_json_bulk_index.py --json "data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --limit-drugs 20 --max-new-chunks 150 --max-embed-chars 150000 --embedding-batch 8 --pause-seconds 0.5 --apply --allow-cloud-embedding
```

命令实际调用 DashScope、产生可能的 Embedding 费用，**不会自动扩大到全部 11,953 款药品**。

每个药品依次执行：

1. JSON 结构切分：字符串字段尽可能保持完整、数组各条单独 Chunk；
2. Neo4j 只读核验：确认来源 URL、Section、原文以及精确字符位置；
3. Chroma 增量预检：如果相同 ID、正文、关键 Metadata 已存在则复用；如有冲突直接拒绝；
4. 若累计新 Chunk / 字符即将超过本次预算，**在开始下一款药品前停止**；
5. 默认拦截在非自然边界强制截断的医学片段（全量预览统计中仅发现 1 个），标记供人工复核；必要时只有人工审查后才可明确使用 `--allow-forced-cuts`，且失败记录需 `--retry-failed` 才会重新尝试；
6. 最多 8 个 Chunk 一批发送给配置的 Embedding 模型；
7. 仅在该药品的所有 Batch 成功后，标记 `done`；
8. 每款药品的完成状态立刻持久保存到 `medical_bulk_state.sqlite3`。

**额度说明**：`--max-new-chunks` 和 `--max-embed-chars` 是每次运行的新增 Embedding 上限，不是人民币费用的上限，也不是累计项目预算。实际 Token 和计费应看 DashScope 账单。已命中的原文不重复计算向量。

## 5. 断点续跑：再次执行相同命令

同一清洗快照、同一工作目录、相同字段与模型配置下，重复执行第 4 步命令即可：

- 已完成药品自动跳过；
- 全部字段为空的药品记作 `no_chunks` 并跳过；
- 先前来源不匹配的药品会留在失败记录中，不会偷偷绕过校验；
- 如果某药品的写入途中断开，数据库检查点不会误标成完成；已有部分 Chunk 仍可在重试时复用；
- 如果更换 JSON、切分参数、Neo4j 目标、Chroma 目录或 Embedding 模型，程序会拒绝跨快照复用检查点。

**查看累计状态（只读，不访问数据库）：**

```powershell
python medical_json_bulk_index.py --status
```

检查点会汇总 `done / no_chunks / alignment_failed / data_error / chroma_conflict / write_failed` 及失败示例。

出现过 `write_failed` 时，排查网络或 API 配置后，**显式允许重试失败的药品**：

```powershell
python medical_json_bulk_index.py --json "data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --limit-drugs 20 --retry-failed --apply --allow-cloud-embedding
```

即使发生了部分成功提交的 Chunk，重试时会先查 Chroma，复用匹配的原有 Chunk，只补交尚未存在的内容。

特别注意：默认只处理与你此前清洗快照相符的 **11,953** 款药品；误将原始未清洗 JSON 用于 `--apply` 会被拒绝。若日志提示“强制截断”，先核实那条原文的医学语义，不建议直接启用 `--allow-forced-cuts`。

如果数据版本、字段或模型真的需要变更，**不要随意删除旧检查点、覆写原 Chroma**。需先确定是否需要新 Collection 和新索引版本，再有计划地迁移；盲目改 `--state` 虽能创建新检查点，但不能解决 Chroma 既有记录的快照冲突。

## 6. 检查 Chroma / Hybrid 检索

每完成一小批，运行你之前的工具：

```powershell
python diagnose_chroma.py
```

然后验证 BM25（本身不会调用云端检索模型）：

```powershell
python medical_hybrid_probe.py --query "阿托伐他汀钙片与动脉粥样硬化有什么关联？" --mode bm25
```

再按需要执行完整 Hybrid + Qwen Reranker（会调用云端）：

```powershell
python medical_hybrid_probe.py --query "阿托伐他汀钙片与动脉粥样硬化有什么关联？" --mode hybrid-rerank --allow-cloud-query
```

这些工具之前已经安装过的话**不需要从这份包重新安装**。原文回溯的技术验证并不代表医疗专家审核或批准适应症认证。

批量完成后再启动 Streamlit：

```powershell
streamlit run "豆馅.py"
```

原 `rag.py` V1.4 会自动读取同一 `rag` Collection，保留 Neo4j + Chroma 联合检索与父级 Section 核验。

## 7. 扩容时要注意的两个技术瓶颈

- 当前 `KnowledgeBaseService.build_bm25_index()` 把 Chroma 中的所有文档 Chunk 加载到内存后建立 BM25。如果数据规模接近 **52,199 Chunk**，Streamlit 启动时间、内存占用和重建成本可能明显增加；请在逐批扩容时监控内存和时长。后期适合把 BM25 改成持久化倒排索引或分区检索，不能只凭这次的六条试点推断全量性能。
- 原始 JSON 中不同药品、字段、记录可能重复；当前严格关联可能拒绝重复匹配或来源不一致的药品。**宁可拒绝自动入库，也不把错误的 Section ID 写进 Chroma。** 失败原因需人工分析后再完善规则。

本轮脚本没有任何 Neo4j 写操作、不删除 Chroma 记录，不修改原图谱药品、疾病候选，也不会将未审核候选升级成确定 `TREATS` 关系。

## 8. 离线测试范围

```powershell
python -m unittest discover -s tests -p "test_medical_bulk.py" -v
```

测试覆盖：已有六条复用、断点恢复、部分写入失败、来源不匹配跳过、同源冲突拒绝、预算边界、字段为空、快照变化和没有云端权限时不允许写入等情形。

**本版只完成了本地离线测试和 JSON 规模分析，没有连接你本机真实 Neo4j / Chroma / DashScope 执行批量入库。** 上一轮的单药六条试点是你本机已实际完成的工程验证。

## 最短操作路线

① 下载并解压；② 先运行 `--verify-neo4j --limit-drugs 20`；③ 再按需 `--apply --allow-cloud-embedding` 写入 20 款药品；④ 用 `--status` 查看断点；⑤ 重复第③步并监控报错、费用和内存。

不要立即全量导入；可以先执行 20、再 50、最后在资源可控的情况下逐步扩大。
