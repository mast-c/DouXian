# 清洗后的 v2 数据同步到 Neo4j

**适用你的现有 RAG_project**。本补丁只增加：

```text
RAG_project/
  kg/
    sync_clean_v2.py      # 新增
```

你的原有代码 `kg/build.py`、`kg/neo4j_import.py`、`.env`、
`data/chroma_db/` 全部保留，不用替换。

## 为什么不直接重复导入？

原版 `kg/build.py` 使用药名生成 `Drug.id`，记录内容生成
`SourceRecord.id`。改名或清洗原始字段后，可能产生新的 ID。
原有 `kg.neo4j_import` 采用 `MERGE`，只能新增/更新，
不会自动删除旧的 SourceRecord / Section / Chunk，导致重复。

本脚本使用 **先全量构建新文件 -> 只读对比 -> 新节点导入 -> 有范围地移除旧节点**。
不会运行整库删除命令，不会修改已存在的 ChromaDB。

## 第 1 步：解压

将这个补丁 ZIP 解压到 `RAG_project` 根目录，
确认目录是：

```text
RAG_project/kg/sync_clean_v2.py
```

## 第 2 步：先备份 Neo4j

在当前 Neo4j 实例的管理工具中保存数据库备份或快照。
若使用 Docker，请备份对应 Neo4j 数据卷；若使用 Neo4j Desktop，
先保留可恢复的数据库副本。

⚠️ 请勿直接运行 `MATCH (n) DETACH DELETE n`，那会删除整个数据库。

## 第 3 步：只读预览差异（建议先运行）

在 VS Code 中进入 `RAG_project` 根目录，
确认 `.env` 有：

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你的数据库密码
```

执行（Windows PowerShell 也可使用同样的单行命令）：

```bash
python -m kg.sync_clean_v2 --input "data/data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2.json" --old-dir "data/kg_output" --out "data/kg_output_clean_v2"
```

脚本只会重新生成 `data/kg_output_clean_v2/` JSONL 并读取 Neo4j；
不会写入数据库。

会看到类似以下统计字段（实际以输出为准）：

```json
{
  "new_drug_count": 12043,
  "db_old_source_count": 12043,
  "new_source_count": 12043,
  "sources_to_add": 1,
  "sources_to_remove": 1,
  "candidate_ids_to_remove": 0,
  "reviewed_claims_blocking_sync": 0
}
```

这是**格式示意**，数字不代表你当前修改后的实际差异！
报告会保存在：

`data/kg_output_clean_v2/sync_plan.json`

如果 `reviewed_claims_blocking_sync > 0`，需要先处理旧证据引用。
如果新记录数量大幅减少，请检查是否误用 300 条抽样文件。

## 第 4 步：真正同步

确认报告正常并备份数据库后，运行：

```bash
python -m kg.sync_clean_v2 --input "data/data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2.json" --old-dir "data/kg_output" --out "data/kg_output_clean_v2" --apply --backup-confirmed
```

脚本将：
1. 通过旧 `kg.neo4j_import.import_kg` MERGE 导入全部新数据；
2. 删除从当前候选相互作用关系指向错误药品/证据的过期边；
3. 根据旧版 `interaction_candidates.jsonl` 清理过期候选；
4. 仅删除属于指定数据集的过期 `SourceRecord` 及其 Section/Chunk；
5. 仅当旧 Drug 节点完全无关系时才删除它；
6. 只清理此前导出且已经没有边的旧 NameMention。

如果旧 `data/kg_output/interaction_candidates.jsonl` 丢失，
无法保证完整清理旧候选相互作用；此时脚本会发出警告，
但仍可安全对齐 SourceRecord 文档记录。

**警告**：此操作不是一个全数据库单事务。
如果导入或删除途中中断，可以检查 `sync_result.json`
并在已备份的前提下再次运行。重复导入采用 MERGE。

## 第 5 步：Neo4j Browser 验证

检查之前残缺名称是否被清除：

```cypher
MATCH (d:Drug)
WHERE d.name IN ['（胶囊','(胶囊']
RETURN d.name AS name, d.id AS id;
```

如果旧 Drug 节点还有其他业务关系，脚本会保留以避免误删，
需要进一步人工检查。

确认当前来源记录数量：

```cypher
MATCH (r:SourceRecord)
WHERE r.dataset='MedGraphRAG-Drug-DB drug_db_v2.json'
RETURN count(r) AS source_records;
```

确认一条完整文档路径：

```cypher
MATCH p=(d:Drug)-[:HAS_RECORD]->(r:SourceRecord)
        -[:HAS_SECTION]->(s:Section {field:'indication'})
        -[:HAS_CHUNK]->(c:Chunk)
RETURN p LIMIT 3;
```

查看修正后药品名称：

```cypher
MATCH (d:Drug)
RETURN d.name AS drug_name ORDER BY drug_name LIMIT 20;
```

## 第 6 步：后续抽取/检索路径更新

新的 JSONL 文件位于 `data/kg_output_clean_v2/`。
第二阶段疾病抽取时使用：

```bash
python -m kg.indication_candidates extract --medical "data/data/raw/drug_db/MedGraphRAG-Drug-DB/medical.json" --in-dir "data/kg_output_clean_v2" --out "data/kg_output_clean_v2/disease_candidates.csv" --limit-drugs 300
```

Neo4j 与你原有 ChromaDB 是两个存储系统：
**本次仅同步 Neo4j，ChromaDB 不会自动同步。**
如之前已向 Chroma 写入旧版说明书向量，需要另外重新建立对应 collection。

## 检查与限制

- `third_party_unverified` 是 SourceRecord 的数据来源状态，不是实体抽取错误。
- 本补丁不进行医学实体识别、不改变临床事实的核验状态。
- 如果之前建立了与旧说明书证据相连的 Claim，脚本会拒绝破坏其引用。
- 为保证可复现，清洗后的 JSON、旧 manifest、新输出和 Neo4j 备份都请保留。
- 仅更新来源标记为 `MedGraphRAG-Drug-DB drug_db_v2.json` 的文档子图。
- 如果你的项目后来接入其他自定义图谱关系，先在备份或测试库验证同步影响。

```
python -m unittest discover -s tests -v
```

这是随补丁附带的离线测试命令（针对测试目录，不连接真实 Neo4j）。
