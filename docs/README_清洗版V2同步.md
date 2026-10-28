# 清洗版 drug_db_v2_cleaned.json → Neo4j 安全同步

**先读完这个说明。** 这是对上一版同步补丁的升级，专门处理你上传的
`drug_db_v2_cleaned.json` 中新增的别名和合并来源记录。

此补丁只新增：
```
RAG_project/
  kg/
    prepare_cleaned_v2.py
    import_cleaned_aliases.py
    sync_cleaned_v2.py
```
不替换原有 `kg/build.py`、`kg/neo4j_import.py`、`.env`，不接触 `data/chroma_db/`。

## 清洗文件结构与统计（本次文件已实际验证）

- 输入 SHA-256：`1850511ebf1a243be1bd6360d3606c71d7d8326b5de98700c50d72f5c2196937`
- `drugs`: 11,953 个主记录（原为 12,043）
- `interactions`: 13,275 条候选（原为 13,316）
- `drug_name_redirects`: 483 条，包含一些残缺旧名；**仅用于审计**
- `drug_aliases`: 339 组，合计 343 个候选别名节点
- `drug_original_names`: 408 组，保留为来源审计
- `additional_source_records`: 7 份，补建为 SourceRecord，不能丢弃
- `name_cleaning_version`: `1.0`

全量重建产出的图谱预期：
- Drug 11,953
- SourceRecord 11,960（含额外 7）
- Section 106,445
- Chunk 107,787
- NameMention 7,949
- InteractionCandidate 13,275
- DrugAlias 343（`review_status=candidate`）

上传的这份清洗文件与旧版数据的**离线**节点 ID 对比（不等于你的本地 DB 统计）：
- Drug：新增 400 / 旧版移除 490
- SourceRecord：新增 407 / 旧版移除 490
- InteractionCandidate：新增 972 / 旧版移除 1,013
- Chunk：新增 2,809 / 旧版移除 3,349

### 0. 复制文件并备份

将已经清洗的 `drug_db_v2_cleaned.json` 放到：

`RAG_project/data/data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2_cleaned.json`

旧的 `drug_db_v2.json` **保留不动**，旧的 `data/kg_output/` **保留不动**。
在 Neo4j 上执行写操作前，先完成可恢复的数据库备份 / 快照。
如数据库中已有人工审核的疾病关系，本程序会在发现旧证据引用时拦截清理。

### 1. 在 VS Code 的 RAG_project 根目录运行只读预览

如之前可以通过 `kg.neo4j_import` 连接 Neo4j，沿用相同 `.env`。

在 Windows PowerShell/终端执行这一整行：

```bash
python -m kg.sync_cleaned_v2 --input "data/data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2_cleaned.json" --old-dir "data/kg_output" --out "data/kg_output_clean_v2"
```

默认只会：
- 读取本地清洗文件并生成全部 JSONL；
- 将数据库旧节点与新 ID 对比；
- 写出 `data/kg_output_clean_v2/sync_plan.json`；
- **不会写入 Neo4j**。

应当查看以下字段：
`new_drug_count`、`new_source_count`、`sources_to_add`、
`sources_to_remove`、`candidate_ids_to_remove`、
`reviewed_claims_blocking_sync` 和 `cleaned_snapshot`。

如果以前已经完整导入 12,043 条原始数据，
且没有其它独立修改，`sources_to_add` 约 407、`sources_to_remove` 约 490。
如果不是，请检查 `sync_plan.json`；不要硬套示例数字。

### 2. 备份确认后应用到 Neo4j

必须先备份数据库，并确认上一步预览合理。

```bash
python -m kg.sync_cleaned_v2 --input "data/data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2_cleaned.json" --old-dir "data/kg_output" --out "data/kg_output_clean_v2" --apply --backup-confirmed
```

脚本将分批：
1. 导入新 Drug/SourceRecord/Section/Chunk/InteractionCandidate；
2. 导入 `DrugAlias -[:ALIAS_CANDIDATE_OF]-> Drug`（不是已核验同义词）；
3. 删除本数据集过期的相互作用候选及其连接；
4. 删除本数据集过期的 SourceRecord 及下属 Section/Chunk；
5. 仅删除没有任何边的旧 Drug 节点；
6. 删除此前 manifest 中已不需要、且没有边的旧 NameMention。

删除范围限制在原项目标记和原 manifest 的 ID 集合；不会执行全库清空。
因为是多批事务，不是一次全局原子更新，出错时应结合备份、差异报告
和 `sync_result.json` 排查，不要无判断地反复执行。

如果检测到旧版已审核 Claim 关联了拟删除的 Section，
脚本会阻止应用，需要人工先处理这些知识证据。
如果找不到旧版 `data/kg_output/interaction_candidates.jsonl`，
脚本也会拒绝删除，以免误留过期候选关系。

### 3. Neo4j Browser 验证同步

**检查残缺药名是否存在：**
```cypher
MATCH (d:Drug)
WHERE d.name IN ['（胶囊','(胶囊']
RETURN d.name AS drug_name, d.id AS drug_id;
```
预期：0 行。如果仍有旧节点，先检查它是否连接了其他业务节点，
不要直接全库删除。

**统计来源文档数量：**
```cypher
MATCH (r:SourceRecord)
WHERE r.dataset='MedGraphRAG-Drug-DB drug_db_v2.json'
RETURN count(r) AS source_records;
```
预期：11,960（包含 7 个合并来源）。

**检查待审核别名：**
```cypher
MATCH (a:DrugAlias)-[:ALIAS_CANDIDATE_OF]->(d:Drug)
RETURN a.name AS alias, d.name AS canonical, a.review_status AS review_status
LIMIT 25;
```

**检查合并来源：**
```cypher
MATCH (d:Drug)-[:HAS_RECORD]->(r:SourceRecord)
WHERE r.source_variant='additional_merged_source'
RETURN d.name, r.source_url
LIMIT 20;
```
预期：7 行。

**检查药品文档路径：**
```cypher
MATCH p=(d:Drug)-[:HAS_RECORD]->(r:SourceRecord)
        -[:HAS_SECTION]->(s:Section {field:'indication'})
        -[:HAS_CHUNK]->(c:Chunk)
RETURN p LIMIT 5;
```

### 4. 下一阶段的路径同步

新 JSONL 位于 `data/kg_output_clean_v2/`。不要再使用旧
`data/kg_output/` 作为疾病实体抽取的输入。

如果你之前已安装 `kg.indication_candidates`，使用：

```bash
python -m kg.indication_candidates extract --medical "data/data/raw/drug_db/MedGraphRAG-Drug-DB/medical.json" --in-dir "data/kg_output_clean_v2" --out "data/kg_output_clean_v2/disease_candidates.csv" --limit-drugs 300
```

**ChromaDB 不会自动同步。** 若旧版文本已建立向量索引，需要单独
建立新的 collection 或安全更新旧 collection 的对应文档。

### 5. 已验证的范围与重要说明

- 已针对实际上传文件全量生成 JSONL，并核验所有 Section/Chunk 偏移，
  药品、别名及相互作用候选的引用完整性，无静态引用错误。
- Python 补丁通过离线测试。
- **未连接你电脑上的 Neo4j 执行过实际写入**；最终结果以你本机
  `sync_plan.json` 和 `sync_result.json` 为准。
- 原始说明书来自第三方整理，数据清洗不等于官方版本核验；
  `DrugAlias` 的关系仅为候选映射，不等于所有剂型或品种可互换。
