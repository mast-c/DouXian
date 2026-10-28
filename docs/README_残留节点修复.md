# Neo4j 残留旧节点精确修复：V2 → Cleaned V2

## 你的当前状态（根据 Neo4j 截图）
| Label | 目前 | 清洗版期望 | 差额 |
|---|---:|---:|---:|
| Drug | 12,040 | 11,953 | +87 |
| SourceRecord（仅本数据集） | 11,960 | 11,960 | 0 |
| InteractionCandidate | 14,288 | 13,275 | +1,013 |

本补丁会基于旧/新 JSON **重新计算哈希 ID**，定位旧版独有节点：
- 1,013 条过期 InteractionCandidate 候选关系；
- 490 个旧 Drug ID 中尚在数据库的节点（预计约 87），仅删除无任何关系的孤立节点；
- 340 个过期 NameMention ID 中无关系的节点。

它不对 Neo4j 做全库清空操作，也不修改现有 ChromaDB。

## 放置补丁
将 ZIP 中的 `kg/repair_residuals.py` 解压到你现有项目根目录：

```
RAG_project/
├─ kg/
│  ├─ build.py              # 已有，保留
│  ├─ neo4j_import.py       # 已有，保留
│  └─ repair_residuals.py   # 新增
├─ data/
│  ├─ raw/drug_db/MedGraphRAG-Drug-DB/
│  │  ├─ drug_db_v2.json
│  │  └─ drug_db_v2_cleaned.json
│  └─ kg_output_clean_v2/
│     ├─ build_report.json
│     ├─ drugs.jsonl
│     ├─ source_records.jsonl
│     └─ interaction_candidates.jsonl
└─ .env
```

注意：使用本机真实路径；这里依据上一份 `sync_plan.json` 的路径。
文件夹中不要覆盖原来的 `kg/build.py`。

## 第一步（只读）：预览残留清理

从 `RAG_project` 根目录运行一整行：

```bash
python -m kg.repair_residuals --old-json "data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2.json" --clean-json "data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2_cleaned.json" --manifest "data/kg_output_clean_v2"
```

默认 **不会修改 Neo4j**。报告输出到：

`data/kg_output_clean_v2/residual_repair_preview.json`

先核对：
- `obsolete_interaction_ids`：应为 1,013
- `obsolete_interactions_present_in_db`：如果没有被提前处理，应为 1,013
- `obsolete_interactions_blocked`：应为 0
- `current_expected_nodes_present`：应是 11,953 Drug / 11,960 SourceRecord / 13,275 InteractionCandidate
- `obsolete_drugs.obsolete_drug_nodes_still_present`：预计 87 左右

如果这些差异不合理，**不要执行 apply**，先检查文件/数据库版本。

## 第二步（需备份）：执行精确清理

先备份 Neo4j（必须有可恢复的数据库备份），然后运行：

```bash
python -m kg.repair_residuals --old-json "data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2.json" --clean-json "data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2_cleaned.json" --manifest "data/kg_output_clean_v2" --apply --backup-confirmed
```

它会：
1. 验证数据库中所有**新版**节点都已存在（新 Drug、SourceRecord、InteractionCandidate 的 ID 完整匹配）；
2. 检查待删除候选的姓名字段、医学审核标记和边类型，遇到额外业务关系则阻止删除；
3. 仅按旧版独有的 InteractionCandidate ID 删除候选节点；
4. 只删除旧版独有且没有其他关系的 Drug 节点；
5. 清理旧版独有且已孤立的 NameMention 节点。

它**不会**自动删除仍连有业务关系的 Drug 节点。
本操作是分批事务，不是一个大型原子事务；操作中断时先看
`residual_repair_result.json`，可根据报告重新预览。

## 第三步：Neo4j 验证

```cypher
MATCH (n:Drug) RETURN count(n) AS drugs;
```

```cypher
MATCH (n:InteractionCandidate) RETURN count(n) AS interactions;
```

```cypher
MATCH (r:SourceRecord)
WHERE r.dataset='MedGraphRAG-Drug-DB drug_db_v2.json'
RETURN count(r) AS sources;
```

预期值分别为 **11,953 / 13,275 / 11,960**（若数据库存在与该项目无关的同标签节点，计数会不同）。

再运行：

```cypher
MATCH (d:Drug)
WHERE d.name IN ['（胶囊', '(胶囊']
RETURN d.name AS old_name, d.id AS id;
```

预期无结果。如果旧 Drug 仍保留，说明它存在其他关系，修复脚本为防止丢失用户业务知识而跳过删除。

## 说明
- 不是临床适应症关系修复，而是**导入旧数据残留**的清理。
- 不读取或发送任何数据库密码、说明书正文到外部服务。
- 目前只做离线脚本测试，没有连接你的 Neo4j 实例。
- 数据依然属于第三方来源，`third_party_unverified` 标记保持不变。
