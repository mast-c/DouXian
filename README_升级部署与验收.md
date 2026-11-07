# GraphRAG v5.3｜图路径召回、证据校验及展示链路修复

本包基于本轮已上传的 **v5.2 medical_semantic_retrieval.py**、`rag.py` 与
`medical_evidence_coverage.py` 进行修改。不是旧版检索器回退。

## 1. 本次实际修复

### A. 检索命中却显示“0条”
- 新增 **`direct_relation_paths`**、**`multi_hop_paths`**、
  **`multi_relation_paths`**、**`same_relation_multi_hop_paths`** 四种独立计数。
  3 跳全为 `ADVERSE_REACTION_MENTION`：多跳=1，但多关系类型=0；这不是空路径。
- 对普通检索、独立任务检索与分类探索，统一只统计已通过来源回查的路径。
- `rag.py` 新增 `trace["graph_path_visibility"]`，分别记录：
  检索层有多少来源验证路径、因回答生成 `evidence_cap` 未进入输入的有多少，
  以及原始路径是否仍在 `trace["semantic_graph"]["paths"]`。
- 新增可选 `medical_graph_path_ui.py`，独立展示 **单跳 + 多跳 + 多种关系类型**，
  不以 `multi_relation_paths > 0` 作为展示开关。只渲染
  `verified_against_original=True` 的已验证路径。

### B. 修复单条坏路径导致整轮失败，同时保持全局安全阻断
- 单条边的**明确局部校验失败**（错误引文、错误端点等），丢弃该候选，继续验证其他路径；
  新增计数 `rejected_invalid_local_edge`、`rejected_malformed_candidate`。
- **整体索引变更、manifest、fingerprint、快照或来源完整性失败**仍然阻断整轮。
  不会为了增加可显示路径而放宽 `_proof()` 或 `verify_source()`。
- 原 `_proof()`、SQLite `_traverse()` 和 `query_plan()` 的函数体没有改动。

### C. Neo4j 路径候选排序与可观测性
- 将 Neo4j 通用 Cypher 查询修改为 **先 `ORDER BY hops ASC, edge_ids ASC`，再 `LIMIT`**。
  不再完全依赖数据库未定义的返回顺序，优先得到短路径。
- 每个要求命中的实体分组单独检查，防止同一个锚点重复经过被错误计算为多个锚点。
- 记录 `raw_candidate_rows`、`neo4j_candidate_limit_hit`、
  `rejected_missing_local_edge_ids`、`rejected_missing_local_node_ids`、
  `rejected_local_relation_mismatch` 等统计项。
- 普通双药检索时，Neo4j 后端不再被仅限 SQLite 的 `patient_pair_paths` 分支抢走。

注意：**排序 + 有界 LIMIT 仍不是完整图穷举**；高出度节点、3～5跳查询在未测试的真实
Neo4j 里仍可能比较慢，需要根据实际查询计划和数据库索引进一步优化。

### D. 多任务规划与证据覆盖关联
- `medical_evidence_coverage.build_answer_targets()` 新增可选的 `semantic_task_plan` 参数，
  老的 3 参数调用继续可用。
- 已存在的同语义回答目标优先关联具体 `semantic_task_id`；否则创建专属目标。
- 来源验证通过的图谱路径必须同时满足任务 ID、实体锚点和对应关系类型才可以成为
  该目标的**原文证据候选**，不构成医学结论。
- `rag.py` 将 LLM 语义任务计划传入回答目标构建；
  合并证据时更容易为不同子任务保留独立已验证证据。
- 对 `text_only` 分类任务，未检索到分类原文之前仍明确保持待补检状态，
  **不能用其他药品路径替代分类事实**。

## 2. 必须复制的文件（5个）

| 文件 | 操作 | 作用 |
|---|---|---|
| `medical_semantic_retrieval.py` | **替换** | 图检索、局部校验失败处理、Neo4j排序及路径统计 |
| `medical_evidence_coverage.py` | **替换** | 子任务到回答目标/证据ID的绑定 |
| `rag.py` | **替换前先对照本机差异** | 将路径独立清单及多任务目标接入流程 |
| `medical_graph_visibility.py` | **新增** | 追踪路径从检索到生成输入各阶段的数量 |
| `medical_graph_path_ui.py` | **新增，可选展示** | Streamlit 展示经过来源验证的单跳/多跳路径 |

**`rag.py` 必须检查本机版本**：本包使用你此前上传到聊天的 `rag.py` 为基线，
如果最近又改动过它，请根据附带 `GraphRAG_v5_3_变更差异.patch` 合入 3 处改动，
不要直接覆盖掉自己最新的业务逻辑。

继续保留原 v5.2 的 `medical_llm_query_understanding.py`、
`medical_semantic_query_planner.py`、`medical_category_graph.py`、
`medical_category_discovery.py`、`evidence_guard.py` 等文件。

## 3. 部署步骤（PowerShell，项目 data 目录）

先备份这三个旧文件：

```powershell
Copy-Item .\medical_semantic_retrieval.py .\medical_semantic_retrieval.py.bak
Copy-Item .\medical_evidence_coverage.py .\medical_evidence_coverage.py.bak
Copy-Item .\rag.py .\rag.py.bak
```

将压缩包中对应的 `.py` 文件复制到 `RAG_project\data`，确认新 `medical_graph_visibility.py`
已同目录存在，再执行：

```powershell
python -m py_compile medical_semantic_retrieval.py medical_evidence_coverage.py rag.py medical_graph_visibility.py medical_graph_path_ui.py
python -m unittest -v test_v52_regressions test_v53_regressions
streamlit run .\豆馅.py
```

完全停止旧 Streamlit/Python 后端再重启，不能只刷新网页。

## 4. 选配：让新单跳/多跳面板在 Streamlit 中真正显示

**注意：你没有上传 `豆馅.py` 或原有 `medical_semantic_graph_view.py`，本包没有改到那个前端。**
`rag.py` 现在会把路径和完整统计保存在 trace 中，但旧面板如果仍只看
`multi_relation_paths`，不会自行改成新面板。

若想追加一个独立且更直观的面板，在 `豆馅.py` **已有 trace 对象形成以后**调用：

```python
from medical_graph_path_ui import render_verified_graph_paths

# trace 为本次 RagService 检索生成的完整 trace dict
render_verified_graph_paths(trace)
```

不要在 `trace` 还没有生成的时候调用。
这只增加原文核验路径的可视化，**不会修改 Evidence Guard 的拒答决定**。
如果你希望替换现有面板而不是并列显示，需要进一步提供 `豆馅.py`
和原路径卡片渲染模块，才能安全地定位现有组件和修改位置。

## 5. 诊断命令

新增 `diagnose_graph_pipeline_v53.py`，不需要 API，也不连接数据库，只读本地 Trace 的摘要。

```powershell
python diagnose_graph_pipeline_v53.py --trace .\medical_trace_082.json
```

此命令只输出路径、证据绑定和管线阶段的数量，**不输出原问题或个人病史内容**。
如果 Trace 中已经有新字段 `graph_path_visibility`，直接查看它更直观。

### 如何解释

| 看到的结果 | 说明 | 下一步 |
|---|---|---|
| `status=source_verification_blocked` | 索引/来源失效，不能用未经验证的路径 | 修复本地SQLite派生索引/Neo4j快照一致性 |
| `verified_path_count>0`, `multi_relation_paths=0` | 可能有单跳或同一关系的多跳 | 看 `direct_relation_paths`/`multi_hop_paths` |
| 检索层 verified > 0，generation 0 | 合并名额/目标绑定没有保留该路径 | 检查 `evidence_flow`/`graph_path_visibility` |
| `neo4j_candidate_limit_hit=True` | 有界候选集可能被截断 | 调整检索策略、索引和预算后真实库复测 |
| `rejected_missing_local_edge_ids>0` | Neo4j关系ID与SQLite派生索引不同步 | 重新核对导入/来源快照 |
| `rejected_invalid_local_edge>0` | 个别候选边与原文位置不一致 | 复核抽取、校验原始记录 |

## 6. 回归测试范围与仍未覆盖的内容

- **23 项离线测试通过**（11项v5.2回归 + 12项v5.3测试）。
- 使用合成 SQLite 原文/边验证无效边拒绝、有效边保留；
  通过模拟 Neo4j session 检查 Cypher排序、分组锚点和 ID 回查；
  测试子任务-目标绑定、证据上限分离以及新 Streamlit 渲染器只显示核验路径。
- 核对原 `_proof()`、SQLite `_traverse()`、`query_plan()` 没有修改。
- 这是**离线修复**：尚未连接你的真实 Neo4j 服务器、SQLite目录、Qwen API；
  也尚未对你真实 `豆馅.py` 的 UI 布局做端到端验证。
- 本版**不自动恢复已被 GraphBlocked 标记失效的索引**，不修复不存在的医学分类边，
  不取消严格来源校验，也不保证每种医学问句都有可用路径。

## 7. 回滚

如本机出现兼容性问题：

```powershell
Copy-Item .\medical_semantic_retrieval.py.bak .\medical_semantic_retrieval.py -Force
Copy-Item .\medical_evidence_coverage.py.bak .\medical_evidence_coverage.py -Force
Copy-Item .\rag.py.bak .\rag.py -Force
```

然后停止并重启 Python 进程。新增辅助文件可保留或删除，但只要用了修复后的 `rag.py`，
**必须同时保留 `medical_graph_visibility.py`**。
