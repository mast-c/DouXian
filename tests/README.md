# tests

当前回归集：患者问题理解 `test_patient_question_understanding.py`，共享查询规划与成本控制 `test_shared_query_retrieval.py`。
证据目标覆盖、合并、审核清单与有界补检：`test_evidence_goal_coverage.py`。
图谱缺口只读诊断、原文位置、共享第三药与分类边界：`test_graph_gap_diagnosis.py`。
完整药名与意图解耦、分类图实体关系、JSON溯源及后端显示：`test_category_graph.py`。
分类驱动的依赖计划、同类/跨类候选、排除条件、双向相互作用连接、不同成分、条件过滤、来源校验及预算：`test_category_discovery.py`。

从 RAG_project 根目录运行：

```powershell
python -m unittest discover -s data/tests -p 'test_*.py' -v
```

覆盖口语药名、用药角色、备选药名与错别字澄清、用药后不适、疗效不佳、时间信息、否定和家人病史。
覆盖共享计划、并发召回、去重后一次精排、阶段超时、失败降级、图路径原文核验和后端接入。
覆盖两药成分不丢失、独立单药路径不能证明相互作用、患者临床问题不被简化为栏目查询、补检原文核验及复审超时、完整后端引用链。
测试使用临时药品目录、临时图索引和接口桩，不连接 Neo4j、Chroma 或模型接口。

历史测试脚本保存在 `../../待清理归档_2026-11-04/测试与评测/`。
