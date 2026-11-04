# tests

当前回归集：患者问题理解 `test_patient_question_understanding.py`，共享查询规划与成本控制 `test_shared_query_retrieval.py`。

从 RAG_project 根目录运行：

```powershell
python -m unittest discover -s data/tests -p 'test_*.py' -v
```

覆盖口语药名、用药角色、备选药名与错别字澄清、用药后不适、疗效不佳、时间信息、否定和家人病史。
覆盖共享计划、并发召回、去重后一次精排、阶段超时、失败降级、图路径原文核验和后端接入。
测试使用临时药品目录、临时图索引和接口桩，不连接 Neo4j、Chroma 或模型接口。

历史测试脚本保存在 `../../待清理归档_2026-11-04/测试与评测/`。
