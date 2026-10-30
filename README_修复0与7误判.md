# DrugKG V2：修复 additional_source_records=0 导致的错误拦截

## 出错原因（已从用户日志确认）

日志显示：

- `schema_version: 0.2.0`
- `counts.source_records: 11960`
- `counts.sections: 106445`
- `counts.chunks: 107787`
- `cleaned_snapshot.additional_source_records: 0`

最新版 `kg.build` 在主构建阶段已经处理了原始清洗数据中的 7 条补充来源。
随后 `kg.prepare_cleaned_v2` 检测到它们已经存在，跳过重复追加，
所以“**本次新增**”是 0；**最终总数**仍然是 11,960。

旧版 `kg.neo4j_drug_graph` 的检查条件
`cleaned_snapshot.additional_source_records != 7` 因此错误地阻止导入。

这个补丁改为验证**真实 JSONL**中的 11,960 来源、106,445 章节、
107,787 Chunk、343 别名及 19,685 疾病候选之间的真实来源证据路径；
不再将“本次新增数”误判成“来源总数”。

本补丁也保留已有 SourceRecord `source_variant`，不会因 CSV/JSONL 缺少此可选字段而清除原属性。

## 操作（Windows PowerShell）

当前目录：
`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

1. 解压补丁，只替换下面一个 Python 文件：

```text
kg/neo4j_drug_graph.py
```

2. **不要重建或删除**：

```text
kg_output_clean_v2_import_fixed
kg_output_clean_v2/disease_candidates_full.csv
kg_output_clean_v2/pilot_v31_merged60.csv
```

3. 明确指定新构建目录，而不是依赖旧的 `$inDir`：

```powershell
python -u -m kg.neo4j_drug_graph preflight --in-dir "kg_output_clean_v2_import_fixed" --candidates "kg_output_clean_v2/disease_candidates_full.csv" --pilot "kg_output_clean_v2/pilot_v31_merged60.csv"
```

预期 `"status": "PREFLIGHT_PASS"`。

4. 可选：正确读取中文 JSON（Windows PowerShell 5.1 也支持）：

```powershell
$report = Get-Content ".\kg_output_clean_v2_import_fixed\build_report.json" -Encoding UTF8 -Raw | ConvertFrom-Json
$report.counts.source_records
$report.cleaned_snapshot.additional_source_records
```

应该依次是 `11960` 和 `0`；这是**正常组合**。

5. 预检通过且独立 Neo4j 已启动后，再执行**明确写入**：

```powershell
python -u -m kg.neo4j_drug_graph apply --in-dir "kg_output_clean_v2_import_fixed" --candidates "kg_output_clean_v2/disease_candidates_full.csv" --pilot "kg_output_clean_v2/pilot_v31_merged60.csv" --uri "bolt://127.0.0.1:7688" --batch-size 400 --apply
```

6. 全量入库完成后运行：

```powershell
python -u -m kg.neo4j_drug_graph verify --in-dir "kg_output_clean_v2_import_fixed" --candidates "kg_output_clean_v2/disease_candidates_full.csv" --pilot "kg_output_clean_v2/pilot_v31_merged60.csv" --uri "bolt://127.0.0.1:7688"
```

预期 `status=VERIFIED`。

## 已验证

- 完整 11,960 来源/19,685 疾病候选的真实本地数据预检原本已通过。
- 本次将本地建图报告模拟成 `schema_version=0.2.0`、
  `cleaned_snapshot.additional_source_records=0` 后，修复版预检通过：
  `PREFLIGHT_PASS`、`source_records=11960`、`disease_candidates=19685`。
- 原有 10 项离线测试通过。
- **尚未连接用户实际 Neo4j 或执行数据库写入。**

### 安全说明

入库的关系是 `DiseaseCandidate`，不等同于医学确认的 `TREATS`。
补丁不会删除已有 Neo4j 节点，不会清空数据库，也不会导入未验证的
“已获批准治疗关系”。仍建议使用原先指定的独立 Neo4j 实例。
