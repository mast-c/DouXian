# DrugKG 第十阶段：清洗版 V2 → Neo4j 全量候选知识图谱导入

## 本阶段真正要做的事

导入的是你已经清洗好的 `drug_db_v2_cleaned.json` 派生图谱。
**不需要重新清洗，不重新调用 Qwen，不先做 300 条评测。**

真实数据已经离线验证：

| 节点类型 | 预期行数 |
|---|---:|
| Drug | 11,953 |
| SourceRecord | 11,960（包含 7 份额外来源） |
| Section | 106,445 |
| Chunk | 107,787 |
| DrugAlias（候选别名） | 343 |
| DiseaseTerm（原始词条，未医学归一） | 1,150 |
| DiseaseCandidate（仅表示原文命中） | 19,685 |
| ModelAssessment（60 条开发集 Qwen 评估） | 60 |

**共约 259,383 个节点，所有入库内容保持候选或第三方未核验状态。**

严格保留源证据关系：
`(Drug)-[:HAS_RECORD]->(SourceRecord)-[:HAS_SECTION]->(Section)-[:HAS_CHUNK]->(Chunk)`

疾病候选关系：
`(Drug)-[:HAS_CANDIDATE]->(DiseaseCandidate)-[:MENTIONS_TERM]->(DiseaseTerm)`
以及 `(DiseaseCandidate)-[:SUPPORTED_BY]->(Section)` 和 `-[:FROM_RECORD]->(SourceRecord)`。

60 条 LLM 结果仅作 `(ModelAssessment)-[:ASSESSES]->(DiseaseCandidate)`，
不建立 `TREATS` 或 `INDICATED_FOR`。请勿将候选关系当成获批适应症。

本版本暂不导入药物相互作用候选；优先跑通药品-疾病-文档 GraphRAG MVP。

## 解压位置

解压 ZIP 到：
`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

会新增（不会覆盖旧 kg.build 等基础模块）：

```
data/
├── kg/
│   └── neo4j_drug_graph.py
├── docker-compose.drugkg-clean-v2.yml
├── requirements-drugkg-import.txt
├── queries/
│   └── drugkg_readonly_examples.cypher
├── tests/
│   └── test_neo4j_drug_graph.py
└── support/
    ├── build.py                 # 仅备用，不要覆盖现有模块
    └── prepare_cleaned_v2.py    # 仅备用，不要覆盖现有模块
```

本机原有的这些文件应当保持不动：
- `kg_output_clean_v2/disease_candidates_full.csv`
- `kg_output_clean_v2/disease_candidates_full.report.json`
- `kg_output_clean_v2/pilot_v31_merged60.csv`
- `kg_output_clean_v2/pilot_v31_merged60.v31_report.json`
- `drug_db_v2_cleaned.json` 原始清洗文件

如果 `kg/prepare_cleaned_v2.py` 不存在，才把
`support/prepare_cleaned_v2.py` 复制到 `kg/`。
若 `kg/build.py` 不存在，则复制 `support/build.py` 到 `kg/`。

## 第一步：本地预检（不连接 Neo4j）

PowerShell 当前目录应当为：
`E:\VsCode_Project\DockerTest_Agent\RAG_project\data>`

先安装驱动：
```powershell
python -m pip install -r "requirements-drugkg-import.txt"
```

检查清洗 JSONL 是否已经完整：
```powershell
$inDir = "kg_output_clean_v2"

python -u -m kg.neo4j_drug_graph preflight --in-dir "$inDir" --candidates "kg_output_clean_v2/disease_candidates_full.csv" --pilot "kg_output_clean_v2/pilot_v31_merged60.csv" --report "kg_output_clean_v2/neo4j_preflight_v2.json"
```

预期 `status=PREFLIGHT_PASS`，
`disease_candidates=19685`、`source_records=11960`、
`sections=106445`、`chunks=107787`、`drug_aliases=343`。

**如果直接通过，不需要执行下一段重建操作。**

如果报告提示“没有 7 份补充来源”“SourceRecord 不是 11960”
或“DrugAlias 不是 343”，说明你此前使用了不完整的
`kg.build`，而不是支持补充来源的 `kg.prepare_cleaned_v2`。
可从清洗文件重新生成**一个新的中间目录**，不覆盖现有文件：

```powershell
# 仅在预检提示缺失补充来源时使用
$cleaned = Get-ChildItem -Path . -Recurse -File -Filter "drug_db_v2_cleaned.json" |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $cleaned) { throw "没有找到清洗文件，请先确认 drug_db_v2_cleaned.json 的位置" }

python -m kg.prepare_cleaned_v2 --input "$cleaned" --out "kg_output_clean_v2_import_ready" --limit 0

$inDir = "kg_output_clean_v2_import_ready"

python -u -m kg.neo4j_drug_graph preflight --in-dir "$inDir" --candidates "kg_output_clean_v2/disease_candidates_full.csv" --pilot "kg_output_clean_v2/pilot_v31_merged60.csv" --report "$inDir/neo4j_preflight_v2.json"
```

**注意**：候选 CSV 和 V3.1 开发集结果仍使用原来的
`kg_output_clean_v2`，因为它们本身已经过验证；
只有图谱 JSONL 输入目录可能需要改为 `kg_output_clean_v2_import_ready`。

## 第二步：启动独立 Neo4j（推荐，避免旧版污染）

你的原有 Neo4j 可能包含旧版 12,043 条数据。
为避免损失老图谱或混入残留节点，推荐单独开一个 DrugKG 实例。

本地先设置 `NEO4J_PASSWORD` 环境变量（不要把真实密码发到聊天中）。
如果已有该变量，直接沿用。

```powershell
docker compose -f "docker-compose.drugkg-clean-v2.yml" up -d
docker compose -f "docker-compose.drugkg-clean-v2.yml" ps
```

它只占用：
- 浏览器：`http://localhost:7475`
- Bolt 连接：`bolt://127.0.0.1:7688`
- 独立 Docker volume：`drugkg_clean_v2_neo4j_data`

原来 Neo4j 的 `7474/7687` 和原有数据 volume 不受影响。
如果容器或端口已被占用，请先核查现有配置，不要删除原数据目录。

使用的数据库名称仍是 Neo4j Community 默认的 `neo4j`；
只是**不同容器和不同存储 volume**。

## 第三步：正式批量导入（明确写入）

必须在第一步 `PREFLIGHT_PASS` 之后运行：

```powershell
python -u -m kg.neo4j_drug_graph apply --in-dir "$inDir" --candidates "kg_output_clean_v2/disease_candidates_full.csv" --pilot "kg_output_clean_v2/pilot_v31_merged60.csv" --uri "bolt://127.0.0.1:7688" --database "neo4j" --batch-size 400 --progress-every 5000 --apply --report "$inDir/neo4j_import_result.json"
```

真实写入必须指定 `apply --apply`。忘记 `--apply` 会直接拒绝写入。
程序会打印 `drugs`、`source_records`、`sections`、`chunks`、
`drug_aliases`、`disease_terms`、`disease_candidates`、
`model_assessments` 的分批进度。

它使用 `MERGE`；中途崩溃后可原命令重复执行，
已经成功创建的同 ID 节点、关系不会重复生成。
脚本不会 `DETACH DELETE`、不会全库清空。
审核相关字段通过 `ON CREATE SET` 初始化，
后续运行不会把人工审核状态重新写成 pending。

但它也**不会自动删除旧图谱**。
因此在开始 schema/数据写入之前，会检查目标数据库：
如果存在非本次清洗快照的 Drug 节点，会拒绝导入。
不要盲目移除这个保护；请改用独立容器或你之前的备份同步工具。

## 第四步：导入验收

```powershell
python -u -m kg.neo4j_drug_graph verify --in-dir "$inDir" --candidates "kg_output_clean_v2/disease_candidates_full.csv" --pilot "kg_output_clean_v2/pilot_v31_merged60.csv" --uri "bolt://127.0.0.1:7688" --database "neo4j" --report "$inDir/neo4j_verify_result.json"
```

预期：`status=VERIFIED`、所有节点数匹配、
`bad_links_by_type` 全部为 0、`bad_evidence_quotes=0`。
本脚本不验证疾病的临床有效性。

然后打开 `http://localhost:7475`，
账户 `neo4j`，密码为本机自行设置的 `NEO4J_PASSWORD`。

打开 `queries/drugkg_readonly_examples.cypher`，
依次运行“药品→疾病候选”“疾病→药品候选”
和“药品→章节→Chunk”查询。

## 错误时如何处理

- `SourceRecord count != 11960`：
  补齐 7 份合并来源，请使用 `kg.prepare_cleaned_v2`，
  不要把 26 条缺引用候选直接删除。
- `old_drugs outside cleaned snapshot`：
  连接到了旧数据库，请改连接隔离实例的 `7688` 端口；
  不要删老节点来绕过校验。
- `Neo4j driver missing`：运行 `python -m pip install neo4j`。
- `NEO4J_PASSWORD not set`：
  在本机终端设置环境变量，不要在命令中写 `--password`。
- 连接失败：查看 `docker compose ... ps`，核对端口/本机防火墙。
- 中断后续跑：原命令再次执行，会重复检查输入并通过 MERGE
  避免相同节点/关系重复写入。
- 数据不完整：先修复输入文件；不要拿 60 条模型建议当作正式适应症。

## 注意限制

- 本包已对清洗版完整真实数据执行本地无数据库写入的预检，
  并通过基础单元测试。
- **没有连接你电脑上的 Neo4j，因此不能声称已经完成线上数据库写入。**
- `DiseaseTerm` 只是从 `medical.json` 匹配出的词，
  部分实际可能是症状、体征或伴随病；未完成术语本体归一。
- 当前来源是第三方整理文档，不是自动完成 NMPA/官方产品说明书核验；
  需保留 `third_party_unverified` 和所有原文证据。
- 正式临床问答不得将 `DiseaseCandidate` 解释为
  确认的 `Drug-[:TREATS]->Disease`。
