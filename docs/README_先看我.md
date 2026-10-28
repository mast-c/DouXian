# 面向现有 RAG_project 的药品知识图谱模块 V1

这是**增量添加模块**，不会替换你原有的 RAG 代码、`.env`、
`data/chroma_db`、`chat_history.sqlite3` 或任意业务脚本。

## 1. 解压位置与目录

把 ZIP 解压到 **RAG_project 项目根目录**，形成：

```text
RAG_project/
├── data/                         # 原有文件夹，不动
│   ├── chroma_db/                # 现有 ChromaDB 不动
│   └── data/
│       └── raw/
│           └── drug_db/
│               └── MedGraphRAG-Drug-DB/
│                   └── drug_db_v2.json    <-- 已经在你的电脑里
├── kg/                           # 新增
│   ├── build.py
│   ├── profile.py
│   ├── neo4j_import.py
│   ├── review_indications.py
│   ├── export_rag_docs.py
│   └── tests/
├── requirements-kg.txt           # 新增
└── docker-compose.kg.yml         # 新增，可选
```

**注意**：以上 `data/data/` 是按你截图显示的层级填写。
如果终端实际进入了 `RAG_project/data/`，输入路径需相应减少一层 `data/`。
最好在 VS Code 终端先运行 `pwd`（macOS/Linux）或 `cd`（Windows）
确认工作目录在 `RAG_project` 根目录。

## 2. 不用 Neo4j，先跑 300 条

图谱构建会先运行 `data/kg/clean_names.py`：它依据同一来源记录的完整标题修复括号残片、别名片段和乱码名称；合法短药名会保留，药品类别/剂型只会作为未解析提及，不会误连到具体 `Drug`。不同来源的同名记录仍保存在多个 `SourceRecord` 中。每次全量构建还会生成 `data/kg_output/name_cleaning_report.json`，其中列出修复、去重、隔离和相互作用端点变更。

重建后可执行完整性检查：

```bash
python -m kg.validate \
  --in-dir "data/kg_output" \
  --source "data/data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2.json" \
  --out "data/kg_output/integrity_report.json"
```

仅当检查结果 `error_count` 为 0 时再导入 Neo4j。被隔离的记录不会静默丢失，详见名称修复报告中的 `quarantine`。

在项目根目录：

```bash
python -m pip install -r requirements-kg.txt

python -m kg.profile \
  --input "data/data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2.json" \
  --out "data/kg_output/source_profile.json"

python -m kg.build \
  --input "data/data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2.json" \
  --out "data/kg_output" \
  --limit 300

python -m kg.neo4j_import --in-dir "data/kg_output"
```

最后一条命令默认为 **dry-run，不修改数据库**。
Windows PowerShell 可把多行命令各自写成一行。

确认 `data/kg_output/build_report.json` 和各个 `.jsonl` 后再处理全量：

```bash
python -m kg.build \
  --input "data/data/raw/drug_db/MedGraphRAG-Drug-DB/drug_db_v2.json" \
  --out "data/kg_output" \
  --limit 0
```

`--limit 300` 用固定随机种子抽样，而不是只选药品名字典开头的 300 条。
`--limit 0` 表示 12,043 种药品全量转换。

## 3. 导入本地 Neo4j

如果已经安装 Neo4j，直接在原有 `.env` 中**追加**：

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你的本地Neo4j密码
```

如果还没有 Neo4j，可以在根目录设置 `NEO4J_PASSWORD` 后运行：

```bash
docker compose -f docker-compose.kg.yml up -d
```

Browser 地址：http://localhost:7474

然后执行：

```bash
python -m kg.neo4j_import --in-dir "data/kg_output" --apply
```

这会创建必要的唯一性约束，使用 `MERGE` 批量写入图谱；
**不会清空你原来数据库中的其他节点**。
但如果已有与本模块同名的 `Drug`、`Section` 标签及 `id` 属性，
应先在单独的 Neo4j 测试数据库中验证，以免发生标签命名冲突。

如使用自定义数据库：

```bash
python -m kg.neo4j_import --in-dir "data/kg_output" --database "neo4j" --apply
```

## 4. 先在 Neo4j Browser 查询

```cypher
MATCH (d:Drug)
RETURN d.name, d.entity_scope
LIMIT 10;
```

```cypher
MATCH (d:Drug)-[:HAS_RECORD]->(r:SourceRecord)
      -[:HAS_SECTION]->(s:Section {field:'indication'})
RETURN d.name AS drug, left(s.text, 250) AS indication,
       r.source_url AS source
LIMIT 20;
```

```cypher
MATCH (c:InteractionCandidate)-[:SOURCE_DRUG]->(d:Drug)
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(s:Section)
RETURN d.name AS source_drug,
       c.target_name AS target_mention,
       c.risk_label_in_source AS dataset_risk,
       c.evidence_status AS evidence_status,
       s.field AS evidence_section
LIMIT 30;
```

**注意**：`InteractionCandidate.risk_label_in_source` 是上游数据集标签，
不是临床有效性或药物相互作用危险等级的独立验证。

## 5. 生成供 RAG / Chroma 使用的检索证据

不需要修改现有 ChromaDB：

```bash
python -m kg.export_rag_docs \
  --in-dir "data/kg_output" \
  --out "data/kg_output/rag_documents.jsonl"
```

每行是 `{id, text, metadata}`，包含药名、原始 URL、字段类型、
source_record_id、section_id、字符偏移。后续可以由你原有的
Embedding/Chroma 流程独立读取并建立一个新 collection。
当前版本**不会自动向已有 `data/chroma_db` 写入任何东西**。

## 6. 从适应症文本建立疾病节点

第一版不把自由文本自动当成已批准适应症。
人工核对候选后制作 `indication_annotations.jsonl`，一行：

```json
{
  "drug_name": "某种药的完整数据集名称",
  "disease_name": "某疾病名称",
  "evidence_quote": "说明书适应症字段里逐字出现、包含该疾病名的引文",
  "review_status": "human_checked"
}
```

将注释填好后运行：

```bash
python -m kg.review_indications \
  --in-dir "data/kg_output" \
  --annotations "indication_annotations.jsonl"

python -m kg.neo4j_import --in-dir "data/kg_output" --apply
```

每次重建原始图谱都会清空旧的审核结果文件，因此请保留单独的人工标注文件，重建后重新运行审核命令。\n\n只有与 `indication` 原文逐字匹配的证据关系才会进入
`IndicationClaim -> DiseaseTerm`，且明确标记
`clinical_status=not_validated`。这不是药监局批准验证。

## 7. 数据使用边界与下一阶段

- 真实数据来源为第三方药品信息整理，不能当作 2026 年最新官方说明书；
- 不要将说明书**非空**字段等同于**有效医学事实**；
- `contraindications` 有混入“慎用”的条目，不直接转成 `CONTRAINDICATED_FOR`；
- `interactions` 只有极少数双方药名在本数据字典中精确匹配，
  保留为候选关系，后面再引入别名和药物类别映射；
- GitHub 仓库代码许可证不自动授权说明书正文再分发；
- 需要 DiseaseTerm 标准化（例如经过授权的 ICD/MeSH 映射）后，
  才建立真正的医学实体关联并开展 GraphRAG 对比。

### 已知真实数据规模

- 12,043 条药品名称级记录
- 13,316 条候选相互作用
- 11,654 条非空适应症
- 全量结构化输出：12,043 Drug / 12,043 SourceRecord /
  106,985 Section / 108,327 Chunk /
  7,978 NameMention / 13,316 InteractionCandidate

详见 `reports/quality_report.json` 和 `reports/full_build_report.json`。
项目附虚构技术示例和测试用例，可执行：

```bash
python -m unittest discover -s kg/tests -v
```

本包中不含第三方说明书原文；用户可在自己已有的 `drug_db_v2.json`
上重新生成所有结构化结果，保证数据来源和过程可复现。
