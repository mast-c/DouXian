# 豆馅医疗 GraphRAG V1.17 — P3-04.4A 疾病百科与 Neo4j 疾病实体对齐

## 目的与完成边界

上一阶段你已经能够从 `disease_knowledge` 查询到独立疾病百科原文，
例如问“过敏性鼻炎是什么”时引用 `medical.json`。

下一步不是把所有同名疾病直接变成 Neo4j 的医学事实，而是：
1. 从现有 8,807 条疾病百科源记录（8,806 个不同名称）中，
   **只读**提取疾病名称；
2. 与现有 Neo4j `DiseaseTerm` 的 `name` 做**严格同名**匹配；
3. 核对原始 `medical.json` 文件 SHA256、P3-04.3 术语对齐审计、
   P3-04.2 完整 Chroma 入库审计；
4. 再次实时只读查询 Neo4j，防止用旧报告误关联已经变化的
   `DiseaseTerm`；
5. 检查 Neo4j 是否已有 ID 相同的 `DiseaseDocument`，避免今后误覆盖；
6. 输出**待人工审核**的 `DiseaseTerm → DiseaseDocument` 候选计划及
   8,807 条源记录的审阅 CSV。

**此包的所有 Neo4j 查询都只有 MATCH/RETURN。**
不会修改 Chroma、Neo4j、SQLite，不调用 Embedding。
不会创建 `Drug-[:TREATS]->Disease`、任何医学确证关系，
也不会修改已有的药品说明书 52,104 条 Chunk。

关系类型只是下一阶段的**计划**：
`(:DiseaseTerm)-[:EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA]->(:DiseaseDocument)`

它明确表示**同名技术匹配，第三方疾病百科资料未经医学验证**；
本阶段不会创建这个节点或关系。

## 包内文件

| 文件 | 用途 |
|---|---|
| `medical_p3043_disease_neo4j_align_v117.py` | 已有的 P3-04.3 实时只读同名对齐程序，重新附带避免文件已被清理 |
| `medical_p3044_disease_graph_link_plan_v117.py` | 新增 P3-04.4A：核对审计哈希、实时验证 Term 及其冲突、输出安全候选计划 |
| `test_medical_p3044_v117.py` | 22 项离线测试；全部使用虚构疾病与模拟 Neo4j |
| `medical_p3041_medical_json_prepare_v117.py` | 现有 `medical.json` JSON Lines 解析工具，线上疾病百科仍然依赖 |
| `README_P3-04-4A_V117_disease_term_graph_alignment.md` | 当前部署与结果说明 |

本包没有改 `rag.py`，也没有改 P3-04.3 在线检索器。

## Windows 使用步骤

工作目录：

`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

将工具包解压到此目录。它可以补回此前清理过的
`medical_p3043_disease_neo4j_align_v117.py`，但不要因此删除
`medical_p3041_medical_json_prepare_v117.py`（它是线上检索运行依赖）。

```powershell
cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"
$py = "E:\Miniconda\envs\rag\python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$raw = ".\data\raw\drug_db\MedGraphRAG-Drug-DB\medical.json"

# 第一步：22 项离线回归测试，不访问数据库
& $py -m unittest -v test_medical_p3044_v117.py

# 检查真实源文件（预期 01aaf5de...55f8dfc49）
Get-FileHash $raw -Algorithm SHA256
```

### 1. 先生成最新的 P3-04.3 疾病术语同名审计

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$align = ".\stage4_reports\p3044a_term_align_$stamp.json"
$links = ".\stage4_reports\p3044a_term_links_$stamp.jsonl"

& $py .\medical_p3043_disease_neo4j_align_v117.py `
  --source $raw `
  --batch-size 400 `
  --report $align `
  --links $links

if ($LASTEXITCODE -ne 0) {
  throw "P3-04.3 同名对齐失败；不要继续。"
}
```

此步会访问你本地 Neo4j 读取 `DiseaseTerm`，只执行 `MATCH/RETURN`。
可能有不少疾病名称没有命中：这是**覆盖率结果**，并非脚本失败。
**不能假定图中的 1,150 个 DiseaseTerm 都能对应百科。**

### 2. 找到真正通过的 P3-04.2 全量 Chroma 审计

```powershell
$auditFile = Get-ChildItem ".\stage4_reports" `
  -Filter "p3042_final_audit_*.json" -File |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if (-not $auditFile) {
  throw "缺少 P3-04.2 完整审计报告，不能生成正式关联计划。"
}

$auditFile.FullName
```

如果有多份审计，选择**针对当前 `medical.json` 和 Chroma Collection 的
真实全量 PASS 报告**。新脚本仍会核查其中的源 SHA256、Chunk 数和来源状态；
不能用手工编造的 PASS 报告。

### 3. P3-04.4A 再次校验 Neo4j 并生成候选计划

```powershell
$planStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$summary = ".\stage4_reports\p3044a_graph_plan_$planStamp.json"
$plan = ".\stage4_reports\p3044a_reviewed_candidates_$planStamp.jsonl"
$review = ".\stage4_reports\p3044a_record_review_$planStamp.csv"

& $py .\medical_p3044_disease_graph_link_plan_v117.py `
  --source $raw `
  --align-report $align `
  --align-links $links `
  --chroma-audit $auditFile.FullName `
  --batch-size 300 `
  --report $summary `
  --plan $plan `
  --review-csv $review

if ($LASTEXITCODE -ne 0) {
  throw "P3-04.4A 核验失败；不要进入 Neo4j 写入阶段。"
}
```

**真正正常的预期 readiness 是：**

`READY_FOR_HUMAN_REVIEW_NO_WRITES`

这不表示医疗审查已经完成，只表示**数据和图谱快照经过技术复核，
可以进入人工审阅**。输出：

- `p3044a_graph_plan_*.json`：疾病术语精确匹配数量、
  图谱节点当前总数、未匹配数、来源与审计状态、冲突统计；
- `p3044a_reviewed_candidates_*.jsonl`：待人工审核的
  `DiseaseTerm → DiseaseDocument` 关联清单；一条记录对应一个独立 OID；
- `p3044a_record_review_*.csv`：覆盖 8,807 条原始疾病记录的审查清单，
  区分严格同名候选、歧义与未决情况。

如果已有文档 ID 冲突，输出
`EXISTING_DOCUMENT_IDS_REQUIRE_REVIEW`；
此时不可随意复用或覆盖旧节点。

如果图谱已发生变化，脚本将 `BLOCKED` 并提示重新跑 P3-04.3。
不删除任何现有节点或检查点。

### 查看关键统计

```powershell
$r = Get-Content $summary -Raw -Encoding UTF8 | ConvertFrom-Json

$r | Select-Object `
  readiness,original_disease_records,unique_disease_names, `
  aligned_unique_names,aligned_records_in_review_plan, `
  new_neo4j_nodes,new_neo4j_relationships

$r.graph_recheck
$r.prior_alignment_counts
```

**请把 `p3044a_graph_plan_*.json` 上传当前对话。**
它只包含统计、源 SHA 和少量冲突 ID，不含 Neo4j 凭据或疾病正文。

## 检索与图谱如何衔接

本阶段不会改变现有 GraphRAG 在线问答行为。疾病问题依旧使用
P3-04.3 的 `disease_knowledge` 精确名称检索、原文校验与来源标注。
药品问题仍走 P3-03B 的说明书证据链。

待审核结果确认后，再考虑 P3-04.4B **只新增明确标为未核验的
`DiseaseDocument` 节点及同名来源关系**，不复制冗余正文，
不创建正式疗效/适应症边。缺少人工审查时应将所有候选保留
`PENDING_HUMAN_REVIEW`。

## 可删除与必须保留

- 项目持续运行必需：`rag.py`、
  `medical_p3043_disease_encyclopedia_runtime_v117.py`、
  `medical_p3041_medical_json_prepare_v117.py`、
  原始 `medical.json`、`chroma_db`、完整 P3-04.2 审计报告。
- 测试结束可归档：`test_medical_p3044_v117.py`。
- 术语审计或重跑计划仍需要：
  `medical_p3043_disease_neo4j_align_v117.py` 和
  `medical_p3044_disease_graph_link_plan_v117.py`。
- 原始药品图谱、已有疾病候选、历史 SQLite 状态和
  `rag.py.pre_p3043_*.bak` 不在本包的删除范围内。

## 验证边界

当前 P3-04.4A 只能证明：
- 源疾病 OID、同名候选及审计哈希一致；
- 当前 Neo4j 精确名称/ID 状态与对齐审计一致；
- 图谱没有与拟建文档 ID 冲突（如果返回 READY）；
- P3-04.2 Chroma 入库审计与源文件一致。

**这不证明疾病同义词关系正确，也不证明 `medical.json`
来自已获得授权的权威医学资料。**
正式建图必须进一步确认具体的数据使用权与审核规则。
