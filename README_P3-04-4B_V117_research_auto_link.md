# 豆馅医疗 GraphRAG V1.17 · P3-04.4B 科研原型自动图谱关联

## 为什么不要求 1,150 条逐条人工医学审核？

你现在做的是**研一 GraphRAG 科研实验**，不是面向患者提供经过权威审核
诊疗结论的医疗产品。对于已有明确技术依据的疾病名称映射，
可以采取 **科研专用自动关联**：

- 只接受 Neo4j `DiseaseTerm.name` 与 `medical.json` 原始名称的
  **严格同名**候选，决不做模糊医学同义词自动确认；
- 严格验证原始 OID、源文件 SHA256、逐行 SHA256、
  P3-04.4A 原始候选审计和 P3-04.2 Chroma 全量审计；
- 在真实 Neo4j 里再次核验 1,150 个 DiseaseTerm 名称和 ID；
- 用 **新创建的** `DiseaseDocument` 存放稳定 OID、来源、索引元数据，
  **不把整份百科文本复制进 Neo4j**；
- 只创建
  `(:DiseaseTerm)-[:EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA]->(:DiseaseDocument)`
  的**未核验百科同名来源关系**；
- 真实写入节点与关系全部保留
  `clinical_verified=false`, `source_status=third_party_unverified`,
  `source_rights_verified=false`, `research_only=true`,
  `review_status=NOT_CLINICALLY_REVIEWED`；
- **不会**创建 `TREATS`、`INDICATED_FOR`、药品治疗事实或获批适应症；
- **不会**删除/重置/更新任何已有 Neo4j 节点、关系、Chroma/SQLite。

人工审核的角色改成**抽样检查工程质量和论文复现实验指标**，
不是逐条人工审核临床内容。数据本身不具备逐条权威来源与使用授权证明，
这一限制须在论文和演示中注明；不宜作为临床诊疗建议。

## 本包包含

1. `medical_p3044b_research_auto_link_v117.py`
   只读预检、20 条试点、断点续建、只读终验；
2. `test_medical_p3044b_research_auto_link_v117.py`
   28 项离线测试，使用假的 Neo4j 和虚构疾病样本；
3. `README_P3-04-4B_V117_research_auto_link.md` 本文档。

依赖原项目已有的 `medical_p3041_medical_json_prepare_v117.py`
（P3-04.3 疾病百科在线检索也依赖此文件，**不要删除**）。

**未打包或修改 `rag.py`，不包含任何自动删除或重置数据库功能。**

## 对你上传的真实候选的核查结果

真实文件 `p3044a_candidates_20261102_164523.jsonl`：
- 共 1,150 条；
- 1,150 个唯一 `disease_term_id` / `document_id` / `disease_name`；
- 全部 `PENDING_HUMAN_REVIEW`，即明确未经过医学审核；
- 全部 `third_party_unverified`、`treatment_relation=false`；
- 整份候选文件原始字节 SHA256：
  `c2010f693a0d2d37e7efd930d973cdd87750ad08a72b43c20a6d25b2a26dc731`；
- 用你上传的完整 `medical.json` 实际核对了全部 1,150 条的
  源 OID、行号、行哈希、疾病名称，结果全部通过。

使用一个**仅供离线测试的模拟 Chroma 审计数据**做输入契约检查，
不能把这个模拟结果当作真实 Chroma 验收。
你本地已有 P3-04.2 真实审计报告，本程序正式运行时必须使用它。

已通过 **28/28 项离线自动化测试**；
还用真实 1,150 条候选在假的 Neo4j 中演练了：
预览 1,150 缺失 → 先建 20 → 补建 1,130 →
全部 1,150 验收 → 再次重跑 0 新建。

**本工具尚未在用户本地真实 Neo4j 上执行，所以不能声明
真实 Neo4j 已经完成建图。**

## Windows 安装

解压 ZIP 内的三个文件到：

`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

请先对 Neo4j 做正常备份/快照，避免与其他可能正在写图的脚本并发运行。
保留 `rag.py`、`chroma_db`、原始 `medical.json`、现有 checkpoint、
所有旧报告，**不要清空或初始化数据库**。

PowerShell：

```powershell
cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"
$py = "E:\Miniconda\envs\rag\python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$raw = ".\data\raw\drug_db\MedGraphRAG-Drug-DB\medical.json"

$plan = ".\stage4_reports\p3044a_candidates_20261102_164523.jsonl"
$planReport = ".\stage4_reports\p3044a_graph_plan_20261102_164523.json"

$audit = ".\stage4_reports\p3042_final_audit_20261102_134326.json"

Test-Path $raw
Test-Path $plan
Test-Path $planReport
Test-Path $audit
```

四个路径都必须为 `True`。这里引用的具体时间戳已经来自你上传的
P3-04.4A 报告，不是随意猜测；如你实际目录存在版本差异，
不要直接用其他历史文件顶替，否则哈希/来源校验会拒绝。

## 第一步：离线测试（不连接 Neo4j）

```powershell
& $py -m unittest -v test_medical_p3044b_research_auto_link_v117.py
```

预期：

```text
Ran 28 tests in ...
OK
```

## 第二步：校验源文件与候选（完全离线，无数据库操作）

```powershell
$base = @(
  ".\medical_p3044b_research_auto_link_v117.py"
  "--source", $raw
  "--plan", $plan
  "--plan-report", $planReport
  "--chroma-audit", $audit
)

& $py @base --offline-check
```

预期：

`[P3-04.4B] OFFLINE_INPUTS_VERIFIED_NO_DB`

这代表候选 SHA、源记录、来源状态、P3-04.2/P3-04.4A
审计身份等静态文件核对通过，不代表已经检查当前 Neo4j。

## 第三步：只读检查真实 Neo4j（默认命令）

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

& $py @base `
  --report ".\stage4_reports\p3044b_dryrun_$stamp.json"
```

预期首次运行：

```text
[P3-04.4B] DRY_RUN_GRAPH_CHECK_OK
Graph candidate count: 1150
Graph before: {'MISSING': 1150}
Graph after: {'MISSING': 1150}
New nodes: 0 new relationships: 0
```

如果某些文档已由这**同一份计划**创建并严格一致，
可能会显示 `EXISTING_EXACT`。发现任何 ID/属性/关系冲突会停止，
绝不自动覆盖。`DiseaseTerm` 数量不再是 1,150、或具体名称/ID
发生变化时，也会停止。

## 第四步：只创建 20 条实验关联

这一步会真的写入 Neo4j 新 `DiseaseDocument` 节点和来源关系。
确认它是你的研究环境，并已备份后执行：

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

& $py @base `
  --apply `
  --confirm-research-only RESEARCH_UNVERIFIED_ONLY `
  --max-new 20 `
  --batch-size 40 `
  --report ".\stage4_reports\p3044b_pilot_$stamp.json"
```

预期：
`PARTIAL_RESEARCH_GRAPH_LINKS`，
`new_nodes_this_run=20`，
`new_relationships_this_run=20`。

这个状态是**预期的 20 条试点**，不是失败。
你可以在 Neo4j Browser 只读查看（可选）：

```cypher
MATCH (t:DiseaseTerm)-[r:EXACT_NAME_MATCH_TO_UNVERIFIED_ENCYCLOPEDIA]->(d:DiseaseDocument)
RETURN t.name AS disease_name,
       d.id AS document_id,
       d.source_status AS source_status,
       d.clinical_verified AS clinical_verified,
       r.link_basis AS link_basis
LIMIT 20;
```

预期标签为 `third_party_unverified`，`clinical_verified=false`。

## 第五步：继续建立其余关联（可重复运行、无覆盖）

只有前一步正常，才执行：

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

& $py @base `
  --apply `
  --confirm-research-only RESEARCH_UNVERIFIED_ONLY `
  --max-new 1150 `
  --batch-size 40 `
  --report ".\stage4_reports\p3044b_full_$stamp.json"
```

程序会先识别已存在的 20 条并严格核验，
再只创建尚未存在的约 1,130 条。
小批次事务保证异常时当前批次回滚；
之前提交成功的批次保留，下次重复执行同样命令即可继续。
**绝不因重跑而自动更新已有节点或关系。**

若出现 `BLOCKED`，不要删除任何节点或清空数据库；
保留终端信息并对照原因。

## 第六步：只读最终图谱关联审计

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

& $py @base `
  --audit `
  --report ".\stage4_reports\p3044b_final_audit_$stamp.json"
```

通过时：

```
[P3-04.4B] FULL_GRAPH_LINK_AUDIT_OK
Graph after: {'EXISTING_EXACT': 1150}
New nodes: 0 new relationships: 0
```

完成后你将拥有：
- 1,150 个通过严格技术匹配的 DiseaseDocument 来源节点；
- 1,150 条精确同名、未经医学审查的来源关系；
- 7,656 个暂未对齐的疾病名称仍可以从 `disease_knowledge`
  进行独立百科检索；
- 原有 52,104 条药品 Chroma Chunk 和 P3-03B 药品疾病提及关系都不变。

注意：**现有 P3-04.3 在线百科检索不会自动改成以这些新 Neo4j
来源关系为入口。** 这次完成的是“图谱元数据关联”，而非再次变更
在线问答路由；后续可以再做疾病图谱/文档联合溯源实验。

## 科研论文建议报告的指标

- 8,806 个百科疾病名称中严格同名对齐：1,150，覆盖率约 13.06%；
- Neo4j 新增来源节点与关系数：各 1,150（完成后以实时审计为准）；
- 自动同名链接异常、重复创建和冲突拒绝次数；
- 原始 `medical.json` 对应关系的 OID/哈希完整性；
- GraphRAG 单跳疾病百科问答的证据准确率、命中率、延迟；
- 混合问答中疾病/药品来源区分与幻觉率。

**技术对齐成功≠临床医学正确。** 在实验论文中写清楚知识库
为第三方未独立审核语料、同名映射不能推断治疗适应症即可。
