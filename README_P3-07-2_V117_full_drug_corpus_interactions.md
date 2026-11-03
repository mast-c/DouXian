# 豆馅医疗 GraphRAG V1.17 · P3-07.2 全库药物相互作用来源检索

## 阶段目标

把 P3-07.1 只支持“华法林＋布洛芬”的单题研究代码，
扩展为**任意两种现有 Neo4j 药品库中的药品**的通用来源检索。

无需为每种药物组合手工编码，也**不采用** `N×N` 的全组合预先建边。

本阶段覆盖范围准确表述为：

> 枚举现有 Neo4j 中全部有有效 `Drug.id` 与 `Drug.name` 的药品节点，
> 为其在 `SourceRecord` / `Section` 中的 `drug_interactions`、
> `warnings`、`contraindications` 三个安全栏目建立独立的
> 可重建 SQLite 派生检索索引；输入两种库内药品后查双方的上述
> 原文章节，提取**直接药名提及**、有证据链接的人工分类词提及，
> 以及无分类映射的**待核验类别共现**。

不等于全世界所有药物、全部上市产品、最新官方说明书或所有可能相互作用。

**特别强调：你已有的 52,104 条 Chroma 药品 chunk
是另一种数据颗粒度，不等于 Neo4j `Section` 数。
本阶段从全部 Neo4j 可识别 Drug/Section 构建索引，
并提供数量覆盖率；不能仅凭 52,104 与 SQLite 的 Section 数
不同，就判定数据丢失，也不能声称已把所有 Chroma-only
历史 chunk 重新建为真实 Neo4j 来源边。**

## 一、交付内容

| 文件 | 作用 |
|---|---|
| `medical_p3072_corpus_index_v117.py` | Neo4j `MATCH/RETURN` 全库游标扫描；创建单独 SQLite 派生索引；统计完整性/跳过项/覆盖率；无 Neo4j 写入 |
| `medical_p3072_pair_retrieval_v117.py` | 任意双药名称解析、双向 Section 原文检索、风险词邻近度、来源链与可选人工分类链接 |
| `medical_p3072_cli_v117.py` | `build` / `audit` / `inspect` 三种命令 |
| `medical_p3072_interaction_ui_v117.py` | 直接显示在现有 `豆馅.py` 原来的“实际命中的知识库证据”折叠区；**没有独立左侧页面** |
| `medical_p3072_patch_douxian_v117.py` | 只在当前前端追加2个钩子，dry-run、自动备份、AST 验证、失败拒绝覆盖 |
| `p3072_curated_drug_classes.example.json` | 默认空的人工药物分类证据模板 |
| `test_medical_p3072_full_corpus_v117.py` | 48 项离线模拟数据库、索引构建、任意双药检索、界面安全与备份测试 |
| `P3-07-2_V117_delivery_manifest.json` | 文件哈希与验收说明 |

只通过真实已有的：

```text
Drug -[:HAS_RECORD]-> SourceRecord -[:HAS_SECTION]-> Section
```

两条 Neo4j 边，**绝不自动创建** `INTERACTS_WITH`、`TREATS`、
`BELONGS_TO_CLASS`、`CONTRAINDICATED_WITH` 等新边。

**药物名称的剂型/盐形式归一化是查找技术，不等于已由药学专家验证
不同制剂、成分、剂量和适用人群完全等价。**

## 二、部署目录

项目根目录：

`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

下载 ZIP 放入该目录，在 PowerShell 执行：

```powershell
cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"
$py = "E:\Miniconda\envs\rag\python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Expand-Archive `
  -LiteralPath ".\medical_P3-07-2_V117_full_drug_corpus_interactions.zip" `
  -DestinationPath "." -Force

& $py -m unittest -v test_medical_p3072_full_corpus_v117.py
```

预期：

`Ran 48 tests ... OK`

这些单元测试使用**虚构 Neo4j 节点和说明书**；并不说明你真实
数据库的全部药品已经索引，真实覆盖必须由下一步构建输出证明。

## 三、扫描全库，建立一次性派生索引

先确认 `.env` 有你之前 P3-06.1 / P3-04.5 使用的：

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你自己的实际密码
NEO4J_DATABASE=neo4j
```

上面是配置字段示例，已存在且可用就不要覆盖；也不要把密码上传到聊天。

启动本地 Neo4j 后运行：

```powershell
& $py .\medical_p3072_cli_v117.py build `
  --root . `
  --index ".\stage4_reports\p3072_full_drug_interaction_index.sqlite" `
  --batch 256
```

特点：

- **Neo4j 只读**：全部是固定的 `MATCH` / `RETURN` 查询。
  使用 `Session.run(Query(...))` + `READ_ACCESS`，避免你之前遇到的
  `Query object is only supported for session.run` 报错；
- 构建只写入本地派生文件 `p3072_*.sqlite`，
  不会修改 Neo4j、Chroma、medical.json、rag.py；
- 先在同文件夹创建临时 SQLite，只有完整扫描、索引和校验成功后
  才原子发布，故不会把中断的扫描当成完整成功；
- 不允许覆盖已有索引。未来更新 Neo4j 后要重建，
  请显式加 `--replace`，旧派生索引会自动单独备份；
- 如 Neo4j 同时有人正在修改药品节点、扫描游标乱序、
  章节过多超出阶段上限等，系统会拒绝把部分扫描称为“全库完整”。

正常完成会打印：

```text
[P3-07.2] FULL_CORPUS_INDEX_BUILT
Drug nodes: [实际数量]
Drug nodes with safety Sections: [实际数量]
Safety Sections indexed: [实际数量]
Drug coverage %: [实际百分比]
Overlong Sections not indexed: [实际数量]
...
```

**计数全部以你实际 Neo4j 为准，不预填或臆测。**

## 四、执行全库索引覆盖审计

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

& $py .\medical_p3072_cli_v117.py audit `
  --root . `
  --index ".\stage4_reports\p3072_full_drug_interaction_index.sqlite" `
  --report ".\stage4_reports\p3072_full_audit_$stamp.json"
```

正常：

`[P3-07.2] FULL_CORPUS_INDEX_AUDIT_OK`

报告包括药品总数、具备安全原文章节药品数、章节和候选别名数、
不合规范的节点/超过长度限制的 Section 以及索引创建时间。

没有对应安全栏目的药品会**保留在药品目录里**，
不会凭空产生一条“安全同服”的医学结论。

## 五、测试任意双药（不是为每组写一次代码）

**案例 A：你已经测试过的组合**

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

& $py .\medical_p3072_cli_v117.py inspect `
  --root . `
  --question "服用华法林期间能不能吃布洛芬？" `
  --report ".\stage4_reports\p3072_pair_warfarin_ibuprofen_$stamp.json"
```

**案例 B：更换任意两种你本机存在的药品**

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

& $py .\medical_p3072_cli_v117.py inspect `
  --root . `
  --question "氯雷他定和左氧氟沙星能一起服用吗？" `
  --report ".\stage4_reports\p3072_pair_anytwo_$stamp.json"
```

第二个组合仅用于验证**检索器是否支持替换实体**，
**不代表我们已经核验该组合具有临床相互作用**。

**案例 C：再换其他库内药品**

例如：

> 阿司匹林与布洛芬能否一起服用？

前提是两种药品的名称或可匹配的剂型名称确实进入了你的索引。
系统不会为未收录的药物编造关系。

`inspect` 输出的重要项目：

- `EXACTLY_TWO_CATALOG_GROUPS_DETECTED`：从目录查出两个名称组；
- `LOCAL_DIRECT_MENTION_WITH_WARNING_WORDS`：某药原文直接提到另一种药和邻近的警告词（字面线索，非临床事实核验）；
- `LOCAL_REVIEWED_CLASS_TEXT_BRIDGE`：另有人工审核及来源说明的类别映射，跨源文字桥接（**仍不是 Neo4j 边**）；
- `LOCAL_UNRESOLVED_CLASS_WARNING_ONLY`：药品说明书提及药物类别警告，但是**尚无证据**将另一药归入该类；
- `NO_PAIR_SPECIFIC_EVIDENCE_FOUND`：本地这三个安全栏目没有发现药物对的相关证据，**绝不代表同服安全**；
- `PARTIAL_COVERAGE_QUERY_BUDGET`：每药扫描的章节超过预算，结果**不是穷举**；
- `AMBIGUOUS_CATALOG_ALIAS`：药名映射不唯一，不自动确定配对；
- `THREE_OR_MORE_DRUGS_NOT_A_PAIR`：提问涉及3种及以上药品，本阶段不擅自抽选两种。

报告中的 `actual_graph_hops=2` 是**来源图谱的跳数**，
不能当成“Drug A → Drug B 之间存在2跳关系”。

## 六、将全库结果显示在原前端证据栏

先停止 Streamlit（`Ctrl+C`），然后**先做 dry-run**：

```powershell
& $py .\medical_p3072_patch_douxian_v117.py `
  --target .\豆馅.py
```

预期 `[P3-07.2] READY_TO_APPLY`。

确认无误后：

```powershell
& $py .\medical_p3072_patch_douxian_v117.py `
  --target .\豆馅.py --apply

if ($LASTEXITCODE -ne 0) {
    throw "P3-07.2 前端增量接入失败"
}

& $py -m py_compile .\豆馅.py
```

会创建 `豆馅.py.pre_p3072_*.bak` 自动备份。
这个补丁既能安装在 P3-06.3C 已修复版本，
也能共存于已安装 P3-07.1 的前端。

在 `.env` 添加：

```dotenv
MEDICAL_P3072_FULL_INTERACTION_ENABLED=1
MEDICAL_P3072_INDEX_PATH=stage4_reports/p3072_full_drug_interaction_index.sqlite

# 如果此前开启了 P3-07.1 仅华法林+布洛芬展示，关闭旧卡片避免重复
MEDICAL_P3071_INTERACTION_ENABLED=0
```

保留原 `MEDICAL_P3063_*` 多跳路径显示设置，
新的全库证据链会紧随原“实际命中的知识库证据”和
既有“多跳图谱来源路径”之后显示。

重启原前端：

```powershell
& $py -m streamlit run ".\豆馅.py"
```

新发一轮用户提问，例如：

> 服用华法林期间能不能吃布洛芬？

界面原有证据区增加：

```text
实际命中的知识库证据（原有条数不变）
    [1] 原始药品说明书 Section
    [2] ...

多跳图谱来源路径（如本轮有路径）
    P01 · 8跳 ...

🔎 全库药物相互作用来源检索
    双药目录识别：华法林 × 布洛芬
    本地章节覆盖：有安全章节的药品数/药品总数
    来源1 · 布洛芬片 / drug_interactions
       Drug -> HAS_RECORD -> SourceRecord -> HAS_SECTION -> Section
       原文候选："双香豆素等抗凝药 ... 增加出血倾向"
       【尚未证明华法林与此类别词具有图谱类别边】
    ...
```

注意不会让左侧再次出现英文独立多跳页面。

## 七、可选：人工审核的药物类别映射

`p3072_curated_drug_classes.example.json` 默认**没有任何条目**。
只有你确实核对过官方药品分类证据、记录审核人与网址，
才可以添加：

```json
{
  "schema_version": 1,
  "entries": [
    {
      "canonical_key": "此处填已人工核对的药名键",
      "class_terms": ["此处填经核对的类别术语"],
      "source_url": "https://对应的权威原始来源网址",
      "reviewed_by": "审核人或课题组代号",
      "status": "reviewed"
    }
  ]
}
```

这只是结构示例，**没有提供未经验证的具体药物分类对应关系**。

如果使用该文件：

```powershell
& $py .\medical_p3072_cli_v117.py inspect `
  --root . `
  --question "服用华法林期间能不能吃布洛芬？" `
  --curated-map ".\p3072_curated_drug_classes.example.json"
```

前端如需同样读取映射，在 `.env` 添加：

```dotenv
MEDICAL_P3072_CURATED_CLASS_MAP=p3072_curated_drug_classes.example.json
```

映射是一个**人工来源管理的文本桥接表**，
没有医学专家审核的映射不代表临床权威，
也不会自动写成图谱的 `BELONGS_TO_CLASS` 关系。

## 八、质量、边界与下一步

离线验收范围：

- 合成 Neo4j 中7个 Drug 节点、6个仿真结构 Section 的完整快照；
- 额外跨页扫描41个模拟药品节点和37个模拟Section，确保全库不漏尾页；
- 多药名、华法林钠/华法林、剂型归一化、两汉字药名；
- 另一组药物对的通用检索，不使用任何硬编码药物对；
- 三药提问、歧义、只有一药、未知药名等失败安全；
- 本地片段 SHA256、精确 Quote 溯源及全索引 SHA256 逐条审计；
- 查询预算截断、超长 Section / 无安全栏目覆盖报告；
- 不修改数据库、单独 SQLite 派生快照的备份与原子发布；
- 现有 `豆馅.py` 增量 AST 插入、Windows BOM/CRLF 保留和文件回滚；
- Python Neo4j Driver 的 `Session.run(Query)` 只读驱动约束测试。

**48/48 项离线测试通过，尚未访问你的真实 Windows Neo4j。**

绝对不要把这版报告写成：

> “系统已确认全库每一种药物组合的临床相互作用或用药安全”。

合理的表述是：

> “构建了覆盖当前 Neo4j 药品实体及指定安全说明书栏目的
> 全库候选相互作用检索器，支持任意两种已收录药品的原文来源
> 路径定位和类别警告证据分层；未核验的类别映射不得自动升级为药物相互作用边或临床建议。”

这一阶段**仍不修改 LLM 回答生成**：
模型若根据自身知识补全中间步骤，原有回答逻辑照旧；
新证据链帮助你辨别哪些步骤确实有数据库来源、
哪些只是跨文本假设。若之后要让模型安全利用所有来源链，
还需要有权威证据白名单和明确的引用策略。

## 九、关闭 / 重建 / 回滚

关闭全库功能：
```dotenv
MEDICAL_P3072_FULL_INTERACTION_ENABLED=0
```

Graph 药品资料修改后重建本地快照（旧快照自动备份）：

```powershell
& $py .\medical_p3072_cli_v117.py build `
  --root . `
  --index ".\stage4_reports\p3072_full_drug_interaction_index.sqlite" `
  --batch 256 --replace
```

完全撤销前端补丁（停止 Streamlit）：

```powershell
$backup=Get-ChildItem . -Filter "豆馅.py.pre_p3072_*.bak" -File |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $backup) { throw "找不到本次补丁前备份" }
Copy-Item $backup.FullName .\豆馅.py -Force
& $py -m py_compile .\豆馅.py
```

原 Neo4j、Chroma、`rag.py` 不需要回滚，因为本阶段没有改变它们。
