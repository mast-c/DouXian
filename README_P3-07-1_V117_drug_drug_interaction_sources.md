# 豆馅医疗 GraphRAG V1.17 · P3-07.1 药物相互作用来源证据链

> 阶段定位：技术来源链与权威标签对照，不是自动医疗决策系统。
> 先用你已经测过的“华法林 + 布洛芬”案例做可复现的安全检索。

## 为什么需要这一阶段？

你之前提问：

> 服用华法林期间能不能吃布洛芬？

系统 Chroma 已返回 3 条布洛芬药品说明书 `drug_interactions`
原文，也通过了 Neo4j SourceRecord/Section 的技术溯源；
但最终 LLM 只用《布洛芬胶囊》的“双香豆素等抗凝药”片段，
自行补充“华法林属于抗凝药”这一桥接知识。

这属于**基于本地检索与 LLM 已有知识的多步推断**；
此前未出现能直接验证的
`华法林 -> 抗凝药类 -> 布洛芬` Neo4j 实体关系路径。

本阶段重点是：**不伪造图谱边，但显式展示每一层证据来自哪里**。

## 新能力

- 从问题中识别华法林和布洛芬（支持中文与英文名称、顺序互换）；
- 独立于旧路由检索准确的三个栏目：
  `drug_interactions`、`warnings`、`contraindications`；
- 仅执行固定的图谱来源路径：

  ```text
  (Drug)-[:HAS_RECORD]->(SourceRecord)-[:HAS_SECTION]->(Section)
  ```

  这条路径**确实是 2 条 Neo4j 边**；它不能被解释成两个药物
  之间存在 2 跳相互作用关系；
- 从返回的 Section 原文中做逐字匹配：直接点名另一个药物，
  或者只点名其药物类别，同时要求风险关键词（出血等）；
- 返回证据 `quote` 的源文本 SHA256 和原图谱节点 ID；
- 单独附上两条美国国家医学图书馆 DailyMed 的官方标签引用，
  供前端跳转查看。它们在本阶段开发时进行了人工在线核对，
  **本地脚本不联网、不实时核验标签版本**；
- 不把 `Section` 到 `DrugClass`、`Warfarin` 到 `Ibuprofen`
  的文字关联伪造成 Neo4j 关系，也不会执行写入；
- 在你的原 `豆馅.py` 的“实际命中的知识库证据”折叠区底部
  追加独立的“⚠ 药物相互作用核验链”，不添加侧边栏页面；
- 本地原文证据的答案引用序号和 `retrieval_summary` 统计不变，
  旧的 5～8 跳来源路径卡片原样保留。

### 关于药学结论

官方 DailyMed Warfarin Sodium 的 7.3 节将 ibuprofen 列为 NSAIDs
中会增加华法林伴随出血风险的药物；官方 Ibuprofen 标签也说明
warfarin+NSAIDs 的严重胃肠道出血危险增加：

- https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=541c9a70-adaf-4ef3-94ba-ad4e70dfa057
- https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=404df13d-4ada-4324-9eed-e21f5d5043a4

**不要自行联用这两类药**；如果实际患者已经服用，
应及时联系医生/药师确认方案，不要自行停用处方华法林。
出现严重或持续出血、呕血、黑便等需要及时就医。

这里的官方标签是美国监管语境的具体产品资料。
给国内真实患者使用时，仍需要核对适用的最新官方说明书和临床规范。
`clinical_review_performed` 保持 `false`。

## 包含的文件

1. `medical_p3071_drug_interaction_v117.py`：
   两药识别、固定只读 Neo4j 查询、原文比对、官方参考网址、来源链报告；
2. `medical_p3071_interaction_preflight_v117.py`：
   单题 CLI 真实图谱预检（JSON 以独占模式保存，不能覆盖旧报告）；
3. `medical_p3071_interaction_ui_v117.py`：
   与聊天原“证据栏”相同 HTML 区域的展示模块，默认关闭；
4. `medical_p3071_patch_douxian_v117.py`：
   针对你 P3-06.3C 后的真实 `豆馅.py` AST 定位、自动备份/回滚；
5. `test_medical_p3071_interaction_v117.py`：
   39 项纯模拟 Neo4j 与 UI 测试；
6. `README_P3-07-1_V117_drug_drug_interaction_sources.md`：
   本中文指南；
7. `P3-07-1_V117_delivery_manifest.json`：
   SHA256 清单、真实图谱验证状态。

**没有**独立的 Streamlit `pages` 页面，
不会让左侧再出现英文的 `Multihop Evidence`。

## 一、安装目录和离线测试

解压到：
`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

```powershell
cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"
$py = "E:\Miniconda\envs\rag\python.exe"
$env:PYTHONUTF8 = "1"

Expand-Archive `
  -LiteralPath ".\medical_P3-07-1_V117_drug_interaction_provenance.zip" `
  -DestinationPath "." -Force

& $py -m unittest -v test_medical_p3071_interaction_v117.py
```

预期：

`Ran 39 tests ... OK`

这里使用模拟 Drug/SourceRecord/Section 节点；
通过离线单测**不等于**证明你的真实 Neo4j
里已存在直接的华法林—布洛芬相互作用边。

## 二、先在真实 Neo4j 上只读预检

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

& $py .\medical_p3071_interaction_preflight_v117.py `
  --root . `
  --question "服用华法林期间能不能吃布洛芬？" `
  --max-per-drug 6 `
  --report ".\stage4_reports\p3071_warfarin_ibuprofen_$stamp.json"
```

可观察：

- `Local graph source paths`：当前 Neo4j 真实的2跳来源记录数；
- `Literal mention types`：`DRUG_CLASS_PLUS_RISK_IN_LEAFLET`
  或 `DIRECT_OTHER_DRUG_PLUS_RISK_IN_LEAFLET`；
- `Official label reference URLs`：默认两条、只是引用指针；
- `Clinical evidence independently reviewed here: NO`：医学评审尚未进行。

如果返回 `GRAPH_UNAVAILABLE`，不说明相互作用无风险；
只表示本机 Neo4j 当前未提供合法的来源章节。检查 .env、
驱动、连接状态后再试，别补造图谱边。

如果返回 `LOCAL_CLASS_RISK_MENTION_FOUND`，
说明第三方药品说明书提到了**抗凝药类别和出血风险**；
这不是直接的华法林—布洛芬关系边。

## 三、接入原证据区（默认先预览）

你之前的 `豆馅.py` 可能已经经过 P3-06.3B/C 热修复，
所以**不要解压另一个全量 `豆馅.py` 直接覆盖当前文件**。
本包只有“增量安全补丁”。

先停止 Streamlit，在项目根目录：

```powershell
& $py .\medical_p3071_patch_douxian_v117.py --target .\豆馅.py
```

预期 `[P3-07.1] READY_TO_APPLY`。

只有通过 dry-run 后：

```powershell
& $py .\medical_p3071_patch_douxian_v117.py `
  --target .\豆馅.py --apply

if ($LASTEXITCODE -ne 0) {
    throw "P3-07.1 前端补丁安装失败"
}

& $py -m py_compile .\豆馅.py
```

自动生成 `豆馅.py.pre_p3071_时间戳.bak`。
**不修改** `rag.py`、`.env` 或你的医学数据库。

## 四、启用独立相互作用证据面板

在 `.env` 添加且只添加一次：

```dotenv
MEDICAL_P3071_INTERACTION_ENABLED=1
```

P3-06.3 的 `MEDICAL_P3063_*` 和 P3-04.5 图谱审计
配置全部保留，不必改动。

使用原来的 Streamlit 命令重启：

```powershell
& $py -m streamlit run ".\豆馅.py"
```

新开一轮提问：

> 服用华法林期间能不能吃布洛芬？

展开原有的“医疗检索完成”证据区，应该看到：

```text
实际命中的知识库证据（原有数量不变）
[1] 布洛芬胶囊 · 药物相互作用
[2] 布洛芬片 · 药物相互作用
...
⚠ 药物相互作用核验链 · 科研展示 / 非自动临床判定

原文 P1 · 布洛芬片 · drug_interactions · DRUG_CLASS_PLUS_RISK_IN_LEAFLET
Drug -> HAS_RECORD -> SourceRecord -> HAS_SECTION -> Section
引用原文中“抗凝药/双香豆素/出血倾向”等准确片段

权威标签引用（外部核实）
- DailyMed · Warfarin Sodium
- DailyMed · Ibuprofen
```

UI 的这些信息是**独立来源解释面板**，不等于答复 LLM
已经自动使用上述链条完成了可审计的图谱药物相互作用推理。

如果没有新增核验链，先检查：
- `.env` 变量是否为 `1`，以及是否有重复同名配置；
- 是否新开问题，历史会话不会自动触发新一次证据链构建；
- 真正的 Neo4j 源数据是否具有 `Drug / SourceRecord / Section`；
- `stage4_reports/p3071_warfarin_ibuprofen_*.json` 的 `status`；
- `豆馅.py` 的 P3-07.1 AST 补丁是否真正 `APPLIED`。

## 五、医疗 GraphRAG 研究中的严谨口径

不能写成：

> “系统从华法林经过抗凝药类别跳到布洛芬，证明了真实4跳交互关系。”

应写成：

> “系统只读检索到药品制剂的 `Drug→SourceRecord→Section`
> **2跳图谱来源链**；以该 Section 中字面出现的药物类别及风险词为
> 文本线索，利用官方标签给出的**直接药物相互作用警告**做外部证据对照。”

当前原图谱没有作为本阶段目标而验证成功的
`Drug)-[:BELONGS_TO_CLASS]->(DrugClass)`、`(:DrugClass)-[:INTERACTS_WITH_CLASS]->...`
或 `(:Drug)-[:INTERACTS_WITH]->(:Drug)`，本包没有创建这些边。

这非常适合论文中“真实可核验图谱路径 vs LLM 推测路径”的消融实验。
可以把评测分为：
1. 是否正确识别双药；
2. 是否正确命中相互作用原文栏目；
3. 是否找到两个药物的直接提及或药物类别提及；
4. 是否能给出来源节点与准确文字片段；
5. 是否对照官方来源，并标记来源的司法辖区、版本、可靠性；
6. 是否错误地画出一个没有在 Neo4j 中出现的 INTERACTS_WITH 边。

## 六、范围限制与后续阶段

**P3-07.1 只支持“华法林×布洛芬”这个经过人工查看的研究样例。**
其他药物对不会因为正好含“抗凝药”等词被自动判定为安全或危险。
不能把 `OUT_OF_SCOPE_PAIR` 解释成“无相互作用”。

本阶段**没有改变模型回答生成与医疗证据评分**：
来源链用于前端技术审计，目的是先确保“真的找到了什么”。
随后再研究 P3-07.2：经权威文档核验的相互作用证据如何安全、
可解释地进入多步问答；不能直接把关键词或未审核图谱边注入模型。

## 七、回滚

**临时关闭功能**：
```dotenv
MEDICAL_P3071_INTERACTION_ENABLED=0
```

重启 Streamlit 即可。

**恢复原来主前端**（先停止 Streamlit）：

```powershell
$backup = Get-ChildItem . -Filter "豆馅.py.pre_p3071_*.bak" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $backup) { throw "没有找到 P3-07.1 安装前备份" }

Copy-Item $backup.FullName ".\豆馅.py" -Force
& $py -m py_compile ".\豆馅.py"
```

数据库与 rag.py 不需要回滚，因为这个阶段没有更改它们。

## 质量保证与实际运行限制

交付验证：
- **39/39** 项离线模拟与 AST 补丁单元测试通过；
- 在上一阶段 P3-06.3C 交付的豆馅.py 完整源文件中，
  已验证增量注入的位置与 Python AST 结构；
- 旧的原文证据卡片计数和多跳 HTML 区域不被重写；
- Neo4j Session.run(Query(...)) + READ_ACCESS 驱动契约测试通过；
- 不提供任何图谱写入代码、不给未审核的医疗关系升级为临床真实事实。

**尚未远程连接你的 Windows Neo4j，也没有验收你本机 Streamlit 运行效果。**
一切“真实本地图谱命中数”需要以你执行的只读预检报告为准。
