# 豆馅 V2.16 — Graph Assist 的关系驱动检索规划（离线先行）

## 你提出的问题

新增第二本《机器学习》以后，词项匹配和候选覆盖虽然增加，
但 V2.14 / V2.15.1 的实测显示新增 Chunk 未必是重要的回答证据。
真正需要改进的是 **Assist 如何识别问题的证据需求、
应该沿哪条有方向的关系继续检索，以及何时不扩展。**

本次做的是 **可运行的独立 V2.16 Graph Assist Planner**，
不是更改 V2.8 里几个正则词，也不是简单合并全部图谱候选。
它已经在真实的旧 29 条 promoted 关系与新 54 条待审关系数据上通过
完全离线的 34 项回归/保护测试。

**最重要的限制：**
- 不修改 `rag.py` 或线上 `graph_assist_v28.py`。
- 当前 `DOUXIAN_GRAPH_V213=1` 时，原 RAG 跳过 V2.8 Assist；
  因此安装本包**不会立即改变在线 Graph ON 问答**。
- 本实验先验证规划合理性，再验证目标 Chunk 被 Hybrid 漏掉/可否
  通过定向检索找回，最后才能考虑实验性在线 A/B 和 Qwen Reranker。
- 29 条 V2.12 已正式 verified 关系可在 verified-only 模式检索。
- V2.15 新增 54 条关系 **全部没有被晋升为 verified**。
  显式 `--explore-pending` 仅表示「离线研究候选」。
- 新关系中的高风险别名/分类/范围限定已按 V2.15.1 原有语义
  筛查文件严格过滤（SHA256 冻结）；`ALSO_CALLED` 和
  `CONDITIONAL_FORM_OF` 等待独立复核，不能直接加入规划推断。

## V2.16 具体新增什么能力？

### 1. 问题意图识别

按优先级识别：理论依据、工作机制、组成结构、类别层级、
有明确直接关系的“关系”问题。对以下情况默认 **零扩展**：

- 定义问题：`吉布斯采样是什么？`
- 缺乏比较所需事实的题目：`随机森林与 Bagging 有什么区别？`
- 识别不到正式图谱概念的问题：例如天文学问题
- 超长问题 / 无子查询 / 明确设置 0 预算

对定向理论问题：
`吉布斯采样的采样机制依赖哪类随机过程？`
识别 `theoretical_basis`，利用
`吉布斯采样 --IS_A--> MCMC --BASED_ON--> 马尔可夫链`
给出独立检索分支 `MCMC 马尔可夫链`。

### 2. 有方向的关系路径

按意图使用不同的严格白名单，例如：

- 理论基础：`IS_A -> BASED_ON`，`BASED_ON`
- 运作机制：`HAS_PART -> USES`，`IS_A -> BASED_ON`，`USES`
- 组成：`HAS_PART`
- 层级：`IS_A`，`SPECIAL_CASE_OF`

不能逆向跨越 `IS_A`、`USES` 来虚构知识。
`ALSO_CALLED -> USES` 则明确标记为
**正式别名导航 + 一条语义关系**，不夸大为两条独立因果关系。

### 3. 独立查询，而不是给原问题追加术语

`plan()` 保持原有 `original_subqueries` 原样返回，另在
`branches` 中输出独立查询和真实原生 Chroma Chunk ID。
每个分支包含：

- `query`：例如 `MCMC 马尔可夫链`
- `path_labels` / `path_predicates`：方向和原来的两条边
- `edges[]`：每条边的原文、教材、章节、行号、Claim ID、Chunk ID
- `target_chunk_id`：此次计划寻找的第二跳原生 Chroma ID
- `source_risk`：旧正式 VERIFIED，还是新教材待审
- `path_kind` / `intermediate_join_type`：区分真正两跳和别名桥接
- `online_eligible=false`：本版本永远不直接接入生产

最多 2 个分支（可配置 0～3），并按 **最终证据 Chunk ID**
去重。两个不同关系得到相同原文 Chunk，不会重复支付 Embedding
检索成本。关系置信状态、方向、是否真正提出新检索词
参与路径筛选。

### 4. 原始 Hybrid Top 20 增量检查（可选）

实验工具并不直接把新 Chunk 塞给 Reranker：
1. 先使用原有 `KnowledgeBaseService.search_hybrid(query,k=20)`，
   不采用 Graph ON 包装器。
2. 使用原有 V2.13 Chroma 实时 ID/正文/source/section 完整性校验，
   验证规划目标 Chunk 的真实身份。
3. 如果目标 Chunk 已经在原 Hybrid Top20，
   或有相同正文+来源的替代 Chunk，**跳过定向分支**。
4. 只有明确使用 `--run-branches` 才允许对遗漏的目标执行
   额外的定向 Hybrid 查询，记录是否找回该 Chunk。
5. 不调用 Qwen Reranker、不生成答案、不更改生产查询分解。
   新增候选 **不等于** 更好答案。

## 安装位置

本 ZIP 已冻结你当前所有正式图谱来源和 SHA256，并且不会写旧目录
里的任何同名文件。它要求前面 V2.15、V2.15.1 的数据包已安装。

将 ZIP **原样解压** 到：

`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

应当能在该目录看到：

- `install_graph_assist_v216.py`
- `manifest_v216_assist.json`
- `v216_payload/`

不要直接覆盖/移动 `v216_payload` 中的原始文件。

### 第一步：安装、测试（不调用 API）

```powershell
conda activate rag
cd E:\VsCode_Project\DockerTest_Agent\RAG_project\data

python .\install_graph_assist_v216.py --check
python .\install_graph_assist_v216.py --install

python -m unittest graph_tools.test_graph_assist_planner_v216 graph_tools.test_eval_graph_assist_v216 -v
```

### 第二步：先运行正式 29 条关系的零 API 实验

```powershell
python -m graph_tools.eval_graph_assist_v216 --offline --out .\graph_reports\assist_v216\v216_offline_verified_run01.json
```

也可以对一个真实问题单独观察检索计划：

```powershell
python -m graph_tools.graph_assist_planner_v216 --question "吉布斯采样的采样机制依赖哪类随机过程？" --out .\graph_reports\assist_v216\v216_single_gibbs_run01.json
```

### 第三步：新教材关系离线探索（可选；零 API）

```powershell
python -m graph_tools.eval_graph_assist_v216 --offline --explore-pending --out .\graph_reports\assist_v216\v216_offline_multibook_run01.json
```

这一步允许 V2.15 的 *部分待审关系* 以探索性身份出现在
`branches` 中，同时保留待审标签和原始证据。
不转为 VERIFIED、不写进旧图谱、不修改在线 RAG。

## 现有离线回归的真实结果（不是回答精度）

| 指标 | V2.16 VERIFIED ONLY | V2.16 研究模式 |
|---|---:|---:|
| 固定测试问题 | 11 | 11 |
| 规划出分支的问题 | 3 | 6 |
| 独立检索分支 | 4 | 7 |
| 负例误触发 | 0 | 0 |
| Qwen API 调用 | 0 | 0 |

旧版 V2.8 的 `RELATION_INTENT` *正则条件本身*只匹配
这 11 道题中的 1 道：一个图谱外的分类问题。
**注意这只是「正则门禁」比较，绝不是 V2.8 实际检索
召回率的完整 A/B 对照。**

固定 11 道题是围绕现有关系人工设计的小规模诊断样本，
不能用上述比例宣传为 GraphRAG 召回率提升。

## 最后：可选的真实 Hybrid 对照（需要查询 Embedding API）

先只检验是否 Hybrid 本来就检索到目标证据，不额外执行分支：

```powershell
python -m graph_tools.eval_graph_assist_v216 --live --case gibbs_random_process --out .\graph_reports\assist_v216\v216_live_baseline_run01.json
```

确认计划正常后，如需检查“定向检索能否找回 Hybrid 漏掉的
目标 Chunk”，显式追加 `--run-branches`：

```powershell
python -m graph_tools.eval_graph_assist_v216 --live --case gibbs_random_process --run-branches --out .\graph_reports\assist_v216\v216_live_branch_run01.json
```

也可以先探索跨教材 MH->MCMC->马尔可夫链，但新教材
关系仍须独立人工语义复核：

```powershell
python -m graph_tools.eval_graph_assist_v216 --live --case mh_crossbook --explore-pending --run-branches --out .\graph_reports\assist_v216\v216_live_crossbook_run01.json
```

注意：
- 以上 live 均可能进行付费 Embedding 查询；
- 仅当目标 Chunk 缺席原 Hybrid Top20 时才真正运行分支查询；
- 每次 `--live` 最多选择 3 道固定题，不生成答案；
- 即使额外的关键词查询找回目标，也只证明**候选召回**，
  不证明其相关性、Reranker Top3 存活或最终答案质量；
- 不要连续使用相同 `--out` 路径，报告采用独占新建。

## 交付文件

本增量包只添加：
- `graph_tools/graph_assist_planner_v216.py`
- `graph_tools/eval_graph_assist_v216.py`
- `graph_tools/test_graph_assist_planner_v216.py`
- `graph_tools/test_eval_graph_assist_v216.py`
- `graph_reports/assist_v216/fixed_eval_cases_v216.json`

安装器还生成独立安装凭据
`graph_reports/assist_v216/v216_assist_install_*.json`。

要安全卸载：

```powershell
python .\install_graph_assist_v216.py --uninstall "安装时显示的完整凭据 JSON 路径"
```

只能删除此安装器**新建且从未修改**的文件。
所有已存在的图谱、RAG、Chroma 和旧版 Assist 永不改动。

## 本阶段通过门禁的含义

- 34 项离线单元测试（方向、合法路径、语义风险、证据绑定、
  定义和比较负例、英文边界、去重、冻结哈希、原始查询不变）；
- 按真实 V2.12 / V2.15 数据，生成 11 题真实离线报告；
- 不声称检索收益：仍需你本地运行 live 报告；
- 不上线改动：后续只有当「确有 Hybrid Top20 遗漏」、
  「目标 Chunk 与问题有关」、「定向查询找回」、
  「Qwen Reranker 最终使用」等条件满足后，再做有开关的在线 A/B。
