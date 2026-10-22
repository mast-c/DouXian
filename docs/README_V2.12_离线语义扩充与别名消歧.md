# 豆馅 V2.12｜离线语义关系扩充与实体别名消歧（增量、可回滚）

> 当前目录与常用命令见 [图谱目录与命令](README_图谱目录与命令.md)。已保存的真实快照复验结果为：原 12 条、新增 17 条、合计 29 条绑定有效；1 条新增候选因歧义未晋升。下文原始 ZIP 的开发测试记录是历史说明，当前无需重新安装。

## 0. 当前交付状态

本包基于 V2.11.1 已通过的 **12/12 真实 Chroma Chunk 绑定**，进行**离线、增量、严格证据约束**的图谱升级。

已经完成的本地开发验证：

- 已从与 V2.7 图谱 SHA256 完全一致的《神经网络与深度学习.md》提取 **18 条新增候选语义关系**；
- 候选涉及 **11 个新概念节点**（包括汇聚函数、马尔可夫随机场、马尔可夫网络、残差网络等）；
- 离线单元/回归测试 **21/21 通过**；
- 用原始教材 + 图谱 + **明确标记为模拟的 Chroma Chunk** 演练过 12+18=30 条关系的 `propose -> link -> promote -> verify` 全流程；
- 原始 ZIP 开发时尚未测试本机真实快照；当前本机快照结果为新增 17/18 通过，以保存的报告和 `verify` 为准。

本包**不会更改** `rag.py`、`豆馅.py`、`knowledge_base.py`、`config_data.py`、Chroma、两本教材、V2.7 图谱或 V2.11.1 现有绑定文件。

## 1. 历史包内四个文件

| 文件 | 用途 |
|---|---|
| `graph_expand_v212.py` | V2.12 命令行主程序：`propose` / `promote` / `verify` |
| `test_graph_expand_v212.py` | 21 项离线测试，包括旧绑定保留、证据篡改、歧义、拒绝覆盖和模拟 30 条全流程 |
| `V212_candidate_catalog_PREVIEW.json` | 基于原始教材已找到的 18 条候选关系（**不是已验证 Chroma 关系**） |
| `README_V2.12_离线语义扩充与别名消歧.md` | 本说明 |

**请不要用这些新脚本替换 `graph_chunk_linker_v211.py`**：V2.12 是旁路新增工具，V2.11.1 继续负责真实 Chroma 绑定。

## 2. Windows PowerShell 运行环境

请在 `E:\VsCode_Project\DockerTest_Agent\RAG_project\data` 目录操作，先关闭 Streamlit（非强制，但这样最容易保持快照一致）。

当前文件已经归类，无需解压旧版 ZIP 或重跑已清理的历史测试脚本。

```powershell
conda activate rag
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"

Test-Path .\graph_tools\graph_expand_v212.py
python -m graph_tools.graph_expand_v212 verify
```

历史离线单元测试为 21 项，相关测试脚本当前已清理。现有数据复验应打印基线 12 条、新增 17 条、合计 29 条；这与历史模拟测试成绩分别记录。

`graph_chunk_linker_v211.py` 必须是此前 **V2.11.1 已验收、支持教材标题根路径归一化**的版本。

## 3. 门禁 A：确认基线 12 条仍然有效（必做）

使用你本机已经通过 V2.11.1 验收的这组文件：

```powershell
python -m graph_tools.graph_chunk_linker_v211 verify `
  --graph ".\graph_data\knowledge_graph_v27.json" `
  --source ".\graph_data\sources\神经网络与深度学习.md" `
  --chunks ".\graph_bindings\chroma_chunks_v211.json" `
  --links ".\graph_bindings\graph_chunk_links_v2111.json"
```

**必须通过**。如果失败，暂停本次升级：不能用 V2.12 跳过旧版安全门禁。

## 4. 门禁 B：提出新增候选（不访问 Chroma）

```powershell
python -m graph_tools.graph_expand_v212 propose `
  --baseline-graph ".\graph_data\knowledge_graph_v27.json" `
  --source ".\graph_data\sources\神经网络与深度学习.md"
```

会生成两个全新文件：

- `graph_data/knowledge_graph_v212_candidates.json`：老图谱原封不动 + 18 条候选关系 + 11 个候选新概念；
- `graph_data/graph_v212_candidate_catalog.json`：原文章节、教材行号、证据片段和候选状态。

**注意：这里的全部 18 条只叫 `pending_real_chunk_binding`，暂不可用于线上问答。**

程序会拒绝覆盖已有输出。若试验过一次，请改输出名称或先把先前生成的文件归档，**不要加 `-Force` 去覆盖正式结果**。

## 5. 门禁 C：用真实 Chroma Chunk 为候选图谱绑定

这一步继续使用 V2.11.1 的**真正 `link` 命令**（不是 `build`）：

```powershell
python -m graph_tools.graph_chunk_linker_v211 link `
  --graph ".\graph_data\knowledge_graph_v212_candidates.json" `
  --source ".\graph_data\sources\神经网络与深度学习.md" `
  --chunks ".\graph_bindings\chroma_chunks_v211.json" `
  --links ".\graph_bindings\graph_chunk_links_v212_candidates.json" `
  --report ".\graph_reports\graph_integrity_report_v212_candidates.json"
```

查看候选绑定情况：

```powershell
(Get-Content ".\graph_reports\graph_integrity_report_v212_candidates.json" `
  -Raw -Encoding UTF8 | ConvertFrom-Json).status_counts
```

理论总计 **30 条候选参与测试**（原 12 + 新 18），但真实 `verified` 数量必须以本机输出为准。

> 如果新增关系出现 `unbound`、`ambiguous` 或 `section_conflict`，这是预期的严格失败路径，不能直接视为事实。`promote` 会将不满足证据条件的新关系排除，并在报告中逐条说明原因。旧 12 条如果不能保持原有唯一 Chunk ID，则 `promote` 整体拒绝执行。

## 6. 门禁 D：只提升已验证的新增关系

```powershell
python -m graph_tools.graph_expand_v212 promote
```

默认文件名与第 3–5 节一致。它会：

1. 核验教材 SHA256、基线 V2.7 图谱哈希以及候选图谱的内置精选规则。
2. 用 V2.11.1 重新计算原 12 条绑定并确认**原有真实 `chunk_id` 全部不变**。
3. 核验全部候选的真实 Chunk 快照、原文、章节、来源和唯一性。
4. 只把**真实 `verified`** 的新增关系纳入 `knowledge_graph_v212.json`。
5. 只保留被新增有效关系引用的概念节点，不生成悬空节点。
6. 重算最终图谱的 Chunk 绑定、生成**有原文证据的同义词组**和歧义检测。
7. 输出四个**不覆盖现有文件**的结果：

| 新文件 | 内容 |
|---|---|
| `graph_data/knowledge_graph_v212.json` | V2.12 正式图谱（只含基线+通过验证的新关系） |
| `graph_bindings/graph_chunk_links_v212.json` | 正式图谱的真实 Chunk 绑定与哈希 |
| `graph_data/graph_alias_registry_v212.json` | 从通过验证的 `ALSO_CALLED` 关系推导的同义词组 |
| `graph_reports/graph_v212_upgrade_report.json` | 新增候选统计、晋升统计、拒绝状态和快照哈希 |

如果 18 个新增候选没有一个能对应唯一真实 Chunk，则**不会输出“成功的 V2.12 正式图谱”**，只保留候选检查报告。

## 7. 门禁 E：正式独立复验

```powershell
python -m graph_tools.graph_expand_v212 verify
```

预期打印：

```text
[OK] V2.12 baseline verified: 12
[OK] V2.12 newly verified: N
[OK] V2.12 total verified: 12+N
[OK] No baseline Chunk ID changed
```

`N` **取决于你本机真实 Chroma 快照，最大是 18**，不能使用模拟环境中 18/18 的结果代替。

`verify` 同时重新核查最终图谱中的每条新增关系是否确实来自这次精选教材规则，避免只对 SHA256 做形式校验。

## 8. 实体别名规则（本版不破坏现有概念 ID）

- 来自 `ALSO_CALLED` 且状态 `verified` 的关系，进入 `verified_synonym_groups`，每组可追溯到已绑定的 Claim ID。
- V2.7 原有 `aliases_for_search`（如“汇聚层”对“池化层”的搜索别名）仍只作为 `search_only_hints`，**不能仅凭搜索别名就宣称实体相同**。
- 如果同一个词同时指向多个不等价的概念组，则记录 `ambiguous_terms`，不自动跨组合并。
- 不修改旧版节点 ID 和原始关系，保留图谱溯源与引用可回滚性。

## 9. 验收时应该看什么

```powershell
$report = Get-Content ".\graph_reports\graph_v212_upgrade_report.json" -Raw -Encoding UTF8 |
    ConvertFrom-Json
$report | Select-Object baseline_verified, candidate_new_count, `
    new_verified, new_not_promoted, final_verified, `
    alias_verified_groups, alias_ambiguous_terms
```

验收条件：
- `baseline_verified = 12`；
- `new_verified >= 1`，且 `final_verified = 12 + new_verified`；
- 旧版基线 `chunk_id` 不变；
- `verify` 通过；
- 所有未晋升候选在报告中可逐条复盘。

**只有第 5–7 节在你本机真实 `chroma_chunks_v211.json` 上执行通过，才算 V2.12 正式验收。**

## 10. 回滚与安全

无需修改或回滚任何旧版 Python 文件。线上问答的 `rag.py`、Fast/Strict 模式、Chroma 持久化目录以及两本教材都不受本版工具影响。

不想使用 V2.12 时，只要让线上系统继续使用旧版 `knowledge_graph_v27.json` 和 `graph_chunk_links_v2111.json` 即可。
若要清理 V2.12，先把本版新增的 `knowledge_graph_v212*.json`、`graph_chunk_links_v212*.json`、`graph_v212*.json`、`graph_alias_registry_v212.json` 归档。

**请保留这些验收基础文件；当前归类位置如下：**

- `graph_data/knowledge_graph_v27.json`
- `graph_data/sources/神经网络与深度学习.md`
- `graph_bindings/chroma_chunks_v211.json`
- `graph_bindings/graph_chunk_links_v2111.json`
- `graph_tools/graph_chunk_linker_v211.py`

本次只做**第一本教材**的同源图谱扩充。第二本《机器学习》需要独立的来源哈希、章节命名空间和证据链；不能直接把第二本的 Claim 塞进仅允许第一本原教材 SHA 的 V2.11.1 图谱中。

## 11. 下一步反馈

请优先贴出以下输出，或者上传文件：

- 第 5 节 `graph_integrity_report_v212_candidates.json` 中的 `status_counts`。
- 第 6 节 `graph_v212_upgrade_report.json`（如果成功生成）。
- 第 7 节 `verify` 终端输出。

不需要上传完整的 `chroma_chunks_v211.json`（里面包含教材原文）。

## 12. 已知不在本版范围内

- 尚未让知识图谱参与线上 Hybrid RAG 召回，因此**不要**把问答质量变化归因于 V2.12。
- B04 回答中 `[svg]` 本地链接、引用编号自相矛盾、公式排版问题属于独立的答案呈现修复任务，不与图谱变量混合修改。
- 不引入 Neo4j、无需 DashScope API 请求、无需重新 Embedding。
