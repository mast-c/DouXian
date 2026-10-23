# 豆馅 DouXian｜V2.13 可信图谱参与在线 Hybrid RAG 检索

## 0. 当前交付的边界

**这是可安装、可回滚的 V2.13 Graph ON/OFF 实验补丁，不是已经完成的真实业务 A/B 验收。**

依据用户上传的最新三个文件定制：

- `data/rag.py`：知识库查询主链，包含 `search_with_rerank` 调用；
- `data/graph_tools/graph_assist_v28.py`：V2.8 旧的图谱扩词功能，默认指向 V2.7；
- `data/graph_tools/graph_paths.py`：四层目录的统一路径常量。

另外，检索接入点依据当前可检查的 `KnowledgeBaseService` 接口设计；安装时 V2.13 会**动态检查当前** `KnowledgeBaseService.search_with_rerank` 是否确实调用 `self.search_hybrid(...)`，不符合时自动拒绝图谱适配，回退原有 Hybrid。

已经通过 **19 个离线单元测试**，使用模拟的图谱、Chroma GET 和 Reranker 检查：图谱注入发生在已有 Reranker 之前、同一条检索只调用一次原 Reranker、真实 Chunk 内容核验、坏哈希/过期快照/不可信关系的回退行为、英文术语与中文别名等。

**没有读取你电脑的 V2.12 正式 29 条关系和 Chroma 数据，也没有调用在线 Qwen，因此上线与效果必须在你本机验收。**

## 1. ZIP 文件放置位置

这个交付包以 `data/` 作为解压根目录：

```text
RAG_project/
└── data/
    ├── rag.py                         # 已有，不会在解压时被覆盖
    ├── install_graph_v213.py          # 新增：安全安装脚本
    ├── README_RAG_V2.13_GraphRAG_ON_OFF.md
    ├── graph_tools/
    │   ├── graph_paths.py             # 已有：保持原样
    │   ├── graph_chunk_linker_v211.py # 已有：保持原样
    │   ├── graph_assist_v28.py        # 已有：保持原样
    │   ├── graph_retriever_v213.py    # 新增
    │   └── test_graph_retriever_v213.py # 新增
    ├── v213_payload/
    │   └── rag.py                     # 审查过的目标版本；由安全安装程序使用
    ├── rag_v213.diff                  # 单独审查主链代码差异
    ├── graph_data/
    │   ├── knowledge_graph_v212.json
    │   ├── graph_alias_registry_v212.json
    │   └── sources/神经网络与深度学习.md
    ├── graph_bindings/
    │   ├── graph_chunk_links_v212.json
    │   └── chroma_chunks_v211.json
    └── graph_reports/
        └── graph_v212_upgrade_report.json
```

**不会修改 `knowledge_base.py`**，不会复制或覆盖 `graph_data/`、`graph_bindings/`、`graph_reports/` 中的正式数据，也不重新计算任何 Embedding。

## 2. 安装前：备份和路径检查

先停掉正在运行的 Streamlit。

在 `E:\VsCode_Project\DockerTest_Agent\RAG_project\data` 下运行：

```powershell
conda activate rag
$env:PYTHONUTF8="1"

# 先在浏览器里手动下载 ZIP 到项目 data 文件夹，文件不会自动出现
Test-Path ".\DouXian_V2.13_Trusted_Graph_Retrieval.zip"

# 上一步返回 True 后才继续
Expand-Archive `
  -LiteralPath ".\DouXian_V2.13_Trusted_Graph_Retrieval.zip" `
  -DestinationPath . -ErrorAction Stop

# 检查新文件是否到位
Test-Path .\graph_tools\graph_retriever_v213.py
Test-Path .\graph_tools\test_graph_retriever_v213.py
Test-Path .\v213_payload\rag.py
```

确认 V2.12 基线数据仍然有效：

```powershell
python -m graph_tools.graph_expand_v212 verify
```

请保留 V2.12 已通过的 **12 条旧关系 + 17 条新关系 = 29 条正式可信关系**。剩下 1 条候选 `ambiguous` 不允许线上使用。

## 3. 先跑无网络测试，后安装主链补丁

```powershell
python -m unittest -v graph_tools.test_graph_retriever_v213

# 只读检查当前 rag.py 是否与上传版本匹配，默认不改文件
python .\install_graph_v213.py --check

# 仅在上一步显示 [OK] 后执行
python .\install_graph_v213.py --install
```

**安装器强制校验整个 `rag.py` 的 SHA256**，当前文件如果和上传给 ChatGPT 的版本不一致，就拒绝覆盖，不会猜测如何合并。

安装时自动备份旧版 `rag.py` 到：

`data/_v213_backups/rag_before_v213_时间戳.py`

后续 `--install` 再次运行发现已是目标版本则提示已安装，不会重复修改。

## 4. A/B 测试，先 OFF 再 ON

每次更改环境变量后，**结束并重新启动 Streamlit 进程**，确保重新初始化知识库。

### 实验 A：Graph OFF（现有 Hybrid 基线）

```powershell
$env:DOUXIAN_GRAPH_V213="0"
$env:DOUXIAN_GRAPH_MODE="off"
streamlit run .\豆馅.py
```

记录相同问题的：
- 回答正确性与完整性；
- 真实引用 Chunk 及来源章节；
- 控制台的 Hybrid + Rerank 时间、完整检索时间；
- 总 Token 和费用（如现有诊断提供）。

### 实验 B：Graph ON（V2.12 正式 29 条关系）

关闭 Streamlit，重新运行：

```powershell
$env:DOUXIAN_GRAPH_V213="1"
$env:DOUXIAN_GRAPH_V213_MAX_EXTRA="2"
$env:DOUXIAN_GRAPH_MODE="off"
streamlit run .\豆馅.py
```

代码在 **`KnowledgeBaseService.search_hybrid()` 返回 Hybrid 候选之后、
`search_with_rerank()` 调用现有 Qwen3-Rerank 之前**加入图谱证据。
单问题默认最多新增 2 条，每个子问题最多 2 条，最多可设置 3 条。

因此 **Reranker 仍然每个子问题调用一次**，但因为候选正文可能变多，
**Reranker 的输入 Token、成本与耗时也可能增加**。

### 建议第一组对照问题

1. 卷积神经网络中，卷积层与池化层分别完成什么运算？它们怎样影响特征表示？
2. 无向图模型与马尔可夫随机场是什么关系？
3. 吉布斯采样与 MCMC、Metropolis-Hastings 算法分别有什么关系？
4. 残差网络中的残差连接起什么作用？
5. 机器学习中的支持向量机是什么？（无第一本图谱精确关系时，应该主要走 Hybrid）

建议每个问题分别记录 OFF/ON 的最终引文编号及 Chunk ID、回答质量、Reranker 候选数和耗时。

## 5. 如何识别 Graph ON 是否真正参与检索？

Graph ON 时，终端出现类似：

```text
[Graph V2.13] status=applied matches=2 added=1
[Graph V2.13] request_status=applied subqueries=2 injected_chunks=1
```

`applied` 表示有额外真实 Chunk 加入 Rerank 候选；
`already_in_hybrid` 表示图谱命中，但对应 Chunk 已经由 Hybrid 找到了；
`no_match` 表示提问没有命中可信图谱关系；
`fallback_error` 表示哈希、来源、绑定、真实 Chroma 或其他条件失败，
已回退原 Hybrid；不能把该次结果记为 Graph ON 成功召回；
`not_active` 表示新适配层未运行（例如知识库 API 不兼容）。

除终端日志外，`rag.py` 返回的 `graph_assist.v213` 包含：
- `runs`：各检索子问题的状态；
- `injected_chunks`：加入候选池的数量；
- `survived_rerank`：被现有 Reranker 保留的新增 Chunk 数（每个子问题）；
- `survived_final`：通过最终多样性合并，进入答案证据集的新增 Chunk 数；
- `elapsed_ms`：图谱检索和候选合并阶段毫秒数。

**只有 `survived_final > 0` 时才能说图谱候选真正进入了该题的最终证据集**。
仍不等于它必然改善了答案质量，效果要靠 A/B 测试确认。

目前不会自动生成假的 A/B 结论，也不保存用户题目的原文日志。

## 6. 安全机制和适用范围

1. 只接受 `graph_data/knowledge_graph_v212.json` 中、在
   `graph_bindings/graph_chunk_links_v212.json` 里 **唯一 `verified`** 的关系；
   已被排除的“平均汇聚 IS_A 汇聚函数”歧义候选不得参与召回。
2. 检查正式图谱、绑定、别名表、教材、Chunk 快照、升级报告之间的 SHA256，
   并调用 V2.11.1 重新复验绑定。第一次成功后按文件状态缓存。
3. `graph_tools/graph_paths.py` 统一负责目录；本补丁直接复用现有四层结构，
   **不改** `graph_paths.py`，也不把图谱文件搬回 `data` 根目录。
4. 对新选中的 `chunk_id` 调用**当前真实 Chroma `.get(ids=...)`**，
   并逐条校验全文内容和来源、章节 Metadata 与已验收快照一致。
5. 只取与用户检索子查询匹配的概念及已验证关系；
   V2.7 `aliases_for_search` 只作为匹配提示，不能单独证明概念等价。
6. 严格按 `chunk_id` 或相同来源正文去重，避免两个候选引用同一条原文。
7. 遇到任何图谱读取/校验错误，保留本轮原始 Hybrid 检索候选。
8. Graph ON 模式下临时跳过旧版 V2.8 `DOUXIAN_GRAPH_MODE=assist` 扩词，
   保证单次实验只改变“增加可信图谱 Chunk”这一主要变量。
9. 保留原始 evidence guard、Self-RAG 重试、前端引用生成、生成模型与 Prompt。

## 7. 如果 Graph ON 不生效，按顺序检查

```powershell
# 必须在 data 目录
python -m graph_tools.graph_expand_v212 verify

# 再验证图谱检索模块和当前 rag.py 是否安装成功
python -m unittest -v graph_tools.test_graph_retriever_v213
python .\install_graph_v213.py --check

# 必须重新启动 Streamlit（只改环境变量不一定更新已缓存的服务）
```

如果控制台出现 `[Graph V2.13] fallback: ...`，先根据异常类型定位；
需要时提供异常日志，不要自行绕过 `verified`、SHA 或章节校验。

## 8. 安全回滚

**立即停用（无需改代码）：**

```powershell
$env:DOUXIAN_GRAPH_V213="0"
$env:DOUXIAN_GRAPH_MODE="off"
# 重启 Streamlit
```

**恢复代码（仅当你明确想恢复）：**

先查看 `_v213_backups/`，然后指定你想恢复的那个备份：

```powershell
# 示例：把下面路径替换成安装器实际打印出来的备份文件名
Copy-Item -LiteralPath ".\_v213_backups\rag_before_v213_这里是时间戳.py" `
  -Destination ".\rag.py" -Force
```

**不要执行大范围删除**，尤其不要清除 `chroma_db/`、`graph_data/`、
`graph_bindings/`、`graph_reports/` 和 V2.12 正式文件。

## 9. 下一步你发回什么？

先执行第 2–3 节的离线测试与安装，再做第 4 节的 Graph ON 测试。

请把下面三部分输出发回来：

- `python -m unittest -v graph_tools.test_graph_retriever_v213` 最后的 `Ran ... tests ... OK`；
- `install_graph_v213.py --install` 的结果；
- 一题 Graph ON 的 `[Graph V2.13]` 控制台日志、`survived_final` 和最终引用变化。

**当前明确状态：开发与模拟测试通过；真实 29 条关系的在线 GraphRAG 验收尚未完成。**
