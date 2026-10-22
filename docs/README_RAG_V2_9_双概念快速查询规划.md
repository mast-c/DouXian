# 豆馅 RAG V2.9：双概念、双方面的 Python 快速查询规划

## 目标与改动边界

这一版**只调整查询规划**：让 B04“卷积层 + 池化层分别做什么运算、怎样影响特征表示”这一类明确的复合问题直接拆成 4 条检索子查询，不再等待 Qwen 规划。

在 V2.8 `rag.py` 中只增加 18 行开关代码，主要规则放在新的 `fast_component_planner_v29.py` 模块。**没有更改** Chroma、BM25、Reranker、候选池大小、生成提示词、Evidence Guard、Self-RAG 或 V2.8 Graph Assist。

### 新增唯一开关

```powershell
# 0：关闭新增规划规则（默认，完全保留 V2.8 的查询规划路径）
# 1：在 KB fast 模式下尝试 V2.9 新规则
$env:DOUXIAN_FAST_COMPONENT_PLAN="1"
```

新规则只有同时满足下列条件才生效：

- `DOUXIAN_KB_MODE=fast`
- `DOUXIAN_FAST_QUERY_PLAN=1`
- `DOUXIAN_FAST_COMPONENT_PLAN=1`
- 输入问题命中限定的“双概念 + 运算与特征影响”句式

未匹配时继续采用原来 Qwen 查询分解；strict 模式不走这条新规则。

## 一、示例

原问题：

> 卷积神经网络中，卷积层与池化层分别完成什么运算？它们怎样影响特征表示？

命中后直接生成：

1. 卷积神经网络中，卷积层完成什么运算？
2. 卷积神经网络中，卷积层怎样影响特征表示？
3. 卷积神经网络中，池化层完成什么运算？
4. 卷积神经网络中，池化层怎样影响特征表示？

支持部分等价说法，例如“汇聚层”“完成哪些操作”“对特征图有什么影响”“CNN 中”等。

**严格的边界**：如果用户额外要求代码、公式推导、复杂条件比较，或询问其他算法，新规则将不匹配、回退旧流程。新规则不预设最终答案，更不会把图谱关系当作模型证据。

## 二、ZIP 文件内容

| 文件 | 作用 |
|---|---|
| `rag.py` | 完整 V2.9，取自 V2.8 后只增加一个可开关的查询分支 |
| `fast_component_planner_v29.py` | 纯 Python、无额外依赖的匹配与 4 条子查询生成 |
| `test_fast_component_plan_v29.py` | 12 组新功能离线测试 |
| `compare_component_plan_v29.py` | OFF/ON 日志耗时、Rerank Token、证据章节比较 |
| `preview_component_plan_v29.py` | 在不调用任何 API 的情况下预览本地规则生成的 4 个子问题 |
| `rag_V2_8_原始备份.py` | V2.8 原版主模块快照 |
| `rag_V2_8_vs_V2_9_单变量查询规划.diff` | 可核验的源码差异 |
| `豆馅_RAG_V2.9_环境变量统一配置说明.md` | 15 个已知环境变量与作用 |
| `README_RAG_V2_9_双概念快速查询规划.md` | 本安装指南 |

**安装前提：已安装完整 V2.8 项目**。本 ZIP 不替换知识库、图谱 JSON、教材快照和 Chroma 索引，不包含你的 `.env`。

## 三、安装到 Windows PowerShell

在 `data` 目录停止 Streamlit（Ctrl+C），建议先备份你自己的实际版本：

```powershell
conda activate rag
Copy-Item .\rag.py .\rag_before_v29_local.py

# 把下载的 ZIP 复制到 data 目录后执行：
Expand-Archive -LiteralPath .\豆馅_RAG_V2.9_双概念快速查询规划.zip -DestinationPath . -Force

# 检查语法，执行免费离线测试
python -m py_compile .\rag.py .\fast_component_planner_v29.py
python .\test_fast_component_plan_v29.py

# 可选：先预览规则是否命中，不会调用 Qwen
python .\preview_component_plan_v29.py --question "卷积神经网络中，卷积层与池化层分别完成什么运算？它们怎样影响特征表示？"
```

预期：

```text
✅ V2.9 查询规划 12 组离线测试通过；没有模型/API 调用。
```

**注意**：如果你电脑上的 `rag.py` 比 V2.8 还多了自定义修改，请先比对 ZIP 中的 `.diff`，不要盲目覆盖；回滚方式见最后。

## 四、单变量 A/B（推荐先关闭图谱，避免混淆）

先在同一个 PowerShell 设置**完全相同**的其余变量：

```powershell
$env:DOUXIAN_KB_MODE="fast"
$env:DOUXIAN_GUARD_FAST="0"
$env:DOUXIAN_FAST_QUERY_PLAN="1"
$env:RERANK_DOC_CHARS="600"
$env:DOUXIAN_FAST_QUALITY_PROMPT="1"
$env:DOUXIAN_GRAPH_MODE="off"
```

**对照组 OFF：**

```powershell
$env:DOUXIAN_FAST_COMPONENT_PLAN="0"
streamlit run .\豆馅.py
```

在新会话问一次完整 B04 问题，保存终端日志，Ctrl+C 停止。

**实验组 ON：**

```powershell
$env:DOUXIAN_FAST_COMPONENT_PLAN="1"
streamlit run .\豆馅.py
```

在另一个新会话再次提出**完全相同**的问题。命中时终端应打印：

```text
[Query Planner] component_fast_path=on, skipped Qwen
```

并能看到 `4 个子问题`。`查询分解（规则/Qwen）` 所报告的耗时应明显减少，但这是**待你本地验证的目标，不是已实测结果**。

### 将运行日志保存为文件（可选）

分别以如下方式代替 `streamlit run`（每份日志只测试一道题，Ctrl+C 结束）：

```powershell
# 对照组
$env:DOUXIAN_FAST_COMPONENT_PLAN="0"
streamlit run .\豆馅.py 2>&1 | Tee-Object -FilePath .\v29_off.log

# 实验组
$env:DOUXIAN_FAST_COMPONENT_PLAN="1"
streamlit run .\豆馅.py 2>&1 | Tee-Object -FilePath .\v29_on.log
```

比较日志，不调用 API：

```powershell
python .\compare_component_plan_v29.py --off .\v29_off.log --on .\v29_on.log
```

该脚本提取查询规划耗时、检索总计、流式生成耗时、Reranker 调用次数和官方 Token，以及最终入选的证据章节。

注意：**日志可能含完整教材 Chunk 或其它私有信息**，分享前请核查。

## 五、正式验收标准

1. ON 在 B04 命中本地规则，且实际查询分解不请求 Qwen。
2. ON 和 OFF 均只有 4 条子查询，Reranker 调用次数不增加。
3. ON 仍能检索到 `5.2.2 卷积层`、`5.2.3 汇聚层`，最好也找到 `5.1.2 互相关`。
4. 最终回答覆盖卷积/池化的**具体运算**及其**对特征表示的影响**。
5. 引文仍需逐句核对，不可把“有助于缓解过拟合”写成“一定防止过拟合”。
6. 以真实用户端体验重新验证 40 秒目标。冷启动和热启动分开记录。
7. 完成规划 ON/OFF 测试后，再固定规划开启，只切换 `DOUXIAN_GRAPH_MODE=off/assist`，衡量图谱的增益。

旧版 B04 Qwen 规划耗时约 25～27 秒，但**这不等于 V2.9 已证明能省下同样时间或维持相同证据质量**。离线测试无法证明在线响应速度、检索质量与回答准确性。

## 六、还原

仅关闭新功能：

```powershell
$env:DOUXIAN_FAST_COMPONENT_PLAN="0"
streamlit run .\豆馅.py
```

需要回滚到你机器上的原版：

```powershell
Copy-Item .\rag_before_v29_local.py .\rag.py -Force
```

Graph Assist 的独立开关 `$env:DOUXIAN_GRAPH_MODE="off"` 或 `"assist"` 继续有效。
