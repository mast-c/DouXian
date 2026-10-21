# 豆馅 RAG｜环境变量统一配置与作用说明（V2.9）

> **适用范围**：基于已核对的豆馅 **V2.9 `rag.py` / `fast_component_planner_v29.py` / `graph_assist_v28.py`**、
> 之前整合包中的 **`knowledge_base.py` / `evidence_guard.py`** 整理。
> 最后整理：2026-10-21。
>
> **本文件是 Markdown 使用说明，不是可直接运行的 `.ps1` 脚本。**
> 请把下方 `powershell` 代码块复制到 Windows PowerShell 后执行。
> 这里的设置只影响当前终端，以及从该终端启动的 Streamlit/Python 进程；关闭终端后通常失效。

---

## 一、日常推荐配置（可直接复制）

在 Windows PowerShell 的项目 `data` 目录、`rag` 环境下执行：

```powershell
# 进入豆馅项目目录（根据你自己电脑的实际路径调整）
cd 'E:\VsCode项目\DockerTest_Agent\RAG项目\data'
conda activate rag

# ① 知识库问答：fast=快速模式；strict=严格证据审核模式
$env:DOUXIAN_KB_MODE="fast"

# ② 严格模式的首轮 Evidence Guard：0=保持模型默认思考；1=关闭思考以尝试加速
#    目前 fast 模式跳过 Guard，因此此变量在 fast 模式下基本无效
$env:DOUXIAN_GUARD_FAST="0"

# ③ 快速查询分解：1=优先尝试 Python 规则；0=关闭该规则快捷路径
#    规则未命中时，复合问题仍可能调用 Qwen 做查询分解
$env:DOUXIAN_FAST_QUERY_PLAN="1"

# ④ Reranker 输入截断：600=每个候选文本最多约 600 字符；0=全文重排
#    只影响发送给 Reranker 的文本，不截断最终提供给回答模型的原始 Chunk
$env:RERANK_DOC_CHARS="600"

# ⑤ V2.6 生成质量提示词：1=开启；0=关闭
#    仅在 fast 模式、且问题命中概率图/贝叶斯网/MCMC 等主题时启用
$env:DOUXIAN_FAST_QUALITY_PROMPT="1"

# ⑥ V2.9 双概念/双方面快速规划：0=关闭（默认）；1=符合规则时跳过 Qwen
#    要求同时设置 DOUXIAN_KB_MODE=fast 和 DOUXIAN_FAST_QUERY_PLAN=1
$env:DOUXIAN_FAST_COMPONENT_PLAN="0"

# ⑦ V2.8 图谱辅助检索：off=关闭；assist=根据已验证图谱补充检索关键词
#    assist 不会替换原 Chunk、增加检索子问题或自动启动图谱推理
$env:DOUXIAN_GRAPH_MODE="assist"

# ⑧ 用于解释“今天”等日期的时区；默认 Asia/Shanghai
$env:DOUXIAN_TIMEZONE="Asia/Shanghai"

# 启动 Streamlit。启动前应已按项目原有 .env 配好模型/API 密钥。
streamlit run .\豆馅.py
```

**如果你要复现 V2.6、不希望图谱影响结果**，只需要把
`$env:DOUXIAN_GRAPH_MODE="assist"` 改成 `"off"`，然后重新启动 Streamlit。

> **安全与性能边界**：上述配置偏重速度，但 fast 模式没有独立的 Evidence Guard / Self-RAG
> 事实校验。`DOUXIAN_FAST_QUALITY_PROMPT=1` 也不是证据充分性审核。
> 历史目标是问答用时低于 40 秒，但并不保证所有新问题都能达到。

---

## 二、已核对的全部 15 个环境变量

### A. 功能控制：7 个

| 变量 | 已知代码默认值 | 可设值 | 作用及建议 |
|---|---|---|---|
| `DOUXIAN_KB_MODE` | `strict` | `fast` / `strict` | **fast** 跳过独立 Evidence Guard、Self-RAG 定向补检，优先降低延迟；**strict** 执行证据审核，通常更慢。日常测试推荐 `fast`。 |
| `DOUXIAN_GUARD_FAST` | `0` | `0` / `1` | 仅影响 **strict 模式首轮 Evidence Guard 模型调用**：`1` 通过 `enable_thinking=False` 禁用思考；`0` 使用模型默认行为。并非关闭 Evidence Guard 的开关。建议保留 `0`。 |
| `DOUXIAN_FAST_QUERY_PLAN` | `1` | `0` / `1` | `1` 优先使用 Python 规则进行符合条件的复合问题拆分；未命中规则时，仍可能调用 Qwen。`0` 关闭 Python 规则快捷路径。 |
| `DOUXIAN_FAST_COMPONENT_PLAN` | `0` | `0` / `1` | **V2.9 新增**：`1` 允许卷积层、池化层两个概念的“运算与特征影响”问法在 Python 中直接拆为 4 个检索问题并跳过 Qwen；只在 `DOUXIAN_KB_MODE=fast` 且 `DOUXIAN_FAST_QUERY_PLAN=1` 时生效，未命中规则回退。默认 `0` 保持 V2.8 行为。 |
| `RERANK_DOC_CHARS` | `0`（优化版 `knowledge_base.py`） | `0` 或整数 **≥300** | `0` 全文送 Reranker；例如 `600` 限制每个候选文本的排序输入字符数，减少 Token 成本，**原始 Chunk 内容与最终引用不因此被截断**。`1`～`299` 会抛出参数异常，非整数按 `0` 处理。 |
| `DOUXIAN_FAST_QUALITY_PROMPT` | `1` | `0` / `1` | `1` 对 fast 模式中特定概率图模型问题补充 MCMC/Gibbs 层级、采样机制、信念传播精确性约束；`0` 关闭。**对其他主题不一定生效**，且不提供事实校验。 |
| `DOUXIAN_GRAPH_MODE` | `off` | `off` / `assist` | **off** 保持 V2.6 原检索词；**assist** 在已有查询分解后用离线知识图谱辅助补充别名、相关术语，再进入原 Hybrid/Rerank 链路。图谱失效时回退原检索。 |

### B. 知识图谱文件位置：2 个（可选）

| 变量 | 未设置时路径 | 作用 |
|---|---|---|
| `DOUXIAN_GRAPH_FILE` | `data\knowledge_graph_v27.json` | 指定已构建的 V2.7 图谱 JSON；只有 `DOUXIAN_GRAPH_MODE=assist` 时会使用。 |
| `DOUXIAN_GRAPH_SOURCE` | 首选 `data\graph_sources\神经网络与深度学习.md`，不存在则查找 `data\神经网络与深度学习.md` | 指定构建图谱时的原始教材快照。图谱加载时会与其 **SHA256** 比对；文件不匹配或不存在，则停用本次图谱增强、退回原检索。 |

如果图谱文件**不在默认位置**，可在 `data` 目录运行：

```powershell
# 可选：只有需要覆盖默认图谱位置时才设置
$env:DOUXIAN_GRAPH_FILE=(Join-Path (Get-Location) 'knowledge_graph_v27.json')

# 可选：指向与图谱 JSON 构建时完全一致的教材原文
$env:DOUXIAN_GRAPH_SOURCE=(Join-Path (Get-Location) 'graph_sources\神经网络与深度学习.md')
```

这里的两条 `DOUXIAN_GRAPH_*` 路径变量不是开关，不要填 `0` 或 `1`。

### C. 模型与 API 连接：5 个（通常写在现有 `.env` 中）

> **不要把真正的 API Key 写在 Markdown 文件里，也不要将包含真实密钥的文件上传公开仓库。**

| 变量 | 读取位置 | 作用 |
|---|---|---|
| `DASHSCOPE_API_KEY` | `rag.py`、`knowledge_base.py` | DashScope 凭证，用于回答模型、Embedding 和 Qwen3-Rerank 等服务；真实值请继续保存在本机已有的 `.env` 中。 |
| `OPENAI_MODEL` | `rag.py` | `ChatOpenAI` 实际要调用的聊天模型名称；名称取决于你当前 DashScope/OpenAI 兼容接口的可用模型。 |
| `OPENAI_BASE_URL` | `rag.py` | `ChatOpenAI` 的 OpenAI 兼容 API 基地址；**不是**图谱文件地址。 |
| `DASHSCOPE_EMBEDDING_MODEL` | `knowledge_base.py` | 创建向量嵌入时使用的 Embedding 模型标识；改动模型前要考虑已有 Chroma 向量和维度是否兼容。 |
| `DASHSCOPE_HTTP_BASE_URL` | `knowledge_base.py` | DashScope 原生 SDK 的 HTTP API 基地址（Reranker 使用）；未设置时保持 SDK 既有默认地址。**不等同于** `OPENAI_BASE_URL`。 |

**建议沿用你原来 `.env` 里的模型名称、接口和密钥，不要仅为了启用知识图谱而改这些变量。**

下列只是**格式示例**，不是你当前项目的真实值，也不能直接连网使用：

```dotenv
# 示例名称与占位符；请保留你当前有效的 .env 配置，不要盲目覆盖
DASHSCOPE_API_KEY=请使用本机已有的真实密钥
OPENAI_MODEL=请填写实际使用的聊天模型名称
OPENAI_BASE_URL=请填写实际使用的OpenAI兼容接口地址
DASHSCOPE_EMBEDDING_MODEL=请填写实际使用的嵌入模型名称
# DASHSCOPE_HTTP_BASE_URL=仅在确实需要更换 SDK API 地址时设置
```

### D. 日期解释：1 个

| 变量 | 默认值 | 作用 |
|---|---|---|
| `DOUXIAN_TIMEZONE` | `Asia/Shanghai` | 控制应用将“今天”等实时问答日期解释为哪个时区的日期。代码读取 `ZoneInfo`，如果时区名无效，会退回当前系统本地日期。不影响图谱、检索或 Qwen 模型配置。 |

---

## 三、切换常用测试方案

### 方案 1：快速问答 + 图谱辅助（推荐用于 V2.8 实验）

```powershell
$env:DOUXIAN_KB_MODE="fast"                 # 跳过独立证据审核
$env:DOUXIAN_GUARD_FAST="0"                 # 如切回 strict，Guard 保留默认思考
$env:DOUXIAN_FAST_QUERY_PLAN="1"
$env:DOUXIAN_FAST_COMPONENT_PLAN="0"            # 优先命中 Python 查询规则
$env:RERANK_DOC_CHARS="600"                 # 排序输入限制 600 字符
$env:DOUXIAN_FAST_QUALITY_PROMPT="1"        # V2.6 领域质量提示
$env:DOUXIAN_GRAPH_MODE="assist"            # 启动 V2.8 图谱术语辅助
streamlit run .\豆馅.py
```

### 方案 2：快速问答，不启用图谱（对照组）

```powershell
$env:DOUXIAN_KB_MODE="fast"
$env:DOUXIAN_GUARD_FAST="0"
$env:DOUXIAN_FAST_QUERY_PLAN="1"
$env:DOUXIAN_FAST_COMPONENT_PLAN="0"
$env:RERANK_DOC_CHARS="600"
$env:DOUXIAN_FAST_QUALITY_PROMPT="1"
$env:DOUXIAN_GRAPH_MODE="off"               # 与方案 1 只改变这一项
streamlit run .\豆馅.py
```

### 方案 3：严格证据审核（耗时可能更长）

```powershell
$env:DOUXIAN_KB_MODE="strict"              # 首轮 Evidence Guard + 按需 Self-RAG 补检
$env:DOUXIAN_GUARD_FAST="0"                # 不禁用 Guard 模型思考
$env:DOUXIAN_FAST_QUERY_PLAN="1"
$env:DOUXIAN_FAST_COMPONENT_PLAN="0"           # 保留查询规划优化
$env:RERANK_DOC_CHARS="600"                # 保留 Reranker 节省 Token 的参数
$env:DOUXIAN_FAST_QUALITY_PROMPT="1"       # 严格模式不会使用该 fast 专属提示词
$env:DOUXIAN_GRAPH_MODE="off"              # 先不叠加图谱变量，便于排查
streamlit run .\豆馅.py
```

**A/B 原则**：比较图谱收益时，最好只调整 `DOUXIAN_GRAPH_MODE` 一个变量。
使用相同问题、新会话，记录总耗时、首字等待、检索章节、引用准确性、官方 Reranker Token。

---

## 四、常见问题与排查命令

### 1. 查看当前是否设置了这些变量

```powershell
# 列出当前 DOUXIAN_ 开关与路径
Get-ChildItem Env:DOUXIAN_* | Sort-Object Name

# 检查非 DOUXIAN_ 的排序截断参数
Get-Item Env:RERANK_DOC_CHARS -ErrorAction SilentlyContinue

# 仅检查密钥是否存在；不输出实际 API Key
if ($env:DASHSCOPE_API_KEY) {
    Write-Host "DASHSCOPE_API_KEY: 已设置（值已隐藏）"
} else {
    Write-Host "DASHSCOPE_API_KEY: 未在当前 PowerShell 环境中设置（程序也可能从 .env 读取）"
}
```

**不要运行 `Get-ChildItem Env:` 后就把全部输出发送到聊天或公开日志**，因为可能暴露密钥。

### 2. 检查 `RERANK_DOC_CHARS=600` 有没有被你本机的代码读取

```powershell
Select-String -Path .\knowledge_base.py -Pattern 'RERANK_DOC_CHARS'
```

若没有匹配结果，说明当前 `knowledge_base.py` 不包含此前实现的截断开关；
即使在 PowerShell 中设置了 `600`，也**不会自动使该功能生效**。
请先核对你最终实际安装的 `knowledge_base.py` 版本。

### 3. 图谱打开后为什么还是 `no_match`？

`DOUXIAN_GRAPH_MODE=assist` 只负责**为部分问题提供额外检索词**。
未匹配到别名/已核验关系时，`no_match` 是正常结果，不表示图谱安装失败。

可以离线预览：

```powershell
python .\preview_graph_v28.py --question "卷积层与池化层分别完成什么运算？"
```

常见日志：

```text
[Graph Assist] status=applied hints=1 耗时=...
```

状态说明：

| 状态 | 含义 |
|---|---|
| `off` | 图谱未启用 |
| `applied` | 已为至少一个子查询添加检索关键词 |
| `no_match` | 当前问题无需或没有可用的扩展 |
| `missing_file` | 图谱 JSON 或原始教材不存在 |
| `invalid_or_stale` | 格式有误或教材 SHA256 与图谱不一致，已回退原检索 |
| `fallback_error` | 图谱辅助出现异常，自动回退原检索 |

### 4. 如何取消某个 PowerShell 变量？

```powershell
# 取消图谱开关，让 V2.8 使用代码默认 off
Remove-Item Env:DOUXIAN_GRAPH_MODE -ErrorAction SilentlyContinue

# 取消显式的 Reranker 截断，优化版代码恢复默认 0（全文重排）
Remove-Item Env:RERANK_DOC_CHARS -ErrorAction SilentlyContinue
```

取消的是**当前终端**中的变量。如果同名项写在项目 `.env` 中，
程序启动时仍可能读到 `.env` 的值。

### 5. 为什么复制 `$env:` 后 Streamlit 没反应？

变量会被从**设置它的 PowerShell 进程启动的子进程**继承。
建议按此顺序操作：

1. 用 `Ctrl+C` 停止原来的 Streamlit；
2. 在**将启动 Streamlit 的同一个 PowerShell 窗口**执行变量赋值；
3. 再运行 `streamlit run .\豆馅.py`；
4. 在程序终端确认 `[Graph Assist]` / `[Query Planner]` 等日志。

---

## 五、哪些目前**不是**环境变量？

在这次已核对的版本里，下列行为是源码里的配置或参数，
**不是**设置 `$env:` 就能控制的开关：

- `search_with_rerank(... recall_k=10, top_n=3)` 等检索调用参数；
- 多子查询的候选池和最终 5 条证据的选择逻辑；
- 图谱语义关系抽取规则、图谱节点清单及关联类型；
- 最终回答生成阶段的 `enable_thinking=False`（当前源码固定设置）；
- 已有的 V2.5 Python 轻量质量检查逻辑（没有独立开关）。

不要随意编造如 `DOUXIAN_GRAPH_TOP_K`、`DOUXIAN_RERANK_MODEL`、
`DOUXIAN_FAST_QUALITY_CHECK` 这样的环境变量名称；当前已核对源码没有读取它们。

## 六、版本与完整性说明

这份清单统计了**已核对源码中的 15 个环境变量**：

- `rag.py`：V2.9 ZIP 内的完整版本；
- `fast_component_planner_v29.py`：V2.9 新增的纯 Python 规则规划器；
- `graph_assist_v28.py`：V2.8 ZIP 内版本；
- `knowledge_base.py`、`evidence_guard.py`：此前完整整合包中的优化版本。

**尚不能核验你电脑上所有模块的最新源码。** 特别是 `config_data.py`、
`search_tool.py`、`vector_stores.py` 等文件没有在本轮完整对应版本中出现，
可能另有项目配置变量。上述 15 项是**已证实的清单**，不是对未知源码的完整性保证。

此外，本次已看到的另一个旧版 `knowledge_base.py` 副本没有读取
`RERANK_DOC_CHARS`。请以上文的 `Select-String` 检查实际项目文件，
而不要仅凭此说明认定功能已安装。

**建议维护习惯**：以后每新增一个 `os.getenv()` 环境变量，就更新本 Markdown，
把变量名称、默认值、可设值、作用以及配置生效范围记录下来。


---

## 七、V2.9 新增：双概念、双方面查询规划的单变量 A/B

仅适合卷积层与池化层明确提出“运算 + 特征表示影响”的问法；
其它复合问题仍可能使用 Qwen 查询分解。这与 V2.8 图谱扩展是**两个独立实验**。

```powershell
$env:DOUXIAN_KB_MODE="fast"
$env:DOUXIAN_FAST_QUERY_PLAN="1"
$env:DOUXIAN_GRAPH_MODE="off"
$env:RERANK_DOC_CHARS="600"

# A：旧流程。使用一轮新会话测试 B04 后，以 Ctrl+C 停止
$env:DOUXIAN_FAST_COMPONENT_PLAN="0"
streamlit run .\豆馅.py

# B：只更改新变量，再次启动并在新会话提问同样的问题
$env:DOUXIAN_FAST_COMPONENT_PLAN="1"
streamlit run .\豆馅.py
```

成功命中时应看到：

```text
[Query Planner] component_fast_path=on, skipped Qwen
```

**规则未命中时回退原来的 Qwen 查询规划；strict 模式不使用这一新增规则。**
Python 规划不代表检索质量和事实准确性已通过审查，仍需核对真实 Chunk。
参见 `README_RAG_V2_9_双概念快速查询规划.md`。
