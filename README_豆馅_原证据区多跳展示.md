# 豆馅 V1.17 · 直接在原“实际命中的知识库证据”中显示多条路径

## 这次修改了哪个文件？

已经根据你上传的完整 `豆馅.py` **直接修改主前端**。
ZIP 中的 `豆馅.py` 就是可替换原文件的完整版本，不是补丁片段。

原本你在文件开头放了：

```python
from medical_p3063_inline_path_evidence_v117 import (
    render_rich_multihop_paths_inside_evidence,
)
```

但是前端 `trace_details_html()` 是 **一次性拼接整个 HTML `<details>`**
（即你截图中的“医疗检索完成…”折叠区），不能直接向字符串里插入
`st.graphviz_chart()`；后者会显示在另一个 Streamlit 容器，
甚至跑到原有证据区之外。

本版本采用与原页面一致的 HTML/CSS 可滚动节点图，
**直接在“实际命中的知识库证据（原有6条）”之后**
插入新的“🔗 多跳图谱来源路径（4条）”卡片组。

你能看到：
- 原文证据 `[1]`、`[2]`...仍显示，不改变引用号码；
- P01、P02、P03、P04 四张独立的 8 跳路径卡片；
- 第一张路径默认展开，横向滚动展示9个节点与8条方向关系；
- 其余路径可点击展开；
- 路径 ID、来源药品、疾病、Section ID、候选片段；
- 逐边完整关系及节点 ID；
- 第三方未经核验、非治疗关系和非临床证据的声明。

只修改前端代码，不更新 Neo4j、Chroma、医学知识库数据、
`rag.py`、SQLite 数据、模型提示词或现有检索统计。

## 两个重要的兼容修复

### 1. 不把路径当作新增原文证据

旧版 P3-06.3 会往 `trace["knowledge_evidence"]` 追加展示卡，
原来6条原文就可能显示为10条“证据”。

新前端会将 `unverified_multihop_path_display_only`
**从原文证据循环中排除**，另行展示路径卡。
所以标题保留“实际命中的知识库证据（6条）”，下方单独出现
“多跳图谱来源路径（4条）”；绝不声称10条独立医学事实。

本轮的质量验收导出也会过滤显示专用路径卡，
但来源路径元数据仍可按 trace 原始结构保留。

### 2. DRUG_ONLY 问题也支持独立来源路径

你此前成功跑8跳的原问题：

> 哪些药品说明书提到了过敏性鼻炎？

现有 P3-04.3 的 question_route 可能把这个问题判为
`DRUG_ONLY`，从而触发旧 P3-06.3 适配器
`SKIPPED_NO_DISEASE_ROUTE`。

本次在问答**完成后**新增一次受控只读兜底：
仅当 `.env` 启用多跳证据展示、当前没有缓存路径，
旧适配器确实跳过了疾病路由，且用户明确问
“药品说明书提及/记载/涉及某个疾病”时，
尝试调用 P3-06.1 原有已审计的只读检索模板。

不会更改已生成的回答，查询失败也不会阻断用户会话。

## 一、安装：三行核心操作

先停止正在运行的 Streamlit（服务窗口中按 Ctrl+C）。

把 ZIP 下载到：

`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

运行：

```powershell
cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"

$py = "E:\Miniconda\envs\rag\python.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

# 一定先备份你原来的主前端
Copy-Item ".\豆馅.py" ".\豆馅.py.pre_multihop_ui_$stamp.bak"

# ZIP 内含修改完的豆馅.py，同名文件覆盖前端，其他文件只增加测试/README
Expand-Archive `
  -LiteralPath ".\豆馅_P3063_原证据区多跳展示_已修改.zip" `
  -DestinationPath "." -Force

# 先检查语法
& $py -m py_compile ".\豆馅.py"

# 再跑完全离线的前端集成测试
& $py -m unittest -v test_douxian_multihop_frontend_v117.py
```

预期：`Ran 20 tests ... OK`。

也可从另一个下载链接取得 `豆馅_已集成多跳证据.py`。
要直接用它，请重命名覆盖你的 `豆馅.py`，同时保留原文件备份。

## 二、.env 配置

保留已经验收通过的 Neo4j / P3-04.5 配置，
特别是 `MEDICAL_P3045_GRAPH_AUDIT_PATH`。

建议：

```dotenv
MEDICAL_P3063_INLINE_PATH_CARDS=1
MEDICAL_P3063_APPEND_GENERIC_CARDS=0

MEDICAL_P3063_PATH_MIN_HOPS=8
MEDICAL_P3063_PATH_MAX_HOPS=8
MEDICAL_P3063_PATH_MAX_PATHS=4
MEDICAL_P3063_PATH_PER_TEMPLATE=4
```

`APPEND_GENERIC_CARDS=0`：只缓存路径元数据，
由刚修改的 `豆馅.py` 在**原证据区域**绘制丰富的节点箭头卡片，
避免旧版通用文本证据卡重复出现。

**不需要重跑 P3-06.3 的 rag.py 自动补丁。**
本次主前端也能在问答完成后独立补齐尚无缓存的路径，
前提是本地已有 P3-06.1A、P3-06.3 相关运行模块
且图谱来源审计与 Neo4j 连接正常。

如果你想看 5～8 跳而非仅8跳，可以调整：

```dotenv
MEDICAL_P3063_PATH_MIN_HOPS=5
MEDICAL_P3063_PATH_MAX_HOPS=8
MEDICAL_P3063_PATH_MAX_PATHS=16
MEDICAL_P3063_PATH_PER_TEMPLATE=4
```

注意实际返回深度仍取决于 Neo4j 中已有的真实关系路径，
不同疾病或问法不会必然命中每个模板。

## 三、重启并重新提问

原来如何启动 Streamlit，现在仍用原命令。示例：

```powershell
& $py -m streamlit run ".\豆馅.py"
```

然后**新发一轮消息**：

> 哪些药品说明书提到了过敏性鼻炎？请展示8跳来源路径。

或更清晰的演示题：

> 过敏性鼻炎是什么？哪些药品说明书提到了它？
> 请展示连接到枯草热等疾病术语的8跳来源路径。

打开回答上方的“医疗检索完成…”折叠区域，
在原有证据卡片下方查看“多跳图谱来源路径”。

**旧的历史消息**如果保存时没有多跳 `trace`，
不会凭空生成相应的历史路径；请重新发送问题验收。

## 四、如果看不到路径，检查这几项

1. `.env` 中 `MEDICAL_P3063_INLINE_PATH_CARDS=1`，没有重复的同名键。
2. 真实 P3-06.1A `medical_p3061_multihop_graph_v117.py`
   已经替换了修复 Neo4j Query 的版本。
3. `stage4_reports` 内有成功的 P3-04.4B 全量图谱审计，
   且 `MEDICAL_P3045_GRAPH_AUDIT_PATH` 指向对应的真实文件。
4. 本轮 trace 的 `p3063_multihop_path_cards.paths` 中
   应保存实际验证的路径；也支持 `disease_encyclopedia.
   multi_hop_graph.verified_paths`。
5. 查询失败时原文证据照旧展示，服务控制台会输出警告；
   不要为凑8跳创建虚构治疗关系。

## 五、关闭或回滚

关闭功能，不删除文件：

```dotenv
MEDICAL_P3063_INLINE_PATH_CARDS=0
```

然后重启 Streamlit。

如要撤销本次前端改动，停止 Streamlit，恢复前面备份：

```powershell
Copy-Item ".\豆馅.py.pre_multihop_ui_你的备份时间戳.bak" ".\豆馅.py" -Force
& $py -m py_compile ".\豆馅.py"
```

## 实际 QA

- 20 个完全离线的前端集成测试全部通过；
- 证据区原文6条 + 路径4条，互相区分，4条8跳共36个节点/32条关系；
- 使用独立的 `path_id` 避免合并药品名相同但实际节点不同的路径；
- 通过真实 P3-06.1 固定路径的 5～8 跳模板结构检查；
- 检查 HTML 转义、不合法治疗关系、路径环路及导出隔离；
- 模拟验证 DRUG_ONLY 触发一次只读路径兜底；
- 未远程运行你的 Windows Streamlit 或 Neo4j；
  8跳实库成功的事实来自你之前提交的测试日志。

**医疗语义边界：第三方来源未经核验；图谱关系只能说明文本提及或来源关联。**
