# 豆馅医疗 GraphRAG V1.17 · P3-06.3 多条路径显示在原证据区域

## 这次改动解决哪一个问题？

你现有的 Streamlit 聊天页面已在“检索与原文核验统计”
下面展示“实际命中的知识库证据（6 条）”，例如：
- `[1] medical.json ...`
- `[2] 氯雷他定片 / 过敏性鼻炎 ...`

但 P3-06.1 在真实 Neo4j 查到的 5～8 跳路径，之前主要存储在
`trace["disease_encyclopedia"]["multi_hop_graph"]["verified_paths"]`，
或独立 `p3061_8hop_*.json`，**没有全部加入这个现有证据列表**。

本阶段通过对现有 `rag.py` 的 **RagService._stream_agent** 进行两个
只读元数据钩子，将多条经 P3-06.1 技术来源校验的路径追加到：

```python
trace["knowledge_evidence"]
```

例如，在你原来的 **6 条原文证据** 后追加：

```text
[7] GraphRAG 多跳路径 P01 · 8 跳（仅展示，不是回答引用）
[8] GraphRAG 多跳路径 P02 · 8 跳（仅展示，不是回答引用）
[9] GraphRAG 多跳路径 P03 · 8 跳（仅展示，不是回答引用）
[10] GraphRAG 多跳路径 P04 · 8 跳（仅展示，不是回答引用）
```

**这4条路径是可视化/来源追溯卡片，不是新增4条医学证据。**
原来“合并去重后 6 条”统计继续保持 6；
UI 下拉列表可能显示 10 张卡片，其中 6 张原文 + 4 张来源路径。

因此我们**不将路径列表写入** `merged_excerpts`、Evidence Guard、
最终 LLM 提示词、知识库真实引用编号或检索统计。
原来 `[1]` 等答案引用的含义、审查结果都不变。

> 重要限制：当前回合没有你实际 Streamlit 主前端文件。
> 本补丁使用已确认的后端 `trace["knowledge_evidence"]` 作为兼容通道，
> 很可能能直接被现有通用证据卡片循环展示，但如果你的主前端
> 过滤未知 `evidence_type`，需要查看该主前端 .py 再作精确补丁。
> 已包含一个可插入原前端的 Graphviz 富展示组件，见“可选嵌入”。

## 先决条件

- 本地 P3-04.4B 全量审计为 `FULL_GRAPH_LINK_AUDIT_OK`；
- P3-04.5 图谱来源检索已安装；
- P3-06.1A 的 Neo4j Query 热修复已替换到本机；
- 你已真实查出 4 条 8 跳路径（此前为过敏性鼻炎）；
- 新的 ZIP 解压到：
  `E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

本包不带 `rag.py` 全量文件、不会重建 Neo4j、不重建 Chroma，
不会向原始 `medical.json` 新增疾病关系。

## 本包的文件

1. `medical_p3063_inline_path_evidence_v117.py`
   - 读取真实 P3-06.1 只读检索结果；
   - 将前4条8跳路径转换为通用 UI 证据卡片；
   - `source` 明确标注“仅展示，不是回答引用”；
   - 在 `trace` 缓存已验证路径，同一轮 Self-RAG 覆盖 UI 列表时重新添加，不再查询 Neo4j；
   - 支持 5～8 跳分层展示、只读失败回退、Graphviz 富展示组件。
2. `medical_p3063_patch_rag_inline_evidence_v117.py`
   - AST 安全定位 `RagService._stream_agent` 中恰好两处 `trace["knowledge_evidence"]` 赋值；
   - 不修改 `_call_tool`、检索逻辑、生成、医学证据审核和任何数据库；
   - dry-run 默认，`--apply` 前自动创建字节级 `.bak` 备份。
3. `test_medical_p3063_inline_path_cards_v117.py`
   - 34 项完全离线测试（默认禁用、4条8跳、去重、缓存、来源状态、恢复、AST与 Windows CRLF）；
4. `README_P3-06-3_V117_inline_multi_path_evidence.md`
5. `P3-06-3_V117_delivery_manifest.json`

## 一、安装（Windows PowerShell）

**停止当前 Streamlit**：在运行它的终端按 `Ctrl + C`。

```powershell
cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"

$py = "E:\Miniconda\envs\rag\python.exe"
$env:PYTHONUTF8 = "1"

Expand-Archive `
  -LiteralPath ".\medical_P3-06-3_V117_inline_multi_path_evidence.zip" `
  -DestinationPath "." -Force
```

先运行离线单测：

```powershell
& $py -m unittest -v test_medical_p3063_inline_path_cards_v117.py
```

预期：

```text
Ran 34 tests ...
OK
```

随后仅查看代码修改方案，不改 `rag.py`：

```powershell
& $py .\medical_p3063_patch_rag_inline_evidence_v117.py `
  --target .\rag.py
```

**只有出现** `[P3-06.3] READY_TO_APPLY` **再执行**：

```powershell
& $py .\medical_p3063_patch_rag_inline_evidence_v117.py `
  --target .\rag.py `
  --apply

if ($LASTEXITCODE -ne 0) {
    throw "P3-06.3 证据路径钩子安装未通过"
}

& $py -m py_compile .\rag.py
```

脚本自动生成类似：
`rag.py.pre_p3063_20261102_XXXXXX_ffffff.bak`。

这份补丁也接受已安装 P3-04.5 但尚未完成 P3-06.1
`rag.py` 集成的版本。多跳路径卡片独立使用 P3-06.1
核心只读检索，不依赖之前是否将 P3-06.1 wrapper 接入 `rag.py`。

## 二、在 .env 启用（最简）

在项目根目录的 `.env` 末尾追加：

```dotenv
MEDICAL_P3063_INLINE_PATH_CARDS=1
MEDICAL_P3063_APPEND_GENERIC_CARDS=1

# 先演示真实验收通过的4条8跳路径
MEDICAL_P3063_PATH_MIN_HOPS=8
MEDICAL_P3063_PATH_MAX_HOPS=8
MEDICAL_P3063_PATH_MAX_PATHS=4
MEDICAL_P3063_PATH_PER_TEMPLATE=4
```

同时保留你之前 `MEDICAL_P3045_*` 的图谱审计文件配置，
特别是 `MEDICAL_P3045_GRAPH_AUDIT_PATH` 必须指向你的真实报告。

说明：
- `MEDICAL_P3063_INLINE_PATH_CARDS=1`：启用这次的独立路径展示；
- `MEDICAL_P3063_APPEND_GENERIC_CARDS=1`：使用现有的
  `knowledge_evidence` 列表，把4条路径追加为普通证据卡片；
- 即使 `MEDICAL_P3061_MULTIHOP_ENABLED=0`，上述路径
  **展示**查询也可以独立执行并复用完整的来源审计；
  但这不等于模型已经用多跳图谱路径进行答案推理。

`.env` 不要重复配置同一个键；如原来已存在，请直接修改旧值。

## 三、重启 Streamlit 后用哪个问题？

**首选演示题**（你已经验证在真实数据库中有8跳路径）：

> 哪些药品说明书提到了过敏性鼻炎？
> 请展示从疾病百科到药品来源、再连接其他疾病百科的8跳
> 来源路径，并说明这只是提及关系，不是药品疗效结论。

你应当在原有“实际命中的知识库证据”区域，看到原有
`medical.json` / 药品资料卡片后，又出现多张：

`GraphRAG 多跳路径 P01 · 8 跳（仅展示，不是回答引用）`

每张路径卡片有：
- 路径 ID，来源疾病、关联药品、另一疾病；
- 9个图谱节点类型与各自 ID；
- 8条边的关系名和方向；
- 可核验的 Section ID（如有）和候选原文片段；
- `clinical_verified=false`、`third_party_unverified` 声明。

若只看到6条原文却没有路径卡片，请检查最终 trace：

```json
"p3063_multihop_path_cards": {
  "status": "MULTIHOP_PATH_CARDS_APPENDED",
  "source_evidence_count": 6,
  "path_cards_count": 4,
  "appended_card_count": 4,
  "displayed_card_count": 10
}
```

如果状态正常，但页面仍不显示，说明你的前端在渲染证据时
使用了额外的 `evidence_type` / ID 过滤条件。此时需要修改
真实 Streamlit 主前端渲染代码，而不是重新查询或重建图谱。
上传前端 .py 源码即可按实际结构完成精确适配。

如果状态是 `NO_VERIFIED_PATHS`、`PATH_QUERY_BLOCKED_KEEP_ORIGINAL_EVIDENCE`
等，则保留原文证据列表、继续检查路径来源审计或 Neo4j 连接。

## 四、需要让证据区域内显示真实箭头图？（可选）

本包已经提供一个 Graphviz 可视化组件：
`render_rich_multihop_paths_inside_evidence(trace)`。

**可以直接把它放进你现有 Streamlit 主页面的**
“实际命中的知识库证据”**循环结束后、外层证据容器关闭前**：

```python
from medical_p3063_inline_path_evidence_v117 import (
    render_rich_multihop_paths_inside_evidence,
)

# 这里在现有的 evidence 卡片循环之后；trace 必须是本轮完整 trace
render_rich_multihop_paths_inside_evidence(trace)
```

再将 `.env` 的：

```dotenv
MEDICAL_P3063_APPEND_GENERIC_CARDS=0
```

这样就由该组件在**原证据区域**按 P01、P02、P03、P04
逐条绘制完整 8 跳的 Graphviz 图；不会在证据列表里重复显示
“普通路径卡片”。它使用 `st.container(border=True)`，
避免已有的外层 `st.expander` 内再嵌套另一个 `st.expander`。
每条路径附“查看逐跳关系与来源”的复选框。

**因为尚未提供实际 Streamlit 主前端源文件，这个最后
一行的“精确插入位置”没有被自动修改。**
因此本包的自动安装方式是原证据区域中的文字版多跳路径卡片，
而图形版需要将这一句挂到你真正的证据组件里。

## 五、如果想同时显示 5、6、7、8 跳

`.env` 改为：

```dotenv
MEDICAL_P3063_PATH_MIN_HOPS=5
MEDICAL_P3063_PATH_MAX_HOPS=8
MEDICAL_P3063_PATH_MAX_PATHS=16
MEDICAL_P3063_PATH_PER_TEMPLATE=4
```

每种路径模板最多返回4条（总上限16），避免前面5～7跳占满
名额导致8跳未执行。所有数字只是查询预算，不保证每种跳数
在每一个问题中都有实际命中。

不建议对每个普通医学问答都打开全范围5～8跳：
多跳检索增加 Neo4j 查询耗时，尤其是7/8跳路径组合较多时。
研究演示时建议先仅8跳，每问最多4张卡片。

## 六、科研边界与统计口径

“6条证据 + 4条路径”不能写成“10条独立医学证据”。

- **6条**是原有检索/引用系统返回的原文和图谱候选；
- **4条**是当前问题下与原始资料同名映射关联的路径；
- 多跳路径可能共享 Drug、SourceRecord、Section、疾病词条；
- 两条路径具有相同药品名与疾病名时，`path_id`
  不同可能来自不同候选实体；保留各自身份；
- 路径技术核验并不代表获得临床专家、官方审查或著作权审核；
- P3-06.3 只扩展 trace UI 表现，不修改回答与引用处理。

## 七、关闭、回滚

**仅关闭路径展示**（不需要重装）：

```dotenv
MEDICAL_P3063_INLINE_PATH_CARDS=0
```

重启 Streamlit，原来的证据列表与 GraphRAG 检索照旧。

**完全撤销代码注入**（先停 Streamlit）：

```powershell
$backup = Get-ChildItem . -Filter "rag.py.pre_p3063_*.bak" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $backup) {
    throw "没找到 P3-06.3 安装前备份"
}

Copy-Item $backup.FullName .\rag.py -Force
& $py -m py_compile .\rag.py
```

只恢复 `rag.py`，不涉及 Neo4j/Chroma 数据回滚。

## 交付测试说明

- 34项离线测试通过（来源属性、路径深度、证据保留、两次 trace
  赋值无额外查询、原文与生成隔离、默认关闭、Windows CRLF与备份）；
- 在真实 V1.17 原文件经 P3-03B → P3-04.3 → P3-04.5 → P3-06.1
  的**内存构造版本**上完成 AST 注入和临时文件备份恢复测试；
- 使用之前的 `medical_trace_007.json` 做了“原文2条+模拟8跳4条”的
  证据列表兼容性测试，原证据评分状态未改变；
- **未访问你的本机 Neo4j 或实际 Streamlit 主页面源码**。
  因此不能声称已在你的截图对应前端完成真实渲染验收；
  实际还需在你的本机重启后观察证据区域。
