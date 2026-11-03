# 豆馅医疗 GraphRAG V1.17 · 8 跳路径证据区显示修复

## 修复了什么？

你当前的“实际命中的知识库证据”区域能看到多条图谱路径，
但即使提问“请展示8跳来源路径”，展示结果仍混合 5、6、7、8 跳，
并可能只显示1条8跳。

根因：前端原先无条件合并 P3-06.1 的在线路径、
P3-06.3 的完整路径缓存和通用路径卡，不按用户指定跳数过滤；
P3-06.1 的在线检索默认最多返回10条，短路径可能先占满名额，
但前端认为“已经有路径”而跳过8跳专项验证。

修复内容：
- 按 `MEDICAL_P3063_PATH_MIN_HOPS/MAX_HOPS` 限制**实际展示**的路径；
- 用户明确写“8跳”而非“5～8跳”时，只展示8跳；
- 优先使用 P3-06.3 的完整图谱路径（有完整原始来源元数据）；
- 如果已有的是混合跳数且没有完成8跳专项查询，完成后只执行
  **至多一次按指定深度的 P3-06.1 只读查询**（最多4条8跳）；
- 使用图谱记录的疾病名称；如果紧凑 trace 中没有名称，
  仅在原图谱 matched_diseases 唯一时采用该核验名称，不乱猜；
- 路径卡标题显示“路径 P01 · 8跳”，便于复制阅读；
- 保持原有 6 条原文证据编号、答案引用和质量统计，图谱路径
  只作为展示，不参与模型生成、临床判断、统计增量或推断治疗关系。

## Windows 安装

先停止 Streamlit，在项目目录运行：

```powershell
cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"
$py = "E:\Miniconda\envs\rag\python.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

# 备份当前豆馅主程序
Copy-Item ".\豆馅.py" ".\豆馅.py.before_8hop_fix_$stamp.bak"

# 把下载的 zip 放在当前目录
Expand-Archive `
  -LiteralPath ".\豆馅_P3063_8跳路径显示修复包.zip" `
  -DestinationPath "." -Force

& $py -m py_compile ".\豆馅.py"
& $py -m unittest -v test_douxian_multihop_fix_v117.py
```

预期 `Ran 16 tests ... OK`。

## .env 配置（同名键只保留一份）

```dotenv
MEDICAL_P3063_INLINE_PATH_CARDS=1
MEDICAL_P3063_APPEND_GENERIC_CARDS=0
MEDICAL_P3063_PATH_MIN_HOPS=8
MEDICAL_P3063_PATH_MAX_HOPS=8
MEDICAL_P3063_PATH_MAX_PATHS=4
MEDICAL_P3063_PATH_PER_TEMPLATE=4
```

继续保留以前 P3-04.5 图谱审计配置。
本包只替换前端 `豆馅.py`，**不用重新打 rag.py 补丁**，
不修改 Neo4j/Chroma/medical.json。

然后用你以前的命令重启 Streamlit：

```powershell
& $py -m streamlit run ".\豆馅.py"
```

重新提问：

> 哪些药品说明书提到了过敏性鼻炎？请展示8跳来源路径。

预期原有“实际命中的知识库证据”保持原始条数。
下方单独显示“多跳图谱来源路径 · 8跳”，只列本轮真实验证的8跳结果；
若图谱仍只有少量路径，**不可强行声称有4条**。

## 特别说明：正文回答为何仍说“无法回答8跳”？

现有 GraphRAG/P3-06.3 把多跳路径保存到追踪数据以供界面渲染。
这些 **display-only 路径不传给模型作为医学事实证据**，
因此模型正文可能仍写“证据不足”，而 UI 的路径卡展示成功。
本修复解决的是**跳数过滤与路径完整展示**，不是多跳链式推理。

如果下一阶段希望让 AI 正文真正讨论“哪几条8跳路径”，需单独构建
可核验的技术路径摘要作为独立回答输入并保持与临床推荐完全隔离。
不要仅凭药品-疾病图谱连通就让模型推断治疗效果。

## 其他判断

`Neo4j 原文核验0条` 属原有独立来源验证统计，
**不能直接理解为多跳图谱路径校验为0**。
`本次耗时38秒` 为总耗时，可能包括药品检索、LLM生成、
多跳读取等；不能凭这一数字单独判断 Neo4j 性能。

## 回滚

停止 Streamlit，从本轮 `.bak` 文件恢复：

```powershell
Copy-Item ".\豆馅.py.before_8hop_fix_你的时间戳.bak" ".\豆馅.py" -Force
& $py -m py_compile ".\豆馅.py"
```

## QA / 安全边界

- 完整主程序 Python AST 解析通过；
- 16个离线测试通过：只8跳过滤、5～8跳配置、
  重名路径去重、缺失疾病名防误判、HTML复制可读性、
  混合路径下只读8跳专项查询、缓存不重复扫描、失败安全回退、
  原文证据条数不增加、治疗关系拒绝。
- 本地没有你的 Windows Neo4j 网络连接，因此尚未替你现场验收真实8跳。
- 所有第三方药品资料仍为 `third_party_unverified`，
  不能声称其已获批准或适用于治疗相关疾病。
