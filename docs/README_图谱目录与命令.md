# 图谱文件目录与常用命令

图谱相关文件按用途集中在 `data` 下的四个目录中。

| 用途 | 目录 | 存放内容 |
|---|---|---|
| 图谱内容 | `graph_data/` | 概念、关系、别名、候选清单，以及 `sources/` 中的教材快照 |
| 证据绑定 | `graph_bindings/` | Chroma Chunk 快照、图谱关系对应的 Chunk ID 和绑定状态 |
| 处理报告 | `graph_reports/` | 绑定统计、失败原因、升级验收结果 |
| 操作工具 | `graph_tools/` | 在线检索辅助模块、离线绑定/扩充/诊断工具、统一路径配置 |

## 先看哪几个文件

- 看正式图谱中的概念和关系：`graph_data/knowledge_graph_v212.json`。
- 看每条关系绑定哪些证据：`graph_bindings/graph_chunk_links_v212.json`，查看 `claims` 中的 `claim_id`、`chunk_ids`、`status`。
- 看升级结果和未通过原因：`graph_reports/graph_v212_upgrade_report.json`。

V2.7 图谱和 V2.11.1 绑定是升级基线，保留用于复验。`*_candidates.json` 与 `V212_candidate_catalog_PREVIEW.json` 是候选材料，不能作为全部已验证的正式关系使用。

当前已保存结果为：基线 12 条、新增通过 17 条、正式合计 29 条；另有 1 条新增候选因证据对应多个 Chunk 而未纳入正式图谱。这里只复验保存的 Chunk 快照，不代表实时数据库内容已同步。

## 操作工具分工

| 文件（均在 `graph_tools/`） | 用途 |
|---|---|
| `graph_assist_v28.py` | 在线 `rag.py` 使用的图谱辅助检索模块 |
| `graph_chunk_linker_v211.py` | `export` 导出快照；`link` 绑定证据；`verify` 复验绑定 |
| `graph_expand_v212.py` | `propose` 提出候选；`promote` 纳入通过验证的关系；`verify` 复验升级 |
| `diagnose_graph_sections_v2111.py` | 只读检查图谱与 Chunk 的章节路径匹配情况 |
| `graph_paths.py` | 统一定义图谱内容、证据绑定、处理报告和教材快照的位置 |
| `__init__.py` | 使工具目录可作为 Python 包导入 |

`graph_assist_v28.py` 会调用绑定工具中的来源校验函数，启用图谱辅助检索时需要保留它们。

## 常用命令

在 `data` 目录执行以下只读验证，不需要调用模型或打开 Chroma 数据库：

```powershell
python -m graph_tools.graph_chunk_linker_v211 verify
python -m graph_tools.graph_expand_v212 verify
python -m graph_tools.diagnose_graph_sections_v2111
```

若终端位于项目根目录 `RAG_project`，可以用脚本路径执行：

```powershell
python .\data\graph_tools\graph_chunk_linker_v211.py verify
python .\data\graph_tools\graph_expand_v212.py verify
```

工具的默认输入输出使用基于脚本位置的绝对路径，不依赖终端当前目录。手动传入 `--graph`、`--chunks` 等相对路径时，仍按终端当前目录解释。

`export` 默认输出到 `graph_bindings/graph_chroma_chunks_v211.jsonl`；现有 `link` 和 `verify` 默认读取 `graph_bindings/chroma_chunks_v211.json`。现有文件虽然扩展名是 `.json`，实际为 JSONL。重新导出后，需要给绑定命令传入对应的 `--chunks` 路径。生成命令会拒绝覆盖已有文件；不要为复验而重新生成已有结果。

## 在线运行与自定义路径

主程序启动方式保持为在 `data` 目录执行 `streamlit run .\豆馅.py`。

`rag.py` 已改为从 `graph_tools.graph_assist_v28` 导入。辅助检索默认仍读取 V2.7 基线图谱；本次归类不会自动切换线上图谱版本或开启辅助检索。

未设置路径变量时使用：

- 图谱：`data/graph_data/knowledge_graph_v27.json`。
- 教材快照：`data/graph_data/sources/神经网络与深度学习.md`。

若在终端或配置中手动设置了 `DOUXIAN_GRAPH_FILE`、`DOUXIAN_GRAPH_SOURCE`，这些设置会优先于默认路径。指向旧位置的设置需要更新；优先使用绝对路径。

## 旧版本说明

其他版本 README 中的 ZIP 交付清单与测试成绩属于对应版本的历史记录。当前项目已清理过部分测试、预览和构建脚本；文件位置与现有验证命令以本说明为准。
