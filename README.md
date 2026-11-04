# data 主程序与数据目录

本目录包含豆馅 Streamlit 主页面、RAG 在线服务及其导入模块。主程序和运行时直接使用的文件保留在此目录；启动时应以本目录作为工作目录。`豆馅.py` 是页面入口，`rag.py` 编排检索与回答。历史测试、阶段报告/日志、补丁备份和过期盘点产物已移到同级的 `../待清理归档_2026-11-04/`。

| 子目录 | 功能 |
| --- | --- |
| `data/` | 原始药品与疾病资料，以及页面会话 SQLite 数据库。 |
| `data/raw/drug_db/MedGraphRAG-Drug-DB/` | 第三方药品与疾病原始数据集及其来源说明。 |
| `chroma_db/` | Chroma 向量集合持久化文件，运行时数据。 |
| `chat_history/` | RAG 对话上下文历史。 |
| `kg/`、`kg_output/`、`kg_output_clean_v2/` | 图谱构建工具、图谱导出及清洗版候选数据；可能被离线流程使用。 |
| `configs/`、`prompts/` | 图谱流程冻结配置及提示词。 |
| `expansion_manifests/` | 图谱扩容批次清单。 |
| `semantic_graph/` | 语义图谱构建/检索数据与配套文件。 |
| `stage3_reports/`、`stage4_reports/`、`reports/` | 阶段运行报告与审计产物。 |
| `review/`、`protocol/` | 人工盲审样本和标注规范。 |
| `queries/` | Neo4j 查询示例。 |
| `docs/` | 图谱、数据同步及维护文档。 |
| `disabled_streamlit_pages/` | 当前未启用的 Streamlit 页面。 |
| `chunk_previews/`、`demo/`、`pages/`、`scripts/`、`support/`、`tests/` | 分块预览、演示、页面扩展和辅助工具目录；当前可能为空。 |

## 主要文件

- `豆馅.py`：Streamlit 主页面、会话交互与证据展示。
- `rag.py`：在线问答流程编排。
- `knowledge_base.py`、`vector_stores.py`：文档入库、向量存储与混合检索。
- `drug_graph_retriever.py`、`medical_semantic_graph.py`：Neo4j 药品/疾病及语义图谱检索。
- `evidence_guard.py`、`self_rag_retry.py`、`medical_answer_quality.py`：证据检查、受限补检和回答质量控制。
- `chat_store.py`、`file_history_store.py`：页面会话和 RAG 消息历史存储。
- `config_data.py`、`.env`：Chroma 路径/集合设置及本机服务配置。
- `medical_*.py`：医疗检索、证据处理、索引、扩容、界面辅助和质量评估模块；在线调用的模块保持在本目录以维持导入路径。
- `README_*.md`、`使用说明_*.txt`、`项目架构与文件功能说明.md`：操作和架构说明。
- `requirements-kg.txt`：图谱离线工具依赖。

`豆馅.py`、`rag.py` 及其同目录模块之间通过 Python 模块名直接导入。运行在线页面时，请先激活项目环境并在 `data/` 执行 `streamlit run 豆馅.py`；不要把这些在线模块拆到子目录，否则需要同步修改导入和启动配置。

## 主要数据文件

- `data/chat_history.sqlite3`：Streamlit 页面会话与消息的 SQLite 持久化数据库。
- `kg_output/`：早期图谱清洗与构建结果；保留作数据追溯。
- `kg_output_clean_v2/`：清洗版药品、来源、章节和关系候选；供离线图谱/评估工作流使用。
- `semantic_graph/medical_graph.sqlite`：本地语义图谱索引与元数据；目录内 PNG/HTML 是图谱预览。
- `configs/`、`expansion_manifests/`、`prompts/`：固定实验配置、扩容名单与抽取/审核提示词。
- `review/`、`protocol/`：人工审核数据和标注规则。

历史已空目录 `chunk_previews/`、`demo/`、`pages/`、`reports/`、`scripts/`、`support/`、`tests/` 和 `v117_route_patch/` 暂保留，供现有工作流后续复用。
