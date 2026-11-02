# Chunk 切分预览目录

导出时间：2026-11-02T10:36:13+08:00。此目录集中保存两套实际数据库切块的查阅快照。

| 文件 | 用途 | 数量 |
| --- | --- | ---: |
| [neo4j_preview.md](neo4j_preview.md) | 图谱切块的完整正文样例 | 31 |
| [chroma_preview.md](chroma_preview.md) | 检索切块的完整正文样例 | 30 |
| [neo4j_chunks.jsonl](neo4j_chunks.jsonl) | 实际 Neo4j Chunk 全量导出，附药名、栏目及来源 | 107,787 |
| [chroma_chunks.jsonl](chroma_chunks.jsonl) | Chroma rag 集合全量导出，保留存储正文与元数据 | 52,104 |
| [manifest.json](manifest.json) | 数量、栏目统计、来源与文件校验值 | — |

JSONL 每行是一块，统一外层结构为 `id`、`document`、`metadata`；Neo4j 的原始 Chunk 属性在 `metadata.chunk_properties` 中。搜索 `drug_name` 可以定位药品，搜索 `section_id` 可以对照两套切块。

Neo4j 与 Chroma 使用不同切分器、参数和栏目范围。Chroma 正文增加了药品与栏目等前缀，两者通过 Section ID 和原文字符位置对应，不要求 Chunk ID 或块数相同。这里的正文都是药品说明书；疾病百科待入库切块不计入本目录。

本目录是导出快照，业务程序不从这里读取。原 `kg_output_clean_v2/chunks.jsonl` 保留在原位置，图谱构建、导入和维护脚本无需改路径。Chroma 数据目录也保持原位置。未更改数据库内容；文件导出不调用模型或 Embedding。

脚本所有输入默认根据脚本所在目录定位，可从任意工作目录调用。刷新时指定新的目录，已有目录拒绝覆盖。例如在 PowerShell 执行：

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
python 'E:\VsCode_Project\DockerTest_Agent\RAG_project\data\scripts\export_chunk_previews.py' --output "E:\VsCode_Project\DockerTest_Agent\RAG_project\data\chunk_previews_$stamp"
```

生成脚本：[export_chunk_previews.py](../scripts/export_chunk_previews.py)。导出期间请避免同时运行数据库入库任务；两个数据库分别读取，导出不构成跨库的同一时刻事务快照。
