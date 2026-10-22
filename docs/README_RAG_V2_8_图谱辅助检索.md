# 豆馅 RAG V2.8｜可开关知识图谱辅助检索

## 1. 这版完成了什么？

基于 V2.6 正式源码和已经离线完成的 V2.7 图谱，本次增加图谱检索辅助。

- 默认 `DOUXIAN_GRAPH_MODE=off`：保留 V2.6 原有检索词和调用路径。
- 可选 `DOUXIAN_GRAPH_MODE=assist`：**查询分解完成后**，根据原教材已验证的知识图谱提供少量别名和概念关系关键词；
  不新增检索子问题，不改变单条/多条问题的 Reranker 参数。
- 任何图谱文件缺失、版本不合或与原 Markdown SHA256 不一致的情况，自动原样使用原来的子查询。
- 图谱**绝不作为答案证据**。最终回答仍只引用原本 Chroma / BM25 / Reranker 选出的真实 Chunk。
- 不改 `knowledge_base.py`、`evidence_guard.py`、`self_rag_retry.py`、前端和现有知识库索引。
- 不新增 Qwen 或 Reranker 调用次数；提示词与原有 V2.5/V2.6 检查规则都保留。

注意：Graph Assist 是**检索词扩展**，不代表已经实现“图谱推理直接回答”或完整 GraphRAG。
某些问题 Qwen 查询分解依然可能耗时 25 秒以上，**本版不会解决 B04 的查询规划瓶颈**。

## 2. 来源、能力边界与安全性

离线图谱由用户提供的《神经网络与深度学习.md》生成：

- 546 个节点：文档 1、章节 510、概念 35；
- 1,258 条关系：其中 510 条章节归属，736 条术语提及；
- **只有 12 条**由程序核对到完整原文出处的明确语义关系（例如 Gibbs 采样 IS_A MCMC）。
- “提及同一术语”或“处于同一章节”绝不等同于语义从属/因果关系。
- 每条边记录来源文件、章节路径、行号和原文证据摘录，V2.7 verify 可校验。

检索词辅助示例：

问题：`卷积层与池化层分别完成什么运算？`
原子查询：`卷积层与池化层分别完成什么运算？`
扩展子查询：`卷积层与池化层分别完成什么运算？ 汇聚层`

规范名“汇聚层”作为检索词帮助找到教材中的标题，并不自动在答案中新增任何解释。
人工映射的查询别名本身不算作从教材抽取的事实，不应直接当作引文。

## 3. 历史 ZIP 文件清单

当前目录与可用命令见 [图谱目录与命令](README_图谱目录与命令.md)。以下是历史交付记录，其中部分测试、预览及构建脚本已不在当前项目中。

- `rag.py`：完整 V2.8 RAG 主模块；在 V2.6 基础上仅添加检索词扩展及日志/trace。
- `graph_assist_v28.py`：离线图谱加载、SHA 核验、关键词扩展，只有标准库。
- `knowledge_graph_v27.py`：从 Markdown 离线构建、查询及来源验证。
- `knowledge_graph_v27.json`：预建图谱，不用再次运行构建。
- `graph_sources/神经网络与深度学习.md`：上传教材的原始快照；单独放入新目录，避免覆盖你项目已有原文件。
- `preview_graph_v28.py`：无需 Qwen 的术语扩展预览。
- `test_graph_assist_v28.py`：9 组 V2.8 离线测试。
- `test_knowledge_graph_v27.py`：7 组 V2.7 离线测试。
- V2.6 原有四组回归测试：`test_fast_answer_prompt_v26.py`、`test_fast_kb_quality_v25.py`、
  `test_kb_fast_strict_v24.py`、`test_answer_footer_dedup_v241.py`。
- `rag_V2_6_原始备份.py`、`rag_V2_6_vs_V2_8_图谱接入.diff`：用于回滚/核对差异。
- 本 README。

## 4. 当前项目的启动与验证

先在 Streamlit 终端按 Ctrl+C 停止进程，切换到：

    cd 'E:\VsCode_Project\DockerTest_Agent\RAG_project\data'
    conda activate rag

当前文件已经按四类目录整理，无需重新解压旧版本 ZIP。启动方式仍是 `streamlit run .\豆馅.py`。

### 离线验证（不收费）

    python -m graph_tools.graph_chunk_linker_v211 verify
    python -m graph_tools.graph_expand_v212 verify

旧版独立预览脚本已不在当前项目中。在线辅助模块位于 `graph_tools/graph_assist_v28.py`，来源校验复用 `graph_tools/graph_chunk_linker_v211.py`。

### V2.8 默认行为：先测 off

    $env:DOUXIAN_KB_MODE="fast"
    $env:DOUXIAN_GUARD_FAST="0"
    $env:DOUXIAN_FAST_QUERY_PLAN="1"
    $env:RERANK_DOC_CHARS="600"
    $env:DOUXIAN_FAST_QUALITY_PROMPT="1"
    $env:DOUXIAN_GRAPH_MODE="off"
    streamlit run .\豆馅.py

用 B04 的提问和 B01 的提问各测试一次，记下：
总耗时、首字时间、引用章节、Rerank total_tokens 和模型回答。

### 开启图谱：assist

停止 Streamlit，**仅改一个变量**：

    $env:DOUXIAN_GRAPH_MODE="assist"
    streamlit run .\豆馅.py

再在新的会话中重问相同问题。终端可能输出：

    [Graph Assist] status=applied hints=1 耗时=0.0030s

首次加载 JSON 和校验整本 Markdown 可能稍慢，后续命中进程内缓存。
如果是 `no_match`：没有必要增加的提示词（属于正常情况）；
如果是 `missing_file`：检查两个路径；
如果是 `invalid_or_stale`：教材内容和图谱不匹配，已回退原始检索。

自定义文件路径：

    $env:DOUXIAN_GRAPH_FILE="E:\你的目录\knowledge_graph_v27.json"
    $env:DOUXIAN_GRAPH_SOURCE="E:\你的目录\神经网络与深度学习.md"

默认路径在 `data/graph_data/knowledge_graph_v27.json` 和
`data/graph_data/sources/神经网络与深度学习.md`。单纯修改教材原文件会使来源 SHA256 校验失败，辅助检索将回退。旧版 `knowledge_graph_v27.py` 构建脚本当前不存在；现有离线工具负责绑定与增量扩充，不能代替从教材重建基线。

**注意**：即使源文件版本一致，也不能证明你当前 Chroma 索引一定使用同一批 Chunk。
V2.8 只是检索词辅助，引用仍以真实检索结果为准。

## 5. 判断是否值得保留 assist

建议进行 B01、B04、B05 的 off/assist A/B，使用相同知识库、模型、
Rerank 配置与新的会话。检查：

- B01：MCMC / Gibbs 层级是否准确，检索证据实际包含相关教材章节。
- B04：是否稳定找到 `5.2.2 卷积层`、`5.2.3 汇聚层`，是否引用真实教材 Chunk。
- B05：无项目正式实验成绩时，是否仍保持证据不足、不编造数值。
- API 消耗：总 Rerank 官方 Tokens 可能随检索候选变化，**调用次数不增加**不等于
  实际 Token 数量必然相等。
- 总耗时目标 40 秒：主要 Qwen 查询分解调用 **不会因图谱查询扩展而被跳过**。
- status=applied 只说明扩展词真的加进查询，不代表最终答案一定改善。

图谱切换状态会进入 `trace["graph_assist"]`（如果知识库正常返回了实际证据）。

## 6. 恢复 V2.6

临时关闭（不需要覆盖代码）：

    $env:DOUXIAN_GRAPH_MODE="off"
    streamlit run .\豆馅.py

恢复原文件：

    Copy-Item .\rag_before_graph_v28.py .\rag.py -Force

`graph_data/`、`graph_tools/graph_assist_v28.py` 不影响 off 运行，
可以保留等待以后继续开发。

## 7. 离线验收范围

在**含完整教材副本**的隔离目录中，执行通过：

- 历史整合源码 72 组；
- V2.4.1 参考文献尾注 5 组；
- V2.5 快速模式轻量质检 8 组；
- V2.6 生成提示约束 7 组；
- V2.7 图谱构建与来源校验 7 组；
- V2.8 辅助检索/真实生产函数模拟 A/B 9 组；
- 总计 **108 组不同的离线测试**，源码编译通过。

没有向真实 DashScope / Qwen 发请求，没有在你的 Windows 环境测实际索引与延迟，
所以**尚未证明图谱在真实问答中能提升召回、准确率或速度**。

下一步收到 off/assist 的真实日志后，再决定是否进入图谱关联 Chunk 的 V2.9。
