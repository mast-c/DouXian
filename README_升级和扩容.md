# 豆馅医疗 Hybrid GraphRAG · V1.8 正式问答质量升级

## 当前已经完成（按用户本机执行结果）

- `drug_db_v2_cleaned.json`：11,953 个药品。
- Neo4j 已包含药品、SourceRecord / Section、疾病提及候选；保留“候选 ≠ 治疗关系”。
- Chroma `rag`：先前 6 条；第二批 18 个药品新增 82 条，合计 **88 条 Chunk**。
- 新增 82 条通过 Neo4j 原文精准匹配、无强制截断；用户反馈后续只读验收与 Hybrid 检索已通过。
- 已有 V1.4 `medical_evidence_bridge` 用于 Chroma Chunk → Neo4j Section 逐字校验。

**这次 V1.8 不再重复导入数据，也不改 `knowledge_base.py` 的 BM25、向量检索、RRF、Qwen Reranker 或 JSON 切分规则。**

## 本版具体改动

只替换 `rag.py` 并新增 `medical_answer_quality.py`。

1. **修复被提前忽略的真实原文：**旧版检测到疾病提及候选、且用户问治疗/适应症时，可能提前返回候选列表，而未采用同时命中的已核验 Chroma/Section 原文；现在只在所有证据*都是*候选关系时走保守固定模板。混合证据交给已有 Evidence Guard + LLM，但未审核候选仍不能直接当成治疗关系。
2. **参考资料合并：**相同真实来源链接的 `[1]`、`[2]` 会放在同一来源下展示，保留证据编号和不同章节，明确并非两份独立医学证据。
3. **清理视觉噪声：**在模型流式输出的完整行中移除空 `-` 列表项和 `[svg](http://localhost:8501/#1)` 这类本地伪引用；不改动正常 `[1]`、Markdown 列表与正文。
4. **减少过长警告：**不再在 LLM 输出前额外插入大段 `⚠️ 图谱说明`；未核验事实仍受到系统规则、证据标记与候选固定模板约束。
5. **医疗回答提示词收敛：**优先结论、原文、状态一句话；待人工审核不能说成审核不通过；不能仅因第三方未核验就宣称该药没有监管批准。
6. **删除计算机知识库遗留的 MCMC/Gibbs 快速模式检查**（保留有效引文编号与长句重复的通用浅层检查）。

本包只是输出质量和检索分支修复，不代表能自动验证临床医学结论。生成模型仍有误述风险，用户需核对原文和正式说明书。

## 安装

这份修改是从上一轮 **V1.4 父级 Section 回溯版本的 rag.py** 制作的。

先备份当前 `rag.py`；如果你本地已经自行继续改过 `rag.py`，优先通过 `rag_v14_to_v18.patch` 合并差异，**不要直接覆盖自己之后的修改**。

把包内两份文件放在 `rag.py` 所在同一目录：

- `rag.py` —— V1.8 更新版（基于 V1.4）
- `medical_answer_quality.py` —— 新的纯 Python 输出清理和引用合并工具

**以下文件不用更换**：`knowledge_base.py`、`config_data.py`、`豆馅.py`、`medical_evidence_bridge.py`、`drug_graph_retriever.py`、`medical_json_bulk_index.py`、`medical_json_chunker.py`、`chroma_db`、Neo4j、`.env`、SQLite 检查点。

在项目目录重启：

```powershell
python -m py_compile rag.py medical_answer_quality.py
streamlit run "豆馅.py"
```

已保存的历史回答是旧文本，不会自动重新排版；在新会话中重新询问即可看到变化。

## 下一步操作：从 88 条 Chunk 安全扩容到数百条

因为你之前的 V1.5 Checkpoint 已有 `done=18`、`no_chunks=2`，再次启动写入模式会自动跳过它们，挑选接下来符合条件的药品。**不用手动清理检查点或重新处理阿托伐他汀钙片。**

建议先预览，再按权限与费用预算进行一小批写入：

```powershell
python medical_json_bulk_index.py --json "data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --limit-drugs 50 --verify-neo4j
```

注意：当前 V1.5 的 `--verify-neo4j` 是只读模式，不加载写入 Checkpoint，可能重新预览头 50 个药品。它不等于“下一个尚未入库的 50 个药品”。若希望验证新一批，需要用 `--start-after "一粒珠"` 辅助选择药名。

实际写入（**会发送原文给 DashScope、产生费用**；确认数据许可后才执行）：

```powershell
python medical_json_bulk_index.py --json "data\raw\drug_db\MedGraphRAG-Drug-DB\drug_db_v2_cleaned.json" --limit-drugs 50 --max-new-chunks 300 --max-embed-chars 160000 --embedding-batch 8 --pause-seconds 0.5 --apply --allow-cloud-embedding
```

这条写入命令会读取原有 `medical_bulk_state.sqlite3` 并自动跳过已经完成的 20 个药品，遇到预算不足时安全停止。单轮可能未处理满 50 药品；这不是错误。

运行后查看：

```powershell
python medical_json_bulk_index.py --status
python diagnose_chroma.py
```

再使用当前唯一“知识库”模式提出：
`5-氨基水杨酸肠溶片的适应症是什么？请给出原文证据。`
以及
`阿托伐他汀钙片与动脉粥样硬化有什么关联？`

期待同一来源的候选与原文在脚注中合并链接；明确提示“提及/风险≠治疗关系”。

## 整体目标（见 ROADMAP_后续目标.md）

最终要做成**基于药品原文、结构化疾病候选和可追溯证据的中文医疗问答系统**，
而不是把未审核疾病提及自动宣布为获批适应症的药品推荐器。

在完成这轮扩容后，优先建设**按题型抽样的问答评测集**和**医学关系人工审核工作流**；
再决定是否扩大到全部 11,953 个药品，最后做检索加速和部署。

## 离线检查

```powershell
python -m unittest discover -s tests -p "test_medical_answer_quality.py" -v
```

10 项离线测试涵盖伪引用过滤、流式数据切块、相同 URL 的引用分组、重复小标题、分支是否保留已核验原文以及医疗安全提示词。已模拟 `rag.py` 的混合证据正式流式生成路径，但没有连接用户本机 Qwen、Chroma、Neo4j 或 Streamlit 实测。
