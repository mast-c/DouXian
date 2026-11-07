# GraphRAG v5.4 | 基于 v5.3 的证据链优先级修复

## 本次基线
- `rag.py`、`medical_evidence_coverage.py` 以交付的 **v5.3** 为基线。
- `evidence_guard.py`、`medical_category_discovery.py` 以 **v5.3 所依赖的 v5.2** 为基线（v5.3 本身没有替换这两份文件）。
- 原 `medical_semantic_retrieval.py`、`_proof()`、Neo4j / SQLite 遍历逻辑及单跳/多跳 UI 组件完全未修改。
- 请先确认本机没有在 v5.3 后单独修改上述四个文件；有修改时应按 `v5_3_to_v5_4.patch` 合并。

## P0 已修复

### 1. 审核证据转换时丢失 `discovery_role`
- `rag.py` 在进入 Evidence Guard 和重建 UI 证据时保留 `discovery_role`、`product`、`partner`、`graph_path`、来源状态和任务身份等字段。
- 新增 `medical_evidence_contract.py`，用于证据元数据传递与稳定来源 ID 核查。
- Trace 新增 `evidence_role_transport`，分别记录审核前、审核证据、补检后的角色数量。

### 2. 完整分类节点名称被错误当作无效引文
- `evidence_guard.py` 对 `DATASET_CATEGORY_MEMBERSHIP → DATASET_CATEGORY_PARENT` 做来源结构限定的引用验证。
- 仅允许经过原检索器 JSON Pointer / SHA-256 来源核验的完整分类节点名称，如“镇痛解热”；**不是**放宽所有短引文要求。
- 当模型漏掉分类节点引用时，只允许恢复已通过完整分类层级来源证明的**本地数据集分类事实**，标记 `validation_method=verified_dataset_json_structure_only`。
- 不自动恢复临床药理结论、药物相互作用、适应症、剂量或安全性。
- 数据集分类 ≠ 官方药理学分类；回答中必须保留来源限定。

### 3. 补检后原证据丢失
- 补检前保存独立证据与审核快照。
- 补检结果如果删掉、改写或重编号原证据，或引用不存在的 ID，则回退到原审核状态。
- Trace 新增 `retry_evidence_integrity`，不会因为恢复证据而绕过 Evidence Guard。

## P1 已修复

### 4. 分类目标误进入说明书栏目补检器
- `dataset_category`、`discovered_sources`、`seed_pair_interaction` 等非说明书字段，不再传入只支持说明书字段的 `medical_targeted_retry.py`。
- 没有可补检的说明书目标时，记录 `skipped_non_section_targets` 而不是触发 `KeyError: dataset_category`。
- **边界：** 这解决了错误路由；不是增加一个全新的数据库分类补检器。分类数据仍来自已经执行并核验的分类图谱检索；真正缺失来源时应保留证据不足状态。

### 5. “同类药之间的示例”被误当成“与起始药的相互作用”
- `medical_category_discovery.py` 当首轮 LLM 生成的相互作用子查询明确命名起始药品时，把它设为实际相互作用端点约束；不再无条件查任意同类药与第三药的示例。
- `medical_evidence_coverage.py` 将目标拆为“起始药分类”“候选药分类成员”“与起始药的具体相互作用”，分别绑定。
- 其他候选药与无关第三方之间的关系只可用于候选检索线索，不能替代与起始药的原文证据。
- 允许使用真实“制剂→成分”证据连接明确相互作用陈述，但仍须独立通过来源验证。

## 必须复制的文件

**替换（4 个）：**
- `rag.py`
- `medical_evidence_coverage.py`
- `evidence_guard.py`
- `medical_category_discovery.py`

**新增（1 个）：**
- `medical_evidence_contract.py`

**可选诊断 / 测试（旧版测试也一并附带，可以直接运行）：**
- `diagnose_evidence_pipeline_v54.py`
- `test_v52_regressions.py`
- `test_v53_regressions.py`
- `test_v54_regressions.py`
- `BASELINE_SHA256.json`
- `v5_3_to_v5_4.patch`

原 v5.3 的 `medical_semantic_retrieval.py`、`medical_graph_visibility.py`、
`medical_graph_path_ui.py` 等**保留原样**。

## Windows PowerShell 安装（位于 RAG_project/data）

先停止原 Streamlit / Python 服务，备份四个文件：
```powershell
Copy-Item .\rag.py .\rag.py.bak_v53
Copy-Item .\medical_evidence_coverage.py .\medical_evidence_coverage.py.bak_v53
Copy-Item .\evidence_guard.py .\evidence_guard.py.bak_v53
Copy-Item .\medical_category_discovery.py .\medical_category_discovery.py.bak_v53
```

将修复包中 5 个核心文件复制到 `RAG_project/data`，保留现有 `.env`。
运行：
```powershell
python -m py_compile rag.py medical_evidence_coverage.py evidence_guard.py medical_category_discovery.py medical_evidence_contract.py
python -m unittest -v test_v52_regressions test_v53_regressions test_v54_regressions
streamlit run .\豆馅.py
```

修复包包含上述全部三份离线测试脚本；将它们与核心模块放在同一项目目录即可执行测试。测试完成后可移动到 `tests/` 目录保存。

## 真实问题验收

重试：
“复方氨酚烷胺片所属的大类与该大类下的什么药物存在相互作用？”

预期：
1. 语义识别继续是 `classification + interaction`；
2. 已验证的分类链仍可用于来源限定的分类回答；
3. `evidence_role_transport.guard_before_retry.roles.seed_category` 不为零（如果检索实际返回起始药分类）；
4. `retry_evidence_integrity.status` 应为 `verified`，或在异常时 `restored_initial_review`；
5. 对仅有其他药品组合示例的证据，“与复方氨酚烷胺片存在相互作用”仍应判为证据不足；
6. 不应因补检器不认识 `dataset_category` 而出现 `retry_error`；
7. 分类源或快照发生变化时，原有 `GraphBlocked` 仍必须阻断无效证据。

诊断新 trace：
```powershell
python diagnose_evidence_pipeline_v54.py --trace ".\medical_trace_最新.json"
```
脚本只输出任务类型、阶段计数和状态，不打印原问题或药品说明书原文。

## 测试与限制
- **37 项离线测试通过**：v5.2 11 项、v5.3 12 项、v5.4 新增 14 项。
- 利用已上传 `medical_trace_017.json` 做只读重放：结构化分类引句可通过；与起始药相互作用的证据仍需单独核实。
- **未连接真实 Qwen、Neo4j、SQLite 分类源、Streamlit 主页面做端到端验收。**
- 对具体分类缺失时不会伪造节点，也不会绕过原始来源校验。
- 由于尚未获得 `medical_targeted_retry.py` 的实际源码，本次采用安全路由避开其不支持的分类字段，未改其内部实现。
