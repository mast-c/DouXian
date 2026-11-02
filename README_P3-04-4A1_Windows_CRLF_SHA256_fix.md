# 豆馅医疗 GraphRAG V1.17 · P3-04.4A.1 Windows CRLF/SHA256 兼容修复

## 你本地这次运行究竟发生了什么？

从你上传的 PowerShell 日志：

- P3-04.4A 的 22 项离线测试中有 **7 FAIL、9 ERROR**；
- 错误全部回溯到相同根因：
  `P3-04.3 映射文件 SHA256 与对齐审计不一致`；
- 而 `medical_p3043_disease_neo4j_align_v117.py` 已经运行成功：
  `EXACT_NAME_ALIGNMENT_COMPLETE_WITH_GAPS`；
- 实际 Neo4j `DiseaseTerm` 数量 1,150，
  `medical.json` 疾病名称 8,806，其中严格同名 1,150，
  未匹配 7,656（不是 7,656 个运行失败）。

## 根因

旧 P3-04.3 对齐脚本在生成 JSONL **写文件前**，按 UTF-8 + `\n` (LF)
字符串计算 `links_sha256`。但它使用 Python 文本模式输出。
Windows 会将写入的 `\n` 自动转换成 `\r\n` (CRLF)，导致：

- 审计报告 `links_sha256`：原始 LF 文本的哈希；
- 实际 JSONL 文件字节 SHA256：Windows CRLF 文本的哈希。

这些文件的 JSON 内容可以完全一致，字节级哈希却不同。
这是跨平台换行兼容问题，**不能据此认定源文件或图谱被篡改**。
但原程序安全地拒绝了不一致，所以还没有生成有风险的图谱关联。

## 本修复的范围

ZIP 中只有：

1. `medical_p3044_disease_graph_link_plan_v117.py`
2. `test_medical_p3044_v117.py`
3. `README_P3-04-4A1_Windows_CRLF_SHA256_fix.md`

**只覆盖前两个 Python 文件，不修改：**
`medical_p3043_disease_neo4j_align_v117.py`、`rag.py`、Neo4j、
`chroma_db`、SQLite、`medical.json`、P3-04.2 全量审计报告或
已经生成的 `p3044a_term_align_*.json` / `p3044a_term_links_*.jsonl`。

新 `medical_p3044_disease_graph_link_plan_v117.py`：

- 首先仍然尝试对映射文件进行**原字节 SHA256 严格比较**；
- 如果不一致，只允许**所有行均为标准 CRLF**，并在将 CRLF
  规范化为 LF 后与历史审计哈希完全一致，才继续；
- 拒绝混合换行、缺失行尾、任意 JSON 内容改动、不同原始疾病 ID；
- 新报告字段 `alignment_link_hash_mode` 为
  `WINDOWS_CRLF_EQUIVALENT` 或 `BYTE_EXACT`，便于审计；
- **仅生成关联计划、未审核 CSV 和本地 JSON 审计报告**；
  实际 Neo4j 查询仍只有 `MATCH/RETURN`。

已用 26 项离线测试（22 项旧用例 + 4 项 Windows 行尾专项测试）验证。

## 安装与验证（PowerShell）

在 Windows 项目目录解压 ZIP：

`E:\VsCode_Project\DockerTest_Agent\RAG_project\data`

建议解压前先备份原来的两个脚本；无需备份/重建数据库。

```powershell
cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"
$py = "E:\Miniconda\envs\rag\python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $py -m unittest -v test_medical_p3044_v117.py

if ($LASTEXITCODE -ne 0) {
  throw "P3-04.4A 单元测试仍未全部通过，停止后续执行。"
}
```

预期：`Ran 26 tests ... OK`。

**你之前的真实 P3-04.3 对齐报告及映射可以直接复用：**

```powershell
$raw = ".\data\raw\drug_db\MedGraphRAG-Drug-DB\medical.json"

$align = ".\stage4_reports\p3044a_term_align_20261102_163244.json"
$links = ".\stage4_reports\p3044a_term_links_20261102_163244.jsonl"

$auditFile = Get-ChildItem ".\stage4_reports" `
  -Filter "p3042_final_audit_*.json" -File |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if (-not $auditFile) {
  throw "找不到 P3-04.2 疾病 Collection 完整审计报告。"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

$summary = ".\stage4_reports\p3044a_graph_plan_$stamp.json"
$plan = ".\stage4_reports\p3044a_candidates_$stamp.jsonl"
$review = ".\stage4_reports\p3044a_review_$stamp.csv"

& $py .\medical_p3044_disease_graph_link_plan_v117.py `
  --source $raw `
  --align-report $align `
  --align-links $links `
  --chroma-audit $auditFile.FullName `
  --report $summary `
  --plan $plan `
  --review-csv $review

if ($LASTEXITCODE -ne 0) {
  throw "P3-04.4A 只读图谱关联规划仍未通过；不要执行任何图谱写入。"
}
```

预期（以实际 Neo4j 当前状态为准）：

`[P3-04.4A] READY_FOR_HUMAN_REVIEW_NO_WRITES`

通过后，打开 `$summary` 可查看
`alignment_link_hash_mode`、`aligned_unique_names`、
`aligned_records_in_review_plan`、`graph_recheck` 等字段。

如果出现新的 `BLOCKED`（例如源文件变化、Neo4j ID 改变、
已有 `DiseaseDocument` ID 冲突），**这是新的安全检查结果**，
不要自动跳过，也不要删除数据库或旧审计报告。

## 哪些暂时不用处理？

- 7,656 个未匹配疾病名称：**现在不需要全部补齐**。
  它们并不影响已有疾病百科在线回答。
  后续可以独立做别名/术语扩容，绝不凭模糊相似度直接确认疾病关系。
- 已经入库的 39,443 条疾病 Chunk 不用重算 Embedding。
- 原药品 `rag` Collection 不用重建。

请将新的 `p3044a_graph_plan_*.json` 上传，确认真实同名覆盖率和
是否存在现有文档 ID 冲突，再安排下一阶段图谱写入。
