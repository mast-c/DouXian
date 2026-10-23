# 豆馅 V2.16 — 安装器兼容性热修复

## 这个补丁修复什么？

用户本机 V2.13.1 运行时代码 `graph_tools/graph_retriever_v213.py` 的 SHA256
和创建 V2.16 时引用的旧版不同；此外 `rag.py` 并不位于项目的 `data` 目录。

原始 V2.16 安装器把上述两项都当作阻塞条件，导致没有安装任何新模块，
随后 unittest 自然报 `ModuleNotFoundError`。

本补丁仅包含：
- `install_graph_assist_v216_compat.py`
- `manifest_v216_assist_compat.json`

不会覆盖你原来的 `install_graph_assist_v216.py` 或
`manifest_v216_assist.json`。

## 修复范围

- **保留所有正式图谱/教材/Chunk 快照/V2.15 审核文件的 SHA256 强校验**
- 将 `graph_retriever_v213.py` 改为“文件必须存在但不强制等于旧版 SHA”
  （允许保留本地已升级的 V2.13.1 运行时文件，**不会修改该文件**）
- 移除错误的 `data/rag.py` 强制存在要求。V2.16 Planner 不依赖这个路径
- 新文件仍采用原来的“完全相同则跳过，内容不同拒绝覆盖”策略
- V2.16 模块和所有五份 payload 的 SHA256 值不变

注意：由于无法在你的电脑上检查真正的 V2.13.1 源码，
此补丁仅修复**安装兼容性**。如果你要运行 `--live`，
它的旧版验证器/API 接口仍需后续单独验证；不意味着已经保证所有在线运行兼容。

## 使用前提

你已经解压过原始 `DouXian_V2.16_Graph_Assist_Planner_Offline.zip`，且在：

`E:\VsCode_Project\DockerTest_Agent\RAG_project\data\v216_payload\`

可以看到 `graph_tools/graph_assist_planner_v216.py` 等文件。

## 操作

**仅将此热修复 ZIP 中两份新文件解压至 `data` 根目录。**
不需要删除原安装器，也不需要修改你本机 V2.13.1 代码。

```powershell
conda activate rag
cd E:\VsCode_Project\DockerTest_Agent\RAG_project\data

python .\install_graph_assist_v216_compat.py --check
python .\install_graph_assist_v216_compat.py --install

python -m unittest graph_tools.test_graph_assist_planner_v216 graph_tools.test_eval_graph_assist_v216 -v

python -m graph_tools.eval_graph_assist_v216 --offline --out .\graph_reports\assist_v216\v216_offline_verified_run01.json
```

如果你之前已经手动生成过同名报告，换一个新名字，例如
`v216_offline_verified_run02.json`。工具拒绝覆盖旧报告。

可继续运行待审核双教材离线探索（零 API 调用）：

```powershell
python -m graph_tools.eval_graph_assist_v216 --offline --explore-pending --out .\graph_reports\assist_v216\v216_offline_multibook_run01.json
```

## 测试

在隔离实验目录中复现：无 `data/rag.py`、给本机 `graph_retriever_v213.py`
附加代码变化使 SHA256 不同。

- 兼容性预检 PASS
- V2.16 增量安装 PASS
- 34 项离线测试 PASS
- 29 条正式关系离线评测 PASS
- 多教材待审关系离线评测 PASS
- 回滚/卸载 PASS
- 所有冻结旧数据 SHA256 保持不变
- 已被修改的 V2.13 运行时文件保持不变

不会修改 `rag.py`、`knowledge_base.py`、Chroma、
V2.8 Assist 或任何历史版本图谱。
