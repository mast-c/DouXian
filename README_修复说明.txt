豆馅医疗 GraphRAG V1.17 · P3-06.3C 前端热修复
==============================================

本包同时解决：
1. 原聊天历史或证据区域报错：
   NameError: name 're' is not defined
2. 左侧导航栏出现不需要的英文页面 “Multihop Evidence”。

原因：
上一个 8 跳显示补丁使用了 re.search()，却漏写 import re。
英文页面则是 earlier P3-06.2 安装在 pages/06_Multihop_Evidence.py，
Streamlit 会自动把 pages/ 目录下的 .py 文件显示到左侧栏。

【本包的两个目标】
- 主前端“豆馅.py”只补一行 import re，其他代码、聊天功能、数据库不改。
- 将 pages/06_Multihop_Evidence.py 移到项目的
  disabled_streamlit_pages/ 下保存，不删除也不影响其他页面。
- 多跳路径仍显示在“实际命中的知识库证据”内，5～8 跳逻辑保留。
- 不修改 rag.py，不访问/更新 Neo4j 和 Chroma，也不修改 .env。

【推荐安装：保留你当前文件的其他自定义修改】
1. 停止运行中的 Streamlit（Ctrl+C）。
2. 把 ZIP 放到项目根目录：
   E:\VsCode_Project\DockerTest_Agent\RAG_project\data
3. 打开 PowerShell 运行：

   cd "E:\VsCode_Project\DockerTest_Agent\RAG_project\data"
   $py = "E:\Miniconda\envs\rag\python.exe"

   Expand-Archive `
     -LiteralPath ".\豆馅_P3063C_修复报错并移除英文多跳页.zip" `
     -DestinationPath "." -Force

   # 先预览，不会修改任何文件
   & $py .\fix_douxian_re_sidebar_v117.py

   # 确认后，添加一行 import re，并备份停用英文页面
   & $py .\fix_douxian_re_sidebar_v117.py --apply

4. 看到 “[P3-06.3C] APPLIED” 后继续验证：

   & $py -m py_compile ".\豆馅.py"
   & $py -m unittest -v test_douxian_re_sidebar_v117.py
   & $py -m unittest -v test_douxian_multihop_fix_v117.py

5. 使用原来的启动命令重启 Streamlit，例如：

   & $py -m streamlit run ".\豆馅.py"

可验收：
- 打开之前导致报错的历史聊天，不再出现 NameError；
- 左侧导航栏没有 “Multihop Evidence”，只保留“豆馅”等你自己的页面；
- 正常问答的“实际命中的知识库证据”下方仍有“多跳图谱来源路径”；
- 只问“8跳”时保持原有8跳过滤，不把5～7跳混入回答的8跳区；
- 原文证据与技术来源路径仍分别统计，不代表新增医学事实。

【备用直接替换文件】
ZIP 内的“修复文件/豆馅.py”是完整修改好的前端。
但优先使用上面的 --apply 方法：它只补 import re，不覆盖自定义代码。
你也可单独下载“豆馅_修复_NameError_多跳仍在证据区.py”手动替换。
手动替换前，请先备份原始豆馅.py；手动替换不会自动停用侧栏页面，
需要另外把 pages/06_Multihop_Evidence.py 移出 pages 文件夹。

【安全备份和回滚】
修复程序自动生成：
  豆馅.py.pre_re_sidebar_fix_时间戳.bak

独立页面会保存到：
  disabled_streamlit_pages/06_Multihop_Evidence_时间戳.py.bak

如果需要回滚，请先停止 Streamlit，用生成的 .bak 恢复“豆馅.py”；
再把对应的旧页面 .bak 文件移回 pages/06_Multihop_Evidence.py。
不要移动/删除其他 pages 页面文件。

【常见状态】
- READY_TO_APPLY：需要执行 --apply。
- APPLIED：已执行修改和页面停用。
- ALREADY_FIXED：已修复，不需要重复操作。
- BLOCKED：你的豆馅.py 结构和此前版本不同，脚本保护性地拒绝覆盖。
  可上传你实际当前的豆馅.py，我们按真实文件再次适配。

【验证范围】
本包在本地用你上一版“豆馅_P3063_8跳路径显示修复包.zip”中的
完整主前端构造并验收；实际 Windows Streamlit 现场尚未运行，
以你完成安装后界面显示为准。所有多跳来源仍为
third_party_unverified；路径展示不等价于临床诊疗证据。
