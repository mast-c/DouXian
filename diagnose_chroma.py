"""只读检查本项目 Chroma 目录、collection_name 和可见 Chunk 数量。

不用 embedding 模型、不连接网络；使用 SQLite mode=ro，不创建或修改 Chroma。
在 config_data.py 同目录执行：python diagnose_chroma.py
"""
from pathlib import Path
import sqlite3

try:
    import config_data as config
except Exception as exc:
    raise SystemExit(f"无法导入 config_data.py：{type(exc).__name__}: {exc}")

print("=== Chroma 只读诊断 ===")
base = Path(config.persist_directory).expanduser().resolve()
name = str(config.collection_name)
db = base / "chroma.sqlite3"
print("persist_directory =", base)
print("collection_name =", name)
print("chroma.sqlite3 存在 =", db.is_file())

if not db.is_file():
    print("结论：配置目录下没有 Chroma SQLite 文件。请核对 config_data.py 的路径；未创建新数据库。")
    raise SystemExit(0)

# 只读 SQLite 连接，不通过 Chroma SDK 创建不存在的 collection。
uri = db.as_uri() + "?mode=ro"
try:
    conn = sqlite3.connect(uri, uri=True, timeout=3)
    conn.execute("PRAGMA query_only = ON")
    tables = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "collections" not in tables:
        print("未找到 collections 表。可能是旧版 Chroma，或数据库结构不同；请检查 Chroma 版本。")
        raise SystemExit(0)
    rows = conn.execute("SELECT id, name FROM collections ORDER BY name").fetchall()
    print("已存在的 collections：")
    if not rows:
        print("  （无）")
    for ident, collection in rows:
        marker = " ← 当前配置" if collection == name else ""
        print("  -", collection, marker)

    target = next((ident for ident, collection in rows if collection == name), None)
    if target is None:
        print("结论：collection_name 与实际 collections 不匹配。")
        raise SystemExit(0)

    if {"segments", "embeddings"}.issubset(tables):
        try:
            count = conn.execute(
                """SELECT count(*)
                   FROM embeddings e JOIN segments s ON e.segment_id = s.id
                   WHERE s.collection = ?""",
                (target,),
            ).fetchone()[0]
            print("当前 collection 的 embeddings 记录数 =", count)
            if count == 0:
                print("结论：Chroma 当前 collection 内没有向量记录；Neo4j 不会自动填充它。")
            elif "embedding_metadata" in tables:
                try:
                    sample = conn.execute(
                        """SELECT DISTINCT m.string_value
                           FROM embeddings e
                           JOIN segments s ON e.segment_id = s.id
                           JOIN embedding_metadata m ON m.id = e.id
                           WHERE s.collection = ? AND m.key = 'source'
                             AND m.string_value IS NOT NULL
                           LIMIT 5""", (target,)
                    ).fetchall()
                    print("样例来源文件名（用于确认是否仍是计算机文档）：")
                    for row in sample:
                        print("  -", str(row[0])[:180])
                except sqlite3.Error:
                    print("不能从当前 schema 读取示例来源名，记录数仍有效。")
        except sqlite3.Error as exc:
            print("本地 SQLite schema 与预期不一致，无法统计记录数：", str(exc)[:150])
    else:
        print("未找到 embeddings/segments 表，无法可靠统计当前版本的 Chunk 数量。")
finally:
    if "conn" in locals():
        conn.close()

print("安全提醒：诊断不会删除或更新任何向量、数据库节点或来源资料。")
