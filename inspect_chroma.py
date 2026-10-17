
from pathlib import Path
import sqlite3
import chromadb
import config_data as config

root = Path(config.persist_directory).resolve()
db = root / "chroma.sqlite3"

print("========== Chroma 诊断 ==========")
print("Chroma 版本：", chromadb.__version__)
print("数据库目录：", root)
print("SQLite 存在：", db.is_file())

if not db.is_file():
    raise SystemExit("未找到数据库，请检查 persist_directory")

# 只读模式，不通过 Chroma 加载 HNSW 索引
conn = sqlite3.connect(
    db.as_uri() + "?mode=ro",
    uri=True,
)

try:
    print("\n========== SQLite 检查 ==========")

    result = conn.execute(
        "PRAGMA quick_check"
    ).fetchone()

    print("完整性检查：", result[0])

    print("\n========== 向量索引目录 ==========")

    segments = conn.execute("""
        SELECT s.id, c.name, s.type
        FROM segments AS s
        JOIN collections AS c
            ON s.collection = c.id
        WHERE s.scope = 'VECTOR'
    """).fetchall()

    for segment_id, name, segment_type in segments:
        folder = root / segment_id

        print("\nCollection：", name)
        print("Segment ID：", segment_id)
        print("Segment Type：", segment_type)
        print("索引目录存在：", folder.is_dir())

        if folder.is_dir():
            for file in sorted(folder.iterdir()):
                if file.is_file():
                    print(
                        "  ",
                        file.name,
                        file.stat().st_size,
                        "bytes",
                    )

    print("\n========== 数据记录 ==========")

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    for table in ("embeddings", "embeddings_queue"):
        if table in tables:
            count = conn.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]

            print(f"{table} 记录数：{count}")

finally:
    conn.close()