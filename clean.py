
from knowledge_base import KnowledgeBaseService

OLD = "../RAG知识库/神经网络与深度学习.md"
NEW = "神经网络与深度学习.md"

service = KnowledgeBaseService()
db = service.chroma

old = db.get(where={"source": OLD}, include=[])
new = db.get(
    where={"source": NEW},
    limit=1,
    include=["documents", "metadatas"],
)

print("旧版本 Chunk:", len(old["ids"]))
print("新版本是否存在:", bool(new["ids"]))

if not old["ids"]:
    print("没有旧记录需要清理")
    raise SystemExit(0)

if not new["ids"]:
    raise RuntimeError("未找到新版本，停止删除")

print("\n新版本示例：")
print("Metadata:", new["metadatas"][0])
print("正文:", new["documents"][0][:200])

confirm = input(
    "\n确认新版本是正确的完整文档后，输入 DELETE："
).strip()

if confirm != "DELETE":
    print("已取消")
    raise SystemExit(0)

ids = old["ids"]

for i in range(0, len(ids), 200):
    db.delete(ids=ids[i:i + 200])

remaining = db.get(where={"source": OLD}, include=[])

assert not remaining["ids"], "旧记录未清理干净"

print(f"清理完成，删除 {len(ids)} 个旧 Chunk")