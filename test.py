
from pathlib import Path
from knowledge_base import KnowledgeBaseService

filename = "../RAG知识库/神经网络与深度学习.md"
file_path = Path(__file__).resolve().parent / filename

if not file_path.is_file():
    raise FileNotFoundError(f"找不到文件：{file_path}")

service = KnowledgeBaseService()

# 3. 检索并检查真实 Metadata
docs = service.search_with_rerank(
    query="卷积神经网络中的卷积层有什么作用？",
    recall_k=10,
    top_n=3,
)

assert docs, "没有检索到知识库内容"

for i, doc in enumerate(docs, 1):
    metadata = doc.metadata or {}

    print(f"\n========== Chunk {i} ==========")
    print("source:", metadata.get("source"))
    print("section_path:", metadata.get("section_path"))
    print("h1:", metadata.get("h1"))
    print("h2:", metadata.get("h2"))
    print("h3:", metadata.get("h3"))
    print("h4:", metadata.get("h4"))
    print("chunk_id:", metadata.get("chunk_id"))
    print("正文前150字:", doc.page_content[:150])