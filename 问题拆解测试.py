from rag import RagService

service = RagService()

question = "贝叶斯网是什么？有什么作用？"

queries = service.decompose_queries(question)

print("========== 查询分解结果 ==========")

for i, query in enumerate(queries, 1):
    print(f"子问题{i}：{query}")