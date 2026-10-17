from rag import RagService

service = RagService()

question = (
    "汇聚层的作用是什么？"
)

queries = service.decompose_queries(question)

print("\n========== 查询分解结果 ==========")

for i, q in enumerate(queries, 1):
    print(f"子问题{i}：{q}")