
import traceback
from rag import RagService

service = RagService()

question = (
    "AlexNet中全连接层与Softmax输出层"
    "是什么关系？网络最终输出什么？"
)

print("\n========== 开始检查知识库 ==========")

try:
    # 直接执行工具，绕过 _call_tool 的异常捕获
    result = service.tool_map[
        "search_knowledge_base"
    ].invoke({
        "question": question
    })

    print("\n========== 检索结果 ==========")
    print(result[:2000])

except Exception:
    print("\n========== 真实异常 ==========")
    traceback.print_exc()