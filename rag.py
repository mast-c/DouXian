import os
from typing import Iterator
from search_tool import web_search
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config  
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from file_history_store import get_history
from weather_tool import get_weather


SYSTEM_PROMPT = """
你的名字叫豆馅，你是一名严谨、友好的中文AI助手。


你拥有三个工具：

1. search_knowledge_base

用途：
查询内部知识库。

使用场景：
- 公司制度
- 上传文件
- 产品资料
- 内部文档


2. get_weather

用途：
查询实时天气。

使用场景：
- 今天/明天天气
- 温度
- 湿度
- 降雨


3. web_search

用途：
搜索互联网最新信息。

使用场景：
- 新闻
- 最新政策
- 股票价格
- 产品价格
- 最近发生的事情
- 网络公开资料


选择规则：

- 内部资料问题：
优先 search_knowledge_base

- 实时天气：
使用 get_weather

- 最新信息：
使用 web_search

- 普通知识：
直接回答


不要为了联网而联网。
不要使用搜索替代天气工具。
"""


class RagService(object):
    def __init__(self):
        # 与原工程一致的 Embedding、向量数据库和聊天模型配置。
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(
                model=os.getenv("DASHSCOPE_EMBEDDING_MODEL"),
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            )
        )
        self.chat_model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.chain = self.__get_chain()

    @staticmethod
    def _text(content) -> str:
        """从流式模型消息提取可直接展示的文本。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        @tool("search_knowledge_base")
        def search_knowledge_base(question: str) -> str:
            """检索豆馅的本地知识库及向量数据库里的文档。

            当用户问上传的文档、知识库内容等问题时使用。
            question 是用于检索的自然语言问题。
            """
            documents: list[Document] = retriever.invoke(question)
            if not documents:
                return "知识库没有检索到相关参考资料。"
            return "\n\n".join(
                f"文档内容：{doc.page_content}\n文档元数据：{doc.metadata}"
                for doc in documents
            )

        self.tools = [search_knowledge_base, get_weather,web_search]
        self.tool_map = {item.name: item for item in self.tools}
        self.model_with_tools = self.chat_model.bind_tools(self.tools)

        # RunnableGenerator 同时支持 .invoke 和逐字/逐块 .stream。
        # RunnableWithMessageHistory 继续使用原工程的文件历史，
        # 并且只保存用户提问和最终回答，不把工具内部消息写成用户历史。
        agent = RunnableGenerator(self._stream_agent)
        return RunnableWithMessageHistory(
            agent,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _stream_agent(self, inputs: Iterator[dict]) -> Iterator[str]:
        """循环处理工具调用，仅向前端输出答案文字。"""
        for payload in inputs:
            question = str(payload["input"])
            history = payload.get("history") or []
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                *history,
                HumanMessage(content=question),
            ]

            # 限制工具调用轮数，防止模型循环请求工具。
            for round_index in range(4):
                combined = None
                saw_tool_chunks = False
                emitted_text = False

                # 在不需要工具时，直接把模型生成的文字 chunk 传给 st.write_stream。
                # 需要工具时，先收集并执行 tool_calls，再进行下一轮模型生成。
                for chunk in self.model_with_tools.stream(messages):
                    combined = chunk if combined is None else combined + chunk
                    if getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None):
                        saw_tool_chunks = True
                    if not saw_tool_chunks:
                        text_chunk = self._text(chunk.content)
                        if text_chunk:
                            emitted_text = True
                            yield text_chunk

                if combined is None:
                    yield "抱歉，模型本次没有返回内容，请稍后重试。"
                    break

                raw_calls = getattr(combined, "tool_calls", None) or []
                if not raw_calls:
                    if not emitted_text:
                        remaining = self._text(combined.content)
                        if remaining:
                            yield remaining
                        elif getattr(combined, "invalid_tool_calls", None):
                            yield "工具调用参数未能正确解析，请重新描述问题。"
                        else:
                            yield "抱歉，模型本次没有返回文本回答。"
                    break

                # 构造符合 OpenAI function-calling 协议的 Assistant/Tool 消息。
                calls = []
                for i, item in enumerate(raw_calls):
                    calls.append({
                        "name": item["name"],
                        "args": item.get("args") or {},
                        "id": item.get("id") or f"call_{round_index}_{i}",
                    })
                messages.append(AIMessage(content=combined.content or "", tool_calls=calls))

                for call in calls:
                    selected_tool = self.tool_map.get(call["name"])
                    if selected_tool is None:
                        result = f"不存在名为 {call['name']} 的工具。"
                    else:
                        try:
                            result = selected_tool.invoke(call["args"])
                        except Exception:
                            result = f"{call['name']} 调用失败，请稍后重试。"
                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=call["id"])
                    )
            else:
                yield "工具调用次数过多，请简化问题后重试。"


if __name__ == "__main__":
    service = RagService()
    config_for_session = {"configurable": {"session_id": "user_001"}}
    result = service.chain.invoke({"input": "北京明天天气怎么样？"}, config_for_session)
    print(result)
