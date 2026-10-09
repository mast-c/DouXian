from langchain_core.tools import tool
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()

client = TavilyClient(
    api_key=os.getenv( "TAVILY_API_KEY")
    )


@tool
def web_search(query:str)->str:
    """
    搜索互联网最新信息。

    用于：
    新闻、政策、价格、
    最新事件。
    """

    result = client.search(
        query=query,
        max_results=5
    )


    return "\n".join(
        [
            item["content"]
            for item in result["results"]
        ]
    )