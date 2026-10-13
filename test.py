
import os
from http import HTTPStatus

import dashscope
from dotenv import load_dotenv

load_dotenv()


def rerank_texts(query: str, documents: list[str], top_n: int = 3):
    """调用 Reranker，返回 (原文索引, 相关性分数)。"""
    if not query.strip() or not documents or top_n < 1:
        return []

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("请先配置 DASHSCOPE_API_KEY")

    base_url = os.getenv("DASHSCOPE_HTTP_BASE_URL")
    if base_url:
        dashscope.base_http_api_url = base_url

    response = dashscope.TextReRank.call(
        model="qwen3-rerank",
        api_key=api_key,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
    )

    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"重排序 API 失败：{response.code} {response.message}"
        )

    ranked = []
    for item in response.output["results"]:
        index = int(item["index"])
        if not 0 <= index < len(documents):
            raise ValueError(f"模型返回无效文档索引：{index}")
        ranked.append((index, float(item["relevance_score"])))

    return ranked


if __name__ == "__main__":
    question = "员工申请病假需要提交什么材料？"

    texts = [
        "考勤制度：员工上班需要按时打卡。",
        "请假制度：员工申请病假需要提交医院出具的诊断证明。",
        "福利制度：员工年假需要提前申请。",
    ]

    results = rerank_texts(question, texts)

    for rank, (index, score) in enumerate(results, start=1):
        print(f"排名 {rank} / 相关分数 {score:.4f} / 原索引 {index}")
        print(texts[index])