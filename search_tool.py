"""通过百炼 Qwen Responses API 进行联网搜索并提供可核验的来源。

只有响应内存在真实的 web_search_call，才认为完成了联网检索。
前端最多展示 10 个去重来源，并保留接口返回的实际来源总数。
"""

import json
import os
from urllib.parse import urlsplit

from langchain_core.tools import tool
from openai import OpenAI

import config_data as config  # noqa: F401：加载项目配置 / .env

DISPLAY_SOURCE_LIMIT = 10


def _field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _response_text(response) -> str:
    value = _field(response, "output_text", "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    parts = []
    for item in _field(response, "output", []) or []:
        if _field(item, "type") != "message":
            continue
        for block in _field(item, "content", []) or []:
            if _field(block, "type") == "output_text":
                value = _field(block, "text", "")
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
    return "\n\n".join(parts)


def _collect_search_sources(response):
    """搜索次数取自真实 web_search_call，来源仅取官方 action.sources。"""
    sources = []
    seen = set()
    search_calls = 0
    completed_calls = 0
    for item in _field(response, "output", []) or []:
        if _field(item, "type") != "web_search_call":
            continue
        search_calls += 1
        status = _field(item, "status", "completed")
        if status not in ("completed", None):
            continue
        completed_calls += 1
        action = _field(item, "action", {}) or {}
        for source in _field(action, "sources", []) or []:
            url = _field(source, "url", "")
            if not isinstance(url, str):
                continue
            url = url.strip()
            try:
                parsed = urlsplit(url)
            except ValueError:
                continue
            if parsed.scheme not in ("http", "https") or not parsed.netloc or url in seen:
                continue
            seen.add(url)
            title = _field(source, "title", "") or parsed.netloc
            sources.append({"url": url, "title": str(title)[:180]})
    return sources, search_calls, completed_calls


def _response(ok=False, answer="", sources=None, search_calls=0, sources_total=0):
    return json.dumps(
        {"ok": bool(ok), "answer": str(answer), "sources": sources or [],
         "search_calls": search_calls, "sources_total": sources_total},
        ensure_ascii=False,
    )


@tool("web_search")
def web_search(query: str) -> str:
    """联网检索最新新闻、政策、行情等公开信息，返回实际搜索证据和来源。"""
    query = str(query or "").strip()
    if not query:
        return _response(answer="缺少搜索关键词。")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "qwen3.8-flash")
    if not api_key or not base_url:
        return _response(answer="联网搜索未配置：请检查 DASHSCOPE_API_KEY 与 OPENAI_BASE_URL。")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        response = client.responses.create(
            model=model,
            input=query,
            tools=[{"type": "web_search"}],
        )
        answer = _response_text(response)
        sources, calls, completed = _collect_search_sources(response)
        ok = completed > 0 and bool(sources) and bool(answer)
        if not ok:
            if calls == 0:
                answer = "百炼接口未报告任何 web_search_call，本次没有证据表明进行了联网搜索。"
            elif completed == 0:
                answer = "百炼接口报告的网页搜索未成功完成。"
            elif not sources:
                answer = "网页搜索调用已完成，但接口未返回可核验的网页来源链接。"
            else:
                answer = "网页搜索调用已完成，但没有生成可展示的回答。"
        return _response(
            ok=ok, answer=answer, sources=sources[:DISPLAY_SOURCE_LIMIT],
            search_calls=calls, sources_total=len(sources),
        )
    except Exception as exc:
        # 不向前端暴露 API Key、请求 URL 或可能包含凭据的异常原文。
        return _response(
            answer=f"千问联网搜索失败（{type(exc).__name__}），请检查接口地域、模型、API Key 或网络连接。"
        )
