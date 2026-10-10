"""通过百炼 Qwen Responses API 进行联网搜索并提供可核验的来源。

只有响应内存在真实的 web_search_call，才认为完成了联网检索。
前端最多展示 5 个去重来源；API 实际检索量不由展示上限控制。
"""

import json
import os
from urllib.parse import urlsplit

from langchain_core.tools import tool
from openai import OpenAI

import config_data as config  # noqa: F401：加载项目配置 / .env

DISPLAY_SOURCE_LIMIT = 5


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
    """提取实际 web_search_call、已完成的来源和真实搜索词。"""
    sources = []
    queries = []
    seen_urls = set()
    seen_queries = set()
    search_calls = 0
    completed_calls = 0

    for item in _field(response, "output", []) or []:
        if _field(item, "type") != "web_search_call":
            continue
        search_calls += 1
        if _field(item, "status", "completed") not in ("completed", None):
            continue

        completed_calls += 1
        action = _field(item, "action", {}) or {}
        raw_queries = _field(action, "query", "")
        if isinstance(raw_queries, str):
            raw_queries = [raw_queries]
        if isinstance(raw_queries, list):
            for raw_query in raw_queries:
                if not isinstance(raw_query, str):
                    continue
                value = " ".join(raw_query.split()).strip()[:160]
                if value and value not in seen_queries:
                    seen_queries.add(value)
                    queries.append(value)

        for source in _field(action, "sources", []) or []:
            url = _field(source, "url", "")
            if not isinstance(url, str):
                continue
            url = url.strip()
            try:
                parsed = urlsplit(url)
            except ValueError:
                continue
            if parsed.scheme not in ("http", "https") or not parsed.netloc or url in seen_urls:
                continue
            seen_urls.add(url)
            title = _field(source, "title", "") or parsed.netloc
            sources.append({"url": url, "title": str(title)[:180]})

    return sources, search_calls, completed_calls, queries


def _select_display_sources(sources, limit=DISPLAY_SOURCE_LIMIT):
    """先取不同网站的来源，再按原始顺序补满，避免同一站刷屏。"""
    if limit <= 0:
        return []
    selected = []
    used_domains = set()
    used_urls = set()

    for source in sources:
        url = source["url"]
        domain = urlsplit(url).netloc.lower().removeprefix("www.")
        if domain in used_domains:
            continue
        selected.append(source)
        used_domains.add(domain)
        used_urls.add(url)
        if len(selected) >= limit:
            return selected

    for source in sources:
        if source["url"] not in used_urls:
            selected.append(source)
            used_urls.add(source["url"])
            if len(selected) >= limit:
                break
    return selected


def _response(ok=False, answer="", sources=None, search_calls=0, sources_total=0, queries=None):
    return json.dumps(
        {"ok": bool(ok), "answer": str(answer), "sources": sources or [],
         "search_calls": search_calls, "sources_total": sources_total,
         "queries": queries or []},
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
            input=(query + "\n\n请精简检索：优先核实官方或高可信来源，"
                   "尽量只选最相关的 3 至 5 篇资料用于回答，避免重复搜索。"),
            tools=[{"type": "web_search"}],
        )
        answer = _response_text(response)
        sources, calls, completed, queries = _collect_search_sources(response)
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
            ok=ok, answer=answer,
            sources=_select_display_sources(sources),
            search_calls=calls, sources_total=len(sources), queries=queries[:8],
        )
    except Exception as exc:
        # 不向前端暴露 API Key、请求 URL 或可能包含凭据的异常原文。
        return _response(
            answer=f"千问联网搜索失败（{type(exc).__name__}），请检查接口地域、模型、API Key 或网络连接。"
        )
