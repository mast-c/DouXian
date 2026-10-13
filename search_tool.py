"""通过百炼 Qwen Responses API 进行联网搜索并提供可核验的来源。

只有响应内存在真实的 web_search_call，才认为完成了联网检索。
前端最多展示 5 个去重来源；API 实际检索量不由展示上限控制。
"""

import json
import os
import re
from urllib.parse import urlsplit

from langchain_core.tools import tool
from openai import OpenAI, APIStatusError, APIConnectionError


DISPLAY_SOURCE_LIMIT = 5


DEFAULT_WEB_SEARCH_MODEL = "qwen3.7-plus"
LEGACY_CHAT_MODELS = frozenset({
    "qwen-plus", "qwen-plus-latest", "qwen-max", "qwen-flash", "qwen-turbo"
})


def _safe_api_error(exc: APIStatusError) -> str:
    """提取可用于排错的状态码、错误码与短消息，不输出请求头或 API Key。"""
    status = getattr(exc, "status_code", "?")
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return f"HTTP {status}"
    problem = body.get("error", body)
    if not isinstance(problem, dict):
        return f"HTTP {status}"
    code = str(problem.get("code") or problem.get("type") or "").strip()
    message = str(problem.get("message") or "").strip()
    message = " ".join(message.split())
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-[已隐藏]", message)
    message = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._-]+", "Bearer [已隐藏]", message)
    # 只保留短错误说明，避免把请求体等敏感上下文写到聊天记录中。
    details = "；".join(p for p in [code[:80], message[:180]] if p)
    return f"HTTP {status}" + (f"：{details}" if details else "")


def _data_inspection_message(exc: APIStatusError) -> str | None:
    """识别百炼内容安全拦截；不把审核报错误诊为密钥、模型或网络配置问题。"""
    body = getattr(exc, "body", None)
    problem = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(problem, dict):
        problem = {}

    details = " ".join(
        str(problem.get(field) or "")
        for field in ("code", "type", "message")
    )
    if not details:
        details = str(exc)

    # 兼容 DataInspectionFailed、data_inspection_failed、
    # InternalError.Algo.DataInspectionFailed 等不同接口返回格式。
    normalized = re.sub(r"[^a-z0-9]", "", details.lower())
    if "datainspectionfailed" not in normalized:
        return None

    desc = details.lower()
    if "output data may contain" in desc:
        where = "本次生成的回答"
    elif "input data may contain" in desc:
        where = "本次提交的查询内容"
    else:
        where = "本次请求或回答内容"

    # 不能因为被拦截就伪造来源、编写新闻，或自动规避审核重试。
    return (
        f"百炼内容安全审核拦截了{where}"
        "（HTTP 400 / DataInspectionFailed），"
        "本轮未得到可核验的联网回答。"
        "这是内容审核结果，并不意味着 API Key 或模型不支持联网。"
        "可以将正常资讯问题限定到某个主题或具体事件，例如“今天的科技新闻”；"
        "若合理查询反复被误拦截，可在百炼控制台查询审核记录并提交工单。"
    )


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
    base_url = (os.getenv("WEB_SEARCH_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip()
    model = (os.getenv("WEB_SEARCH_MODEL") or DEFAULT_WEB_SEARCH_MODEL).strip()
    if not api_key or not base_url:
        return _response(answer="联网搜索未配置：请检查 DASHSCOPE_API_KEY 与 OPENAI_BASE_URL。")
    if not model:
        return _response(answer="WEB_SEARCH_MODEL 不能为空。")
    if model.lower() in LEGACY_CHAT_MODELS:
        return _response(
            answer=f"当前联网模型 {model} 适用于普通聊天接口，但不适用于该 Responses 联网工具；"
                   f"请将 .env 中 WEB_SEARCH_MODEL 改为 {DEFAULT_WEB_SEARCH_MODEL}，"
                   "保留 OPENAI_MODEL 作为普通聊天模型。"
        )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        if not hasattr(client, "responses"):
            return _response(answer="当前 openai Python SDK 版本过旧，不支持 Responses API；请升级 openai。")
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
    except APIStatusError as exc:
        # 内容安全拦截与接口配置失败是不同问题；准确分类后提示下一步操作。
        inspection_message = _data_inspection_message(exc)
        if inspection_message:
            return _response(answer=inspection_message)
        # 其他 HTTP 错误保留脱敏的状态码与原因，方便针对性排查。
        return _response(
            answer=f"千问联网搜索失败：{_safe_api_error(exc)}。"
                   "请根据服务端错误码核对模型、地域和账号权限。"
        )
    except APIConnectionError:
        return _response(answer="联网请求无法连接到百炼服务，请检查网络、代理或服务地址。")
    except Exception as exc:
        # 避免暴露 API Key、完整 URL 或模型请求内容。
        return _response(
            answer=f"千问联网搜索失败（{type(exc).__name__}）。请检查 SDK 版本与服务配置。"
        )
