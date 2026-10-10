import html
import uuid
import time
from urllib.parse import urlsplit
import streamlit as st
from chat_store import ChatStore
from file_history_store import delete_history
from rag import RagService
import config_data as config


st.set_page_config(
    page_title="豆馅",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 会话工具函数
# =========================

def create_conversation(state):
    """创建并持久化会话，再把它设为当前会话。"""
    sid = str(uuid.uuid4())
    state["chat_store"].create_conversation(sid)
    state["conversations"][sid] = {
        "title": "新对话",
        "messages": [],
    }
    state["current_session"] = sid
    state["pending_prompt"] = None
    state["delete_confirm_sid"] = None
    return sid


def delete_conversation(state, sid):
    """
    删除指定会话。

    - 删除非当前会话：当前会话保持不变。
    - 删除当前会话：切到剩余会话中最新的一条。
    - 删除最后一条会话：自动创建一个新的空会话，避免 current_chat() 找不到会话。
    """
    conversations = state["conversations"]

    if sid not in conversations:
        state["delete_confirm_sid"] = None
        return False

    deleting_current = sid == state.get("current_session")

    # 先删除 SQLite 中的会话及消息；成功后再更新页面内存。
    if not state["chat_store"].delete_conversation(sid):
        return False
    # UI 消息在 SQLite；后台模型上下文在 JSON。两处必须同时清理。
    try:
        delete_history(sid)
    except OSError as exc:
        # 不将失败的磁盘清理伪装为已完成；记录在服务日志，便于手工处理。
        import logging
        logging.getLogger(__name__).warning("清理 RAG 上下文失败：%s", exc)
    del conversations[sid]

    if deleting_current:
        # 当前正在等待回答的 prompt 属于被删会话时，不应继续处理。
        state["pending_prompt"] = None

        if conversations:
            # dict 保持创建顺序，取剩余会话里最新创建的一条。
            state["current_session"] = next(reversed(conversations))
            state["chat_store"].set_active_session(state["current_session"])
        else:
            create_conversation(state)

    state["delete_confirm_sid"] = None
    return True


# =========================
# 会话状态
# =========================

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

# 首次启动从 SQLite 加载；热更新时尽量迁移旧的内存会话。
if "chat_store" not in st.session_state:
    store = ChatStore()
    old_conversations = st.session_state.get("conversations", {})
    store.import_existing(old_conversations)
    old_selected = st.session_state.get("current_session")
    if old_selected in old_conversations:
        store.set_active_session(old_selected)
    st.session_state["chat_store"] = store
    st.session_state["conversations"] = store.load_conversations()

if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None

if "delete_confirm_sid" not in st.session_state:
    st.session_state["delete_confirm_sid"] = None

# 不只判断 current_session 是否存在，还保证它确实指向一个现有会话。
if (
    "current_session" not in st.session_state
    or st.session_state["current_session"]
    not in st.session_state["conversations"]
):
    restored_sid = st.session_state["chat_store"].get_active_session()
    if restored_sid in st.session_state["conversations"]:
        st.session_state["current_session"] = restored_sid
    elif st.session_state["conversations"]:
        st.session_state["current_session"] = next(
            reversed(st.session_state["conversations"])
        )
        st.session_state["chat_store"].set_active_session(
            st.session_state["current_session"]
        )
    else:
        create_conversation(st.session_state)


def current_chat():
    return st.session_state["conversations"][
        st.session_state["current_session"]
    ]


def safe_text(text):
    return html.escape(str(text)).replace("\n", "<br>")


def short_title(text, limit=20):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def display_title(chat):
    """兼容旧数据里标题为空字符串或只有空格的情况。"""
    title = str(chat.get("title", "")).strip()
    return title or "新对话"


def show_user(text):
    st.markdown(
        f"""
        <div class="user-row">
            <div class="user-bubble">
                {safe_text(text)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    r"""
<style>
:root{
  --sidebar:#f7f7f8;
  --line:#e8e8eb;
  --text:#202124;
  --muted:#8d9199;
  --bubble:#f4f4f5;
  --blue:#2457ff;
}

/* 只给页面根节点设置正文字体，不能覆盖 Streamlit 内部图标元素。 */
html,body,.stApp{
  font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
}

.stApp{
  background:#fff;
  color:var(--text);
}

/* 隐藏主工具栏 */
#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"]{
  display:none !important;
}


/* =========================
   左侧栏
   ========================= */

[data-testid="stSidebar"]{
  background:var(--sidebar);
  border-right:1px solid var(--line);
}

[data-testid="stSidebar"][aria-expanded="true"]{
  min-width:260px;
}

/* 侧边栏顶部容器：缩小顶部间距，内容更靠上 */
[data-testid="stSidebar"] > div:first-child{
  padding-top:10px;
}

/* 品牌名：更精致的排版 + 蓝色圆点装饰 */
.brand{
  font-size:22px;
  font-weight:800;
  letter-spacing:-0.5px;
  color:#17191e;
  padding:4px 12px 12px;
  display:flex;
  align-items:center;
  gap:7px;
}

/* 品牌名前面的蓝色圆点装饰 */
.brand::before{
  content:"";
  display:inline-block;
  width:9px;
  height:9px;
  border-radius:50%;
  background:var(--blue);
  flex-shrink:0;
  box-shadow:0 0 0 3px rgba(36,87,255,.12);
}

/* "最近对话" 标题：更精致的标签样式 */
.side-title{
  font-size:11px;
  font-weight:600;
  color:var(--muted);
  text-transform:uppercase;
  letter-spacing:0.8px;
  padding:4px 12px 4px;
  margin-bottom:2px;
}

/* 分割线：品牌区与对话列表之间 */
.side-title::after{
  content:"";
  display:block;
  height:1px;
  background:var(--line);
  margin:6px 0 4px;
}

/* 新对话按钮：干净的虚线边框按钮 */
.st-key-new-chat button{
  width:100%;
  min-height:38px;
  border-radius:9px;
  border:1.5px dashed #c8cad0;
  background:transparent;
  color:var(--blue);
  font-weight:600;
  font-size:14px;
  justify-content:flex-start;
  padding:0 12px;
  margin:2px 8px 6px;
  transition:all .15s ease;
}

.st-key-new-chat button:hover{
  border-color:var(--blue);
  background:rgba(36,87,255,.05);
  color:var(--blue);
  box-shadow:0 2px 8px rgba(36,87,255,.08);
}

/* =========================
   历史会话行
   ========================= */

[class*="st-key-history-row-"]{
  border-radius:8px;
  margin:1px 8px 1px;
  transition:background .12s ease;
  position:relative;
}

[class*="st-key-history-row-"]:hover{
  background:transparent;
}

/* 让一行里的标题和 ⋯ 靠得更紧 */
[class*="st-key-history-row-"] [data-testid="stHorizontalBlock"]{
  gap:2px !important;
  align-items:center;
}

/* 历史标题按钮 */
[class*="st-key-history-button-"] button{
  width:100%;
  min-height:36px;
  padding:7px 10px;
  border:none !important;
  box-shadow:none !important;
  background:transparent !important;
  border-radius:7px;
  justify-content:flex-start;
  color:#4f535b !important;
  font-size:13.5px;
  font-weight:450;
  text-align:left;
  transition:all .12s ease;
}

[class*="st-key-history-button-"] button:hover{
  border:none !important;
  box-shadow:none !important;
  background:rgba(36,87,255,.06) !important;
  color:#17191e !important;
}

[class*="st-key-history-button-"] button p{
  width:100%;
  margin:0;
  overflow:hidden;
  white-space:nowrap;
  text-overflow:ellipsis;
  text-align:left;
}

/* 右侧 ⋯ 菜单按钮 */
[class*="st-key-history-menu-"] button{
  width:30px;
  min-width:30px;
  min-height:30px;
  padding:0;
  border:none !important;
  box-shadow:none !important;
  background:transparent !important;
  border-radius:6px;
  color:#a1a4aa !important;
  font-size:16px;
  font-weight:500;
  opacity:0;
  transition:opacity .15s ease, background .15s ease, color .15s ease;
}

[class*="st-key-history-row-"]:hover
[class*="st-key-history-menu-"] button{
  opacity:0.7;
}

[class*="st-key-history-menu-"] button:hover{
  opacity:1 !important;
  background:rgba(36,87,255,.08) !important;
  color:var(--blue) !important;
}

/* 删除确认区域 */
[class*="st-key-delete-panel-"]{
  margin:2px 8px 6px;
  padding:8px 9px 7px;
  border:1px solid #f0c8c8;
  border-radius:9px;
  background:#fff;
  box-shadow:0 4px 12px rgba(180,35,24,.06);
}

[class*="st-key-delete-panel-"] [data-testid="stCaptionContainer"]{
  margin-bottom:5px;
}

[class*="st-key-confirm-delete-"] button{
  min-height:32px;
  border:1px solid #f0c8c8 !important;
  background:#fff5f5 !important;
  color:#b42318 !important;
  font-weight:600;
  font-size:13px;
}

[class*="st-key-confirm-delete-"] button:hover{
  border-color:#e9adad !important;
  background:#ffeaea !important;
  color:#981b12 !important;
}

[class*="st-key-cancel-delete-"] button{
  min-height:32px;
  border:1px solid #e1e2e5 !important;
  background:#fff !important;
  color:#4f535b !important;
  font-weight:500;
  font-size:13px;
}

[class*="st-key-cancel-delete-"] button:hover{
  background:#f3f3f5 !important;
}
/* =========================
   主内容
   ========================= */

.main .block-container{
  max-width:920px;
  padding-top:26px;
  padding-bottom:132px;
}

.topbar{
  display:flex;
  justify-content:space-between;
  align-items:center;
  height:44px;
  margin-bottom:18px;
}

.model{
  font-size:16px;
  font-weight:650;
}

.badge{
  font-size:12px;
  color:var(--muted);
  border:1px solid #ececef;
  border-radius:999px;
  padding:4px 9px;
}

/* 首页欢迎区 */
.hero{
  min-height:29vh;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:flex-end;
  text-align:center;
  padding-bottom:24px;
}

.hero-title{
  font-size:32px;
  font-weight:680;
  letter-spacing:-.7px;
  color:#17191e;
}

.logo{
  color:var(--blue);
  font-size:30px;
  margin-right:8px;
  vertical-align:-2px;
}

.hero-sub{
  font-size:14px;
  color:var(--muted);
  margin-top:10px;
}

/* 输入框 */
[data-testid="stChatInput"]{
  border:1px solid #dedfe3;
  border-radius:24px;
  background:#fff;
  box-shadow:0 8px 30px rgba(20,25,35,.055);
  overflow:hidden;
}

[data-testid="stChatInput"]:focus-within{
  border-color:#c8cad0;
  box-shadow:0 8px 32px rgba(20,25,35,.075);
}

[data-testid="stChatInput"] textarea{
  font-size:15px;
  line-height:1.55;
  min-height:58px;
  padding-top:17px;
}

.st-key-home-composer{
  max-width:820px;
  margin:0 auto;
}

.st-key-home-composer [data-testid="stChatInput"]{
  min-height:108px;
  border-radius:26px;
}

.st-key-home-composer [data-testid="stChatInput"] textarea{
  min-height:88px;
}

.quick-tools{
  max-width:790px;
  margin:10px auto 0;
  text-align:center;
  color:#777b84;
  font-size:13px;
}

[data-testid="stBottomBlockContainer"]{
  background:linear-gradient(to top,#fff 74%,rgba(255,255,255,0));
  padding-bottom:18px;
  animation:composerDrop .32s cubic-bezier(.22,.8,.24,1) both;
}

@keyframes composerDrop{
  from{
    opacity:0;
    transform:translateY(-72px);
  }
  to{
    opacity:1;
    transform:translateY(0);
  }
}

/* 对话 */
.user-row{
  display:flex;
  justify-content:flex-end;
  margin:18px 0 22px;
}

.user-bubble{
  max-width:min(76%,690px);
  background:var(--bubble);
  border-radius:18px;
  padding:11px 16px;
  line-height:1.7;
  font-size:15px;
  overflow-wrap:anywhere;
}

[class*="st-key-assistant-msg-"] p,
[class*="st-key-assistant-live-"] p{
  line-height:1.78;
  font-size:15px;
}

[class*="st-key-assistant-msg-"] pre,
[class*="st-key-assistant-live-"] pre{
  border-radius:12px;
}

@media(max-width:780px){
  [data-testid="stSidebar"]{
    min-width:240px;
    max-width:240px;
  }

  .main .block-container{
    padding-left:18px;
    padding-right:18px;
    padding-bottom:118px;
  }

  .hero-title{
    font-size:26px;
  }

  .user-bubble{
    max-width:88%;
  }
}

/* 不再用 Streamlit 自动生成的哈希类名隐藏控件：版本更新后可能误伤图标。 */
/* 第一次提问后，立即隐藏已经失效的首页输入框 */
.st-key-home-composer[data-stale="true"],
.st-key-home-composer:has([data-stale="true"]),
[data-stale="true"]:has(.st-key-home-composer){
    display:none !important;
}
/* =========================
   AI 思考中
   ========================= */

.thinking-row{
  display:flex;
  align-items:center;
  gap:7px;
  margin:8px 0 14px;
  color:#8d9199;
  font-size:14px;
}

.thinking-spark{
  color:#2457ff;
  font-size:15px;
}

.thinking-dots{
  display:inline-flex;
  align-items:center;
  gap:4px;
  margin-left:1px;
}

.thinking-dots i{
  width:4px;
  height:4px;
  border-radius:50%;
  background:currentColor;
  opacity:.25;
  animation:thinkingDot 1.05s infinite ease-in-out;
}

.thinking-dots i:nth-child(2){
  animation-delay:.14s;
}

.thinking-dots i:nth-child(3){
  animation-delay:.28s;
}

@keyframes thinkingDot{
  0%,60%,100%{
    transform:translateY(0);
    opacity:.25;
  }

  30%{
    transform:translateY(-3px);
    opacity:.9;
  }
}

@media (prefers-reduced-motion: reduce){
  .thinking-dots i{
    animation:none;
    opacity:.55;
  }
}

/* 思考记录的折叠箭头是 CSS 画的，不依赖任何图标字体。 */
.trace-live{
  display:flex;align-items:center;gap:8px;
  margin:8px 0 14px;padding:10px 13px;
  background:#f8f9fb;border:1px solid #e9ebef;border-radius:10px;
  color:#667085;font-size:13px;line-height:1.6;
}
.trace-live-spark{
  color:var(--blue);font-size:15px;
  animation:tracePulse 1.3s ease-in-out infinite;
}
@keyframes tracePulse{
  0%,100%{opacity:.48;transform:scale(.94)}
  50%{opacity:1;transform:scale(1.08)}
}
.trace-panel{
  margin:8px 0 14px;background:#f9fafc;
  border:1px solid #e8ebf1;border-radius:10px;
  color:#475467;overflow:hidden;
}
.trace-panel > summary{
  display:flex;align-items:center;gap:10px;
  padding:10px 13px;cursor:pointer;list-style:none;
  color:#475467;font-size:13px;line-height:1.5;
  user-select:none;
}
.trace-panel > summary::-webkit-details-marker{display:none}
.trace-panel > summary::before{
  content:"";display:inline-block;flex-shrink:0;
  width:7px;height:7px;
  border-right:1.7px solid #667085;border-bottom:1.7px solid #667085;
  transform:rotate(-45deg);transition:transform .16s ease;
}
.trace-panel[open] > summary::before{transform:rotate(45deg)}
.trace-panel > summary:hover{background:#f1f4f9}
.trace-panel > summary:focus-visible{
  outline:2px solid var(--blue);outline-offset:-3px;
}
.trace-panel-body{
  border-top:1px solid #e9ebef;padding:10px 15px 12px;
  font-size:12.5px;line-height:1.8;
}
.trace-note{margin:0 0 8px;color:#858d9a;font-size:12px}
.trace-events{list-style:none;padding:0;margin:0 0 8px}
.trace-events li{margin:4px 0;overflow-wrap:anywhere}
.trace-source-title{
  margin:10px 0 5px;font-size:12px;
  font-weight:650;color:#4f596b;
}
.trace-sources{
  max-height:230px;overflow:auto;overscroll-behavior:contain;
  margin:0 0 8px;padding-left:20px;
}
.trace-sources li{margin:3px 0;overflow-wrap:anywhere}
.trace-sources a{color:#2457c6;text-decoration:none}
.trace-sources a:hover{text-decoration:underline}
.trace-total{margin:8px 0 0;color:#858d9a;font-size:12px}
@media (prefers-reduced-motion:reduce){
  .trace-live-spark{animation:none}
  .trace-panel > summary::before{transition:none}
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================
# AI 可验证的执行步骤（不是模型内部推理）
# =========================

def elapsed_label(seconds):
    try:
        value = max(0, int(float(seconds) + 0.5))
    except (TypeError, ValueError, OverflowError):
        value = 0
    return f"{value // 60}m {value % 60}s"


def _web_sources(trace):
    if not isinstance(trace, dict):
        return []
    result = []
    seen = set()
    for source in (trace.get("sources") or []):
        if not isinstance(source, dict):
            continue
        url = source.get("url", "")
        if not isinstance(url, str):
            continue
        try:
            parsed = urlsplit(url)
        except ValueError:
            continue
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            continue
        if url in seen:
            continue
        seen.add(url)
        result.append({"url": url, "title": str(source.get("title") or parsed.netloc)})
    return result


def trace_heading(trace):
    seconds = trace.get("think_seconds", 0) if isinstance(trace, dict) else 0
    label = f"✦ 思考了 {elapsed_label(seconds)}"
    events = (trace.get("events") or []) if isinstance(trace, dict) else []
    searched = any(
        isinstance(event, dict) and event.get("type") == "end"
        and event.get("tool") == "web_search"
        for event in events
    )
    if searched:
        success = any(
            isinstance(event, dict) and event.get("type") == "end"
            and event.get("tool") == "web_search" and event.get("ok")
            for event in events
        )
        count = trace.get("sources_total") or len(_web_sources(trace))
        label += (f" · 返回 {count} 条网页来源" if count else " · 已联网检索") if success else " · 联网检索未成功"
    return label


def render_html_fragment(body, holder=None):
    """优先用 st.html 展示安全转义后的 HTML；兼容没有 st.html 的旧版。"""
    target = holder if holder is not None else st
    if hasattr(target, "html"):
        target.html(body)
    else:
        target.markdown(body, unsafe_allow_html=True)


def live_trace_html(label):
    """运行时状态：不使用 st.status，也就没有内部 Material 箭头。"""
    return (
        '<div class="trace-live" role="status" aria-live="polite">'
        '<span class="trace-live-spark" aria-hidden="true">✦</span>'
        f'<span>{html.escape(str(label))}</span>'
        '</div>'
    )

TOOL_LABELS = {
    "search_knowledge_base": "知识库检索",
    "get_weather": "天气查询",
    "web_search": "联网搜索",
}


def describe_tool_event(event):
    """仅使用工具调用的实际元数据，不生成假想的思考文字或来源数量。"""
    if not isinstance(event, dict):
        return ""
    name = TOOL_LABELS.get(event.get("tool"), "外部工具")
    if event.get("type") == "start":
        query = str(event.get("query", "")).strip()
        return f"⏳ 开始{name}" + (f"：{query}" if query else "")
    if event.get("type") != "end":
        return ""
    if not event.get("ok", False):
        return f"⚠️ {name}未成功"
    duration = elapsed_label(event.get("duration", 0))
    if event.get("tool") == "web_search":
        if "search_calls" in event:
            count = event.get("sources_total") or len(event.get("sources") or [])
            if count:
                return f"✓ {name}完成 · 返回 {count} 条来源 · 耗时 {duration}"
            try:
                search_calls = int(event.get("search_calls") or 0)
            except (ValueError, TypeError):
                search_calls = 0
            if search_calls == 0:
                return f"✓ {name}请求完成（接口未报告实际搜索） · 耗时 {duration}"
            return f"✓ {name}完成（接口未返回来源链接） · 耗时 {duration}"
        return f"✓ {name}完成（工具未提供来源统计） · 耗时 {duration}"
    return f"✓ {name}完成 · 耗时 {duration}"


def trace_details_html(trace):
    """生成可展开的真实工具步骤。所有动态内容先 HTML 转义。"""
    if not isinstance(trace, dict) or not trace:
        return ""

    heading = html.escape(trace_heading(trace))
    events = trace.get("events")
    events = events if isinstance(events, list) else []
    steps = []
    for event in events:
        if isinstance(event, dict) and event.get("type") == "end":
            description = describe_tool_event(event)
            if description:
                steps.append(
                    '<li class="trace-event">'
                    + html.escape(description)
                    + '</li>'
                )
    if not steps:
        steps.append('<li class="trace-event">本次没有记录到外部工具调用。</li>')

    content = [
        '<p class="trace-note">实际工具执行记录，不包含模型内部推理。</p>',
        '<ul class="trace-events">' + "".join(steps) + '</ul>',
    ]
    sources = _web_sources(trace)
    if sources:
        content.append(
            '<div class="trace-source-title">联网来源链接（展示 '
            + str(len(sources)) + ' 条'
            + (('，接口共返回 ' + str(trace.get("sources_total")) + ' 条')
               if (trace.get("sources_total") or 0) > len(sources) else '')
            + '，已去重）</div>'
        )
        links = []
        for source in sources:
            # URL 已在 _web_sources 中过滤 http/https；属性与文本继续转义。
            url = html.escape(source["url"], quote=True)
            title = html.escape(source["title"])
            links.append(
                f'<li><a href="{url}" target="_blank" rel="noopener noreferrer">'
                f'{title}</a></li>'
            )
        content.append('<ol class="trace-sources">' + "".join(links) + '</ol>')
    if "total_seconds" in trace:
        content.append(
            '<p class="trace-total">整次回答耗时 '
            + html.escape(elapsed_label(trace["total_seconds"]))
            + '（含文字输出）</p>'
        )

    return (
        '<details class="trace-panel">'
        f'<summary><span>{heading}</span></summary>'
        '<div class="trace-panel-body">'
        + "".join(content)
        + '</div></details>'
    )


def render_saved_trace(trace):
    body = trace_details_html(trace)
    if body:
        render_html_fragment(body)

# =========================
# 当前历史会话选中效果
# =========================

active_sid = st.session_state["current_session"]

st.markdown(
    f"""
    <style>
    .st-key-history-row-{active_sid} {{
        background: rgba(36,87,255,.06);
    }}
    .st-key-history-row-{active_sid}::before {{
        content:"";
        position:absolute;
        left:0;
        top:50%;
        transform:translateY(-50%);
        width:3px;
        height:60%;
        background:var(--blue);
        border-radius:0 3px 3px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 左侧历史会话
# =========================

with st.sidebar:
    st.markdown(
        '<div class="brand">豆馅</div>',
        unsafe_allow_html=True,
    )

    if st.button("＋ 新对话", key="new-chat"):
        create_conversation(st.session_state)
        st.rerun()

    st.markdown(
        '<div class="side-title">最近对话</div>',
        unsafe_allow_html=True,
    )

    for sid, chat_item in reversed(
        list(st.session_state["conversations"].items())
    ):
        title = display_title(chat_item)

        with st.container(key=f"history-row-{sid}"):
            title_col, menu_col = st.columns([0.86, 0.14], gap="small")

            with title_col:
                if st.button(
                    title,
                    key=f"history-button-{sid}",
                    help=title,
                ):
                    st.session_state["chat_store"].set_active_session(sid)
                    st.session_state["current_session"] = sid
                    st.session_state["delete_confirm_sid"] = None
                    st.rerun()

            with menu_col:
                is_confirming = (
                    st.session_state.get("delete_confirm_sid") == sid
                )

                if st.button(
                    "×" if is_confirming else "⋯",
                    key=f"history-menu-{sid}",
                    help="取消删除" if is_confirming else "删除对话",
                ):
                    st.session_state["delete_confirm_sid"] = (
                        None if is_confirming else sid
                    )
                    st.rerun()

        if st.session_state.get("delete_confirm_sid") == sid:
            with st.container(key=f"delete-panel-{sid}"):
                st.caption(f"删除“{short_title(title, 12)}”？此操作不可撤销。")

                delete_col, cancel_col = st.columns(2, gap="small")

                with delete_col:
                    if st.button(
                        "删除",
                        key=f"confirm-delete-{sid}",
                        use_container_width=True,
                    ):
                        delete_conversation(st.session_state, sid)
                        st.rerun()

                with cancel_col:
                    if st.button(
                        "取消",
                        key=f"cancel-delete-{sid}",
                        use_container_width=True,
                    ):
                        st.session_state["delete_confirm_sid"] = None
                        st.rerun()


# =========================
# 当前聊天
# =========================

chat = current_chat()
messages = chat["messages"]


for msg in messages:
    if msg["role"] == "user":
        show_user(msg["content"])
    else:
        render_saved_trace(msg.get("trace"))
        st.markdown(msg["content"])


if not messages:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                ✦ 你好，我是豆馅
            </div>
            <div class="hero-sub">
                基于你的知识库回答问题
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 没有消息：首页输入框显示在欢迎语下面
    with st.container(key="home-composer"):
        prompt = st.chat_input(
            "向豆馅提问",
            key=f"home_prompt_{st.session_state['current_session']}",
        )

else:
    # 有消息以后，输入框固定到底部
    prompt = st.chat_input(
        "给豆馅发送消息",
        key=f"chat_prompt_{st.session_state['current_session']}",
    )


# =========================
# 用户消息先显示，立即切到底部
# =========================

if prompt:
    # 消息先落盘，保证随后的 st.rerun 不会丢失这次提问。
    new_title = short_title(prompt) if display_title(chat) == "新对话" else None
    st.session_state["chat_store"].append_message(
        st.session_state["current_session"],
        "user", prompt,
        new_title=new_title,
    )
    chat["messages"].append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    if new_title is not None:
        chat["title"] = new_title

    st.session_state["pending_prompt"] = prompt
    st.rerun()


# =========================
# AI 流式回答
# =========================

pending = st.session_state.get("pending_prompt")

if pending:
    st.session_state["pending_prompt"] = None

    # 记录从请求发出到首段可见答案的耗时；不是模型内部 token 推理时长。
    started = time.perf_counter()
    trace = {"version": 1, "events": [], "sources": []}
    clock = {"first_answer_at": None}

    # 只用页面自己的 HTML 状态条，避免 Streamlit 内部图标字体失效露出英文。
    live_slot = st.empty()
    render_html_fragment(live_trace_html("豆馅正在思考…"), live_slot)

    def show_progress(event):
        if clock["first_answer_at"] is None:
            action = TOOL_LABELS.get(event.get("tool"), "处理请求")
            phase = "正在" + action if event.get("type") == "start" else action + "已返回"
            label = phase + " · " + elapsed_label(time.perf_counter() - started)
            render_html_fragment(live_trace_html(label), live_slot)

    # 以 SQLite 恢复的当前会话为模型上下文的唯一 UI 来源；
    # 当前用户消息已经在 messages 末尾保存，因此不再重复放入 history。
    previous_messages = messages[:-1] if messages and messages[-1].get("role") == "user" else messages
    stream = st.session_state["rag"].chain.stream(
        {
            "input": pending,
            "history": previous_messages,
            "trace": trace,
            "on_progress": show_progress,
        },
        {
            "configurable": {
                "session_id": st.session_state["current_session"]
            }
        },
    )

    def stream_with_timing(source):
        for chunk in source:
            if clock["first_answer_at"] is None and (
                isinstance(chunk, str) and bool(chunk.strip())
            ):
                clock["first_answer_at"] = time.perf_counter()
                label = "思考了 " + elapsed_label(clock["first_answer_at"] - started)
                render_html_fragment(live_trace_html(label), live_slot)
            yield chunk

    try:
        answer = st.write_stream(stream_with_timing(stream))
    except Exception:
        render_html_fragment(live_trace_html("回答未完成 · 请检查服务日志"), live_slot)
        # 不伪造回答，也不把失败记录写成成功的历史消息。
        st.error("AI 回答过程中发生错误，请检查服务日志后重试。")
        st.stop()

    total_seconds = time.perf_counter() - started
    think_seconds = (
        clock["first_answer_at"] - started
        if clock["first_answer_at"] is not None
        else total_seconds
    )
    trace["think_seconds"] = round(think_seconds, 2)
    trace["total_seconds"] = round(total_seconds, 2)

    render_html_fragment(trace_details_html(trace), live_slot)

    # 先保存答案 + 耗时 + 执行事件，刷新后仍可展开历史记录。
    if answer is None:
        answer = ""
    st.session_state["chat_store"].append_message(
        st.session_state["current_session"], "assistant", str(answer),
        trace=trace,
    )
    chat["messages"].append({
        "role": "assistant", "content": str(answer), "trace": trace,
    })
    st.rerun()
