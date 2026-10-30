import html
import uuid
import time
from urllib.parse import urlsplit
import streamlit as st
import streamlit.components.v1 as components
from chat_store import ChatStore
from file_history_store import delete_history
from rag import RagService
from search_tool import DISPLAY_SOURCE_LIMIT
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

/* 与千问思考记录接近的轻量布局：透明、无外框，无背景卡片。 */
.trace-live,
.trace-panel{
  margin:9px 0 14px;
  padding:0;
  color:#667085;
  background:transparent !important;
  border:0 !important;
  border-radius:0;
  box-shadow:none !important;
  font-size:13px;
}
.trace-live{
  display:flex;
  align-items:center;
  gap:9px;
  min-height:26px;
  line-height:1.6;
}
.trace-live-spark{
  color:#6d83b8;
  font-size:15px;
  animation:tracePulse 1.4s ease-in-out infinite;
}
@keyframes tracePulse{
  0%,100%{opacity:.45}
  50%{opacity:1}
}
.trace-panel > summary{
  display:flex;
  align-items:center;
  gap:9px;
  padding:3px 0;
  cursor:pointer;
  list-style:none;
  background:transparent !important;
  color:#657084;
  font-size:13.5px;
  line-height:1.65;
  user-select:none;
}
.trace-panel > summary::-webkit-details-marker{display:none}
.trace-panel > summary::marker{content:""}
.trace-panel > summary:hover{color:#35445e}
.trace-panel > summary:focus-visible{
  outline:2px solid #8eaaff;
  outline-offset:4px;
  border-radius:3px;
}
.trace-search-icon{
  display:inline-block;
  width:17px;
  height:17px;
  position:relative;
  flex:none;
  color:#738097;
}
.trace-search-icon::before{
  content:"";
  position:absolute;
  left:1px;
  top:1px;
  width:10px;
  height:10px;
  border:1.35px solid currentColor;
  border-radius:50%;
}
.trace-search-icon::after{
  content:"";
  position:absolute;
  left:12px;
  top:11px;
  width:6px;
  height:1.35px;
  background:currentColor;
  transform:rotate(47deg);
  transform-origin:left center;
}
.trace-summary-text{min-width:0}
.trace-chevron{
  margin-left:3px;
  flex:none;
  width:7px;
  height:7px;
  border-right:1.3px solid #8b96a5;
  border-bottom:1.3px solid #8b96a5;
  transform:rotate(45deg);
  transition:transform .18s ease;
}
.trace-panel[open] .trace-chevron{transform:rotate(225deg)}
.trace-panel-body{
  padding:8px 0 4px 25px;
  border:0 !important;
  background:transparent !important;
  color:#697384;
  font-size:13px;
  line-height:1.7;
}
.trace-step-title{
  display:flex;
  align-items:center;
  gap:8px;
  font-weight:580;
  color:#5e687a;
}
.trace-step-icon{
  color:#8190a7;
  font-size:14px;
  font-weight:500;
}
.trace-queries{
  display:flex;
  flex-wrap:wrap;
  gap:5px 15px;
  margin:9px 0 3px 21px;
  color:#8b7067;
}
.trace-query{
  display:inline-block;
  max-width:100%;
  overflow-wrap:anywhere;
}
.trace-source-chips{
  display:flex;
  flex-wrap:wrap;
  align-items:center;
  gap:9px;
  margin:12px 0 5px 21px;
}
.trace-source-chip{
  display:inline-flex;
  align-items:center;
  gap:7px;
  min-width:0;
  max-width:min(100%,315px);
  padding:5px 11px;
  background:#f5f6f8;
  border:0;
  border-radius:999px;
  color:#606d80;
  text-decoration:none !important;
  font-size:12.5px;
  line-height:1.55;
  transition:background .12s ease,color .12s ease;
}
.trace-source-chip:hover{
  background:#eaf0ff;
  color:#2b56bc;
}
.trace-chip-icon{
  flex:none;
  color:#7897cf;
  font-size:12px;
}
.trace-chip-text{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}
.trace-events{
  margin:9px 0 0;
  padding:0;
  list-style:none;
}
.trace-events li{margin:5px 0;overflow-wrap:anywhere}
.trace-alert{color:#aa6b31;margin:7px 0}
.trace-note,.trace-total{
  margin:10px 0 0;
  font-size:11.5px;
  color:#9aa1ab;
}
@media(max-width:780px){
  .trace-panel-body{padding-left:18px}
  .trace-queries,.trace-source-chips{margin-left:0}
  .trace-source-chip{max-width:100%}
}
@media(prefers-reduced-motion:reduce){
  .trace-live-spark{animation:none}
  .trace-chevron,.trace-source-chip{transition:none}
}

/* 知识库真实检索片段：与现有思考记录一起折叠展示 */
.kb-evidence-list{
  display:grid;
  gap:10px;
  margin:10px 0 12px 21px;
}
.kb-evidence-item{
  padding:3px 0 3px 11px;
  border-left:2px solid #dce5f4;
  overflow-wrap:anywhere;
}
.kb-evidence-title{
  color:#526177;
  font-size:12.5px;
  font-weight:650;
}
.kb-evidence-section{
  margin-top:4px;
  color:#758298;
  font-size:12px;
}
.kb-evidence-preview{
  margin-top:5px;
  color:#8a929f;
  font-size:12px;
  line-height:1.65;
}
.kb-chunk-details{
  margin:7px 0 2px;
  color:#667b9e;
}
.kb-chunk-details > summary{
  display:inline-flex;
  align-items:center;
  gap:5px;
  cursor:pointer;
  font-size:12.5px;
  line-height:1.9;
  user-select:none;
  list-style:none;
  border-radius:4px;
}
.kb-chunk-details > summary::-webkit-details-marker{display:none}
.kb-chunk-details > summary::marker{content:""}
.kb-chunk-details > summary::before{
  content:"▸";
  font-size:12px;
  transition:transform .15s ease;
}
.kb-chunk-details[open] > summary::before{
  transform:rotate(90deg);
}
.kb-chunk-details > summary:focus-visible{
  outline:2px solid #8eaaff;
  outline-offset:3px;
}
.kb-chunk-details .kb-label-close{display:none}
.kb-chunk-details[open] .kb-label-open{display:none}
.kb-chunk-details[open] .kb-label-close{display:inline}
.kb-chunk-full{
  box-sizing:border-box;
  max-height:400px;
  overflow-y:auto;
  overflow-wrap:anywhere;
  white-space:pre-wrap;
  user-select:text;
  background:#f6f8fb;
  border:1px solid #e6ebf4;
  border-radius:8px;
  margin:8px 0 5px;
  padding:13px 15px;
  color:#3e4c62;
  font-size:12.5px;
  line-height:1.8;
}
.kb-evidence-old-note{
  color:#9a8b74;
  font-size:12px;
  margin-top:7px;
}


/* GraphRAG 候选证据：只有展开检索详情时才出现，配色与“豆馅”一致。 */
.kb-graph-evidence-item{
  padding:15px 17px;
  border:1px solid #e2eaf6;
  border-left:3px solid #8aa9e6;
  border-radius:12px;
  background:#fbfcff;
  box-shadow:0 4px 18px rgba(20,48,100,.035);
}
.kb-graph-evidence-top{
  display:flex;gap:8px;align-items:center;flex-wrap:wrap;
  margin-bottom:8px;
}
.kb-graph-evidence-name{font-size:13.5px;font-weight:650;color:#293b59;}
.kb-graph-tag{
  padding:2px 9px;border-radius:20px;background:#fff4e3;color:#956820;
  font-size:11px;line-height:1.6;font-weight:600;
}
.kb-graph-evidence-quote{
  margin:8px 0 9px;border-left:2px solid #ccd8ed;padding:6px 0 6px 12px;
  color:#3d526b;font-size:13px;line-height:1.8;white-space:pre-wrap;overflow-wrap:anywhere;
}
.kb-graph-meta{font-size:12px;color:#8190a5;margin-top:7px;}
.kb-graph-source{
  display:inline-flex;align-items:center;gap:5px;margin-top:7px;
  font-size:12px;font-weight:550;color:#2457cd;text-decoration:none;
}
.kb-graph-source:hover{text-decoration:underline;color:#173c92;}

/* 只调整运行时状态行，已保存的搜索思考记录样式不受影响。 */
.st-key-douxian-live-progress .trace-live{
  margin:2px 0 0;
  min-height:32px;
}
.st-key-douxian-live-progress [data-testid="stHorizontalBlock"]{
  align-items:center;
  gap:6px !important;
}
.st-key-douxian-live-progress iframe{
  display:block;
  border:0;
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
    """限制展示条数，也兼容修改前存到 SQLite 的旧记录。"""
    if not isinstance(trace, dict):
        return []
    result = []
    seen = set()
    sources = trace.get("sources")
    for source in (sources if isinstance(sources, list) else []):
        if not isinstance(source, dict):
            continue
        url = source.get("url", "")
        if not isinstance(url, str):
            continue
        try:
            parsed = urlsplit(url)
        except ValueError:
            continue
        if parsed.scheme not in ("http", "https") or not parsed.netloc or url in seen:
            continue
        seen.add(url)
        result.append({"url": url, "title": str(source.get("title") or parsed.netloc)})
        if len(result) >= DISPLAY_SOURCE_LIMIT:
            break
    return result


def _web_queries(trace):
    """只展示 Tavily 实际执行的检索关键词，不猜测模型内部思考。"""
    if not isinstance(trace, dict):
        return []
    candidates = []
    if isinstance(trace.get("search_queries"), list):
        candidates.extend(trace["search_queries"])
    for event in trace.get("events") or []:
        if isinstance(event, dict) and event.get("tool") == "web_search":
            values = event.get("queries")
            if isinstance(values, list):
                candidates.extend(values)
    result = []
    for value in candidates:
        if isinstance(value, str):
            value = " ".join(value.split())[:160]
            if value and value not in result:
                result.append(value)
        if len(result) >= 8:
            break
    return result


def trace_heading(trace):
    if not isinstance(trace, dict):
        return "已完成回答"
    events = trace.get("events") or []
    ends = [event for event in events if isinstance(event, dict)
            and event.get("type") == "end"]
    web_ends = [event for event in ends if event.get("tool") == "web_search"]
    failure = any(not event.get("ok", False) for event in ends)
    if web_ends and any(event.get("ok") for event in web_ends):
        count = len(_web_sources(trace))
        text = f"已完成分析，展示 {count} 篇参考资料，相关操作已执行" if count else "已完成联网分析"
        return text + ("（部分操作未成功）" if failure else "")
    kb_ends = [event for event in ends
               if event.get("tool") in ("search_knowledge_base", "search_drug_graph")]
    if kb_ends and any(event.get("ok") for event in kb_ends):
        report = trace.get("retrieval_summary")
        if isinstance(report, dict):
            graph_hits = report.get("search_drug_graph", {}).get("hits", 0)
            chroma_hits = report.get("search_knowledge_base", {}).get("hits", 0)
            try:
                graph_hits, chroma_hits = int(graph_hits), int(chroma_hits)
            except (ValueError, TypeError):
                graph_hits, chroma_hits = 0, 0
            label = f"已检索医疗知识库 · Neo4j {graph_hits} 条 / Chroma {chroma_hits} 条"
            validation = report.get("search_knowledge_base", {}).get("medical_validation")
            if isinstance(validation, dict) and validation.get("verified"):
                label += f" · 原文核验 {validation['verified']} 条"
        elif trace.get("graph_answer_kind") == "unverified_candidates":
            label = f"已检索药品图谱 · {trace.get('graph_candidates_found', 0)} 条候选证据"
        else:
            excerpts = trace.get("knowledge_evidence")
            count = len(excerpts) if isinstance(excerpts, list) else 0
            label = f"已完成知识库检索 · {count} 条证据" if count else "已完成知识库检索"
        return label + ("（部分操作未成功）" if failure else "")
    if failure:
        return "已完成分析，部分操作未成功"
    if ends:
        return "已完成分析，相关操作已执行"
    return "已完成回答"


def render_html_fragment(body, holder=None):
    """st.html 支持原生 details；老版本 Streamlit 回退到 st.markdown。"""
    target = holder if holder is not None else st
    if hasattr(target, "html"):
        target.html(body)
    else:
        target.markdown(body, unsafe_allow_html=True)


def live_trace_html(label):
    """轻量状态行：无边框、无色块，且不会展示虚构的思考步骤。"""
    return (
        '<div class="trace-live" role="status" aria-live="polite">'
        '<span class="trace-live-spark" aria-hidden="true">✦</span>'
        f'<span>{html.escape(str(label))}</span>'
        '</div>'
    )


def live_timer_html(label, elapsed_ms):
    """在浏览器里单独计时；Python 等待 Tavily 时数字也会每秒变化。"""
    element_id = "douxian_clock_" + uuid.uuid4().hex
    elapsed_ms = max(0, int(elapsed_ms))
    total_seconds = elapsed_ms // 1000
    fallback = f"{total_seconds // 60}m {total_seconds % 60}s"
    safe_label = html.escape(str(label))
    spark = '<span class="dx-spark" aria-hidden="true">✦</span>' if safe_label else ""
    prefix = safe_label + " · " if safe_label else ""
    return f"""
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<style>
html, body {{margin:0;padding:0;overflow:hidden;background:transparent}}
.dx-timer {{
  display:flex;align-items:center;gap:8px;height:32px;
  color:#718095;font:13.5px/1.55 Inter,-apple-system,BlinkMacSystemFont,
  "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
}}
.dx-spark {{color:#7189c0;font-size:15px;flex:none}}
.dx-time {{font-variant-numeric:tabular-nums;white-space:nowrap}}
</style></head><body>
<div class="dx-timer" role="status">
 {spark}
 <span>{prefix}<span class="dx-time" id="{element_id}">{fallback}</span></span>
</div>
<script>
(()=>{{
  "use strict";
  const node = document.getElementById("{element_id}");
  const startedAt = performance.now();
  const initialMs = {elapsed_ms};
  function update(){{
    const total = Math.max(0, Math.floor((initialMs + performance.now() - startedAt) / 1000));
    node.textContent = `${{Math.floor(total/60)}}m ${{total%60}}s`;
  }}
  update();
  const id = window.setInterval(update, 250);
  window.addEventListener("pagehide", () => window.clearInterval(id), {{once:true}});
}})();
</script></body></html>
"""


def render_live_timer(label, started_at):
    """计时 iframe 每轮只创建一次，不参与服务器阶段更新时的容器替换。"""
    elapsed_ms = int(max(0, time.perf_counter() - started_at) * 1000)
    components.html(live_timer_html(label, elapsed_ms), height=36, scrolling=False)


TOOL_LABELS = {
    "search_knowledge_base": "知识库检索",
    "search_drug_graph": "Neo4j 图谱检索",
    "get_weather": "天气查询",
    "web_search": "联网搜索",
}


def describe_tool_event(event):
    """保留真实的工具运行结果，不显示几十条底层网页计数。"""
    if not isinstance(event, dict) or event.get("type") != "end":
        return ""
    name = TOOL_LABELS.get(event.get("tool"), "外部工具")
    if not event.get("ok", False):
        return f"⚠ {name}未成功"
    return f"✓ 已完成{name}"


def graph_evidence_html(item, index):
    """GraphRAG 专用的折叠证据卡；HTML 全部转义，不执行原文中的标签。"""
    drug = html.escape(str(item.get("drug_name") or "未知药品")[:120])
    disease = html.escape(str(item.get("disease_term") or "未知疾病词条")[:120])
    quote = html.escape(str(item.get("evidence_quote") or "原文片段缺失")[:500])
    raw_url = str(item.get("source_url") or "").strip()
    url_html = ''
    if len(raw_url) <= 2048:
        try:
            parts = urlsplit(raw_url)
            if parts.scheme in ("http", "https") and parts.netloc:
                url_html = (
                    '<a class="kb-graph-source" href="'
                    + html.escape(raw_url, quote=True)
                    + '" target="_blank" rel="noopener noreferrer">'
                    '查看原始药品资料 ↗</a>'
                )
        except ValueError:
            pass
    if not url_html:
        url_html = '<div class="kb-graph-meta">原始来源链接未提供</div>'

    raw_review = str(item.get("candidate_status") or "")
    raw_clinical = str(item.get("clinical_status") or "")
    raw_source_status = str(item.get("source_status") or "")
    review = "待人工复核" if raw_review == "pending_human_review" else "审核状态待核对"
    clinical = "医学有效性未验证" if raw_clinical == "not_validated" else "医学验证状态待核对"
    source_label = "第三方来源待核验" if raw_source_status == "third_party_unverified" else "来源状态待核验"

    full = item.get("content")
    full_details = (
        '<details class="kb-chunk-details"><summary>查看原始结构化记录</summary>'
        '<div class="kb-chunk-full">' + html.escape(full[:7000]) + '</div></details>'
    ) if isinstance(full, str) and full else ''
    return (
        '<div class="kb-evidence-item kb-graph-evidence-item">'
        '<div class="kb-graph-evidence-top">'
        f'<span class="kb-graph-evidence-name">[{index}] {drug} / {disease}</span>'
        f'<span class="kb-graph-tag">{review}</span>'
        '</div>'
        '<div class="kb-graph-evidence-quote">“' + quote + '”</div>'
        '<div class="kb-graph-meta">' + clinical + ' · ' + source_label + '</div>'
        + url_html + full_details + '</div>'
    )


def trace_details_html(trace):
    """千问风格的原生折叠区：实际检索词 + 最多 5 个紧凑来源标签。"""
    if not isinstance(trace, dict) or not trace:
        return ""

    heading = html.escape(trace_heading(trace))
    events = trace.get("events") if isinstance(trace.get("events"), list) else []
    ends = [event for event in events
            if isinstance(event, dict) and event.get("type") == "end"]
    web_ends = [event for event in ends if event.get("tool") == "web_search"]
    sources = _web_sources(trace)
    queries = _web_queries(trace)
    content = []

    if web_ends:
        search_label = (
            f"搜索 {len(queries)} 组关键词" if queries else "已执行联网搜索"
        )
        if sources:
            search_label += f"，展示 {len(sources)} 篇参考资料"
        content.append(
            '<div class="trace-step-title">'
            '<span class="trace-step-icon" aria-hidden="true">⌕</span>'
            f'<span>{html.escape(search_label)}</span></div>'
        )
        if queries:
            keyword_html = "".join(
                f'<span class="trace-query">“{html.escape(query)}”</span>'
                for query in queries
            )
            content.append('<div class="trace-queries">' + keyword_html + '</div>')

        if sources:
            chips = []
            for source in sources:
                url = html.escape(source["url"], quote=True)
                title = html.escape(source["title"])
                chips.append(
                    f'<a class="trace-source-chip" href="{url}" '
                    'target="_blank" rel="noopener noreferrer">'
                    '<span class="trace-chip-icon" aria-hidden="true">↗</span>'
                    f'<span class="trace-chip-text">{title}</span></a>'
                )
            content.append('<div class="trace-source-chips">' + "".join(chips) + '</div>')
        if any(not event.get("ok", False) for event in web_ends):
            content.append('<p class="trace-alert">⚠ 部分联网操作未成功，请检查服务配置。</p>')

    # 与回答 [1]、[2] 引用对应的真实知识库证据；所有字段均进行 HTML 转义。
    knowledge_evidence = trace.get("knowledge_evidence")
    if isinstance(knowledge_evidence, list) and knowledge_evidence:
        cards = []
        for item in knowledge_evidence[:5]:
            if not isinstance(item, dict):
                continue
            index = item.get("id")
            index = str(index) if type(index) is int and index > 0 else "?"
            if item.get("evidence_type") == "unverified_drug_disease_candidate":
                cards.append(graph_evidence_html(item, index))
                continue
            source = html.escape(str(item.get("source") or "未知文件"))
            section = html.escape(str(item.get("section_path") or "未标注章节"))
            preview = html.escape(str(item.get("preview") or ""))
            full_text = item.get("content")
            if isinstance(full_text, str) and full_text:
                # 证据作为纯文本渲染，保留换行，同时防止 Markdown 中的 HTML 注入。
                safe_full_text = html.escape(full_text)
                chunk_details = (
                    '<details class="kb-chunk-details">'
                    '<summary><span class="kb-label-open">展开完整 Chunk</span>'
                    '<span class="kb-label-close">收起完整 Chunk</span></summary>'
                    f'<div class="kb-chunk-full">{safe_full_text}</div>'
                    '</details>'
                )
            else:
                # 兼容以前的 SQLite 历史消息：旧记录只存过 240 字摘要。
                chunk_details = (
                    '<div class="kb-evidence-old-note">'
                    '旧消息仅保存了摘要，重新提问后即可展开完整 Chunk。'
                    '</div>'
                )
            raw_url = str(item.get("source_url") or "").strip()
            source_link_html = ""
            if len(raw_url) <= 2048:
                try:
                    parsed = urlsplit(raw_url)
                    if parsed.scheme in ("http", "https") and parsed.netloc:
                        source_link_html = (
                            '<a class="kb-graph-source" target="_blank" '
                            'rel="noopener noreferrer" href="'
                            + html.escape(raw_url, quote=True)
                            + '">查看药品资料来源 ↗</a>'
                        )
                except ValueError:
                    pass
            verified_label = (
                '<div class="kb-evidence-section">原文回溯：Neo4j Section 精确匹配 ✓ '
                '（不代表临床审核）</div>'
                if item.get("parent_verified") else ""
            )
            cards.append(
                '<div class="kb-evidence-item">'
                f'<div class="kb-evidence-title">[{index}] {source}</div>'
                f'<div class="kb-evidence-section">章节：{section}</div>'
                + verified_label
                + (f'<div class="kb-evidence-preview">摘要：{preview}</div>' if preview else "")
                + source_link_html
                + chunk_details
                + '</div>'
            )
        if cards:
            content.append(
                '<div class="trace-step-title">'
                '<span class="trace-step-icon" aria-hidden="true">⌕</span>'
                f'<span>实际命中的知识库证据（{len(cards)} 条）</span></div>'
            )
            content.append('<div class="kb-evidence-list">' + "".join(cards) + '</div>')

    report = trace.get("retrieval_summary")
    if isinstance(report, dict):
        graph_status = report.get("search_drug_graph", {}).get("status", "未执行")
        chroma_status = report.get("search_knowledge_base", {}).get("status", "未执行")
        status_chinese = {
            "ok": "已命中", "no_match": "未命中", "empty_collection": "文档库为空",
            "error": "检索失败", "invalid_response": "响应异常",
            "provenance_rejected": "原文定位未通过，已拒绝引用",
            "no_relevant_sections": "本次命中栏目不相关",
        }
        parts = [
            "Neo4j：" + status_chinese.get(graph_status, "状态待核对"),
            "Chroma：" + status_chinese.get(chroma_status, "状态待核对"),
        ]
        if chroma_status == "empty_collection":
            parts.append("请检查 collection_name 与医疗文档入库情况")
        medical_validation = report.get("search_knowledge_base", {}).get(
            "medical_validation")
        if isinstance(medical_validation, dict):
            verified = int(medical_validation.get("verified") or 0)
            rejected = int(medical_validation.get("failed_provenance") or 0)
            skipped = int(medical_validation.get("skipped_section") or 0)
            if verified or rejected or skipped:
                parts.append(
                    f"原文回溯通过 {verified} 条，栏目不相关 {skipped} 条，"
                    f"来源校验拒绝 {rejected} 条"
                )
        content.append(
            '<p class="trace-note">' + html.escape(" · ".join(parts)) + '</p>'
        )

    other_ends = [event for event in ends if event.get("tool") != "web_search"]
    if other_ends:
        steps = "".join(
            '<li>' + html.escape(describe_tool_event(event)) + '</li>'
            for event in other_ends
        )
        content.append('<ul class="trace-events">' + steps + '</ul>')

    if not ends:
        content.append('<p class="trace-note">本次直接生成回答，未调用外部工具。</p>')

    if "total_seconds" in trace:
        content.append(
            '<p class="trace-total">本次耗时 '
            + html.escape(elapsed_label(trace["total_seconds"]))
            + '（含回答生成）</p>'
        )

    return (
        '<details class="trace-panel">'
        '<summary><span class="trace-search-icon" aria-hidden="true"></span>'
        f'<span class="trace-summary-text">{heading}</span>'
        '<span class="trace-chevron" aria-hidden="true"></span></summary>'
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

    # 历史会话可能记录旧 GraphRAG 模式；统一归入“知识库”。
    if st.session_state.get("douxian_answer_mode") == "GraphRAG":
        st.session_state["douxian_answer_mode"] = "知识库"

    st.radio(
        "回答模式",
        options=["知识库", "自动", "联网"],
        key="douxian_answer_mode",
        horizontal=True,
        help="知识库：同时检索 Neo4j 药品图谱和原文章节、Chroma Hybrid 文档；自动：由 Agent 选择工具；联网：检索最新公开信息。",
    )

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

    # 状态文本和实时计时器分开放置：
    # 状态变化只替换左侧的原生 HTML，绝不反复卸载右侧的 iframe。
    # 计时 iframe 仅在本轮启动时创建一次，由其内部 JS 持续更新秒数。
    with st.container(key="douxian-live-progress"):
        status_column, elapsed_column = st.columns([1.6, 5.0], gap="small")
        with status_column:
            phase_slot = st.empty()
            render_html_fragment(live_trace_html("豆馅正在思考"), phase_slot)
        with elapsed_column:
            render_live_timer("", started)

    def show_progress(event):
        if not isinstance(event, dict):
            return
        action = TOOL_LABELS.get(event.get("tool"), "处理请求")
        if event.get("type") == "start":
            phase = "正在" + action
        elif event.get("type") == "end":
            phase = action + ("已返回" if event.get("ok") else "未成功")
        else:
            return
        # 这里只更新普通文本，右侧 iframe 保持不变。
        render_html_fragment(live_trace_html(phase), phase_slot)

    # 以 SQLite 恢复的当前会话为模型上下文的唯一 UI 来源；
    # 当前用户消息已经在 messages 末尾保存，因此不再重复放入 history。
    previous_messages = messages[:-1] if messages and messages[-1].get("role") == "user" else messages
    stream = st.session_state["rag"].chain.stream(
        {
            "input": pending,
            "mode": st.session_state.get("douxian_answer_mode", "知识库"),
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
                render_html_fragment(live_trace_html("正在生成回答"), phase_slot)
            yield chunk

    try:
        answer = st.write_stream(stream_with_timing(stream))
    except Exception:
        render_html_fragment(live_trace_html("回答未完成 · 请检查服务日志"), phase_slot)
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

    # 不要在这里把正在计时的 iframe 再替换成 <details>。
    # 先保存完整数据，随后只执行一次 st.rerun()，
    # 新一轮页面会正常绘制历史消息下的折叠思考记录。

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
