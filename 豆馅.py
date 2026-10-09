import html
import uuid
import streamlit as st
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
    """创建一个新会话，并把它设为当前会话。"""
    sid = str(uuid.uuid4())
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
    del conversations[sid]

    if deleting_current:
        # 当前正在等待回答的 prompt 属于被删会话时，不应继续处理。
        state["pending_prompt"] = None

        if conversations:
            # dict 保持插入顺序，取剩余会话里最新创建的一条。
            state["current_session"] = next(reversed(conversations))
        else:
            create_conversation(state)

    state["delete_confirm_sid"] = None
    return True


# =========================
# 会话状态
# =========================

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

if "conversations" not in st.session_state:
    st.session_state["conversations"] = {}

if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None
    
if "force_search" not in st.session_state:
    st.session_state["force_search"] = False
    
if "delete_confirm_sid" not in st.session_state:
    st.session_state["delete_confirm_sid"] = None

# 不只判断 current_session 是否存在，还保证它确实指向一个现有会话。
if (
    "current_session" not in st.session_state
    or st.session_state["current_session"]
    not in st.session_state["conversations"]
):
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

html,body,[class*="st-"]{
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

/* 保留你原来的隐藏规则 */
span.st-emotion-cache-5r6ut5{
  display:none !important;
  visibility:hidden !important;
  opacity:0 !important;
  font-size:0 !important;
  width:0 !important;
  height:0 !important;
  color:transparent !important;
}

button:has(> span.st-emotion-cache-5r6ut5){
  display:none !important;
  pointer-events:none !important;
}
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
</style>
""",
    unsafe_allow_html=True,
)


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
    chat["messages"].append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # 兼容旧数据：空标题也视为“新对话”。
    if display_title(chat) == "新对话":
        chat["title"] = short_title(prompt)

    st.session_state["pending_prompt"] = prompt
    st.rerun()


# =========================
# AI 流式回答
# =========================

pending = st.session_state.get("pending_prompt")

if pending:
  
    st.session_state["pending_prompt"] = None


    # =========================
    # AI 思考提示
    # =========================

    thinking_box = st.empty()

    thinking_box.markdown(
    '<div class="thinking-row">'
    '<span class="thinking-spark">✦</span>'
    '<span>豆馅正在思考</span>'
    '<span class="thinking-dots" aria-hidden="true">'
    '<i></i><i></i><i></i>'
    '</span>'
    '</div>',
    unsafe_allow_html=True,
    )


    # =========================
    # 调用 RAG
    # =========================

    stream = (
        st.session_state["rag"]
        .chain
        .stream(
            {
                "input": pending
            },
            {
                "configurable": {
                    "session_id":
                        st.session_state[
                            "current_session"
                        ]
                }
            },
        )
    )


    # =========================
    # 第一段回答出来时
    # 自动删除“思考中”
    # =========================

    def stream_after_thinking(source):

        first_content_seen = False

        try:

            for chunk in source:

                has_content = (
                    chunk is not None
                    and (
                        not isinstance(
                            chunk,
                            str
                        )
                        or bool(
                            chunk.strip()
                        )
                    )
                )


                if (
                    has_content
                    and
                    not first_content_seen
                ):

                    thinking_box.empty()

                    first_content_seen = True


                yield chunk


        finally:

            # 即使模型没有返回内容
            # 或流式过程中出现异常，
            # 思考提示也不会一直留着
            thinking_box.empty()


    # =========================
    # AI 流式回答
    # =========================

    answer = st.write_stream(
        stream_after_thinking(
            stream
        )
    )


    if answer is None:

        answer = ""


    chat["messages"].append(
        {
            "role": "assistant",
            "content": str(answer),
        }
    )


    st.rerun()
