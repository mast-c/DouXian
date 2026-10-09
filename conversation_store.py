"""豆馅页面会话持久化：本地 SQLite，无额外第三方依赖。

会话标题、用户/助手消息、当前选中的会话保存到 chat_history/conversations.sqlite3。
RAG 的 LangChain 历史仍由 file_history_store.py 以相同 session_id 保存。
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_HISTORY_DIR = Path(__file__).resolve().parent / "chat_history"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _valid_session_id(value):
    """只导入本 UI 生成的 UUID 会话，不导入 RAG 的测试 session_id。"""
    try:
        return str(uuid.UUID(str(value))) == str(value)
    except (ValueError, AttributeError, TypeError):
        return False


def _title_from_question(question, limit=20):
    question = " ".join(str(question).split())
    return question[:limit] + "…" if len(question) > limit else question or "新对话"


def _read_legacy_messages(file_path):
    """读取旧版 file_history_store.py 的 LangChain JSON，不依赖 LangChain 导入。"""
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError, UnicodeError):
        return []

    if not isinstance(data, list):
        return []

    messages = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = {"human": "user", "ai": "assistant"}.get(item.get("type"))
        if role is None:
            continue
        payload = item.get("data") or {}
        content = payload.get("content") if isinstance(payload, dict) else None
        if isinstance(content, list):
            # 兼容少数模型返回的多文本块格式。
            content = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


class ConversationStore:
    """以 SQLite 为页面显示历史的持久化来源。"""

    def __init__(self, db_path=None, history_dir=None):
        self.history_dir = Path(history_dir) if history_dir else DEFAULT_HISTORY_DIR
        self.history_dir = self.history_dir.resolve()
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else self.history_dir / "conversations.sqlite3"
        self.db_path = self.db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        # sqlite3.Connection 的内置 with 只负责提交事务，不负责关闭连接。
        # 必须显式关闭，避免 Streamlit rerun 导致 Windows 数据库被占用。
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES conversations(session_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS ix_messages_session
                    ON messages(session_id, message_id);
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            # 仅在首次初始化时导入旧 chat_history 里的 UUID 文件。
            # 防止以后已经删除的历史会话被反复导入。
            migrated = conn.execute(
                "SELECT value FROM app_state WHERE key='legacy_import_completed'"
            ).fetchone()
            if migrated is None:
                self._import_legacy_histories(conn)
                conn.execute(
                    "INSERT OR IGNORE INTO app_state(key,value) VALUES(?,?)",
                    ("legacy_import_completed", "1"),
                )

    def _import_legacy_histories(self, conn):
        candidates = []
        for file_path in self.history_dir.iterdir():
            if not file_path.is_file() or not _valid_session_id(file_path.name):
                continue
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, file_path))

        for mtime, file_path in sorted(candidates, key=lambda item: (item[0], item[1].name)):
            messages = _read_legacy_messages(file_path)
            if not messages:
                continue
            first_question = next((m["content"] for m in messages if m["role"] == "user"), "")
            ts = datetime.fromtimestamp(mtime, timezone.utc).isoformat(timespec="microseconds")
            cursor = conn.execute(
                "INSERT OR IGNORE INTO conversations(session_id,title,created_at,updated_at) "
                "VALUES(?,?,?,?)",
                (file_path.name, _title_from_question(first_question), ts, ts),
            )
            if cursor.rowcount == 0:
                continue
            conn.executemany(
                "INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)",
                [(file_path.name, m["role"], m["content"], ts) for m in messages],
            )

    def load_all(self):
        """返回与原 Streamlit state 相同格式、按最后活动时间正序排列的 dict。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id,title FROM conversations "
                "ORDER BY updated_at ASC, rowid ASC"
            ).fetchall()
            chats = {
                row["session_id"]: {"title": row["title"], "messages": []}
                for row in rows
            }
            for row in conn.execute(
                "SELECT session_id,role,content FROM messages ORDER BY message_id"
            ):
                if row["session_id"] in chats:
                    chats[row["session_id"]]["messages"].append({
                        "role": row["role"], "content": row["content"]
                    })
            return chats

    def get_active(self):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM app_state WHERE key='current_session'"
            ).fetchone()
            return row["value"] if row else None

    @staticmethod
    def _set_active(conn, sid):
        conn.execute(
            "INSERT INTO app_state(key,value) VALUES('current_session',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (sid,),
        )

    def set_active(self, sid):
        with self._connect() as conn:
            found = conn.execute(
                "SELECT 1 FROM conversations WHERE session_id=?", (sid,)
            ).fetchone()
            if not found:
                return False
            self._set_active(conn, sid)
            return True

    def create(self, sid, title="新对话"):
        if not _valid_session_id(sid):
            raise ValueError("会话 ID 必须为 UUID")
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations(session_id,title,created_at,updated_at) "
                "VALUES(?,?,?,?)", (sid, title or "新对话", now, now)
            )
            self._set_active(conn, sid)

    def add_message(self, sid, role, content, title=None):
        if role not in ("user", "assistant"):
            raise ValueError("只允许保存 user / assistant 消息")
        if not isinstance(content, str):
            content = str(content)
        now = _now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT title FROM conversations WHERE session_id=?", (sid,)
            ).fetchone()
            if row is None:
                raise KeyError(f"会话不存在：{sid}")
            conn.execute(
                "INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)",
                (sid, role, content, now),
            )
            if title is None:
                conn.execute(
                    "UPDATE conversations SET updated_at=? WHERE session_id=?", (now, sid)
                )
            else:
                conn.execute(
                    "UPDATE conversations SET title=?,updated_at=? WHERE session_id=?",
                    (title.strip() or "新对话", now, sid),
                )

    def delete(self, sid):
        """永久删除会话标题/页面消息，并把已选会话切换到现存最新一条。"""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE session_id=?", (sid,))
            if cursor.rowcount == 0:
                return False
            selected = conn.execute(
                "SELECT value FROM app_state WHERE key='current_session'"
            ).fetchone()
            if selected is not None and selected["value"] == sid:
                latest = conn.execute(
                    "SELECT session_id FROM conversations ORDER BY updated_at DESC,rowid DESC LIMIT 1"
                ).fetchone()
                if latest is None:
                    conn.execute("DELETE FROM app_state WHERE key='current_session'")
                else:
                    self._set_active(conn, latest["session_id"])
            return True

    def merge_session_state(self, memory_chats):
        """代码热更新时接住旧 Streamlit 内存里的对话，避免未刷新的消息丢失。"""
        if not isinstance(memory_chats, dict):
            return
        with self._connect() as conn:
            for sid, chat in memory_chats.items():
                if not _valid_session_id(sid) or not isinstance(chat, dict):
                    continue
                incoming = [
                    (m["role"], str(m["content"]))
                    for m in chat.get("messages", [])
                    if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                    and isinstance(m.get("content"), str)
                ]
                row = conn.execute(
                    "SELECT title FROM conversations WHERE session_id=?", (sid,)
                ).fetchone()
                now = _now()
                title = str(chat.get("title") or "新对话").strip() or "新对话"
                if row is None:
                    conn.execute(
                        "INSERT INTO conversations(session_id,title,created_at,updated_at) VALUES(?,?,?,?)",
                        (sid, title, now, now),
                    )
                    existing = []
                else:
                    existing = [tuple(r) for r in conn.execute(
                        "SELECT role,content FROM messages WHERE session_id=? ORDER BY message_id",
                        (sid,),
                    )]
                    if row["title"] == "新对话" and title != "新对话":
                        conn.execute(
                            "UPDATE conversations SET title=? WHERE session_id=?", (title, sid)
                        )
                # 仅补存尾部新增消息，不重复写入数据库现有消息。
                if incoming[:len(existing)] != existing:
                    continue
                extra = incoming[len(existing):]
                if extra:
                    conn.executemany(
                        "INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)",
                        [(sid, role, text, now) for role, text in extra],
                    )
                    conn.execute(
                        "UPDATE conversations SET updated_at=? WHERE session_id=?", (now, sid)
                    )
