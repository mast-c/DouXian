"""豆馅的 SQLite 会话存储。

所有数据库连接均在调用结束后关闭；数据库文件默认保存在项目目录 data/chat_history.sqlite3。
该方案面向单用户本地部署。若对外提供服务，请先增加用户身份与数据隔离。
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class ChatStore:
    """负责保存和恢复会话标题、消息、当前会话。"""

    def __init__(self, db_path: Optional[str] = None):
        custom_path = db_path or os.getenv("DOUXIAN_CHAT_DB")
        self.db_path = (
            Path(custom_path).expanduser()
            if custom_path
            else Path(__file__).resolve().parent / "data" / "chat_history.sqlite3"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self):
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (session_id)
                        REFERENCES conversations(session_id) ON DELETE CASCADE
                )
            """)
            # 自动迁移旧数据库：不清除已有对话和消息。
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(messages)")
            }
            if "metadata_json" not in columns:
                conn.execute(
                    "ALTER TABLE messages ADD COLUMN metadata_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                )
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_order
                ON messages(session_id, message_id)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

    def load_conversations(self) -> dict:
        """以创建顺序加载，兼容现有 UI 的 reversed(dict.items())。"""
        with self._connection() as conn:
            conversations = {
                row["session_id"]: {"title": row["title"], "messages": []}
                for row in conn.execute("""
                    SELECT session_id, title FROM conversations
                    ORDER BY created_at ASC, rowid ASC
                """)
            }
            for row in conn.execute("""
                SELECT session_id, role, content, metadata_json FROM messages
                ORDER BY message_id ASC
            """):
                item = conversations.get(row["session_id"])
                if item is not None:
                    message = {
                        "role": row["role"], "content": row["content"]
                    }
                    try:
                        trace = json.loads(row["metadata_json"] or "{}")
                    except (ValueError, TypeError):
                        trace = {}
                    if row["role"] == "assistant" and isinstance(trace, dict) and trace:
                        message["trace"] = trace
                    item["messages"].append(message)
            return conversations

    def get_active_session(self) -> Optional[str]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM app_state WHERE key = ?",
                ("active_session",),
            ).fetchone()
            return row["value"] if row else None

    def set_active_session(self, session_id: str):
        with self._connection() as conn:
            if not conn.execute(
                "SELECT 1 FROM conversations WHERE session_id = ?", (session_id,)
            ).fetchone():
                raise KeyError(f"会话不存在：{session_id}")
            conn.execute("""
                INSERT INTO app_state(key, value) VALUES ('active_session', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (session_id,))

    def create_conversation(self, session_id: str, title: str = "新对话"):
        """创建会话，同时持久化当前会话选中状态。"""
        stamp = _utc_now()
        with self._connection() as conn:
            conn.execute("""
                INSERT INTO conversations(session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, title, stamp, stamp))
            conn.execute("""
                INSERT INTO app_state(key, value) VALUES ('active_session', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (session_id,))

    def append_message(
        self, session_id: str, role: str, content: str,
        *, new_title: Optional[str] = None, trace: Optional[dict] = None,
    ):
        """消息和（可选）首问标题在同一个事务中提交。"""
        if role not in ("user", "assistant"):
            raise ValueError("role 必须是 user 或 assistant")
        stamp = _utc_now()
        # 工具步骤只随 assistant 消息保存；JSON 可随数据库一起备份和恢复。
        metadata_json = json.dumps(
            trace if role == "assistant" and isinstance(trace, dict) else {},
            ensure_ascii=False,
        )
        with self._connection() as conn:
            updated = conn.execute("""
                UPDATE conversations
                SET updated_at = ?, title = COALESCE(?, title)
                WHERE session_id = ?
            """, (stamp, new_title, session_id))
            if updated.rowcount != 1:
                raise KeyError(f"会话不存在：{session_id}")
            conn.execute("""
                INSERT INTO messages(
                    session_id, role, content, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, str(content), stamp, metadata_json))

    def delete_conversation(self, session_id: str) -> bool:
        """级联删除本数据库里的所有消息，并移除失效的选中标记。"""
        with self._connection() as conn:
            result = conn.execute(
                "DELETE FROM conversations WHERE session_id = ?", (session_id,)
            )
            if result.rowcount:
                conn.execute("""
                    DELETE FROM app_state
                    WHERE key = 'active_session' AND value = ?
                """, (session_id,))
            return result.rowcount > 0

    def import_existing(self, conversations: dict):
        """热更新兼容：只迁移尚未写入数据库的旧 Session State 会话。"""
        if not isinstance(conversations, dict):
            return
        with self._connection() as conn:
            for sid, chat in conversations.items():
                if not isinstance(sid, str) or not sid or not isinstance(chat, dict):
                    continue
                title = str(chat.get("title", "") or "").strip() or "新对话"
                stamp = _utc_now()
                inserted = conn.execute("""
                    INSERT OR IGNORE INTO conversations(
                        session_id, title, created_at, updated_at
                    ) VALUES (?, ?, ?, ?)
                """, (sid, title, stamp, stamp))
                if inserted.rowcount != 1:
                    continue  # 已有持久记录时避免重复导入消息
                for message in chat.get("messages", []):
                    if not isinstance(message, dict):
                        continue
                    role = message.get("role")
                    if role not in ("user", "assistant"):
                        continue
                    trace = message.get("trace", {}) if role == "assistant" else {}
                    metadata_json = json.dumps(
                        trace if isinstance(trace, dict) else {}, ensure_ascii=False
                    )
                    conn.execute("""
                        INSERT INTO messages(
                            session_id, role, content, created_at, metadata_json
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        sid, role, str(message.get("content", "")),
                        _utc_now(), metadata_json,
                    ))
