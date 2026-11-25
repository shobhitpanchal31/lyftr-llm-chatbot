from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional


def resolve_sqlite_path(database_url: str) -> str:
    if database_url == ":memory:" or database_url == "sqlite:///:memory:":
        return ":memory:"
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///") :]
    if database_url.startswith("sqlite://"):
        return database_url[len("sqlite://") :]
    return database_url


class Database:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.db_path = resolve_sqlite_path(database_url)
        if self.db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self.db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        # Keep a persistent connection when using in-memory sqlite for tests
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            yield self._memory_conn
            return

        conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    metadata    TEXT,
                    created_at  TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, id);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id    TEXT PRIMARY KEY,
                    system_prompt TEXT,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def check_health(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1;").fetchone()
            return True
        except Exception:
            return False

    def get_session_system_prompt(self, session_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?;",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return row["system_prompt"]

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC;
                """,
                (session_id,),
            ).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            item: Dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
            }
            if row["role"] == "assistant" and row["metadata"]:
                item["metadata"] = json.loads(row["metadata"])
            results.append(item)
        return results

    def save_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        assistant_metadata: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(assistant_metadata)

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?;",
                (session_id,),
            ).fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, system_prompt, created_at, updated_at)
                    VALUES (?, ?, ?, ?);
                    """,
                    (session_id, system_prompt, now, now),
                )
            else:
                effective_prompt = (
                    system_prompt if system_prompt is not None else existing["system_prompt"]
                )
                conn.execute(
                    """
                    UPDATE sessions
                    SET system_prompt = ?, updated_at = ?
                    WHERE session_id = ?;
                    """,
                    (effective_prompt, now, session_id),
                )

            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'user', ?, NULL, ?);
                """,
                (session_id, user_content, now),
            )
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'assistant', ?, ?, ?);
                """,
                (session_id, assistant_content, metadata_json, now),
            )
            conn.commit()
