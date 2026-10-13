"""Persistent personal memory system for Suzy.

Stores user preferences, ongoing projects, work context, and explicitly saved
facts in a local SQLite database (data/memory.db).
Safeguards: passwords, access tokens, and sensitive secrets are strictly rejected.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.paths import APP_DIR

log = logging.getLogger("suzy.memory")

# Sensitive patterns that must NEVER be saved in personal conversational memory
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)\b(password|passwd|pwd)\b\s*(\bis\b|[:=])\s*\S+"),
    re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|bearer)\b\s*(\bis\b|[:=])\s*\S+"),
    re.compile(r"gsk_[a-zA-Z0-9_-]{20,}"),          # Groq key
    re.compile(r"AIza[0-9A-Za-z-_]{35}"),            # Google key
    re.compile(r"\b[A-Za-z0-9+/]{40,}\b"),           # Generic base64 / token
    re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),       # Credit card
]


class MemorySecurityError(ValueError):
    """Raised when memory content contains sensitive credentials or secrets."""


@dataclass
class MemoryRecord:
    id: int
    category: str
    key: str
    content: str
    created_at: str
    updated_at: str
    access_count: int = 0
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "key": self.key,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "is_active": self.is_active,
        }


class MemoryStore:
    """Thread-safe SQLite persistent memory store."""

    def __init__(self, db_path: Path | str | None = None, clock: Callable[[], datetime] = datetime.now) -> None:
        if db_path is None:
            data_dir = APP_DIR / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = data_dir / "memory.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL DEFAULT 'fact',
                    key TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_cat ON memories(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active)")
            conn.commit()

    @staticmethod
    def check_sensitive(content: str) -> None:
        """Reject saving secrets, tokens, or passwords."""
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(content):
                raise MemorySecurityError(
                    "Security safeguard: This information looks like a password, API key, "
                    "or sensitive token. I do not store sensitive credentials in memory."
                )

    def remember(self, content: str, category: str = "fact", key: str = "") -> MemoryRecord:
        """Save a new memory or update an existing memory if duplicate key exists."""
        text = (content or "").strip()
        if not text:
            raise ValueError("Memory content cannot be empty.")
        self.check_sensitive(text)

        cat = (category or "fact").strip().lower()
        key_clean = (key or "").strip().lower()
        now_iso = self.clock().isoformat(timespec="seconds")

        with self._get_connection() as conn:
            # If a specific key is provided within category, update it
            if key_clean:
                row = conn.execute(
                    "SELECT id FROM memories WHERE category = ? AND key = ? AND is_active = 1",
                    (cat, key_clean),
                ).fetchone()
                if row:
                    rec_id = row["id"]
                    conn.execute(
                        "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                        (text, now_iso, rec_id),
                    )
                    conn.commit()
                    log.info("Updated memory #%d [%s:%s]", rec_id, cat, key_clean)
                    return MemoryRecord(rec_id, cat, key_clean, text, now_iso, now_iso, 0, True)

            cursor = conn.execute(
                """INSERT INTO memories (category, key, content, created_at, updated_at, access_count, is_active)
                   VALUES (?, ?, ?, ?, ?, 0, 1)""",
                (cat, key_clean, text, now_iso, now_iso),
            )
            rec_id = cursor.lastrowid
            conn.commit()
            log.info("Saved memory #%d [%s]", rec_id, cat)
            return MemoryRecord(rec_id, cat, key_clean, text, now_iso, now_iso, 0, True)

    def recall(self, query: str = "", category: str | None = None, limit: int = 10) -> list[MemoryRecord]:
        """Search stored memories by query string or category."""
        query_text = (query or "").strip().lower()
        cat = category.strip().lower() if category and category != "all" else None

        sql = "SELECT id, category, key, content, created_at, updated_at, access_count, is_active FROM memories WHERE is_active = 1"
        params: list[Any] = []

        if cat:
            sql += " AND category = ?"
            params.append(cat)

        if query_text:
            sql += " AND (LOWER(content) LIKE ? OR LOWER(key) LIKE ?)"
            params.extend([f"%{query_text}%", f"%{query_text}%"])

        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        results: list[MemoryRecord] = []
        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            ids_to_bump = []
            for r in rows:
                rec = MemoryRecord(
                    id=r["id"],
                    category=r["category"],
                    key=r["key"],
                    content=r["content"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    access_count=r["access_count"] + 1,
                    is_active=bool(r["is_active"]),
                )
                results.append(rec)
                ids_to_bump.append(r["id"])

            if ids_to_bump:
                conn.executemany(
                    "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                    [(i,) for i in ids_to_bump],
                )
                conn.commit()

        return results

    def forget(self, memory_id: int) -> bool:
        """Mark a memory as inactive/forgotten."""
        with self._get_connection() as conn:
            cursor = conn.execute("UPDATE memories SET is_active = 0 WHERE id = ?", (memory_id,))
            conn.commit()
            success = cursor.rowcount > 0
            if success:
                log.info("Forgotten memory #%d", memory_id)
            return success

    def delete(self, memory_id: int) -> bool:
        """Permanently delete a memory record."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    def list_all(self, active_only: bool = True) -> list[MemoryRecord]:
        """List memories for user inspection and editing."""
        sql = "SELECT id, category, key, content, created_at, updated_at, access_count, is_active FROM memories"
        if active_only:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY category ASC, updated_at DESC"

        with self._get_connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [
                MemoryRecord(
                    id=r["id"],
                    category=r["category"],
                    key=r["key"],
                    content=r["content"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    access_count=r["access_count"],
                    is_active=bool(r["is_active"]),
                )
                for r in rows
            ]

    def get_context_summary(self, max_items: int = 8) -> str:
        """Format top active memories for system prompt context."""
        mems = self.list_all(active_only=True)[:max_items]
        if not mems:
            return ""
        lines = [f"- [{m.category}] {m.content}" for m in mems]
        return "USER SAVED MEMORIES & PREFERENCES:\n" + "\n".join(lines)


# Global singleton instance
_GLOBAL_STORE: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = MemoryStore()
    return _GLOBAL_STORE
