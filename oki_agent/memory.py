from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class MemoryItem:
    kind: str
    content: str
    importance: int
    created_at: str


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add(self, kind: str, content: str, *, importance: int = 3, source: str = "chat") -> None:
        content = content.strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories(kind, content, importance, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (kind, content, importance, source, now, now),
            )

    def search(self, query: str, *, limit: int = 5) -> list[MemoryItem]:
        tokens = [token.lower() for token in query.split() if len(token) > 1]
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT kind, content, importance, created_at
                FROM memories
                ORDER BY importance DESC, updated_at DESC
                LIMIT 100
                """
            ).fetchall()
        ranked: list[tuple[int, MemoryItem]] = []
        for row in rows:
            content_lower = row["content"].lower()
            score = int(row["importance"]) * 2 + sum(token in content_lower for token in tokens)
            if score > 0:
                ranked.append(
                    (
                        score,
                        MemoryItem(
                            kind=row["kind"],
                            content=row["content"],
                            importance=row["importance"],
                            created_at=row["created_at"],
                        ),
                    )
                )
        return [item for _, item in sorted(ranked, key=lambda pair: pair[0], reverse=True)[:limit]]

    def list_recent(self, *, limit: int = 10) -> list[MemoryItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT kind, content, importance, created_at
                FROM memories
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            MemoryItem(
                kind=row["kind"],
                content=row["content"],
                importance=row["importance"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 3,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_rank
                ON memories(importance DESC, updated_at DESC)
                """
            )
