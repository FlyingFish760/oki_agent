"""SQLite-backed local memory for Oki."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_DB_PATH = Path.home() / ".oki" / "memory.sqlite3"
MEMORY_TYPES = {"semantic_fact", "episodic_event", "preference", "procedure"}


@dataclass(frozen=True)
class Memory:
    id: int
    type: str
    content: str
    source_turn: str
    created_at: str
    updated_at: str
    importance: int
    is_active: bool


class MemoryStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_turn TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 3,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active)")

    def add(self, memory_type: str, content: str, source_turn: str, importance: int = 3) -> int:
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unknown memory type: {memory_type}")
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories(type, content, source_turn, created_at, updated_at, importance, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (memory_type, content.strip(), source_turn.strip(), now, now, int(importance)),
            )
            return int(cursor.lastrowid)

    def list(self, include_inactive: bool = False) -> list[Memory]:
        query = "SELECT * FROM memories"
        if not include_inactive:
            query += " WHERE is_active = 1"
        query += " ORDER BY updated_at DESC, importance DESC"
        with self._connect() as connection:
            return [self._row_to_memory(row) for row in connection.execute(query)]

    def delete(self, memory_id: int) -> bool:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE memories SET is_active = 0, updated_at = ? WHERE id = ? AND is_active = 1",
                (now, int(memory_id)),
            )
            return cursor.rowcount > 0

    def search(self, query: str, limit: int = 5) -> list[Memory]:
        words = _keywords(query)
        memories = self.list()
        scored: list[tuple[float, Memory]] = []
        for index, memory in enumerate(memories):
            content_words = _keywords(memory.content)
            overlap = len(words & content_words)
            if overlap == 0 and words:
                continue
            recency_bonus = 1.0 / max(1, index + 1)
            score = overlap * 2.0 + memory.importance + recency_bonus
            scored.append((score, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored[:limit]]

    def extract_from_turn(self, user_text: str, assistant_text: str) -> list[int]:
        """Heuristic memory extraction for the demo.

        A production path would call the tuned model to extract facts. This
        keeps the demo offline-friendly and transparent.
        """

        candidates: list[tuple[str, str, int]] = []
        lowered = user_text.lower()
        preference_markers = ["我喜欢", "我偏好", "我希望", "i like", "i prefer"]
        fact_markers = ["我是", "我在", "我的", "i am", "i work", "my "]
        if any(marker in lowered or marker in user_text for marker in preference_markers):
            candidates.append(("preference", user_text, 4))
        elif any(marker in lowered or marker in user_text for marker in fact_markers):
            candidates.append(("semantic_fact", user_text, 3))
        if "以后" in user_text or "下次" in user_text or "always" in lowered:
            candidates.append(("procedure", user_text, 4))
        if not candidates and len(user_text) > 40 and "记住" in user_text:
            candidates.append(("episodic_event", user_text, 3))
        return [self.add(kind, content, user_text + "\n" + assistant_text, importance) for kind, content, importance in candidates]

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=int(row["id"]),
            type=str(row["type"]),
            content=str(row["content"]),
            source_turn=str(row["source_turn"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            importance=int(row["importance"]),
            is_active=bool(row["is_active"]),
        )


def _keywords(text: str) -> set[str]:
    return {item.lower() for item in re.findall(r"[\w\u4e00-\u9fff]+", text) if len(item) > 1}
