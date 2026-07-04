"""混合存储：SQLite（结构化事实/时间戳/来源）+ 向量库（语义检索）。

向量库默认用 Chroma（本地、易上手）。这里只搭接口骨架，
Day5 再把 embedding 与向量检索接上。用户掌控:
所有记忆都可 list / edit / delete（见对应方法）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import MemoryItem, MemoryType

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content       TEXT NOT NULL,
    mtype         TEXT NOT NULL,
    importance    REAL NOT NULL DEFAULT 0.5,
    created_at    REAL NOT NULL,
    last_accessed REAL NOT NULL,
    source        TEXT,
    subject_key   TEXT,
    superseded_by INTEGER
);
CREATE INDEX IF NOT EXISTS idx_mem_subject ON memories(subject_key);
CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(mtype);
"""


class MemoryStore:
    def __init__(self, db_path: str | Path, vector_dir: str | Path | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.vector_dir = Path(vector_dir) if vector_dir else None
        self._vector = None  # TODO(Day5): 惰性初始化 Chroma collection

    # ---- 写入 ---------------------------------------------------------
    def add(self, item: MemoryItem) -> int:
        row = item.to_row()
        cur = self.conn.execute(
            """INSERT INTO memories
               (content, mtype, importance, created_at, last_accessed, source, subject_key, superseded_by)
               VALUES (:content, :mtype, :importance, :created_at, :last_accessed, :source, :subject_key, :superseded_by)""",
            row,
        )
        self.conn.commit()
        item.id = cur.lastrowid
        # TODO(Day5): 情景记忆同时写入向量库（embedding + metadata=id）
        return item.id

    def supersede(self, old_id: int, new_id: int) -> None:
        """冲突与更新：新记忆覆盖旧记忆（保留历史，标记 superseded_by）。"""
        self.conn.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (new_id, old_id))
        self.conn.commit()

    # ---- 读取 / 用户掌控 ---------------------------------------------
    def list(self, mtype: MemoryType | None = None, include_superseded: bool = False) -> list[sqlite3.Row]:
        q = "SELECT * FROM memories WHERE 1=1"
        args: list = []
        if mtype:
            q += " AND mtype = ?"
            args.append(mtype.value)
        if not include_superseded:
            q += " AND superseded_by IS NULL"
        q += " ORDER BY created_at DESC"
        return self.conn.execute(q, args).fetchall()

    def edit(self, mem_id: int, content: str) -> None:
        self.conn.execute("UPDATE memories SET content = ? WHERE id = ?", (content, mem_id))
        self.conn.commit()

    def delete(self, mem_id: int) -> None:
        self.conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        self.conn.commit()
        # TODO(Day5): 同步从向量库删除

    def close(self) -> None:
        self.conn.close()
