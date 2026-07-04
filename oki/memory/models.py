"""记忆数据模型。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(str, Enum):
    EPISODIC = "episodic"      # 情景：发生过的对话/事件
    SEMANTIC = "semantic"      # 语义：关于用户的事实
    PROCEDURAL = "procedural"  # 程序性：用户习惯的做事方式


@dataclass
class MemoryItem:
    content: str
    mtype: MemoryType
    importance: float = 0.5          # 0~1，写入时由 LLM 打分，用于检索排序与遗忘
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    source: str | None = None        # 来源（哪轮对话/哪个工具）
    # 冲突与更新：同一 subject 的事实用 key 归组，新覆盖旧
    subject_key: str | None = None
    superseded_by: int | None = None  # 被哪条记忆取代（None = 生效中）
    id: int | None = None

    def to_row(self) -> dict:
        return {
            "content": self.content,
            "mtype": self.mtype.value,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "source": self.source,
            "subject_key": self.subject_key,
            "superseded_by": self.superseded_by,
        }
