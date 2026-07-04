"""巩固与遗忘（"睡眠期整理"）。

  - 遗忘: 对低重要性 + 久未访问的记忆做时间衰减淘汰，避免检索被陈年琐事污染。
  - 巩固: 定期把零散记忆摘要成更高层的用户画像（self-portrait），
          供 persona.system_prompt 使用，实现人格随交互缓慢演化。

Day5/Day6 再接实际逻辑；此处为骨架。
"""

from __future__ import annotations

import time

from ..inference.ollama_client import OllamaClient
from .store import MemoryStore


class Consolidator:
    def __init__(self, store: MemoryStore, client: OllamaClient | None = None):
        self.store = store
        self.client = client

    def forget(self, importance_floor: float = 0.2, idle_days: float = 90.0) -> int:
        """淘汰低价值 + 长期未访问的记忆，返回删除条数。"""
        cutoff = time.time() - idle_days * 86400.0
        rows = [
            r for r in self.store.list()
            if r["importance"] < importance_floor and r["last_accessed"] < cutoff
        ]
        for r in rows:
            self.store.delete(r["id"])
        return len(rows)

    def build_self_portrait(self) -> str:
        """把语义/程序性记忆摘要成一段用户画像。"""
        if self.client is None:
            return ""
        facts = [r["content"] for r in self.store.list()]
        if not facts:
            return ""
        prompt = "把以下关于用户的信息摘要成一段简洁的用户画像：\n" + "\n".join(facts)
        return self.client.complete([{"role": "user", "content": prompt}])
