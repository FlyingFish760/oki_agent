"""检索：不能只靠相似度。综合 相关性 + 时间新近度 + 重要性 打分。

score = w_rel * relevance + w_rec * recency + w_imp * importance

Day5 把 relevance 接到向量库余弦相似度；这里先给可跑的骨架，
relevance 缺省用简单的关键词重叠占位。
"""

from __future__ import annotations

import math
import time

from .store import MemoryStore


class MemoryRetriever:
    def __init__(
        self,
        store: MemoryStore,
        w_rel: float = 0.6,
        w_rec: float = 0.25,
        w_imp: float = 0.15,
        half_life_days: float = 30.0,
    ):
        self.store = store
        self.w_rel, self.w_rec, self.w_imp = w_rel, w_rec, w_imp
        self.half_life = half_life_days * 86400.0

    def _recency(self, created_at: float, now: float) -> float:
        # 指数时间衰减，half_life 天后降到 0.5
        return math.exp(-math.log(2) * (now - created_at) / self.half_life)

    def _relevance(self, query: str, content: str) -> float:
        # TODO(Day5): 换成向量余弦相似度
        q = set(query.lower().split())
        c = set(content.lower().split())
        return len(q & c) / len(q) if q else 0.0

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        now = time.time()
        scored: list[tuple[str, float]] = []
        for row in self.store.list():
            rel = self._relevance(query, row["content"])
            rec = self._recency(row["created_at"], now)
            imp = row["importance"]
            score = self.w_rel * rel + self.w_rec * rec + self.w_imp * imp
            scored.append((row["content"], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def as_context(self, query: str, top_k: int = 5) -> str:
        hits = self.retrieve(query, top_k)
        return "\n".join(f"- {c}" for c, _ in hits)
