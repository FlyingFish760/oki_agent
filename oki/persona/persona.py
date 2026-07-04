"""人格装配：persona card + 可调旋钮 -> system prompt。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PersonaKnobs:
    """用户可调的人格旋钮（呼应"用户掌控"）。0.0 ~ 1.0。"""

    formality: float = 0.4      # 0 活泼随意 <-> 1 正式严肃
    verbosity: float = 0.4      # 0 简洁 <-> 1 话多
    warmth: float = 0.7         # 0 冷静克制 <-> 1 热情

    def describe(self) -> str:
        def band(v: float, low: str, mid: str, high: str) -> str:
            return low if v < 0.34 else (mid if v < 0.67 else high)

        return (
            f"语气正式度: {band(self.formality, '活泼随意', '适度', '正式严肃')}; "
            f"话量: {band(self.verbosity, '简洁', '适中', '详尽')}; "
            f"温度: {band(self.warmth, '克制', '友好', '热情')}。"
        )


class Persona:
    """把静态人格卡 + 动态旋钮 + 记忆画像组装为 system prompt。"""

    def __init__(self, card: dict[str, Any], knobs: PersonaKnobs | None = None):
        self.card = card
        self.knobs = knobs or PersonaKnobs()

    @property
    def name(self) -> str:
        return self.card.get("name", "oki")

    def system_prompt(self, self_portrait: str | None = None, memory_context: str | None = None) -> str:
        """生成本轮 system prompt。

        self_portrait: memory 巩固出的用户/自我画像（人格层 3）。
        memory_context: 本轮检索到的相关记忆（由 orchestrator 注入）。
        """
        parts: list[str] = []
        parts.append(self.card.get("system_prompt", f"你是 {self.name}，一个本地个人助手。"))
        parts.append("【当前人格旋钮】" + self.knobs.describe())
        if self_portrait:
            parts.append("【你对用户的了解（自我画像）】\n" + self_portrait)
        if memory_context:
            parts.append("【与本轮相关的记忆】\n" + memory_context)
        return "\n\n".join(parts)

    # TODO(Day6): detect_drift(history) —— 多轮中检测人格漂移，必要时重注入人格卡。
