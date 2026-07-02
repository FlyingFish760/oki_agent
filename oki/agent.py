"""Conversation orchestration for Oki."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from oki.memory import Memory, MemoryStore
from oki.models import ChatMessage, ModelClient
from oki.tools import ToolRegistry


DEFAULT_PERSONA_CARD = Path("configs/persona_card.yaml")


class OkiAgent:
    def __init__(
        self,
        model: ModelClient,
        memory: MemoryStore,
        tools: ToolRegistry,
        persona_path: Path | str = DEFAULT_PERSONA_CARD,
    ) -> None:
        self.model = model
        self.memory = memory
        self.tools = tools
        self.persona = Path(persona_path).read_text(encoding="utf-8") if Path(persona_path).exists() else ""
        self.history: list[ChatMessage] = []

    def stream_turn(self, user_text: str) -> Iterable[str]:
        tool_result = self.tools.maybe_run(user_text)
        retrieved = self.memory.search(user_text, limit=4)
        messages = self._build_messages(user_text, retrieved)
        if tool_result:
            messages.append(ChatMessage("system", f"Tool result from {tool_result.name}:\n{tool_result.content}"))
        chunks: list[str] = []
        try:
            for chunk in self.model.stream_chat(messages):
                chunks.append(chunk)
                yield chunk
        except RuntimeError as exc:
            fallback = f"\n[model unavailable] {exc}\n请设置 OKI_OFFLINE_DEMO=1 使用离线演示模式，或启动 Ollama。"
            chunks.append(fallback)
            yield fallback
        assistant_text = "".join(chunks).strip()
        if tool_result and tool_result.requires_confirmation:
            assistant_text += "\n[confirmation required]"
        self.history.extend([ChatMessage("user", user_text), ChatMessage("assistant", assistant_text)])
        self.memory.extract_from_turn(user_text, assistant_text)

    def _build_messages(self, user_text: str, memories: list[Memory]) -> list[ChatMessage]:
        memory_text = "\n".join(f"- [{item.type}] {item.content}" for item in memories) or "- No relevant memory found."
        system = (
            "You are Oki, a local personal assistant. Follow the persona card and keep user data local.\n\n"
            f"Persona card:\n{self.persona}\n\n"
            f"Relevant user memory:\n{memory_text}\n\n"
            "Use tool results when provided. Do not claim to perform irreversible actions without confirmation."
        )
        return [ChatMessage("system", system), *self.history[-8:], ChatMessage("user", user_text)]
