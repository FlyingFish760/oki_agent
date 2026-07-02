from __future__ import annotations

from oki_agent.llm import LLMBackend, Message
from oki_agent.memory import MemoryStore
from oki_agent.persona import PersonaCard
from oki_agent.tools import LocalTools


class OkiAssistant:
    def __init__(
        self,
        *,
        persona: PersonaCard,
        memory: MemoryStore,
        tools: LocalTools,
        llm: LLMBackend,
    ) -> None:
        self.persona = persona
        self.memory = memory
        self.tools = tools
        self.llm = llm

    def respond(self, user_input: str) -> str:
        tool_response = self._maybe_handle_tool(user_input)
        if tool_response is not None:
            return tool_response

        memories = self.memory.search(user_input)
        messages: list[Message] = [
            {"role": "system", "content": self._system_prompt(memories)},
            {"role": "user", "content": user_input},
        ]
        response = self.llm.chat(messages).content
        self._capture_lightweight_memory(user_input)
        return response

    def _system_prompt(self, memories: object) -> str:
        memory_lines = []
        for item in memories:
            memory_lines.append(f"- [{item.kind}, importance={item.importance}] {item.content}")
        memory_context = "\n".join(memory_lines) if memory_lines else "- No relevant long-term memory yet."
        return (
            self.persona.system_prompt()
            + "\n\nLong-term user memory retrieved for this turn:\n"
            + memory_context
            + "\n\nKeep private data local. Ask for confirmation before risky local actions."
        )

    def _maybe_handle_tool(self, user_input: str) -> str | None:
        stripped = user_input.strip()
        if stripped.startswith("/remember "):
            return self.tools.remember(stripped.removeprefix("/remember ")).content
        if stripped.startswith("/search "):
            return self.tools.search_files(stripped.removeprefix("/search ")).content
        if stripped == "/train-persona":
            return self.tools.queue_persona_training().content
        if stripped == "/memories":
            memories = self.memory.list_recent()
            if not memories:
                return "No memories saved yet."
            return "\n".join(f"- [{item.kind}] {item.content}" for item in memories)
        return None

    def _capture_lightweight_memory(self, user_input: str) -> None:
        lowered = user_input.lower()
        cues = ("我喜欢", "我不喜欢", "记住", "偏好", "prefer", "remember")
        if any(cue in lowered for cue in cues):
            self.memory.add("semantic", user_input, importance=4, source="chat")
