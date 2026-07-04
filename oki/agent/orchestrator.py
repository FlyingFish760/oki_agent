"""对话主循环：把人格、记忆、工具串起来。

一轮的流程:
  1. 检索相关记忆 -> 注入 system prompt（人格 + 自我画像 + 记忆）
  2. 流式生成回复（thinking 由交互层决定展示/折叠/隐藏）
  3. 若模型请求工具调用 -> 经安全 gate -> 执行 -> 回灌
  4. 轮末让 extractor 抽取值得记的东西写入记忆

Day6 把这些接满；此处为可运行的最小骨架。
"""

from __future__ import annotations

from collections.abc import Iterator

from ..inference.ollama_client import OllamaClient
from ..memory.extractor import MemoryExtractor
from ..memory.retriever import MemoryRetriever
from ..memory.store import MemoryStore
from ..persona.persona import Persona
from .planner import Planner
from .tools import ToolRegistry


class Orchestrator:
    def __init__(
        self,
        client: OllamaClient,
        persona: Persona,
        store: MemoryStore,
        tools: ToolRegistry | None = None,
    ):
        self.client = client
        self.persona = persona
        self.store = store
        self.retriever = MemoryRetriever(store)
        self.extractor = MemoryExtractor(client)
        self.tools = tools or ToolRegistry()
        self.planner = Planner(client, self.tools)
        self.history: list[dict[str, str]] = []

    def _build_messages(self, user_input: str) -> list[dict[str, str]]:
        mem_ctx = self.retriever.as_context(user_input)
        system = self.persona.system_prompt(memory_context=mem_ctx or None)
        return [{"role": "system", "content": system}, *self.history, {"role": "user", "content": user_input}]

    def chat(self, user_input: str) -> Iterator[str]:
        """流式返回本轮回复增量。"""
        messages = self._build_messages(user_input)
        reply_parts: list[str] = []
        for delta in self.client.chat(messages, stream=True):
            reply_parts.append(delta)
            yield delta
        reply = "".join(reply_parts)

        # 更新工作记忆（context window）
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})

        # 轮末：抽取并写入长期记忆
        dialogue = f"用户: {user_input}\n{self.persona.name}: {reply}"
        for item in self.extractor.extract(dialogue):
            self.store.add(item)
