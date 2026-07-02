"""Model adapters used by the Oki demo.

The demo can talk to a local Ollama-compatible server when available. It also
ships with a deterministic fallback so the memory and tool flow can be shown
without a 35B model on the interview machine.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass


BASELINE_MODEL = "qwen3.6-35b-a3b"
PERSONA_MODEL = "qwen3.6-35b-a3b-oki-lora"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ModelClient:
    """Small interface for streaming chat completions."""

    def stream_chat(self, messages: list[ChatMessage]) -> Iterable[str]:
        raise NotImplementedError


class OllamaClient(ModelClient):
    """Ollama chat API client.

    Set OKI_OLLAMA_URL to override the default local endpoint.
    """

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OKI_OLLAMA_URL") or "http://localhost:11434").rstrip("/")

    def stream_chat(self, messages: list[ChatMessage]) -> Iterable[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "stream": True,
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    event = json.loads(raw_line.decode("utf-8"))
                    content = event.get("message", {}).get("content")
                    if content:
                        yield content
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama model call failed: {exc}") from exc


class DemoClient(ModelClient):
    """Offline fallback used for smoke tests and model-less demos."""

    def __init__(self, model_kind: str) -> None:
        self.model_kind = model_kind

    def stream_chat(self, messages: list[ChatMessage]) -> Iterable[str]:
        latest = next((item.content for item in reversed(messages) if item.role == "user"), "")
        memory_hint = next((item.content for item in messages if item.role == "system" and "Relevant user memory" in item.content), "")
        if self.model_kind == "persona":
            response = self._persona_response(latest, memory_hint)
        else:
            response = self._base_response(latest)
        for token in response.split(" "):
            yield token + " "

    def _base_response(self, latest: str) -> str:
        return f"I can help with that. You asked: {latest.strip()}"

    def _persona_response(self, latest: str, memory_hint: str) -> str:
        memory_line = "我会先照顾你的偏好和上下文。"
        if memory_hint.strip() and "No relevant memory found" not in memory_hint:
            memory_line = "我查到一条与你相关的本地记忆，会把它纳入判断。"
        cleaned = latest.strip().rstrip("。.!！？?")
        return (
            f"好，我来稳稳接住这件事。{memory_line} "
            f"你的请求是：{cleaned}。我会先给出可执行的小步骤，再提醒需要你确认的风险点。"
        )


def build_model_client(kind: str, offline: bool = False) -> ModelClient:
    if kind not in {"base", "persona"}:
        raise ValueError("model kind must be 'base' or 'persona'")
    if offline or os.getenv("OKI_OFFLINE_DEMO") == "1":
        return DemoClient(kind)
    model_name = BASELINE_MODEL if kind == "base" else PERSONA_MODEL
    return OllamaClient(model_name)



