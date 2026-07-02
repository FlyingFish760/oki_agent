from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.error
import urllib.request


Message = dict[str, str]


@dataclass(frozen=True)
class LLMResponse:
    content: str


class LLMBackend:
    def chat(self, messages: list[Message]) -> LLMResponse:
        raise NotImplementedError


class OfflineBackend(LLMBackend):
    def chat(self, messages: list[Message]) -> LLMResponse:
        user_message = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
        return LLMResponse(
            "我先用本地离线 demo 模式回应："
            f"你刚才说「{user_message}」。"
            "如果接上 Qwen3.6-35B-A3B 的本地推理端点，我会用同一套人格、记忆和工具上下文生成正式回答。"
        )


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, endpoint: str, model: str, api_key: str | None = None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key

    def chat(self, messages: list[Message]) -> LLMResponse:
        payload = json.dumps({"model": self.model, "messages": messages, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=payload,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM endpoint request failed: {exc}") from exc
        return LLMResponse(data["choices"][0]["message"]["content"])

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
