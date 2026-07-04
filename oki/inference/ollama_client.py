"""Ollama 推理客户端封装。

demo 用 Qwen3.6-35B-A3B 的 GGUF Q4_K_M（约 20GB，可跑在 32GB 显存上）。
支持:
  - 流式 chat（本地首 token 有延迟，流式对体感很重要）
  - base vs base+persona-adapter 的模型切换（A/B 演示用）
  - thinking 内容的分离（展示 / 折叠 / 隐藏由交互层决定）

依赖: `pip install ollama`
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

try:
    import ollama
except ImportError:  # 允许在未装 ollama 的环境下 import 本模块
    ollama = None


class OllamaClient:
    def __init__(self, model: str, host: str | None = None, options: dict[str, Any] | None = None):
        if ollama is None:
            raise ImportError("未安装 ollama，请先 `pip install ollama` 并启动 `ollama serve`。")
        self.model = model
        self.options = options or {}
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = True,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """流式返回文本增量。

        messages: [{"role": "system"|"user"|"assistant", "content": ...}, ...]
        """
        opts = {**self.options, **(options or {})}
        response = self._client.chat(
            model=self.model,
            messages=messages,
            stream=stream,
            options=opts,
        )
        if stream:
            for chunk in response:
                yield chunk["message"]["content"]
        else:
            yield response["message"]["content"]

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """非流式，收集为完整字符串。"""
        return "".join(self.chat(messages, stream=False, **kwargs))
