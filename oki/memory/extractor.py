"""写入判断：不是所有对话都该记。

用 LLM 从一轮/多轮对话里抽取值得长期保存的实体、事实、偏好，
给每条打 importance 分，并归类为 episodic/semantic/procedural。
Day5 接上真实 LLM 调用；这里先定义接口与 prompt 骨架。
"""

from __future__ import annotations

import json

from ..inference.ollama_client import OllamaClient
from .models import MemoryItem, MemoryType

EXTRACT_PROMPT = """你是记忆抽取器。阅读下面这轮对话，抽取值得长期保存的信息。
只保留：关于用户的稳定事实、明确偏好、习惯做法、重要事件。忽略寒暄与临时信息。
以 JSON 数组输出，每个元素形如：
{{"content": "...", "mtype": "semantic|episodic|procedural", "importance": 0.0-1.0, "subject_key": "可选归组键"}}
若无可记，输出 []。

对话：
{dialogue}
"""


class MemoryExtractor:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client

    def extract(self, dialogue: str) -> list[MemoryItem]:
        if self.client is None:
            return []  # 未接模型时空跑
        raw = self.client.complete(
            [{"role": "user", "content": EXTRACT_PROMPT.format(dialogue=dialogue)}]
        )
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> list[MemoryItem]:
        try:
            data = json.loads(raw[raw.find("[") : raw.rfind("]") + 1])
        except (ValueError, json.JSONDecodeError):
            return []
        items: list[MemoryItem] = []
        for d in data:
            try:
                items.append(
                    MemoryItem(
                        content=d["content"],
                        mtype=MemoryType(d.get("mtype", "episodic")),
                        importance=float(d.get("importance", 0.5)),
                        subject_key=d.get("subject_key"),
                        source="extractor",
                    )
                )
            except (KeyError, ValueError):
                continue
        return items
