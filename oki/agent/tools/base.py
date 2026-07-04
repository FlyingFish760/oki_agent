"""工具基类、注册表与安全 gate。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolError(Exception):
    pass


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]
    parameters: dict[str, Any]        # JSON-schema 风格，供 function-calling
    dangerous: bool = False           # 危险操作（写/删/发送等）需确认
    dry_run: Callable[..., str] | None = None  # 可选：预演，返回"将要做什么"

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# confirm 回调签名: (工具名, 参数, 预演文本) -> 是否放行
ConfirmFn = Callable[[str, dict, str], bool]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"未知工具: {name}")
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def call(self, name: str, args: dict, confirm: ConfirmFn = lambda *_: True) -> str:
        tool = self.get(name)
        if tool.dangerous:
            preview = tool.dry_run(**args) if tool.dry_run else f"{name}({args})"
            if not confirm(name, args, preview):
                return f"[已取消] 用户未确认危险操作: {name}"
        return tool.func(**args)
