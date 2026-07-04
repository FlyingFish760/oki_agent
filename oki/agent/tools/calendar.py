"""日历/提醒工具（骨架）。

Day5 先做一个最小的本地提醒：写入本地 SQLite/JSON，
到点由交互层轮询提示。此处仅登记工具，实际存储 Day5 补。
"""

from __future__ import annotations

from .base import Tool, ToolRegistry


def _add_reminder(when: str, text: str) -> str:
    # TODO(Day5): 落地到本地存储（reminders 表）
    return f"[提醒已登记] {when} -> {text}"


def register_calendar_tools(reg: ToolRegistry) -> None:
    reg.register(Tool(
        name="add_reminder",
        description="添加一条本地提醒。",
        func=_add_reminder,
        parameters={"type": "object", "properties": {
            "when": {"type": "string", "description": "时间，如 2026-07-04 09:00"},
            "text": {"type": "string"}}, "required": ["when", "text"]},
    ))
