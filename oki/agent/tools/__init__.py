"""工具层：本地能力 + 安全 gate。

安全是重点（在用户电脑上执行真实命令有风险）:
  - 每个工具声明 dangerous=True/False
  - 危险操作执行前必须过 confirm 回调（权限确认）
  - 尽量提供 dry-run 与可撤销
这直接呼应公司"用户掌控数据"的卖点。
"""

from .base import Tool, ToolRegistry, ToolError
from .files import register_file_tools
from .calendar import register_calendar_tools


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_file_tools(reg)
    register_calendar_tools(reg)
    return reg


__all__ = ["Tool", "ToolRegistry", "ToolError", "default_registry"]
