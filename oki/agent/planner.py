"""任务规划：简单命令直接调工具，复杂任务 plan-execute / ReAct 分解。

Day5/Day6 接实际的 function-calling / ReAct 循环；此处定义接口骨架。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..inference.ollama_client import OllamaClient
from .tools import ToolRegistry


@dataclass
class Step:
    tool: str
    args: dict
    rationale: str = ""


class Planner:
    def __init__(self, client: OllamaClient, tools: ToolRegistry):
        self.client = client
        self.tools = tools

    def plan(self, goal: str) -> list[Step]:
        """把目标分解为若干工具调用步骤。

        TODO(Day6): 用模型的 function-calling 产出 Step 列表；
        简单意图可直接返回单步。
        """
        return []

    def execute(self, steps: list[Step], confirm=lambda s: True) -> list[str]:
        """按序执行；危险步骤经 confirm 回调确认（见 tools 安全 gate）。"""
        results: list[str] = []
        for step in steps:
            results.append(self.tools.call(step.tool, step.args, confirm=confirm))
        return results
