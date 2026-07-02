"""Local tools with explicit safety boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolResult:
    name: str
    content: str
    requires_confirmation: bool = False


class ToolRegistry:
    def __init__(self, workspace: Path | str | None = None) -> None:
        self.workspace = Path(workspace or Path.cwd()).resolve()

    def maybe_run(self, user_text: str) -> ToolResult | None:
        lowered = user_text.lower()
        if user_text.startswith("/search "):
            return self.search_files(user_text.removeprefix("/search ").strip())
        if user_text.startswith("/read "):
            return self.read_file(user_text.removeprefix("/read ").strip())
        if user_text.startswith("/remind "):
            return self.reminder_draft(user_text.removeprefix("/remind ").strip())
        if "删除" in user_text or "delete " in lowered or "发送邮件" in user_text:
            return ToolResult(
                "safety_gate",
                "这个动作可能产生真实副作用。Demo 版本只生成执行计划，不直接删除文件或发送外部消息。",
                requires_confirmation=True,
            )
        return None

    def search_files(self, query: str, limit: int = 8) -> ToolResult:
        if not query:
            return ToolResult("search_files", "请在 /search 后面提供关键词。")
        matches: list[str] = []
        for path in self.workspace.rglob("*"):
            if len(matches) >= limit:
                break
            if path.is_file() and query.lower() in path.name.lower():
                matches.append(str(path.relative_to(self.workspace)))
        if not matches:
            return ToolResult("search_files", f"没有在 {self.workspace} 下找到文件名包含 `{query}` 的文件。")
        return ToolResult("search_files", "找到这些文件：\n" + "\n".join(f"- {item}" for item in matches))

    def read_file(self, relative_path: str, max_chars: int = 4000) -> ToolResult:
        if not relative_path:
            return ToolResult("read_file", "请在 /read 后面提供相对路径。")
        target = (self.workspace / relative_path).resolve()
        if not _is_relative_to(target, self.workspace):
            return ToolResult("read_file", "拒绝读取工作区外的路径。", requires_confirmation=True)
        if not target.exists() or not target.is_file():
            return ToolResult("read_file", f"文件不存在：{relative_path}")
        content = target.read_text(encoding="utf-8", errors="replace")[:max_chars]
        return ToolResult("read_file", f"{relative_path} 的前 {max_chars} 字符：\n{content}")

    def reminder_draft(self, text: str) -> ToolResult:
        if not text:
            return ToolResult("reminder_draft", "请在 /remind 后面描述提醒内容。")
        return ToolResult(
            "reminder_draft",
            f"提醒草稿：{text}\nDemo 版本不会写入系统日历；需要用户确认后再接真实日历工具。",
            requires_confirmation=True,
        )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
