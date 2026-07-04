"""文件工具：读 / 搜索 / 写。写为危险操作，需确认。"""

from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolRegistry


def _read_file(path: str, max_chars: int = 8000) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        return f"[错误] 文件不存在: {path}"
    return p.read_text(encoding="utf-8", errors="replace")[:max_chars]


def _search_files(directory: str, keyword: str, max_hits: int = 20) -> str:
    root = Path(directory).expanduser()
    if not root.is_dir():
        return f"[错误] 目录不存在: {directory}"
    hits: list[str] = []
    for p in root.rglob("*"):
        if p.is_file() and keyword.lower() in p.name.lower():
            hits.append(str(p))
            if len(hits) >= max_hits:
                break
    return "\n".join(hits) or "无匹配。"


def _write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {p} ({len(content)} 字符)。"


def _write_dry_run(path: str, content: str) -> str:
    return f"将向 {path} 写入 {len(content)} 字符（覆盖已有内容）。"


def register_file_tools(reg: ToolRegistry) -> None:
    reg.register(Tool(
        name="read_file",
        description="读取本地文本文件内容。",
        func=_read_file,
        parameters={"type": "object", "properties": {
            "path": {"type": "string"}, "max_chars": {"type": "integer"}}, "required": ["path"]},
    ))
    reg.register(Tool(
        name="search_files",
        description="在目录下按文件名关键字搜索。",
        func=_search_files,
        parameters={"type": "object", "properties": {
            "directory": {"type": "string"}, "keyword": {"type": "string"}}, "required": ["directory", "keyword"]},
    ))
    reg.register(Tool(
        name="write_file",
        description="写入/覆盖本地文件（危险操作，需确认）。",
        func=_write_file,
        dangerous=True,
        dry_run=_write_dry_run,
        parameters={"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    ))
