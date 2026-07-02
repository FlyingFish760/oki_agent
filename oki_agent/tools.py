from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oki_agent.memory import MemoryStore
from oki_agent.persona import PersonaCard
from oki_agent.training import LocalTrainingQueue, TrainingJob


@dataclass(frozen=True)
class ToolResult:
    name: str
    content: str


class LocalTools:
    def __init__(self, root: Path, memory: MemoryStore, persona: PersonaCard) -> None:
        self.root = root.resolve()
        self.memory = memory
        self.persona = persona

    def remember(self, content: str, *, kind: str = "semantic", importance: int = 3) -> ToolResult:
        self.memory.add(kind, content, importance=importance, source="tool")
        return ToolResult("remember", f"Saved memory: {content}")

    def search_files(self, text: str, *, limit: int = 5) -> ToolResult:
        if not text.strip():
            raise ValueError("search text cannot be empty")
        matches: list[str] = []
        for path in self.root.rglob("*"):
            if len(matches) >= limit:
                break
            if not path.is_file() or ".git" in path.parts or ".oki" in path.parts:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if text.lower() in content.lower():
                matches.append(str(path.relative_to(self.root)))
        body = "\n".join(matches) if matches else "No matches found."
        return ToolResult("search_files", body)

    def queue_persona_training(self) -> ToolResult:
        job: TrainingJob = LocalTrainingQueue(self.root).prepare_persona_job(self.persona, self.memory)
        return ToolResult(
            "queue_persona_training",
            f"Prepared local idle-time training manifest: {job.manifest_path}",
        )
