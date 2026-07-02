from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "qwen3.6-35b-a3b"


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    endpoint: str | None
    api_key: str | None
    persona_path: Path
    memory_path: Path
    workspace_root: Path
    offline: bool

    @classmethod
    def from_env(
        cls,
        *,
        persona_path: Path | None = None,
        memory_path: Path | None = None,
        workspace_root: Path | None = None,
        offline: bool = False,
    ) -> "RuntimeConfig":
        root = workspace_root or Path.cwd()
        return cls(
            model=os.getenv("OKI_LLM_MODEL", DEFAULT_MODEL),
            endpoint=os.getenv("OKI_LLM_ENDPOINT"),
            api_key=os.getenv("OKI_LLM_API_KEY"),
            persona_path=persona_path or root / "persona" / "oki_persona.toml",
            memory_path=memory_path or root / ".oki" / "memory.sqlite3",
            workspace_root=root,
            offline=offline or os.getenv("OKI_OFFLINE", "").lower() in {"1", "true", "yes"},
        )
