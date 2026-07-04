"""统一配置加载：读取 configs/*.yaml 并提供简单的点访问。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 项目根目录（.../oki_agent）
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Config:
    """运行时全局配置。各子系统从这里取自己的段。"""

    model: dict[str, Any] = field(default_factory=dict)
    persona: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        return cls(
            model=_load_yaml("model.yaml"),
            persona=_load_yaml("persona.yaml"),
            memory=_load_yaml("memory.yaml"),
        )


def load_config() -> Config:
    return Config.load()
