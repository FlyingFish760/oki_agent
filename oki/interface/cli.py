"""CLI 交互：流式输出 + 危险操作确认 gate。

thinking 的展示策略（展示/折叠/隐藏）在这里决定，默认折叠。
"""

from __future__ import annotations

import sys

from ..config import load_config
from ..agent.orchestrator import Orchestrator
from ..agent.tools import default_registry
from ..inference.ollama_client import OllamaClient
from ..memory.store import MemoryStore
from ..persona.persona import Persona, PersonaKnobs


def _confirm(name: str, args: dict, preview: str) -> bool:
    print(f"\n[需要确认] 即将执行危险操作: {name}\n  预演: {preview}")
    return input("允许执行? [y/N] ").strip().lower() == "y"


def build_orchestrator() -> Orchestrator:
    cfg = load_config()
    model = cfg.model.get("chat_model", "qwen3-35b-a3b")
    client = OllamaClient(model=model, host=cfg.model.get("host"))

    persona_card = cfg.persona.get("card", {"name": "oki"})
    knobs_cfg = cfg.persona.get("knobs", {})
    persona = Persona(persona_card, PersonaKnobs(**knobs_cfg))

    db_path = cfg.memory.get("db_path", "data/memory.db")
    vector_dir = cfg.memory.get("vector_dir", "data/chroma")
    store = MemoryStore(db_path, vector_dir)

    orch = Orchestrator(client, persona, store, tools=default_registry())
    return orch


def run_cli() -> None:
    orch = build_orchestrator()
    print(f"{orch.persona.name} 已就绪（输入 /exit 退出）。\n")
    while True:
        try:
            user = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            break

        print(f"{orch.persona.name} > ", end="", flush=True)
        for delta in orch.chat(user):
            sys.stdout.write(delta)
            sys.stdout.flush()
        print("\n")

    orch.store.close()
