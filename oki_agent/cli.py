from __future__ import annotations

import argparse
from pathlib import Path

from oki_agent.assistant import OkiAssistant
from oki_agent.config import RuntimeConfig
from oki_agent.llm import OfflineBackend, OpenAICompatibleBackend
from oki_agent.memory import MemoryStore
from oki_agent.persona import PersonaCard
from oki_agent.tools import LocalTools


def build_assistant(config: RuntimeConfig) -> OkiAssistant:
    persona = PersonaCard.load(config.persona_path)
    memory = MemoryStore(config.memory_path)
    tools = LocalTools(config.workspace_root, memory, persona)
    llm = (
        OfflineBackend()
        if config.offline or not config.endpoint
        else OpenAICompatibleBackend(config.endpoint, config.model, config.api_key)
    )
    return OkiAssistant(persona=persona, memory=memory, tools=tools, llm=llm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Oki assistant demo.")
    parser.add_argument("--once", help="Send one message and exit.")
    parser.add_argument("--offline", action="store_true", help="Use deterministic offline demo responses.")
    parser.add_argument("--persona", type=Path, help="Path to a persona TOML card.")
    parser.add_argument("--memory", type=Path, help="Path to the SQLite memory database.")
    args = parser.parse_args()

    config = RuntimeConfig.from_env(
        persona_path=args.persona,
        memory_path=args.memory,
        offline=args.offline,
    )
    assistant = build_assistant(config)

    if args.once:
        print(assistant.respond(args.once))
        return

    print("Oki local assistant. Commands: /remember, /search, /memories, /train-persona, /exit")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user_input in {"/exit", "exit", "quit"}:
            return
        if not user_input:
            continue
        print(f"oki> {assistant.respond(user_input)}")
