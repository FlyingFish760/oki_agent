"""Command line interface for the Oki demo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from oki.agent import OkiAgent
from oki.memory import DEFAULT_DB_PATH, MemoryStore
from oki.models import BASELINE_MODEL, PERSONA_MODEL, build_model_client
from oki.tools import ToolRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oki", description="Oki local personal assistant demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat")
    chat_parser.add_argument("--model", choices=["base", "persona"], default="persona")
    chat_parser.add_argument("--offline", action="store_true", help="Use deterministic local fallback instead of Ollama")
    chat_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite memory path")

    eval_parser = subparsers.add_parser("eval-persona", help="Run a small A/B persona evaluation")
    eval_parser.add_argument("--offline", action="store_true", help="Use deterministic local fallback instead of Ollama")

    memory_parser = subparsers.add_parser("memory", help="Inspect local memory")
    memory_parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite memory path")
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command", required=True)
    memory_subparsers.add_parser("list", help="List active memories")
    delete_parser = memory_subparsers.add_parser("delete", help="Soft-delete a memory")
    delete_parser.add_argument("id", type=int)

    args = parser.parse_args(argv)
    if args.command == "chat":
        return run_chat(args.model, Path(args.db), args.offline)
    if args.command == "eval-persona":
        return run_eval(args.offline)
    if args.command == "memory":
        return run_memory(args.memory_command, Path(args.db), getattr(args, "id", None))
    parser.error("unknown command")
    return 2


def run_chat(model_kind: str, db_path: Path, offline: bool) -> int:
    if offline:
        os.environ["OKI_OFFLINE_DEMO"] = "1"
    print(f"Oki demo chat. model={model_kind} base={BASELINE_MODEL} persona={PERSONA_MODEL}")
    print("Commands: /search <name>, /read <path>, /remind <text>, /memory, /exit")
    agent = OkiAgent(
        model=build_model_client(model_kind, offline=offline),
        memory=MemoryStore(db_path),
        tools=ToolRegistry(Path.cwd()),
    )
    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not user_text:
            continue
        if user_text in {"/exit", "exit", "quit"}:
            print("bye")
            return 0
        if user_text == "/memory":
            print_memories(agent.memory)
            continue
        print("oki> ", end="", flush=True)
        for chunk in agent.stream_turn(user_text):
            print(chunk, end="", flush=True)
        print()


def run_memory(command: str, db_path: Path, memory_id: int | None) -> int:
    store = MemoryStore(db_path)
    if command == "list":
        print_memories(store)
        return 0
    if command == "delete":
        if memory_id is None:
            raise SystemExit("memory delete requires an id")
        deleted = store.delete(memory_id)
        print("deleted" if deleted else "not found")
        return 0 if deleted else 1
    raise SystemExit(f"unknown memory command: {command}")


def print_memories(store: MemoryStore) -> None:
    memories = store.list()
    if not memories:
        print("No active memories.")
        return
    for item in memories:
        print(f"{item.id}: [{item.type}] importance={item.importance} {item.content}")


def run_eval(offline: bool) -> int:
    from oki.evaluation import run_persona_eval

    rows = run_persona_eval(offline=offline)
    print("| prompt | base | persona |")
    print("| --- | --- | --- |")
    for row in rows:
        print(f"| {row['prompt']} | {row['base']} | {row['persona']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
