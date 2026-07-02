from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter

from oki_agent.cli import build_assistant
from oki_agent.config import RuntimeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run persona baseline prompts against Oki.")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("evals/persona_baseline_prompts.jsonl"),
        help="JSONL prompt set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path. Defaults to .oki/evals/persona_baseline_<timestamp>.jsonl.",
    )
    parser.add_argument("--offline", action="store_true", help="Use deterministic offline demo responses.")
    parser.add_argument("--persona", type=Path, help="Path to a persona TOML card.")
    parser.add_argument(
        "--memory",
        type=Path,
        help="Path to the SQLite memory database. Defaults to an isolated eval memory file.",
    )
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(".oki") / "evals" / f"persona_baseline_{stamp}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    config = RuntimeConfig.from_env(
        persona_path=args.persona,
        memory_path=args.memory or output.parent / f"persona_baseline_memory_{stamp}.sqlite3",
        offline=args.offline,
    )
    assistant = build_assistant(config)
    prompts = _load_prompts(args.prompts)

    with output.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            start = perf_counter()
            response = assistant.respond(prompt["prompt"])
            elapsed_ms = round((perf_counter() - start) * 1000)
            row = {
                "run_id": stamp,
                "model": config.model,
                "endpoint": "offline" if config.offline or not config.endpoint else config.endpoint,
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "prompt": prompt["prompt"],
                "response": response,
                "latency_ms": elapsed_ms,
                "checks": prompt["checks"],
                "manual_scores": {
                    "persona_consistency": None,
                    "safety": None,
                    "local_privacy_positioning": None,
                    "instruction_following": None,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{prompt['id']}: {elapsed_ms} ms")

    print(f"Wrote persona baseline results to {output}")


def _load_prompts(path: Path) -> list[dict[str, object]]:
    prompts: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            _validate_prompt(item, line_number)
            prompts.append(item)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def _validate_prompt(item: dict[str, object], line_number: int) -> None:
    required = {"id", "category", "prompt", "checks"}
    missing = required.difference(item)
    if missing:
        raise ValueError(f"prompt line {line_number} missing fields: {sorted(missing)}")
    if not isinstance(item["checks"], list) or not item["checks"]:
        raise ValueError(f"prompt line {line_number} must include non-empty checks")


if __name__ == "__main__":
    main()
