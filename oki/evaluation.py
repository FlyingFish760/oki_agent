"""A/B persona evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from oki.models import ChatMessage, build_model_client


EVAL_PATH = Path("data/eval_prompts.jsonl")


def run_persona_eval(offline: bool = False) -> list[dict[str, str]]:
    prompts = _load_prompts()
    base = build_model_client("base", offline=offline)
    persona = build_model_client("persona", offline=offline)
    rows: list[dict[str, str]] = []
    for prompt in prompts:
        messages = [ChatMessage("user", prompt)]
        rows.append(
            {
                "prompt": prompt,
                "base": _collect(base.stream_chat(messages)),
                "persona": _collect(persona.stream_chat(messages)),
            }
        )
    return rows


def _load_prompts() -> list[str]:
    if not EVAL_PATH.exists():
        return ["我今天状态很乱，帮我安排接下来两小时。"]
    prompts: list[str] = []
    for line in EVAL_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        prompts.append(str(item["prompt"]))
    return prompts


def _collect(chunks) -> str:
    return "".join(chunks).strip().replace("|", "\\|")
