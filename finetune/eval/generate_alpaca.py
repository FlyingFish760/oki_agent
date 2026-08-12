"""Generate Alpaca-format responses for instruction JSONL using Ollama."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from finetune.data.capability_card import (
        CapabilityCard,
        CapabilityRequirement,
        parse_capability_card,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from finetune.data.capability_card import (
        CapabilityCard,
        CapabilityRequirement,
        parse_capability_card,
    )


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OUTPUT = Path("finetune/eval/model_outputs.json")


def load_instruction_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: each JSONL line must be an object"
                )
            instruction = record.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(
                    f"{path}:{line_number}: 'instruction' must be a "
                    "non-empty string"
                )
            requirement_id = record.get("requirement_id")
            if not isinstance(requirement_id, str) or not requirement_id.strip():
                raise ValueError(
                    f"{path}:{line_number}: 'requirement_id' must be a "
                    "non-empty string"
                )
            record["instruction"] = instruction.strip()
            record["requirement_id"] = requirement_id.strip()
            records.append(record)

    if not records:
        raise ValueError(f"Instruction dataset is empty: {path}")
    return records


def get_requirement(
    card: CapabilityCard,
    requirement_id: str,
    *,
    source_id: str | None,
) -> CapabilityRequirement:
    normalized_id = requirement_id.strip().lower()
    for requirement in card.requirements:
        if requirement.requirement_id.lower() == normalized_id:
            return requirement
    available = ", ".join(
        requirement.requirement_id for requirement in card.requirements
    )
    source_label = source_id or "unknown source instruction"
    raise ValueError(
        f"{source_label}: unknown requirement_id {requirement_id!r}; "
        f"available IDs: {available}"
    )


def call_ollama(
    *,
    instruction: str,
    model: str,
    host: str,
    temperature: float,
    top_p: float,
    thinking: bool,
    timeout: float,
) -> str:
    """Send the instruction unchanged as the Ollama generation prompt."""
    payload = {
        "model": model,
        "prompt": instruction,
        "stream": False,
        "think": thinking,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP {exc.code} {exc.reason}: {error_body}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON") from exc

    output = data.get("response")
    if not isinstance(output, str) or not output.strip():
        raise RuntimeError("Ollama response is missing or empty")
    return output.strip()


def generate_with_retries(
    *,
    instruction: str,
    args: argparse.Namespace,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            return call_ollama(
                instruction=instruction,
                model=args.model,
                host=args.host,
                temperature=args.temperature,
                top_p=args.top_p,
                thinking=args.thinking,
                timeout=args.timeout,
            )
        except (
            RuntimeError,
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep)

    raise RuntimeError(
        f"Generation failed after {args.max_retries} attempts: {last_error}"
    ) from last_error


def build_alpaca_record(
    source: dict[str, Any],
    *,
    record_id: int,
    capability_card: Path,
    requirement: CapabilityRequirement,
    output: str,
    model: str,
) -> dict[str, Any]:
    return {
        "id": f"alpaca_eval_{record_id:06d}",
        "source_instruction_id": source.get("id"),
        "capability_card": capability_card.name,
        "requirement_id": requirement.requirement_id,
        "requirement_title": requirement.title,
        "requirement_description": requirement.description,
        "instruction_teacher_model": source.get("teacher_model"),
        "generator": model,
        "split": source.get("split"),
        "instruction": source["instruction"],
        "output": output,
    }


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def partial_output_path(path: Path) -> Path:
    """Return the live JSONL path used while building the final JSON file."""
    return path.with_name(f"{path.name}.partial.jsonl")


def append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def run(args: argparse.Namespace) -> None:
    if not args.instructions.exists():
        raise FileNotFoundError(
            f"Instruction JSONL not found: {args.instructions}"
        )
    if not args.capability_card.exists():
        raise FileNotFoundError(
            f"Capability card not found: {args.capability_card}"
        )
    records = load_instruction_records(args.instructions)
    card = parse_capability_card(args.capability_card)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    if args.dry_run:
        source = records[0]
        requirement = get_requirement(
            card,
            source["requirement_id"],
            source_id=source.get("id"),
        )
        print(f"requirement_id: {requirement.requirement_id}")
        print(f"direct_prompt: {source['instruction']}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial_path = partial_output_path(args.out)
    generated: list[dict[str, Any]] = []
    print(f"Live JSONL output: {partial_path}")
    with partial_path.open("w", encoding="utf-8") as partial_file:
        for index, source in enumerate(records, start=1):
            requirement = get_requirement(
                card,
                source["requirement_id"],
                source_id=source.get("id"),
            )
            output = generate_with_retries(
                instruction=source["instruction"],
                args=args,
            )
            generated_record = build_alpaca_record(
                source,
                record_id=index,
                capability_card=args.capability_card,
                requirement=requirement,
                output=output,
                model=args.model,
            )
            generated.append(generated_record)
            append_jsonl(partial_file, generated_record)
            print(f"[{index}/{len(records)}] generated", end="\r")

    write_json(args.out, generated)
    partial_path.unlink()
    print(
        f"\nGeneration complete: wrote {len(generated)} records to {args.out}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Ollama responses from instruction JSONL and write an "
            "Alpaca-format JSON array."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--instructions",
        type=Path,
        required=True,
        help="Input instruction JSONL path.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name.",
    )
    parser.add_argument(
        "--capability-card",
        type=Path,
        required=True,
        help=(
            "Capability card used to resolve requirement title and "
            "description from each input requirement_id."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Alpaca-format JSON output path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of input records to process.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode for supported Ollama models.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Ollama request timeout in seconds.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=0.5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first direct prompt without calling Ollama.",
    )
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_retries <= 0:
        parser.error("--max-retries must be positive")
    if args.retry_sleep < 0:
        parser.error("--retry-sleep must be non-negative")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
