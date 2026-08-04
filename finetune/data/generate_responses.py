"""Generate conversational SFT responses for instruction JSONL records."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    from capability_card import (
        CapabilityCard,
        CapabilityRequirement,
        parse_capability_card,
    )
    from generate_instructions import DEFAULT_IDENTITY_DESCRIPTION
    from generation_common import (
        TeacherPrompt,
        add_teacher_arguments,
        append_jsonl,
        call_teacher,
        configure_teacher_args,
        prompt_template_name,
    )
except ModuleNotFoundError:
    from .capability_card import (
        CapabilityCard,
        CapabilityRequirement,
        parse_capability_card,
    )
    from .generate_instructions import DEFAULT_IDENTITY_DESCRIPTION
    from .generation_common import (
        TeacherPrompt,
        add_teacher_arguments,
        append_jsonl,
        call_teacher,
        configure_teacher_args,
        prompt_template_name,
    )


DEFAULT_PROMPT_TEMPLATE = Path(
    "finetune/data/templates/response_genereation_template.txt"
)
DEFAULT_OUTPUT = Path(
    "finetune/data/datasets/capability_sft.jsonl"
)
DEFAULT_REJECTED_OUTPUT = Path(
    "finetune/data/datasets/capability_sft_rejected.jsonl"
)

SYSTEM_MARKER = "[SYSTEM PROMPT]"
USER_MARKER = "[USER PROMPT]"
REQUIRED_FIELDS = (
    "instruction",
    "requirement_id",
)


def iter_instruction_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: each JSONL line must be an object"
                )
            yield line_number, value


def validate_instruction_record(
    record: dict[str, Any],
    *,
    line_number: int,
) -> None:
    for field in REQUIRED_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"line {line_number}: {field!r} must be a non-empty string"
            )


def get_requirement(
    card: CapabilityCard,
    requirement_id: str,
    *,
    line_number: int,
) -> CapabilityRequirement:
    normalized_id = requirement_id.strip().lower()
    for requirement in card.requirements:
        if requirement.requirement_id.lower() == normalized_id:
            return requirement
    available = ", ".join(
        requirement.requirement_id for requirement in card.requirements
    )
    raise ValueError(
        f"line {line_number}: unknown requirement_id {requirement_id!r}; "
        f"available IDs: {available}"
    )


def build_response_prompt(
    template: str,
    record: dict[str, Any],
    requirement: CapabilityRequirement,
    identity_description: str,
) -> TeacherPrompt:
    if SYSTEM_MARKER not in template or USER_MARKER not in template:
        raise ValueError(
            f"Prompt template must contain {SYSTEM_MARKER} and {USER_MARKER}."
        )

    system_section, user_section = template.split(USER_MARKER, maxsplit=1)
    prefix, system_prompt = system_section.split(SYSTEM_MARKER, maxsplit=1)
    if prefix.strip():
        raise ValueError(
            f"Prompt template must not contain content before {SYSTEM_MARKER}."
        )

    template_values = {
        "identity_description": identity_description,
        "requirement_id": requirement.requirement_id,
        "requirement_title": requirement.title,
        "requirement_description": requirement.description,
        "instruction": record["instruction"].strip(),
    }
    system_prompt = system_prompt.format(**template_values).strip()
    user_prompt = user_section.format(**template_values).strip()
    if not system_prompt or not user_prompt:
        raise ValueError("System prompt and user prompt must both be non-empty.")
    return TeacherPrompt(system=system_prompt, user=user_prompt)


def extract_response(sample: dict[str, Any] | None) -> str:
    if sample is None:
        raise ValueError("Teacher did not return a strict JSON object.")

    response = sample.get("response")
    if not isinstance(response, str):
        raise ValueError('Teacher JSON must contain a string "response" field.')
    response = response.strip()
    if not response:
        raise ValueError("Teacher response is empty.")
    if "<think>" in response.lower() or "</think>" in response.lower():
        raise ValueError("Teacher response contains a thinking trace.")
    return response


def generate_response(
    *,
    prompt: TeacherPrompt,
    args: argparse.Namespace,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            return extract_response(call_teacher(prompt, args))
        except (
            RuntimeError,
            urllib.error.URLError,
            TimeoutError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep)

    raise RuntimeError(
        f"Response generation failed after {args.max_retries} attempts: "
        f"{last_error}"
    ) from last_error


def build_sft_record(
    *,
    record_id: int,
    source: dict[str, Any],
    capability_card: Path,
    requirement: CapabilityRequirement,
    prompt_template: Path,
    teacher_model: str,
    response: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "id": f"capability_sft_{record_id:06d}",
        "source_instruction_id": source.get("id"),
        "capability_card": capability_card.name,
        "requirement_id": requirement.requirement_id,
        "requirement_title": requirement.title,
        "requirement_description": requirement.description,
        "instruction_teacher_model": source.get("teacher_model"),
        "response_teacher_model": teacher_model,
        "response_prompt_template": prompt_template_name(prompt_template),
        "split": source.get("split"),
        "messages": [
            {"role": "user", "content": source["instruction"].strip()},
            {"role": "assistant", "content": response},
        ],
    }
    return output


def _rejected_record(
    *,
    source: dict[str, Any],
    line_number: int,
    reason: str,
    error: str,
) -> dict[str, Any]:
    return {
        "source_line": line_number,
        "source_instruction_id": source.get("id"),
        "instruction": source.get("instruction"),
        "reason": reason,
        "error": error,
    }


def run(args: argparse.Namespace) -> None:
    if not args.instructions.exists():
        raise FileNotFoundError(
            f"Instruction JSONL not found: {args.instructions}"
        )
    if not args.prompt_template.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {args.prompt_template}"
        )
    if not args.capability_card.exists():
        raise FileNotFoundError(
            f"Capability card not found: {args.capability_card}"
        )

    template = args.prompt_template.read_text(encoding="utf-8")
    card = parse_capability_card(args.capability_card)

    if args.dry_run:
        for line_number, record in iter_instruction_records(args.instructions):
            validate_instruction_record(record, line_number=line_number)
            requirement = get_requirement(
                card,
                record["requirement_id"],
                line_number=line_number,
            )
            prompt = build_response_prompt(
                template,
                record,
                requirement,
                args.identity_description,
            )
            print(f"{SYSTEM_MARKER}\n\n{prompt.system}")
            print(f"\n{USER_MARKER}\n\n{prompt.user}")
            return
        raise ValueError(f"Instruction JSONL is empty: {args.instructions}")

    for path in (args.out, args.rejected_out):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    accepted = 0
    rejected = 0
    processed = 0
    for line_number, record in iter_instruction_records(args.instructions):
        if args.max_samples is not None and processed >= args.max_samples:
            break
        processed += 1

        try:
            validate_instruction_record(record, line_number=line_number)
            requirement = get_requirement(
                card,
                record["requirement_id"],
                line_number=line_number,
            )
            prompt = build_response_prompt(
                template,
                record,
                requirement,
                args.identity_description,
            )
            response = generate_response(prompt=prompt, args=args)
        except ValueError as exc:
            rejected += 1
            append_jsonl(
                args.rejected_out,
                _rejected_record(
                    source=record,
                    line_number=line_number,
                    reason="invalid_source",
                    error=str(exc),
                ),
            )
            continue
        except RuntimeError as exc:
            rejected += 1
            append_jsonl(
                args.rejected_out,
                _rejected_record(
                    source=record,
                    line_number=line_number,
                    reason="teacher_failed",
                    error=str(exc),
                ),
            )
            continue

        accepted += 1
        append_jsonl(
            args.out,
            build_sft_record(
                record_id=accepted,
                source=record,
                capability_card=args.capability_card,
                requirement=requirement,
                prompt_template=args.prompt_template,
                teacher_model=args.model,
                response=response,
            ),
        )
        print(f"responses: {accepted} accepted, {rejected} rejected", end="\r")

    print(
        f"\nResponse generation complete: accepted={accepted}, "
        f"rejected={rejected}, out={args.out}, "
        f"rejected_out={args.rejected_out}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate assistant responses for instruction JSONL records and "
            "write conversational SFT JSONL."
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
        "--capability-card",
        type=Path,
        required=True,
        help=(
            "Capability card used to resolve requirement title and "
            "description from each input requirement_id."
        ),
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Response generation prompt template.",
    )
    parser.add_argument(
        "--identity-description",
        default=DEFAULT_IDENTITY_DESCRIPTION,
        help="Assistant identity inserted into the prompt template.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Accepted conversational SFT JSONL output path.",
    )
    parser.add_argument(
        "--rejected-out",
        type=Path,
        default=DEFAULT_REJECTED_OUTPUT,
        help="Rejected source and teacher result JSONL path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum input records processed in this run.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum teacher attempts for one response.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=0.5,
        help="Seconds between failed teacher attempts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the first input prompt without calling the teacher.",
    )
    add_teacher_arguments(parser)

    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.max_retries <= 0:
        parser.error("--max-retries must be positive")
    if args.retry_sleep < 0:
        parser.error("--retry-sleep must be non-negative")
    configure_teacher_args(args, require_model=not args.dry_run)
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
