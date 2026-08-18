"""Generate a JSONL instruction dataset for every capability requirement."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from capability_card import (
        CapabilityCard,
        CapabilityRequirement,
        parse_capability_card,
    )
    from generate_instructions import (
        DEFAULT_IDENTITY_DESCRIPTION,
        DEFAULT_PROMPT_TEMPLATE,
        build_instruction_prompt,
        extract_instructions,
    )
    from generation_common import (
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
    from .generate_instructions import (
        DEFAULT_IDENTITY_DESCRIPTION,
        DEFAULT_PROMPT_TEMPLATE,
        build_instruction_prompt,
        extract_instructions,
    )
    from .generation_common import (
        add_teacher_arguments,
        append_jsonl,
        call_teacher,
        configure_teacher_args,
        prompt_template_name,
    )


DEFAULT_OUTPUT = Path(
    "finetune/data/datasets/capability_instructions.jsonl"
)


@dataclass(frozen=True)
class ResumeState:
    record_count: int
    requirement_counts: dict[str, int]

    @property
    def next_record_id(self) -> int:
        return self.record_count + 1


def distribute_instruction_counts(
    card: CapabilityCard,
    total_instructions: int,
) -> dict[str, int]:
    requirement_count = len(card.requirements)
    if requirement_count == 0:
        raise ValueError("Capability card contains no requirements.")
    if total_instructions < requirement_count:
        raise ValueError(
            "--total-instructions must be at least the number of requirements "
            f"({requirement_count}) so every requirement receives a sample."
        )

    base_count, remainder = divmod(total_instructions, requirement_count)
    return {
        requirement.requirement_id: base_count + (index < remainder)
        for index, requirement in enumerate(card.requirements)
    }


def load_resume_state(
    *,
    path: Path,
    card: CapabilityCard,
    card_path: Path,
    targets: dict[str, int],
    prompt_template: Path,
    teacher_model: str,
) -> ResumeState:
    if not path.exists():
        raise FileNotFoundError(f"Resume output does not exist: {path}")

    expected_requirements = [
        requirement.requirement_id
        for requirement in card.requirements
        for _ in range(targets[requirement.requirement_id])
    ]
    expected_card = card_path.name
    expected_template = prompt_template_name(prompt_template)
    counts: Counter[str] = Counter()
    record_count = 0

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(
                    f"{path}:{line_number}: blank lines are not allowed when "
                    "resuming"
                )
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
            if line_number > len(expected_requirements):
                raise ValueError(
                    f"{path} already contains more than "
                    f"{sum(targets.values())} target records"
                )

            expected_id = f"capability_instruction_{line_number:06d}"
            if record.get("id") != expected_id:
                raise ValueError(
                    f"{path}:{line_number}: expected id {expected_id!r}, "
                    f"found {record.get('id')!r}"
                )

            expected_requirement = expected_requirements[line_number - 1]
            requirement_id = record.get("requirement_id")
            if requirement_id != expected_requirement:
                raise ValueError(
                    f"{path}:{line_number}: expected requirement_id "
                    f"{expected_requirement!r}, found {requirement_id!r}"
                )

            metadata_checks = {
                "capability_card": expected_card,
                "prompt_template": expected_template,
                "teacher_model": teacher_model,
            }
            for field, expected_value in metadata_checks.items():
                if record.get(field) != expected_value:
                    raise ValueError(
                        f"{path}:{line_number}: expected {field}="
                        f"{expected_value!r}, found {record.get(field)!r}"
                    )

            instruction = record.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(
                    f"{path}:{line_number}: instruction must be a non-empty "
                    "string"
                )

            counts[requirement_id] += 1
            record_count += 1

    requirement_counts = {
        requirement.requirement_id: counts[requirement.requirement_id]
        for requirement in card.requirements
    }
    return ResumeState(
        record_count=record_count,
        requirement_counts=requirement_counts,
    )


def _generate_batch(
    *,
    args: argparse.Namespace,
    template: str,
    requirement: CapabilityRequirement,
    batch_size: int,
) -> list[str]:
    prompt = build_instruction_prompt(
        template,
        requirement,
        batch_size,
        args.identity_description,
    )
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            sample = call_teacher(prompt, args)
            instructions = extract_instructions(sample, batch_size)
            return instructions
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
        f"Failed to generate {batch_size} instructions for "
        f"{requirement.requirement_id} after {args.max_retries} attempts: "
        f"{last_error}"
    ) from last_error


def _instruction_record(
    *,
    record_id: int,
    card_path: Path,
    requirement: CapabilityRequirement,
    prompt_template: Path,
    teacher_model: str,
    instruction: str,
) -> dict[str, Any]:
    return {
        "id": f"capability_instruction_{record_id:06d}",
        "capability_card": card_path.name,
        "requirement_id": requirement.requirement_id,
        "requirement_title": requirement.title,
        "requirement_description": requirement.description,
        "prompt_template": prompt_template_name(prompt_template),
        "teacher_model": teacher_model,
        "instruction": instruction,
    }


def _print_generation_plan(
    card: CapabilityCard,
    targets: dict[str, int],
    instructions_per_call: int,
) -> None:
    print(f"Requirements: {len(card.requirements)}")
    print(f"Total instructions: {sum(targets.values())}")
    print(f"Instructions per teacher call: {instructions_per_call}")
    for requirement in card.requirements:
        target = targets[requirement.requirement_id]
        calls = math.ceil(target / instructions_per_call)
        print(
            f"{requirement.requirement_id}: target={target}, "
            f"teacher_calls={calls}"
        )


def run(args: argparse.Namespace) -> None:
    if not args.capability_card.exists():
        raise FileNotFoundError(
            f"Capability card not found: {args.capability_card}"
        )
    if not args.prompt_template.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {args.prompt_template}"
        )

    card = parse_capability_card(args.capability_card)
    targets = distribute_instruction_counts(
        card,
        args.total_instructions,
    )
    _print_generation_plan(card, targets, args.num_instructions)
    if args.dry_run:
        return

    template = args.prompt_template.read_text(encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        resume_state = load_resume_state(
            path=args.out,
            card=card,
            card_path=args.capability_card,
            targets=targets,
            prompt_template=args.prompt_template,
            teacher_model=args.model,
        )
        existing_counts = resume_state.requirement_counts
        record_id = resume_state.next_record_id
        print(f"Resume: found {resume_state.record_count} valid records")
        print(f"Next record ID: capability_instruction_{record_id:06d}")
    else:
        args.out.write_text("", encoding="utf-8")
        existing_counts = {
            requirement.requirement_id: 0
            for requirement in card.requirements
        }
        record_id = 1

    for requirement in card.requirements:
        target = targets[requirement.requirement_id]
        generated_for_requirement = existing_counts[requirement.requirement_id]

        if generated_for_requirement == target:
            print(
                f"{requirement.requirement_id}: "
                f"{generated_for_requirement}/{target}, already complete"
            )
            continue
        if generated_for_requirement:
            print(
                f"{requirement.requirement_id}: resuming from "
                f"{generated_for_requirement}/{target}"
            )

        while generated_for_requirement < target:
            remaining = target - generated_for_requirement
            batch_size = min(args.num_instructions, remaining)
            instructions = _generate_batch(
                args=args,
                template=template,
                requirement=requirement,
                batch_size=batch_size,
            )

            for instruction in instructions:
                append_jsonl(
                    args.out,
                    _instruction_record(
                        record_id=record_id,
                        card_path=args.capability_card,
                        requirement=requirement,
                        prompt_template=args.prompt_template,
                        teacher_model=args.model,
                        instruction=instruction,
                    ),
                )
                record_id += 1
                generated_for_requirement += 1

            print(
                f"{requirement.requirement_id}: "
                f"{generated_for_requirement}/{target}"
            )

    generated_total = record_id - 1
    if generated_total != args.total_instructions:
        raise RuntimeError(
            f"Generated {generated_total} instructions, expected "
            f"{args.total_instructions}."
        )
    print(
        f"Generation complete: wrote {generated_total} instructions to "
        f"{args.out}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate instructions for every requirement in a capability card "
            "and write one instruction record per JSONL line."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--capability-card",
        type=Path,
        required=True,
        help="Capability card to parse.",
    )
    parser.add_argument(
        "--num-instructions",
        type=int,
        required=True,
        help="Maximum instructions requested in one teacher call.",
    )
    parser.add_argument(
        "--total-instructions",
        type=int,
        required=True,
        help="Exact total number of instructions generated across the card.",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Instruction generation prompt template.",
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
        help="JSONL output path.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum teacher attempts for one batch.",
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
        help="Print the distribution plan without calling the teacher.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Validate and continue an existing output JSONL instead of "
            "truncating it."
        ),
    )
    add_teacher_arguments(parser)

    args = parser.parse_args()
    if args.num_instructions <= 0:
        parser.error("--num-instructions must be positive")
    if args.total_instructions <= 0:
        parser.error("--total-instructions must be positive")
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
