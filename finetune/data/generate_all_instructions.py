"""Generate a JSONL instruction dataset for every capability requirement."""

from __future__ import annotations

import argparse
import math
import time
import urllib.error
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
    args.out.write_text("", encoding="utf-8")

    record_id = 1

    for requirement in card.requirements:
        target = targets[requirement.requirement_id]
        generated_for_requirement: list[str] = []

        while len(generated_for_requirement) < target:
            remaining = target - len(generated_for_requirement)
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
                generated_for_requirement.append(instruction)

            print(
                f"{requirement.requirement_id}: "
                f"{len(generated_for_requirement)}/{target}"
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
