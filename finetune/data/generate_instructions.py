"""Generate new user instructions for one capability requirement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from capability_card import (
        CapabilityCard,
        CapabilityRequirement,
        parse_capability_card,
    )
    from generation_common import (
        TeacherPrompt,
        add_teacher_arguments,
        call_teacher,
        clean_text,
        configure_teacher_args,
        prompt_template_name,
        write_json,
    )
except ModuleNotFoundError:
    from .capability_card import (
        CapabilityCard,
        CapabilityRequirement,
        parse_capability_card,
    )
    from .generation_common import (
        TeacherPrompt,
        add_teacher_arguments,
        call_teacher,
        clean_text,
        configure_teacher_args,
        prompt_template_name,
        write_json,
    )


DEFAULT_CAPABILITY_CARD = Path(
    "finetune/data/Privacy_and_Data_Security_Persona_Card.md"
)
DEFAULT_PROMPT_TEMPLATE = Path(
    "finetune/data/templates/instruction_generation_template.txt"
)
DEFAULT_OUTPUT = Path(
    "finetune/data/datasets/privacy_security_instructions.json"
)
DEFAULT_IDENTITY_DESCRIPTION = (
    "The user's local personal assistant running on the user's own computer."
)

SYSTEM_MARKER = "[SYSTEM PROMPT]"
USER_MARKER = "[USER PROMPT]"


def select_requirement(
    card: CapabilityCard,
    selector: str,
) -> CapabilityRequirement:
    normalized = selector.strip().lower()
    if normalized.isdigit():
        normalized = f"requirement_{int(normalized):02d}"

    matches = [
        requirement
        for requirement in card.requirements
        if requirement.requirement_id.lower() == normalized
    ]
    if not matches:
        available = ", ".join(
            requirement.requirement_id for requirement in card.requirements
        )
        raise ValueError(
            f"Unknown requirement {selector!r}. Available requirements: "
            f"{available}"
        )
    return matches[0]


def format_few_shot_instructions(instructions: list[str]) -> str:
    return "\n".join(
        f'Example{index}:\n"instruction": '
        f"{json.dumps(instruction, ensure_ascii=False)}"
        for index, instruction in enumerate(instructions, start=1)
    )


def build_instruction_prompt(
    template: str,
    requirement: CapabilityRequirement,
    num_instructions: int,
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
        "few_shot_instructions": format_few_shot_instructions(
            requirement.few_shot_instructions
        ),
        "num_instructions": num_instructions,
    }
    system_prompt = system_prompt.format(**template_values)
    user_prompt = user_section.format(
        **template_values,
    )
    if not system_prompt.strip() or not user_prompt.strip():
        raise ValueError("System prompt and user prompt must both be non-empty.")
    return TeacherPrompt(
        system=system_prompt.strip(),
        user=user_prompt.strip(),
    )


def extract_instructions(
    sample: dict[str, Any] | None,
    expected_count: int,
) -> list[str]:
    if sample is None:
        raise ValueError("Teacher did not return a strict JSON object.")

    raw_instructions = sample.get("instructions")
    if not isinstance(raw_instructions, list):
        raise ValueError(
            'Teacher JSON must contain an "instructions" array.'
        )

    instructions: list[str] = []
    for index, value in enumerate(raw_instructions, start=1):
        if not isinstance(value, str):
            raise ValueError(f"Instruction {index} must be a string.")
        instruction = clean_text(value)
        if not instruction:
            raise ValueError(f"Instruction {index} is empty.")
        instructions.append(instruction)

    if len(instructions) != expected_count:
        raise ValueError(
            f"Teacher returned {len(instructions)} instructions; "
            f"expected exactly {expected_count}."
        )
    return instructions


def build_output(
    *,
    card_path: Path,
    requirement: CapabilityRequirement,
    prompt_template: Path,
    teacher_model: str,
    instructions: list[str],
) -> dict[str, Any]:
    return {
        "capability_card": card_path.name,
        "requirement_id": requirement.requirement_id,
        "requirement_title": requirement.title,
        "requirement_description": requirement.description,
        "prompt_template": prompt_template_name(prompt_template),
        "teacher_model": teacher_model,
        "instructions": instructions,
    }


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
    requirement = select_requirement(card, args.requirement)
    template = args.prompt_template.read_text(encoding="utf-8")
    prompt = build_instruction_prompt(
        template,
        requirement,
        args.num_instructions,
        args.identity_description,
    )

    if args.dry_run_prompt:
        print(f"{SYSTEM_MARKER}\n\n{prompt.system}")
        print(f"\n{USER_MARKER}\n\n{prompt.user}")
        return

    sample = call_teacher(prompt, args)
    instructions = extract_instructions(sample, args.num_instructions)
    output = build_output(
        card_path=args.capability_card,
        requirement=requirement,
        prompt_template=args.prompt_template,
        teacher_model=args.model,
        instructions=instructions,
    )
    write_json(args.out, output)
    print(
        f"Generated {len(instructions)} instructions for "
        f"{requirement.requirement_id} at {args.out}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a batch of new English user instructions for one "
            "capability requirement."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--capability-card",
        type=Path,
        default=DEFAULT_CAPABILITY_CARD,
        help="Capability card containing requirements and few-shot instructions.",
    )
    parser.add_argument(
        "--requirement",
        required=True,
        help="Requirement number or ID, for example 1 or requirement_01.",
    )
    parser.add_argument(
        "--num-instructions",
        type=int,
        default=20,
        help="Exact number of new instructions requested from the teacher.",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Template containing separate system and user prompt sections.",
    )
    parser.add_argument(
        "--identity-description",
        default=DEFAULT_IDENTITY_DESCRIPTION,
        help="Assistant identity inserted into the instruction prompt template.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path.",
    )
    parser.add_argument(
        "--dry-run-prompt",
        action="store_true",
        help="Render the prompt without calling a teacher or writing output.",
    )
    add_teacher_arguments(parser)

    args = parser.parse_args()
    if args.num_instructions <= 0:
        parser.error("--num-instructions must be positive")
    configure_teacher_args(args, require_model=not args.dry_run_prompt)
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
