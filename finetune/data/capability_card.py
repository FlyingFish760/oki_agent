"""Parse capability cards into structured requirements and few-shot instructions."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


_CORE_SECTION = "## Core Requirements"
_FEW_SHOT_SECTION = "# Few-shot Instructions"
_CORE_REQUIREMENT_RE = re.compile(r"^###\s+(\d+)\.\s+(.+?)\s*$")
_FEW_SHOT_REQUIREMENT_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$")
_EXAMPLE_RE = re.compile(r"^###\s+Example\s+\d+\s*$", re.IGNORECASE)
_INSTRUCTION_RE = re.compile(r"^\*\*Instruction:\*\*\s*\\?\s*$", re.IGNORECASE)
_EXPECTED_BEHAVIOR_RE = re.compile(
    r"^\*\*Expected Behavior:\*\*\s*\\?\s*$",
    re.IGNORECASE,
)
_SEPARATOR_RE = re.compile(r"^-{3,}\s*$")


@dataclass
class CapabilityRequirement:
    requirement_id: str
    number: int
    title: str
    description: str
    few_shot_instructions: list[str]


@dataclass
class CapabilityCard:
    title: str
    requirements: list[CapabilityRequirement]


def _line_label(path: Path, line_number: int) -> str:
    return f"{path}:{line_number}"


def _clean_block(lines: list[str]) -> str:
    cleaned: list[str] = []
    for line in lines:
        text = line.strip()
        if not text or _SEPARATOR_RE.fullmatch(text):
            continue
        text = re.sub(r"\\\s*$", "", text).strip()
        if text:
            cleaned.append(text)
    return " ".join(cleaned)


def _parse_title(lines: list[str], path: Path) -> str:
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped == _CORE_SECTION:
            break
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                return title
            raise ValueError(f"{_line_label(path, line_number)}: card title is empty")
    raise ValueError(f"{path}: missing card title before {_CORE_SECTION!r}")


def _section_index(lines: list[str], heading: str, path: Path) -> int:
    matches = [index for index, line in enumerate(lines) if line.strip() == heading]
    if not matches:
        raise ValueError(f"{path}: missing section {heading!r}")
    if len(matches) > 1:
        line_numbers = ", ".join(str(index + 1) for index in matches)
        raise ValueError(
            f"{path}: section {heading!r} appears more than once "
            f"(lines {line_numbers})"
        )
    return matches[0]


def _parse_core_requirements(
    lines: list[str],
    start: int,
    end: int,
    path: Path,
) -> tuple[list[CapabilityRequirement], dict[int, CapabilityRequirement]]:
    requirements: list[CapabilityRequirement] = []
    requirements_by_number: dict[int, CapabilityRequirement] = {}
    current_number: int | None = None
    current_title = ""
    current_description: list[str] = []
    current_line = 0

    def finish_current() -> None:
        nonlocal current_number, current_title, current_description, current_line
        if current_number is None:
            return
        description = _clean_block(current_description)
        if not description:
            raise ValueError(
                f"{_line_label(path, current_line)}: requirement "
                f"{current_number} has no description"
            )
        requirement = CapabilityRequirement(
            requirement_id=f"requirement_{current_number:02d}",
            number=current_number,
            title=current_title,
            description=description,
            few_shot_instructions=[],
        )
        requirements.append(requirement)
        requirements_by_number[current_number] = requirement
        current_number = None
        current_title = ""
        current_description = []
        current_line = 0

    for index in range(start, end):
        stripped = lines[index].strip()
        heading = _CORE_REQUIREMENT_RE.fullmatch(stripped)
        if heading:
            finish_current()
            number = int(heading.group(1))
            if number in requirements_by_number:
                raise ValueError(
                    f"{_line_label(path, index + 1)}: duplicate core requirement "
                    f"number {number}"
                )
            title = heading.group(2).strip()
            if not title:
                raise ValueError(
                    f"{_line_label(path, index + 1)}: requirement {number} "
                    "has an empty title"
                )
            current_number = number
            current_title = title
            current_line = index + 1
            continue

        if current_number is not None:
            current_description.append(lines[index])

    finish_current()
    if not requirements:
        raise ValueError(f"{path}: {_CORE_SECTION!r} contains no requirements")
    return requirements, requirements_by_number


def _parse_few_shot_instructions(
    lines: list[str],
    start: int,
    requirements_by_number: dict[int, CapabilityRequirement],
    path: Path,
) -> None:
    current_requirement: CapabilityRequirement | None = None
    current_instruction: list[str] | None = None
    current_instruction_line = 0
    seen_sections: set[int] = set()

    def finish_instruction() -> None:
        nonlocal current_instruction, current_instruction_line
        if current_instruction is None:
            return
        instruction = _clean_block(current_instruction)
        if not instruction:
            assert current_requirement is not None
            raise ValueError(
                f"{_line_label(path, current_instruction_line)}: empty instruction "
                f"for requirement {current_requirement.number}"
            )
        assert current_requirement is not None
        current_requirement.few_shot_instructions.append(instruction)
        current_instruction = None
        current_instruction_line = 0

    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        requirement_heading = _FEW_SHOT_REQUIREMENT_RE.fullmatch(stripped)
        if requirement_heading:
            finish_instruction()
            number = int(requirement_heading.group(1))
            if number in seen_sections:
                raise ValueError(
                    f"{_line_label(path, index + 1)}: duplicate few-shot section "
                    f"for requirement {number}"
                )
            try:
                requirement = requirements_by_number[number]
            except KeyError as exc:
                raise ValueError(
                    f"{_line_label(path, index + 1)}: few-shot section references "
                    f"unknown requirement {number}"
                ) from exc
            few_shot_title = requirement_heading.group(2).strip()
            if few_shot_title != requirement.title:
                raise ValueError(
                    f"{_line_label(path, index + 1)}: few-shot title "
                    f"{few_shot_title!r} does not match core requirement "
                    f"title {requirement.title!r}"
                )
            seen_sections.add(number)
            current_requirement = requirement
            continue

        if _EXAMPLE_RE.fullmatch(stripped):
            finish_instruction()
            continue

        if _INSTRUCTION_RE.fullmatch(stripped):
            finish_instruction()
            if current_requirement is None:
                raise ValueError(
                    f"{_line_label(path, index + 1)}: instruction appears before "
                    "a numbered few-shot section"
                )
            current_instruction = []
            current_instruction_line = index + 1
            continue

        if _EXPECTED_BEHAVIOR_RE.fullmatch(stripped):
            finish_instruction()
            continue

        if current_instruction is not None:
            current_instruction.append(lines[index])

    finish_instruction()

    for requirement in requirements_by_number.values():
        if requirement.number not in seen_sections:
            raise ValueError(
                f"{path}: missing few-shot section for requirement "
                f"{requirement.number}"
            )
        if not requirement.few_shot_instructions:
            raise ValueError(
                f"{path}: requirement {requirement.number} has no few-shot "
                "instructions"
            )


def parse_capability_card(path: Path) -> CapabilityCard:
    """Parse a Markdown capability card without retaining expected behaviors."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Unable to read capability card {path}: {exc}") from exc

    title = _parse_title(lines, path)
    core_index = _section_index(lines, _CORE_SECTION, path)
    few_shot_index = _section_index(lines, _FEW_SHOT_SECTION, path)
    if core_index >= few_shot_index:
        raise ValueError(
            f"{path}: {_CORE_SECTION!r} must appear before "
            f"{_FEW_SHOT_SECTION!r}"
        )

    requirements, requirements_by_number = _parse_core_requirements(
        lines,
        core_index + 1,
        few_shot_index,
        path,
    )
    _parse_few_shot_instructions(
        lines,
        few_shot_index + 1,
        requirements_by_number,
        path,
    )
    return CapabilityCard(title=title, requirements=requirements)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a capability card and print a summary of its requirements "
            "and few-shot instruction counts."
        )
    )
    parser.add_argument("card", type=Path, help="Path to the capability card.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    card = parse_capability_card(args.card)
    print(f"Card: {card.title}")
    print(f"Requirements: {len(card.requirements)}")
    total_instructions = 0
    for requirement in card.requirements:
        count = len(requirement.few_shot_instructions)
        total_instructions += count
        print(
            f"{requirement.requirement_id}: {requirement.title} "
            f"({count} instructions)"
        )
    print(f"Total instructions: {total_instructions}")


if __name__ == "__main__":
    main()
