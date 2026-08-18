from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..generate_all_instructions import (
    distribute_instruction_counts,
    load_resume_state,
    run,
)
from ..generate_instructions import (
    DEFAULT_IDENTITY_DESCRIPTION,
    DEFAULT_PROMPT_TEMPLATE,
    extract_instructions,
)
from ..capability_card import parse_capability_card


CARD_PATH = (
    Path(__file__).resolve().parents[2]
    / "persona_cards"
    / "Privacy_and_Data_Security_Persona_Card.md"
)


class GenerateAllInstructionsTest(unittest.TestCase):
    def test_duplicate_instructions_are_preserved(self) -> None:
        instructions = extract_instructions(
            {"instructions": ["Repeat this.", "Repeat this."]},
            expected_count=2,
        )

        self.assertEqual(instructions, ["Repeat this.", "Repeat this."])

    def test_distributes_total_across_every_requirement(self) -> None:
        card = parse_capability_card(CARD_PATH)

        targets = distribute_instruction_counts(card, 42)

        self.assertEqual(sum(targets.values()), 42)
        self.assertEqual(list(targets.values()), [6, 6, 5, 5, 5, 5, 5, 5])

    def test_writes_one_jsonl_record_for_every_requirement(self) -> None:
        call_number = 0

        def fake_teacher(*_args, **_kwargs):
            nonlocal call_number
            call_number += 1
            return {
                "instructions": [
                    (
                        "Please review the privacy settings for my account "
                        f"in scenario {call_number}."
                    )
                ]
            }

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "instructions.jsonl"
            args = argparse.Namespace(
                capability_card=CARD_PATH,
                num_instructions=3,
                total_instructions=8,
                prompt_template=DEFAULT_PROMPT_TEMPLATE,
                identity_description=DEFAULT_IDENTITY_DESCRIPTION,
                out=output_path,
                max_retries=1,
                retry_sleep=0,
                dry_run=False,
                resume=False,
                teacher="openai",
                model="mock-teacher",
                base_url="https://example.invalid",
                api_key="",
                json_response_format=True,
                temperature=0.8,
                top_p=0.9,
            )

            with patch(
                "finetune.data.generate_all_instructions.call_teacher",
                side_effect=fake_teacher,
            ):
                run(args)

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 8)
        self.assertEqual(
            {row["requirement_id"] for row in rows},
            {f"requirement_{number:02d}" for number in range(1, 9)},
        )
        self.assertTrue(all(row["teacher_model"] == "mock-teacher" for row in rows))
        self.assertTrue(all(isinstance(row["instruction"], str) for row in rows))

    def test_resume_skips_complete_requirements_and_ignores_titles(self) -> None:
        card = parse_capability_card(CARD_PATH)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "instructions.jsonl"
            existing = []
            for index, requirement in enumerate(card.requirements[:7], start=1):
                existing.append(
                    {
                        "id": f"capability_instruction_{index:06d}",
                        "capability_card": CARD_PATH.name,
                        "requirement_id": requirement.requirement_id,
                        "requirement_title": "An intentionally stale title",
                        "requirement_description": "Intentionally stale text",
                        "prompt_template": DEFAULT_PROMPT_TEMPLATE.stem,
                        "teacher_model": "mock-teacher",
                        "instruction": f"Existing instruction {index}.",
                    }
                )
            output_path.write_text(
                "".join(json.dumps(row) + "\n" for row in existing),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                capability_card=CARD_PATH,
                num_instructions=3,
                total_instructions=8,
                prompt_template=DEFAULT_PROMPT_TEMPLATE,
                identity_description=DEFAULT_IDENTITY_DESCRIPTION,
                out=output_path,
                max_retries=1,
                retry_sleep=0,
                dry_run=False,
                resume=True,
                teacher="openai",
                model="mock-teacher",
                base_url="https://example.invalid",
                api_key="",
                json_response_format=True,
                temperature=0.8,
                top_p=0.9,
            )

            with patch(
                "finetune.data.generate_all_instructions.call_teacher",
                return_value={"instructions": ["Final instruction."]},
            ) as teacher:
                run(args)

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        teacher.assert_called_once()
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[-1]["id"], "capability_instruction_000008")
        self.assertEqual(rows[-1]["requirement_id"], "requirement_08")

    def test_resume_rejects_metadata_mismatch_without_modifying_file(self) -> None:
        card = parse_capability_card(CARD_PATH)
        targets = distribute_instruction_counts(card, 8)
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "instructions.jsonl"
            original = json.dumps(
                {
                    "id": "capability_instruction_000001",
                    "capability_card": CARD_PATH.name,
                    "requirement_id": "requirement_01",
                    "prompt_template": DEFAULT_PROMPT_TEMPLATE.stem,
                    "teacher_model": "different-teacher",
                    "instruction": "Existing instruction.",
                }
            ) + "\n"
            output_path.write_text(original, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "teacher_model"):
                load_resume_state(
                    path=output_path,
                    card=card,
                    card_path=CARD_PATH,
                    targets=targets,
                    prompt_template=DEFAULT_PROMPT_TEMPLATE,
                    teacher_model="mock-teacher",
                )

            self.assertEqual(output_path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
