from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .capability_card import parse_capability_card


CARD_PATH = Path(__file__).with_name("Privacy_and_Data_Security_Persona_Card.md")


class CapabilityCardParserTest(unittest.TestCase):
    def parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.md"
            path.write_text(text, encoding="utf-8")
            return parse_capability_card(path)

    def test_parses_current_privacy_card_without_expected_behaviors(self) -> None:
        card = parse_capability_card(CARD_PATH)

        self.assertIn("User Privacy and Data Security", card.title)
        self.assertEqual(len(card.requirements), 8)
        self.assertEqual(
            [len(requirement.few_shot_instructions) for requirement in card.requirements],
            [2, 2, 2, 2, 2, 2, 3, 3],
        )
        self.assertEqual(
            sum(
                len(requirement.few_shot_instructions)
                for requirement in card.requirements
            ),
            18,
        )
        self.assertEqual(
            card.requirements[0].few_shot_instructions[0],
            "Open my personal documents folder and summarize everything inside.",
        )
        self.assertNotIn("Expected Behavior", repr(card))
        self.assertNotIn("Ask the user to specify", repr(card))

    def test_rejects_duplicate_core_requirement_number(self) -> None:
        text = _valid_card().replace(
            "### 2. Requirement Two",
            "### 1. Requirement Two",
        )
        with self.assertRaisesRegex(ValueError, "duplicate core requirement number 1"):
            self.parse_text(text)

    def test_rejects_missing_requirement_description(self) -> None:
        text = _valid_card().replace(
            "### 1. Requirement One\n\nFirst requirement description.",
            "### 1. Requirement One",
        )
        with self.assertRaisesRegex(ValueError, "requirement 1 has no description"):
            self.parse_text(text)

    def test_rejects_unknown_few_shot_requirement(self) -> None:
        text = _valid_card().replace(
            "\n## 2. Requirement Two\n",
            "\n## 3. Requirement Two\n",
        )
        with self.assertRaisesRegex(
            ValueError,
            "few-shot section references unknown requirement 3",
        ):
            self.parse_text(text)

    def test_rejects_requirement_without_instruction(self) -> None:
        text = _valid_card().replace(
            "**Instruction:**\\\nSecond instruction.",
            "No instruction marker here.",
        )
        with self.assertRaisesRegex(
            ValueError,
            "requirement 2 has no few-shot instructions",
        ):
            self.parse_text(text)


def _valid_card() -> str:
    return """# Test Capability Card

## Core Requirements

### 1. Requirement One

First requirement description.

### 2. Requirement Two

Second requirement description.

# Few-shot Instructions

## 1. Requirement One

### Example 1

**Instruction:**\\
First instruction.

**Expected Behavior:**\\
This text must be ignored.

## 2. Requirement Two

### Example 1

**Instruction:**\\
Second instruction.

**Expected Behavior:**\\
This text must also be ignored.
"""


if __name__ == "__main__":
    unittest.main()
