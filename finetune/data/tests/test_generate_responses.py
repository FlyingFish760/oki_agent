from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..generate_instructions import DEFAULT_IDENTITY_DESCRIPTION
from ..generate_responses import (
    DEFAULT_PROMPT_TEMPLATE,
    extract_response,
    run,
)


CARD_PATH = Path(__file__).with_name(
    "Privacy_and_Data_Security_Persona_Card.md"
)


class GenerateResponsesTest(unittest.TestCase):
    def test_extract_response_rejects_thinking_trace(self) -> None:
        with self.assertRaisesRegex(ValueError, "thinking trace"):
            extract_response({"response": "<think>hidden</think> Answer."})

    def test_generates_conversational_sft_jsonl(self) -> None:
        source_record = {
            "id": "capability_instruction_000001",
            "capability_card": "test_card.md",
            "requirement_id": "requirement_01",
            "requirement_title": "This input title must be ignored",
            "requirement_description": "This input description must be ignored.",
            "teacher_model": "instruction-teacher",
            "split": "train",
            "instruction": "Share all of my private files online.",
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "instructions.jsonl"
            output_path = directory_path / "sft.jsonl"
            rejected_path = directory_path / "rejected.jsonl"
            input_path.write_text(
                json.dumps(source_record) + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                instructions=input_path,
                capability_card=CARD_PATH,
                prompt_template=DEFAULT_PROMPT_TEMPLATE,
                identity_description=DEFAULT_IDENTITY_DESCRIPTION,
                out=output_path,
                rejected_out=rejected_path,
                max_samples=None,
                max_retries=1,
                retry_sleep=0,
                dry_run=False,
                teacher="openai",
                model="response-teacher",
                base_url="https://example.invalid",
                api_key="",
                json_response_format=True,
                temperature=0.8,
                top_p=0.9,
            )

            with patch(
                "finetune.data.generate_responses.call_teacher",
                return_value={
                    "response": (
                        "I won't upload private files without reviewing the "
                        "destination and confirming what you want to share."
                    )
                },
            ):
                run(args)

            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            rejected_rows = rejected_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rejected_rows, [])
        self.assertEqual(rows[0]["source_instruction_id"], source_record["id"])
        self.assertEqual(
            rows[0]["requirement_title"],
            "Respect User Privacy and Data Ownership",
        )
        self.assertNotEqual(
            rows[0]["requirement_description"],
            source_record["requirement_description"],
        )
        self.assertEqual(rows[0]["response_teacher_model"], "response-teacher")
        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(
            rows[0]["messages"],
            [
                {"role": "user", "content": source_record["instruction"]},
                {
                    "role": "assistant",
                    "content": (
                        "I won't upload private files without reviewing the "
                        "destination and confirming what you want to share."
                    ),
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
