from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from finetune.data.capability_card import parse_capability_card
from .generate_alpaca import build_alpaca_record, call_ollama, run


CARD_PATH = Path(__file__).parents[1] / "data" / "Privacy_and_Data_Security_Persona_Card.md"


class GenerateAlpacaTest(unittest.TestCase):
    def test_call_ollama_uses_instruction_as_direct_prompt(self) -> None:
        response = unittest.mock.MagicMock()
        response.read.return_value = json.dumps(
            {"response": "Generated response."}
        ).encode("utf-8")
        response.__enter__.return_value = response

        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            output = call_ollama(
                instruction="Direct instruction.",
                model="test-model",
                host="http://localhost:11434/",
                temperature=0.0,
                top_p=1.0,
                thinking=True,
                timeout=10,
            )

        self.assertEqual(output, "Generated response.")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://localhost:11434/api/generate")
        self.assertEqual(payload["prompt"], "Direct instruction.")
        self.assertIs(payload["think"], True)
        self.assertNotIn("system", payload)
        self.assertNotIn("messages", payload)

    def test_build_alpaca_record_preserves_metadata(self) -> None:
        source = {
            "id": "capability_instruction_000001",
            "requirement_id": "requirement_01",
            "requirement_title": "This input title must be ignored",
            "requirement_description": "This input description must be ignored",
            "instruction": "User instruction.",
            "split": "test",
            "teacher_model": "instruction-model",
            "messages": [],
        }
        requirement = parse_capability_card(CARD_PATH).requirements[0]

        record = build_alpaca_record(
            source,
            record_id=1,
            capability_card=CARD_PATH,
            requirement=requirement,
            output="Model response.",
            model="new-model",
        )

        self.assertEqual(
            record,
            {
                "id": "alpaca_eval_000001",
                "source_instruction_id": "capability_instruction_000001",
                "capability_card": "Privacy_and_Data_Security_Persona_Card.md",
                "requirement_id": "requirement_01",
                "requirement_title": "Respect User Privacy and Data Ownership",
                "requirement_description": requirement.description,
                "instruction_teacher_model": "instruction-model",
                "generator": "new-model",
                "split": "test",
                "instruction": "User instruction.",
                "output": "Model response.",
            },
        )

    def test_run_writes_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "input.jsonl"
            output_path = directory_path / "output.json"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "requirement_id": "requirement_01",
                        "instruction": "User instruction.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                instructions=input_path,
                capability_card=CARD_PATH,
                model="test-model",
                host="http://localhost:11434",
                out=output_path,
                max_samples=None,
                temperature=0.0,
                top_p=1.0,
                thinking=False,
                timeout=10,
                max_retries=1,
                retry_sleep=0,
                dry_run=False,
            )

            with patch(
                "finetune.eval.generate_alpaca.call_ollama",
                return_value="Model response.",
            ):
                run(args)

            output = json.loads(output_path.read_text(encoding="utf-8"))
            partial_path = output_path.with_name(
                f"{output_path.name}.partial.jsonl"
            )
            self.assertFalse(partial_path.exists())

        self.assertIsInstance(output, list)
        self.assertEqual(output[0]["id"], "alpaca_eval_000001")
        self.assertEqual(output[0]["instruction"], "User instruction.")
        self.assertEqual(output[0]["output"], "Model response.")
        self.assertEqual(output[0]["generator"], "test-model")

    def test_run_flushes_each_record_to_partial_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "input.jsonl"
            output_path = directory_path / "output.json"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "requirement_id": "requirement_01",
                        "instruction": "User instruction.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                instructions=input_path,
                capability_card=CARD_PATH,
                model="test-model",
                host="http://localhost:11434",
                out=output_path,
                max_samples=None,
                temperature=0.0,
                top_p=1.0,
                thinking=False,
                timeout=10,
                max_retries=1,
                retry_sleep=0,
                dry_run=False,
            )
            partial_path = output_path.with_name(
                f"{output_path.name}.partial.jsonl"
            )

            def inspect_partial(**_: object) -> str:
                self.assertTrue(partial_path.exists())
                return "Model response."

            with patch(
                "finetune.eval.generate_alpaca.call_ollama",
                side_effect=inspect_partial,
            ):
                run(args)

            self.assertFalse(partial_path.exists())

    def test_failed_run_keeps_completed_records_in_partial_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "input.jsonl"
            output_path = directory_path / "output.json"
            input_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": source_id,
                            "requirement_id": "requirement_01",
                            "instruction": instruction,
                        }
                    )
                    for source_id, instruction in (
                        ("one", "First instruction."),
                        ("two", "Second instruction."),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                instructions=input_path,
                capability_card=CARD_PATH,
                model="test-model",
                host="http://localhost:11434",
                out=output_path,
                max_samples=None,
                temperature=0.0,
                top_p=1.0,
                thinking=False,
                timeout=10,
                max_retries=1,
                retry_sleep=0,
                dry_run=False,
            )

            with patch(
                "finetune.eval.generate_alpaca.call_ollama",
                side_effect=["First response.", RuntimeError("failed")],
            ):
                with self.assertRaisesRegex(RuntimeError, "Generation failed"):
                    run(args)

            partial_path = output_path.with_name(
                f"{output_path.name}.partial.jsonl"
            )
            partial_records = [
                json.loads(line)
                for line in partial_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertFalse(output_path.exists())
            self.assertEqual(len(partial_records), 1)
            self.assertEqual(partial_records[0]["id"], "alpaca_eval_000001")


if __name__ == "__main__":
    unittest.main()
