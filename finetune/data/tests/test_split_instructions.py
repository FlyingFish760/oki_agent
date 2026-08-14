from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from ..split_instructions import (
    SPLIT_NAMES,
    largest_remainder_counts,
    load_instruction_records,
    run,
    split_records,
    validate_ratios,
)


RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def make_records(per_requirement: int = 10) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    record_number = 1
    for requirement_number in (1, 2):
        for item_number in range(per_requirement):
            records.append(
                {
                    "id": f"instruction_{record_number:04d}",
                    "requirement_id": f"requirement_{requirement_number:02d}",
                    "instruction": (
                        f"Instruction {item_number} for requirement "
                        f"{requirement_number}."
                    ),
                }
            )
            record_number += 1
    return records


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class SplitInstructionsTest(unittest.TestCase):
    def test_largest_remainder_counts(self) -> None:
        self.assertEqual(
            largest_remainder_counts(10, RATIOS),
            {"train": 8, "valid": 1, "test": 1},
        )
        self.assertEqual(
            largest_remainder_counts(11, RATIOS),
            {"train": 9, "valid": 1, "test": 1},
        )

    def test_split_is_deterministic_disjoint_and_complete(self) -> None:
        records = make_records()

        first, counts = split_records(records, ratios=RATIOS, seed=42)
        second, _ = split_records(records, ratios=RATIOS, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(
            counts,
            {
                "requirement_01": {
                    "total": 10,
                    "train": 8,
                    "valid": 1,
                    "test": 1,
                },
                "requirement_02": {
                    "total": 10,
                    "train": 8,
                    "valid": 1,
                    "test": 1,
                },
            },
        )

        ids_by_split = {
            split_name: {record["id"] for record in first[split_name]}
            for split_name in SPLIT_NAMES
        }
        self.assertTrue(ids_by_split["train"].isdisjoint(ids_by_split["valid"]))
        self.assertTrue(ids_by_split["train"].isdisjoint(ids_by_split["test"]))
        self.assertTrue(ids_by_split["valid"].isdisjoint(ids_by_split["test"]))
        self.assertEqual(
            set().union(*ids_by_split.values()),
            {record["id"] for record in records},
        )
        for split_name, split_records_for_name in first.items():
            self.assertTrue(
                all(
                    record["split"] == split_name
                    for record in split_records_for_name
                )
            )

    def test_run_writes_three_files_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "instructions.jsonl"
            output_dir = directory_path / "splits"
            write_jsonl(input_path, make_records())
            args = argparse.Namespace(
                input=input_path,
                train_ratio=0.8,
                valid_ratio=0.1,
                test_ratio=0.1,
                seed=42,
                output_dir=output_dir,
                prefix="privacy",
                dry_run=False,
            )

            run(args)

            for split_name in SPLIT_NAMES:
                self.assertTrue(
                    (output_dir / f"privacy_{split_name}.jsonl").exists()
                )
            manifest = json.loads(
                (output_dir / "privacy_split_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            manifest["totals"],
            {"all": 20, "train": 16, "valid": 2, "test": 2},
        )
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["ratios"], RATIOS)

    def test_rejects_invalid_ratios(self) -> None:
        with self.assertRaisesRegex(ValueError, "must sum to 1.0"):
            validate_ratios({"train": 0.8, "valid": 0.2, "test": 0.2})
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_ratios({"train": 1.1, "valid": -0.1, "test": 0.0})

    def test_rejects_missing_fields_duplicate_ids_and_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)

            missing_path = directory_path / "missing.jsonl"
            write_jsonl(missing_path, [{"id": "one", "instruction": "Text"}])
            with self.assertRaisesRegex(ValueError, "requirement_id"):
                load_instruction_records(missing_path)

            duplicate_path = directory_path / "duplicate.jsonl"
            write_jsonl(
                duplicate_path,
                [
                    {
                        "id": "same",
                        "instruction": "First",
                        "requirement_id": "requirement_01",
                    },
                    {
                        "id": "same",
                        "instruction": "Second",
                        "requirement_id": "requirement_01",
                    },
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                load_instruction_records(duplicate_path)

            empty_path = directory_path / "empty.jsonl"
            empty_path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset is empty"):
                load_instruction_records(empty_path)

    def test_rejects_category_too_small_for_non_empty_splits(self) -> None:
        with self.assertRaisesRegex(ValueError, "Generate more records"):
            largest_remainder_counts(5, RATIOS)


if __name__ == "__main__":
    unittest.main()
