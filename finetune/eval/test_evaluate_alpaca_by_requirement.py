from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from .evaluate_alpaca_by_requirement import (
    forward_slash_path,
    pair_and_group,
    run,
)


METRIC_HEADER = (
    ",win_rate,standard_error,mode,avg_length,n_wins,n_wins_base,"
    "n_draws,n_total,discrete_win_rate,length_controlled_winrate,"
    "lc_standard_error\n"
)


def record(number: int, requirement_id: str, generator: str) -> dict[str, str]:
    return {
        "id": f"alpaca_eval_{number:06d}",
        "source_instruction_id": f"source_{number:06d}",
        "requirement_id": requirement_id,
        "requirement_title": f"Title {requirement_id}",
        "instruction": f"Instruction {number}",
        "output": f"Output {number}",
        "generator": generator,
    }


class EvaluateAlpacaByRequirementTest(unittest.TestCase):
    def test_forward_slash_path_normalizes_windows_paths(self) -> None:
        self.assertEqual(
            forward_slash_path(r"D:\models\outputs.json"),
            "D:/models/outputs.json",
        )

    def test_pair_and_group_is_independent_of_input_order(self) -> None:
        model = [record(2, "requirement_02", "model"), record(1, "requirement_01", "model")]
        reference = [record(1, "requirement_01", "reference"), record(2, "requirement_02", "reference")]
        groups = pair_and_group(model, reference)
        self.assertEqual(list(groups), ["requirement_01", "requirement_02"])
        self.assertEqual(groups["requirement_01"][0][0]["instruction"], "Instruction 1")

    def test_pair_and_group_rejects_instruction_mismatch(self) -> None:
        model = [record(1, "requirement_01", "model")]
        reference = [record(1, "requirement_01", "reference")]
        reference[0]["instruction"] = "Different"
        with self.assertRaisesRegex(ValueError, "Instruction mismatch"):
            pair_and_group(model, reference)

    def test_run_evaluates_every_requirement_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.json"
            reference_path = root / "reference.json"
            output_dir = root / "results"
            models = [
                record(1, "requirement_01", "candidate"),
                record(2, "requirement_02", "candidate"),
            ]
            references = [
                record(2, "requirement_02", "reference"),
                record(1, "requirement_01", "reference"),
            ]
            model_path.write_text(json.dumps(models), encoding="utf-8")
            reference_path.write_text(json.dumps(references), encoding="utf-8")
            commands: list[Sequence[str]] = []

            def fake_runner(command: Sequence[str]) -> None:
                commands.append(command)
                result_dir = Path(command[command.index("--output_path") + 1])
                (result_dir / "leaderboard.csv").write_text(
                    METRIC_HEADER
                    + "candidate,75.0,5.0,community,120,3,1,0,4,75.0,"
                    "70.0,6.0\n",
                    encoding="utf-8",
                )

            args = argparse.Namespace(
                model_outputs=model_path,
                reference_outputs=reference_path,
                annotators_config=r"configs\test_xiongmao",
                input_keys="requirement_title,requirement_description",
                output_dir=output_dir,
                alpaca_eval_command=r"D:\env\Scripts\alpaca_eval.exe",
            )
            summary_path = run(args, command_runner=fake_runner)
            with summary_path.open(encoding="utf-8", newline="") as source:
                summary = list(csv.DictReader(source))

        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertNotIn("\\", command[0])
            for flag in (
                "--model_outputs",
                "--reference_outputs",
                "--annotators_config",
                "--output_path",
            ):
                self.assertNotIn("\\", command[command.index(flag) + 1])
        self.assertEqual(
            [
                Path(command[command.index("--output_path") + 1]).name
                for command in commands
            ],
            ["requirement_01", "requirement_02"],
        )
        self.assertTrue(
            all(
                command[command.index("--input_keys") + 1]
                == "requirement_title,requirement_description"
                for command in commands
            )
        )
        self.assertEqual(
            [row["requirement_id"] for row in summary],
            ["requirement_01", "requirement_02"],
        )
        self.assertEqual(summary[0]["generator"], "candidate")
        self.assertEqual(summary[0]["win_rate"], "75.0")
        self.assertEqual(summary[0]["n_total"], "4")


if __name__ == "__main__":
    unittest.main()
