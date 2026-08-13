"""Evaluate model outputs against references per capability requirement."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence


LEADERBOARD_METRICS = (
    "win_rate",
    "standard_error",
    "mode",
    "avg_length",
    "n_wins",
    "n_wins_base",
    "n_draws",
    "n_total",
    "discrete_win_rate",
    "length_controlled_winrate",
    "lc_standard_error",
)


def forward_slash_path(value: str | Path) -> str:
    """Serialize a path-like CLI value with forward slashes."""
    if isinstance(value, Path):
        return value.as_posix()
    return value.replace("\\", "/")


def load_outputs(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError(f"{label} must be a non-empty JSON array: {path}")

    required = ("instruction", "output", "requirement_id")
    for index, record in enumerate(data, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{label} record {index} must be a JSON object")
        for field in required:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{label} record {index} has invalid or missing {field!r}"
                )
    return data


def pairing_key(record: dict[str, Any]) -> tuple[str, str]:
    for field in ("source_instruction_id", "id"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return field, value.strip()
    return "instruction", record["instruction"].strip()


def index_unique(
    records: list[dict[str, Any]], *, label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = pairing_key(record)
        if key in indexed:
            raise ValueError(f"{label} contains duplicate pairing key {key}")
        indexed[key] = record
    return indexed


def pair_and_group(
    model_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    model_index = index_unique(model_records, label="model outputs")
    reference_index = index_unique(reference_records, label="reference outputs")
    if model_index.keys() != reference_index.keys():
        missing_reference = sorted(model_index.keys() - reference_index.keys())
        missing_model = sorted(reference_index.keys() - model_index.keys())
        raise ValueError(
            "Model and reference records do not have identical pairing keys; "
            f"missing reference={missing_reference[:5]}, "
            f"missing model={missing_model[:5]}"
        )

    grouped_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, model in model_index.items():
        reference = reference_index[key]
        model_instruction = model["instruction"].strip()
        reference_instruction = reference["instruction"].strip()
        if model_instruction != reference_instruction:
            raise ValueError(f"Instruction mismatch for pairing key {key}")

        model_requirement = model["requirement_id"].strip()
        reference_requirement = reference["requirement_id"].strip()
        if model_requirement != reference_requirement:
            raise ValueError(f"Requirement mismatch for pairing key {key}")
        grouped_model[model_requirement].append(model)
        grouped_reference[model_requirement].append(reference)

    return {
        requirement_id: (
            grouped_model[requirement_id],
            grouped_reference[requirement_id],
        )
        for requirement_id in sorted(grouped_model)
    }


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def default_command_runner(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def find_leaderboard(result_dir: Path) -> Path:
    direct = result_dir / "leaderboard.csv"
    if direct.exists():
        return direct
    candidates = list(result_dir.rglob("leaderboard.csv"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one AlpacaEval leaderboard under {result_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def read_model_metrics(path: Path, generator: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        model_column = reader.fieldnames[0] if reader.fieldnames else None
    if model_column is None:
        raise ValueError(f"Leaderboard has no model column: {path}")

    matching = [row for row in rows if row.get(model_column) == generator]
    if not matching and len(rows) == 1:
        matching = rows
    if len(matching) != 1:
        raise ValueError(
            f"Could not select one row for generator {generator!r} from {path}"
        )
    missing = [field for field in LEADERBOARD_METRICS if field not in matching[0]]
    if missing:
        raise ValueError(f"Leaderboard is missing metric columns: {missing}")
    return {field: matching[0].get(field, "") for field in LEADERBOARD_METRICS}


def requirement_title(records: list[dict[str, Any]]) -> str:
    titles = {
        value.strip()
        for record in records
        if isinstance((value := record.get("requirement_title")), str)
        and value.strip()
    }
    if len(titles) > 1:
        raise ValueError(f"Conflicting requirement titles: {sorted(titles)}")
    return next(iter(titles), "")


def run(
    args: argparse.Namespace,
    *,
    command_runner: Callable[[Sequence[str]], None] = default_command_runner,
) -> Path:
    model_records = load_outputs(args.model_outputs, label="model outputs")
    reference_records = load_outputs(
        args.reference_outputs, label="reference outputs"
    )
    groups = pair_and_group(model_records, reference_records)

    generators = {
        record["generator"].strip()
        for record in model_records
        if isinstance(record.get("generator"), str)
        and record["generator"].strip()
    }
    if len(generators) != 1:
        raise ValueError(
            "Model outputs must contain exactly one non-empty generator; "
            f"found {sorted(generators)}"
        )
    generator = next(iter(generators))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []
    for index, (requirement_id, (models, references)) in enumerate(
        groups.items(), start=1
    ):
        result_dir = args.output_dir / requirement_id
        if result_dir.exists() and any(result_dir.iterdir()):
            raise FileExistsError(
                f"Result directory is not empty: {result_dir}. Choose a new "
                "--output-dir or remove the previous results explicitly."
            )
        result_dir.mkdir(parents=True, exist_ok=True)
        model_path = result_dir / "model_outputs.json"
        reference_path = result_dir / "reference_outputs.json"
        write_json(model_path, models)
        write_json(reference_path, references)

        command = [
            forward_slash_path(args.alpaca_eval_command),
            "evaluate",
            "--model_outputs",
            forward_slash_path(model_path),
            "--reference_outputs",
            forward_slash_path(reference_path),
            "--annotators_config",
            forward_slash_path(args.annotators_config),
            "--input_keys",
            args.input_keys,
            "--output_path",
            forward_slash_path(result_dir),
        ]
        print(
            f"[{index}/{len(groups)}] Evaluating {requirement_id} "
            f"({len(models)} samples)"
        )
        command_runner(command)
        metrics = read_model_metrics(find_leaderboard(result_dir), generator)
        summary_rows.append(
            {
                "requirement_id": requirement_id,
                "requirement_title": requirement_title(models),
                "generator": generator,
                **metrics,
            }
        )

    summary_path = args.output_dir / "leaderboard.csv"
    fieldnames = [
        "requirement_id",
        "requirement_title",
        "generator",
        *LEADERBOARD_METRICS,
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Evaluation complete: {forward_slash_path(summary_path)}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AlpacaEval separately for every requirement category and "
            "write a requirement-level leaderboard."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-outputs", type=Path, required=True)
    parser.add_argument("--reference-outputs", type=Path, required=True)
    parser.add_argument("--annotators-config", required=True)
    parser.add_argument(
        "--input-keys",
        default="instruction,requirement_title,requirement_description",
        help=(
            "Comma-separated input field names passed to AlpacaEval as "
            "--input_keys."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("finetune/eval/results/by_requirement"),
    )
    parser.add_argument(
        "--alpaca-eval-command",
        default="alpaca_eval",
        help="AlpacaEval executable name or path.",
    )
    args = parser.parse_args()
    if not args.input_keys.strip():
        parser.error("--input-keys must not be empty")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
