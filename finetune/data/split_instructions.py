"""Stratify instruction JSONL into train, valid, and test datasets."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from generation_common import write_json, write_jsonl
except ModuleNotFoundError:
    from .generation_common import write_json, write_jsonl


SPLIT_NAMES = ("train", "valid", "test")
RATIO_TOLERANCE = 1e-9


def load_instruction_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: each JSONL line must be an object"
                )

            for field_name in ("instruction", "requirement_id"):
                field_value = value.get(field_name)
                if not isinstance(field_value, str) or not field_value.strip():
                    raise ValueError(
                        f"{path}:{line_number}: {field_name!r} must be a "
                        "non-empty string"
                    )
                value[field_name] = field_value.strip()

            record_id = value.get("id")
            if record_id is not None:
                if not isinstance(record_id, str) or not record_id.strip():
                    raise ValueError(
                        f"{path}:{line_number}: 'id' must be a non-empty "
                        "string when present"
                    )
                record_id = record_id.strip()
                if record_id in seen_ids:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate id {record_id!r}"
                    )
                value["id"] = record_id
                seen_ids.add(record_id)

            records.append(value)

    if not records:
        raise ValueError(f"Instruction dataset is empty: {path}")
    return records


def validate_ratios(ratios: dict[str, float]) -> None:
    for split_name in SPLIT_NAMES:
        ratio = ratios[split_name]
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError(
                f"{split_name} ratio must be a finite non-negative number"
            )
    ratio_sum = sum(ratios.values())
    if not math.isclose(
        ratio_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=RATIO_TOLERANCE,
    ):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")


def largest_remainder_counts(
    total: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    validate_ratios(ratios)
    raw_counts = {
        split_name: total * ratios[split_name]
        for split_name in SPLIT_NAMES
    }
    counts = {
        split_name: math.floor(raw_counts[split_name])
        for split_name in SPLIT_NAMES
    }
    remaining = total - sum(counts.values())
    ranked_splits = sorted(
        SPLIT_NAMES,
        key=lambda split_name: (
            -(raw_counts[split_name] - counts[split_name]),
            SPLIT_NAMES.index(split_name),
        ),
    )
    for split_name in ranked_splits[:remaining]:
        counts[split_name] += 1

    missing = [
        split_name
        for split_name in SPLIT_NAMES
        if ratios[split_name] > 0 and counts[split_name] == 0
    ]
    if missing:
        raise ValueError(
            f"A category with {total} records cannot provide a non-empty "
            f"{', '.join(missing)} split at the requested ratios. Generate "
            "more records for this requirement."
        )
    return counts


def split_records(
    records: list[dict[str, Any]],
    *,
    ratios: dict[str, float],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    validate_ratios(ratios)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["requirement_id"]].append(record)

    rng = random.Random(seed)
    split_rows = {split_name: [] for split_name in SPLIT_NAMES}
    requirement_counts: dict[str, dict[str, int]] = {}

    for requirement_id in sorted(grouped):
        group = list(grouped[requirement_id])
        rng.shuffle(group)
        counts = largest_remainder_counts(len(group), ratios)
        requirement_counts[requirement_id] = {
            "total": len(group),
            **counts,
        }

        offset = 0
        for split_name in SPLIT_NAMES:
            next_offset = offset + counts[split_name]
            for record in group[offset:next_offset]:
                split_rows[split_name].append(
                    {**record, "split": split_name}
                )
            offset = next_offset

    return split_rows, requirement_counts


def output_paths(
    output_dir: Path,
    prefix: str,
) -> tuple[dict[str, Path], Path]:
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("--prefix must be a non-empty filename prefix")
    split_paths = {
        split_name: output_dir / f"{prefix}_{split_name}.jsonl"
        for split_name in SPLIT_NAMES
    }
    manifest_path = output_dir / f"{prefix}_split_manifest.json"
    return split_paths, manifest_path


def build_manifest(
    *,
    input_path: Path,
    seed: int,
    ratios: dict[str, float],
    split_rows: dict[str, list[dict[str, Any]]],
    requirement_counts: dict[str, dict[str, int]],
    split_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "input": str(input_path),
        "seed": seed,
        "ratios": ratios,
        "totals": {
            "all": sum(len(rows) for rows in split_rows.values()),
            **{
                split_name: len(split_rows[split_name])
                for split_name in SPLIT_NAMES
            },
        },
        "requirements": requirement_counts,
        "outputs": {
            split_name: str(split_paths[split_name])
            for split_name in SPLIT_NAMES
        },
    }


def print_summary(manifest: dict[str, Any]) -> None:
    print(
        "Split totals: "
        + ", ".join(
            f"{split_name}={manifest['totals'][split_name]}"
            for split_name in SPLIT_NAMES
        )
    )
    for requirement_id, counts in manifest["requirements"].items():
        print(
            f"{requirement_id}: total={counts['total']}, "
            + ", ".join(
                f"{split_name}={counts[split_name]}"
                for split_name in SPLIT_NAMES
            )
        )


def run(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Instruction JSONL not found: {args.input}")

    ratios = {
        "train": args.train_ratio,
        "valid": args.valid_ratio,
        "test": args.test_ratio,
    }
    records = load_instruction_records(args.input)
    split_rows, requirement_counts = split_records(
        records,
        ratios=ratios,
        seed=args.seed,
    )
    split_paths, manifest_path = output_paths(args.output_dir, args.prefix)
    manifest = build_manifest(
        input_path=args.input,
        seed=args.seed,
        ratios=ratios,
        split_rows=split_rows,
        requirement_counts=requirement_counts,
        split_paths=split_paths,
    )
    print_summary(manifest)

    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        write_jsonl(split_paths[split_name], split_rows[split_name])
    write_json(manifest_path, manifest)
    print(f"Split files written to {args.output_dir}")
    print(f"Manifest written to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shuffle instructions within each requirement category and split "
            "them into train, valid, and test JSONL datasets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Complete instruction JSONL dataset.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for split JSONL files and manifest.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Filename prefix for split outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print split counts without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
