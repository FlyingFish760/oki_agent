"""Generate English conversational SFT data from open-source user prompts."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from generation_common import (
        TeacherPrompt,
        add_teacher_arguments,
        append_jsonl,
        call_teacher,
        clean_text,
        configure_teacher_args,
        is_mostly_english,
        prompt_template_name,
        write_jsonl,
    )
except ModuleNotFoundError:
    from .generation_common import (
        TeacherPrompt,
        add_teacher_arguments,
        append_jsonl,
        call_teacher,
        clean_text,
        configure_teacher_args,
        is_mostly_english,
        prompt_template_name,
        write_jsonl,
    )


DEFAULT_PERSONA_PROFILE = Path("finetune/data/persona_profile-v2.1-en.md")
DEFAULT_PROMPT_TEMPLATE = Path(
    "finetune/data/templates/teacher_prompt_template.txt"
)
DATASET_MIX = [("OpenAssistant/oasst1", 1.0)]

BAD_ASSISTANT_PATTERNS = [
    "as an ai language model",
    "as a language model",
    "i am an ai",
    "i'm an ai",
    "i am only an ai",
    "happy to assist",
    "happy to help",
    "feel free to contact me",
    "please don't hesitate",
]

UNSAFE_SOURCE_PATTERNS = [
    "delete all",
    "remove all",
    "password",
    "hack",
    "bypass",
    "pirate",
    "crack",
    "malware",
    "phishing",
    "steal",
    "private key",
    "token",
]


@dataclass(frozen=True)
class SourcePrompt:
    source_dataset: str
    text: str


@dataclass
class GenerationStats:
    accepted_train: int = 0
    accepted_eval: int = 0
    rejected: int = 0
    teacher_failed: int = 0


def _is_source_candidate(text: str) -> tuple[bool, str]:
    text = clean_text(text)
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if len(words) < 8:
        return False, "too_short"
    if len(words) > 220:
        return False, "too_long"
    if not is_mostly_english(text):
        return False, "not_english"

    lowered = text.lower()
    if any(pattern in lowered for pattern in UNSAFE_SOURCE_PATTERNS):
        return False, "unsafe_or_sensitive"
    if "http://" in lowered or "https://" in lowered:
        return False, "contains_url"
    return True, "ok"


def _load_dataset(name: str, split: str, cache_dir: Path | None = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install finetune dependencies first: uv sync --extra finetune"
        ) from exc
    return load_dataset(
        name,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def _iter_oasst(
    max_source_rows: int,
    cache_dir: Path | None = None,
) -> Iterable[SourcePrompt]:
    dataset = _load_dataset("OpenAssistant/oasst1", "train", cache_dir)
    for index, row in enumerate(dataset):
        if index >= max_source_rows:
            break
        if row.get("lang") not in {None, "en"}:
            continue
        if row.get("role") != "prompter":
            continue
        text = clean_text(str(row.get("text", "")))
        accepted, _ = _is_source_candidate(text)
        if accepted:
            yield SourcePrompt("OpenAssistant/oasst1", text)


def _collect_source_prompts(args: argparse.Namespace) -> list[SourcePrompt]:
    target = args.n_train + args.n_eval
    by_dataset: dict[str, list[SourcePrompt]] = {
        name: [] for name, _ in DATASET_MIX
    }
    loaders = {"OpenAssistant/oasst1": _iter_oasst}

    for name, ratio in DATASET_MIX:
        desired = max(1, round(target * ratio * args.source_oversample))
        rows: list[SourcePrompt] = []
        for prompt in loaders[name](args.max_source_rows, args.cache_dir):
            rows.append(prompt)
            if len(rows) >= desired:
                break
        by_dataset[name] = rows

    selected: list[SourcePrompt] = []
    seen: set[str] = set()
    for name, ratio in DATASET_MIX:
        desired = max(1, round(target * ratio))
        rows = by_dataset[name]
        random.shuffle(rows)
        for prompt in rows:
            key = prompt.text.lower()
            if key in seen:
                continue
            seen.add(key)
            selected.append(prompt)
            selected_for_dataset = sum(
                item.source_dataset == name for item in selected
            )
            if selected_for_dataset >= desired:
                break

    leftovers = [
        prompt
        for rows in by_dataset.values()
        for prompt in rows
        if prompt.text.lower() not in seen
    ]
    random.shuffle(leftovers)
    for prompt in leftovers:
        if len(selected) >= target * args.source_oversample:
            break
        key = prompt.text.lower()
        if key not in seen:
            selected.append(prompt)
            seen.add(key)

    random.shuffle(selected)
    if len(selected) < target:
        raise RuntimeError(
            f"Only collected {len(selected)} source prompts, but need {target}. "
            "Increase --max-source-rows or relax filters."
        )
    return selected


def _profile_for_prompt(profile: str, max_chars: int) -> str:
    profile = re.sub(r"<!--.*?-->", "", profile, flags=re.DOTALL)
    profile = re.sub(r"\n{3,}", "\n\n", profile).strip()
    if len(profile) <= max_chars:
        return profile
    return profile[:max_chars].rsplit(" ", 1)[0].strip()


def _build_teacher_prompt(
    template: str,
    persona_profile: str,
    user_instruction: str,
) -> TeacherPrompt:
    rendered = template.format(
        persona_profile=persona_profile,
        user_instruction=user_instruction,
    )
    marker = "[SOURCE USER MESSAGE]"
    if marker not in rendered:
        raise ValueError(
            f"Prompt template must contain {marker} to split system and user prompts."
        )

    system_prompt, user_prompt = rendered.split(marker, 1)
    user_prompt = clean_text(user_prompt)
    if not system_prompt.strip() or not user_prompt:
        raise ValueError(
            "Prompt template must include non-empty system and user sections "
            f"around {marker}."
        )
    return TeacherPrompt(system=system_prompt.strip(), user=user_prompt)


def _assistant_from_sample(sample: dict[str, Any]) -> str:
    assistant = sample.get("assistant")
    return clean_text(str(assistant)) if isinstance(assistant, str) else ""


def _validate_assistant(
    assistant: str,
    max_assistant_words: int,
) -> tuple[bool, str]:
    if not assistant:
        return False, "empty_assistant"
    if "<think>" in assistant.lower() or "</think>" in assistant.lower():
        return False, "thinking_trace"
    if not is_mostly_english(assistant):
        return False, "assistant_not_english"
    lowered = assistant.lower()
    if any(pattern in lowered for pattern in BAD_ASSISTANT_PATTERNS):
        return False, "bad_assistant_phrase"
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", assistant)
    if len(words) > max_assistant_words:
        return False, "assistant_too_verbose"
    return True, "ok"


def _sft_record(
    record_id: int,
    source: SourcePrompt,
    assistant: str,
    persona_profile: Path,
    prompt_template: Path,
    teacher_model: str,
) -> dict[str, Any]:
    return {
        "id": f"instruction_en_{record_id:06d}",
        "source_dataset": source.source_dataset,
        "persona_profile": persona_profile.name,
        "prompt_template": prompt_template_name(prompt_template),
        "teacher_model": teacher_model,
        "messages": [
            {"role": "user", "content": source.text},
            {"role": "assistant", "content": assistant},
        ],
    }


def run(args: argparse.Namespace) -> None:
    if not args.persona_profile.exists():
        raise FileNotFoundError(
            f"Persona profile not found: {args.persona_profile}"
        )
    if not args.prompt_template.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {args.prompt_template}"
        )

    random.seed(args.seed)
    profile_text = args.persona_profile.read_text(encoding="utf-8")
    persona_profile = _profile_for_prompt(
        profile_text,
        args.persona_profile_max_chars,
    )
    prompt_template = args.prompt_template.read_text(encoding="utf-8")
    sources = _collect_source_prompts(args)
    random.shuffle(sources)

    if args.dry_run_sources:
        rows = (
            {
                "id": f"instruction_en_source_{index:06d}",
                "source_dataset": source.source_dataset,
                "user_instruction": source.text,
            }
            for index, source in enumerate(
                sources[: args.n_train + args.n_eval],
                start=1,
            )
        )
        written = write_jsonl(args.source_preview_out, rows)
        print(
            f"Source dry-run complete: wrote {written} prompts to "
            f"{args.source_preview_out}"
        )
        return

    for path in (args.out, args.eval_out, args.rejected_out):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    stats = GenerationStats()
    record_id = 1
    max_attempts = min(
        len(sources),
        (args.n_train + args.n_eval) * args.max_attempts_multiplier,
    )
    for source in sources[:max_attempts]:
        if (
            stats.accepted_train >= args.n_train
            and stats.accepted_eval >= args.n_eval
        ):
            break
        target_path = (
            args.out
            if stats.accepted_train < args.n_train
            else args.eval_out
        )
        prompt = _build_teacher_prompt(
            prompt_template,
            persona_profile,
            source.text,
        )
        try:
            sample = call_teacher(prompt, args)
        except (
            RuntimeError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            stats.teacher_failed += 1
            append_jsonl(
                args.rejected_out,
                {
                    "source_dataset": source.source_dataset,
                    "user_instruction": source.text,
                    "reason": "teacher_failed",
                    "error": str(exc),
                },
            )
            time.sleep(args.retry_sleep)
            continue

        if sample is None:
            stats.rejected += 1
            append_jsonl(
                args.rejected_out,
                {
                    "source_dataset": source.source_dataset,
                    "user_instruction": source.text,
                    "reason": "invalid_json",
                },
            )
            continue

        assistant = _assistant_from_sample(sample)
        accepted, reason = _validate_assistant(
            assistant,
            args.max_assistant_words,
        )
        if not accepted:
            stats.rejected += 1
            append_jsonl(
                args.rejected_out,
                {
                    "source_dataset": source.source_dataset,
                    "user_instruction": source.text,
                    "reason": reason,
                    "sample": sample,
                },
            )
            continue

        append_jsonl(
            target_path,
            _sft_record(
                record_id,
                source,
                assistant,
                args.persona_profile,
                args.prompt_template,
                args.model,
            ),
        )
        record_id += 1
        if target_path == args.out:
            stats.accepted_train += 1
        else:
            stats.accepted_eval += 1

    print(
        "SFT generation complete: "
        f"train={stats.accepted_train}/{args.n_train} "
        f"eval={stats.accepted_eval}/{args.n_eval} "
        f"rejected={stats.rejected} "
        f"teacher_failed={stats.teacher_failed}"
    )
    if (
        stats.accepted_train < args.n_train
        or stats.accepted_eval < args.n_eval
    ):
        raise RuntimeError(
            "Generation ended before target counts were reached. "
            "See --rejected-out for details."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate English conversational SFT data from OpenAssistant/oasst1 "
            "user prompts and teacher-generated oki replies."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--persona-profile",
        type=Path,
        default=DEFAULT_PERSONA_PROFILE,
        help="Persona profile inserted into the teacher system prompt.",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Template containing {persona_profile} and {user_instruction}.",
    )
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("finetune/data/datasets/instruction_en_train.jsonl"),
    )
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=Path("finetune/data/datasets/instruction_en_eval.jsonl"),
    )
    parser.add_argument(
        "--rejected-out",
        type=Path,
        default=Path("finetune/data/datasets/instruction_en_rejected.jsonl"),
    )
    parser.add_argument(
        "--dry-run-sources",
        action="store_true",
        help="Sample source prompts without calling the teacher.",
    )
    parser.add_argument(
        "--source-preview-out",
        type=Path,
        default=Path(
            "finetune/data/datasets/instruction_en_source_preview.jsonl"
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-source-rows", type=int, default=20000)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--source-oversample", type=int, default=4)
    parser.add_argument("--max-attempts-multiplier", type=int, default=8)
    parser.add_argument("--max-assistant-words", type=int, default=180)
    parser.add_argument("--persona-profile-max-chars", type=int, default=3500)
    parser.add_argument("--retry-sleep", type=float, default=0.3)
    add_teacher_arguments(parser)

    args = parser.parse_args(argv)
    if args.n_train < 0 or args.n_eval < 0:
        parser.error("--n-train and --n-eval must be non-negative")
    if args.n_train + args.n_eval == 0:
        parser.error("At least one of --n-train or --n-eval must be positive")
    configure_teacher_args(args, require_model=not args.dry_run_sources)
    return args


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
