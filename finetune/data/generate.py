"""Day2 data generation.

This script has two modes:

1. Legacy persona mode, kept for the original project skeleton:

    python finetune/data/generate.py --mode persona --n 500

2. English instruction mode, based on open-source English datasets and a
   standalone teacher prompt template:

    python finetune/data/generate.py \
        --persona-profile finetune/data/persona_profile-v2.1-en.md \
        --mode instruction-en \
        --n-train 300 \
        --n-eval 50 \
        --teacher openai \
        --model <teacher-model-name> \
        --out finetune/data/datasets/instruction_en_train.jsonl \
        --eval-out finetune/data/datasets/instruction_en_eval.jsonl

The instruction-en mode uses datasets only as sources of English user intent. It does
not train on original assistant answers, does not translate, does not localize,
and does not add category labels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEGACY_PERSONA_CARD = Path(__file__).resolve().parent.parent / "persona_card-v2.0.md"
INSTRUCTION_PERSONA_PROFILE = Path("finetune/data/persona_profile-v2.1-en.md")
TEACHER_PROMPT_TEMPLATE = Path("finetune/data/templates/teacher_prompt_template.txt")

DATASET_MIX = [
    ("OpenAssistant/oasst1", 1.0),
]

INTENTS = [
    "闲聊",
    "任务执行",
    "拒绝不合理请求",
    "情绪安抚",
    "危险操作确认",
    "承认不知道",
    "多轮追问",
    "工具调用",
]

LEGACY_GEN_SYSTEM = """你在为一个名叫 oki 的本地助手生成人格训练对话。
严格遵循下面的人格卡，生成一段自然的多轮对话（2~4 轮）。
输出 JSON: {{"messages": [{{"role": "user|assistant", "content": "..."}}, ...]}}
人格卡：
{card}
本条对话的意图：{intent}
"""

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

UNSAFE_PATTERNS = [
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


@dataclass(frozen=True)
class TeacherPrompt:
    system: str
    user: str


@dataclass
class GenerationStats:
    accepted_train: int = 0
    accepted_eval: int = 0
    rejected: int = 0
    teacher_failed: int = 0


def _strip_thinking(messages: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    if mode == "keep":
        return messages
    out = []
    for msg in messages:
        content = msg["content"]
        if msg["role"] == "assistant":
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        out.append({**msg, "content": content})
    return out


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_strict_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text or text.startswith("```"):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {error_body}") from exc


def _extract_openai_response_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text

    chunks: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "".join(chunks)


def _openai_input_text(prompt: TeacherPrompt) -> str:
    return f"{prompt.user}\n\nReturn JSON only."


def _call_openai(prompt: TeacherPrompt, args: argparse.Namespace) -> dict[str, Any] | None:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --teacher openai")

    payload = {
        "model": args.model,
        "instructions": prompt.system,
        "input": _openai_input_text(prompt),
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.json_response_format:
        payload["text"] = {"format": {"type": "json_object"}}
    data = _post_json(
        f"{args.base_url.rstrip('/')}/responses",
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    content = _extract_openai_response_text(data)
    return _parse_strict_json(content)


def _call_ollama(prompt: TeacherPrompt, args: argparse.Namespace) -> dict[str, Any] | None:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": args.temperature, "top_p": args.top_p},
    }
    data = _post_json(f"{args.base_url.rstrip('/')}/api/chat", payload)
    content = data.get("message", {}).get("content", "")
    return _parse_strict_json(content)


def _call_gemini(prompt: TeacherPrompt, args: argparse.Namespace) -> dict[str, Any] | None:
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for --teacher gemini")

    model = args.model if args.model.startswith("models/") else f"models/{args.model}"
    payload: dict[str, Any] = {
        "systemInstruction": {
            "parts": [{"text": prompt.system}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt.user}],
            }
        ],
        "generationConfig": {
            "temperature": args.temperature,
            "topP": args.top_p,
        },
    }
    if args.json_response_format:
        payload["generationConfig"]["response_mime_type"] = "application/json"

    data = _post_json(f"{args.base_url.rstrip('/')}/{model}:generateContent?key={api_key}", payload)
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    content = "".join(str(part.get("text", "")) for part in parts)
    return _parse_strict_json(content)


def _call_teacher(prompt: TeacherPrompt, args: argparse.Namespace) -> dict[str, Any] | None:
    if args.teacher == "openai":
        return _call_openai(prompt, args)
    if args.teacher == "ollama":
        return _call_ollama(prompt, args)
    if args.teacher == "gemini":
        return _call_gemini(prompt, args)
    raise ValueError(f"unknown teacher: {args.teacher}")


def _call_legacy_teacher(system: str) -> dict[str, Any] | None:
    """Legacy hook for the original persona generation flow."""
    raise NotImplementedError("请实现 _call_legacy_teacher：接入 teacher 模型。")


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("`\"' ")
    return text


def _is_mostly_english(text: str) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    if letters == 0:
        return False
    ascii_letters = sum(("a" <= ch.lower() <= "z") for ch in text)
    return ascii_letters / letters >= 0.85


def _is_source_candidate(text: str) -> tuple[bool, str]:
    text = _clean_text(text)
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if len(words) < 8:
        return False, "too_short"
    if len(words) > 220:
        return False, "too_long"
    if not _is_mostly_english(text):
        return False, "not_english"

    low = text.lower()
    if any(pattern in low for pattern in UNSAFE_PATTERNS):
        return False, "unsafe_or_sensitive"
    if "http://" in low or "https://" in low:
        return False, "contains_url"
    return True, "ok"


def _load_dataset_safe(name: str, split: str, cache_dir: Path | None = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install finetune dependencies first: uv sync --extra finetune") from exc
    return load_dataset(name, split=split, cache_dir=str(cache_dir) if cache_dir else None)


def _iter_oasst(max_source_rows: int, cache_dir: Path | None = None) -> Iterable[SourcePrompt]:
    ds = _load_dataset_safe("OpenAssistant/oasst1", "train", cache_dir)
    for idx, row in enumerate(ds):
        if idx >= max_source_rows:
            break
        if row.get("lang") not in {None, "en"}:
            continue
        if row.get("role") != "prompter":
            continue
        text = _clean_text(str(row.get("text", "")))
        ok, _ = _is_source_candidate(text)
        if ok:
            yield SourcePrompt("OpenAssistant/oasst1", text)

def _collect_source_prompts(args: argparse.Namespace) -> list[SourcePrompt]:
    target = args.n_train + args.n_eval
    by_dataset: dict[str, list[SourcePrompt]] = {name: [] for name, _ in DATASET_MIX}
    loaders = {
        "OpenAssistant/oasst1": _iter_oasst,
    }

    for name, ratio in DATASET_MIX:
        desired = max(1, round(target * ratio * args.source_oversample))
        rows = []
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
            if sum(1 for item in selected if item.source_dataset == name) >= desired:
                break

    leftovers = [p for rows in by_dataset.values() for p in rows if p.text.lower() not in seen]
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


def _persona_profile_for_prompt(profile: str, max_chars: int) -> str:
    profile = re.sub(r"<!--.*?-->", "", profile, flags=re.DOTALL)
    profile = re.sub(r"\n{3,}", "\n\n", profile).strip()
    if len(profile) <= max_chars:
        return profile
    return profile[:max_chars].rsplit(" ", 1)[0].strip()


def _prompt_template_name(path: Path) -> str:
    return path.stem


def _build_teacher_prompt(template: str, persona_profile: str, user_instruction: str) -> TeacherPrompt:
    rendered = template.format(
        persona_profile=persona_profile,
        user_instruction=user_instruction,
    )
    marker = "[SOURCE USER MESSAGE]"
    if marker not in rendered:
        raise ValueError(f"Prompt template must contain {marker} to split system and user prompts.")

    system_prompt, user_prompt = rendered.split(marker, 1)
    user_prompt = _clean_text(user_prompt)
    if not system_prompt.strip() or not user_prompt:
        raise ValueError(f"Prompt template must include non-empty system and user sections around {marker}.")
    return TeacherPrompt(system=system_prompt.strip(), user=user_prompt)


def _assistant_from_teacher_sample(sample: dict[str, Any]) -> str:
    assistant = sample.get("assistant")
    return _clean_text(str(assistant)) if isinstance(assistant, str) else ""


def _validate_assistant(assistant: str, max_assistant_words: int) -> tuple[bool, str]:
    if not assistant:
        return False, "empty_assistant"
    if "<think>" in assistant.lower() or "</think>" in assistant.lower():
        return False, "thinking_trace"
    if not _is_mostly_english(assistant):
        return False, "assistant_not_english"
    low = assistant.lower()
    if any(pattern in low for pattern in BAD_ASSISTANT_PATTERNS):
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
) -> dict[str, Any]:
    return {
        "id": f"instruction_en_{record_id:06d}",
        "source_dataset": source.source_dataset,
        "persona_profile": persona_profile.name,
        "prompt_template": _prompt_template_name(prompt_template),
        "messages": [
            {"role": "user", "content": source.text},
            {"role": "assistant", "content": assistant},
        ],
    }


def run_instruction_en(args: argparse.Namespace) -> None:
    if not args.persona_profile.exists():
        raise FileNotFoundError(
            f"Persona profile not found: {args.persona_profile}. "
            "Create or pass an English persona profile before running instruction-en generation."
        )
    if not args.prompt_template.exists():
        raise FileNotFoundError(f"Prompt template not found: {args.prompt_template}")
    random.seed(args.seed)

    profile = args.persona_profile.read_text(encoding="utf-8")
    persona_profile = _persona_profile_for_prompt(profile, args.persona_profile_max_chars)
    prompt_template = args.prompt_template.read_text(encoding="utf-8")
    sources = _collect_source_prompts(args)
    random.shuffle(sources)

    if args.dry_run_sources:
        preview_rows = (
            {
                "id": f"instruction_en_source_{idx:06d}",
                "source_dataset": source.source_dataset,
                "user_instruction": source.text,
            }
            for idx, source in enumerate(sources[: args.n_train + args.n_eval], start=1)
        )
        written = _write_jsonl(args.source_preview_out, preview_rows)
        print(f"source dry-run complete: wrote {written} prompts to {args.source_preview_out}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    args.rejected_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("", encoding="utf-8")
    args.eval_out.write_text("", encoding="utf-8")
    args.rejected_out.write_text("", encoding="utf-8")

    stats = GenerationStats()
    record_id = 1
    max_attempts = min(len(sources), (args.n_train + args.n_eval) * args.max_attempts_multiplier)
    for source in sources[:max_attempts]:
        target_path = args.out if stats.accepted_train < args.n_train else args.eval_out
        if stats.accepted_train >= args.n_train and stats.accepted_eval >= args.n_eval:
            break

        prompt = _build_teacher_prompt(prompt_template, persona_profile, source.text)
        try:
            sample = _call_teacher(prompt, args)
        except (RuntimeError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            stats.teacher_failed += 1
            _append_jsonl(
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
            _append_jsonl(
                args.rejected_out,
                {
                    "source_dataset": source.source_dataset,
                    "user_instruction": source.text,
                    "reason": "invalid_json",
                },
            )
            continue

        assistant = _assistant_from_teacher_sample(sample)
        ok, reason = _validate_assistant(assistant, args.max_assistant_words)
        if not ok:
            stats.rejected += 1
            _append_jsonl(
                args.rejected_out,
                {
                    "source_dataset": source.source_dataset,
                    "user_instruction": source.text,
                    "reason": reason,
                    "sample": sample,
                },
            )
            continue

        record = _sft_record(record_id, source, assistant, args.persona_profile, args.prompt_template)
        _append_jsonl(target_path, record)
        record_id += 1
        if target_path == args.out:
            stats.accepted_train += 1
        else:
            stats.accepted_eval += 1

    print(
        "instruction-en generation complete: "
        f"train={stats.accepted_train}/{args.n_train} "
        f"eval={stats.accepted_eval}/{args.n_eval} "
        f"rejected={stats.rejected} teacher_failed={stats.teacher_failed} "
        f"out={args.out} eval_out={args.eval_out} rejected_out={args.rejected_out}"
    )
    if stats.accepted_train < args.n_train or stats.accepted_eval < args.n_eval:
        raise RuntimeError("Generation ended before target counts were reached. See rejected_out for reasons.")


def run_legacy_persona(args: argparse.Namespace) -> None:
    card = LEGACY_PERSONA_CARD.read_text(encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for _ in range(args.n):
            intent = random.choice(INTENTS)
            sample = _call_legacy_teacher(LEGACY_GEN_SYSTEM.format(card=card, intent=intent))
            if not sample:
                continue
            sample["messages"] = _strip_thinking(sample["messages"], args.thinking)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
    print(f"已生成 {written} 条到 {args.out}。记得人工筛一遍，剔除跑偏样本。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SFT JSONL data for oki. The recommended Day2 path is "
            "`--mode instruction-en`: sample English user prompts from open-source datasets, "
            "then use a teacher model plus the persona profile to generate oki replies."
        ),
        epilog="""Examples:
  1) Preview English source prompts only, without calling a teacher:
     python finetune/data/generate.py --mode instruction-en --dry-run-sources --n-train 10 --n-eval 5

  2) Generate the default 300 train / 50 eval English instruction dataset with the official OpenAI Responses API:
     python finetune/data/generate.py --mode instruction-en --persona-profile finetune/data/persona_profile-v2.1-en.md --prompt-template finetune/data/templates/teacher_prompt_template.txt --teacher openai --model <teacher-model-name>

  3) Generate with Gemini API:
     python finetune/data/generate.py --mode instruction-en --teacher gemini --model gemini-2.0-flash

  4) Generate with a local Ollama teacher:
     python finetune/data/generate.py --mode instruction-en --teacher ollama --model <ollama-model-name>

  5) Legacy persona generator entrypoint, kept for the original skeleton:
     python finetune/data/generate.py --mode persona --n 500 --thinking strip

Outputs for instruction-en:
  train:    finetune/data/datasets/instruction_en_train.jsonl
  eval:     finetune/data/datasets/instruction_en_eval.jsonl
  rejected: finetune/data/datasets/instruction_en_rejected.jsonl

Source mix for instruction-en:
  OpenAssistant/oasst1: 100%

Notes:
  - instruction-en keeps source text in English; it does not translate or localize.
  - Source datasets provide only user intent/context. Original assistant answers are not used.
  - The teacher must return strict JSON with an assistant reply; generate.py builds the final messages.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["persona", "instruction-en"],
        default="persona",
        help="Generation mode. Use instruction-en for the current English instruction pipeline. Default: %(default)s.",
    )

    # Legacy persona mode.
    legacy = parser.add_argument_group("legacy persona mode")
    legacy.add_argument(
        "--n",
        type=int,
        default=500,
        help="Number of samples to generate in legacy persona mode. Default: %(default)s.",
    )
    legacy.add_argument(
        "--thinking",
        choices=["strip", "keep"],
        default="strip",
        help="How to handle assistant <think> traces in legacy persona mode. Default: %(default)s.",
    )

    # English instruction mode.
    instruction = parser.add_argument_group("instruction-en data settings")
    instruction.add_argument(
        "--persona-profile",
        type=Path,
        default=INSTRUCTION_PERSONA_PROFILE,
        help="English persona profile inserted into the prompt template. Default: %(default)s.",
    )
    instruction.add_argument(
        "--prompt-template",
        type=Path,
        default=TEACHER_PROMPT_TEMPLATE,
        help=(
            "Prompt template file for instruction-en generation. It must contain "
            "{persona_profile}, {user_instruction}, and [SOURCE USER MESSAGE]. "
            "Default: %(default)s."
        ),
    )
    instruction.add_argument(
        "--n-train",
        type=int,
        default=300,
        help="Accepted training samples to write in instruction-en mode. Default: %(default)s.",
    )
    instruction.add_argument(
        "--n-eval",
        type=int,
        default=50,
        help="Accepted held-out evaluation samples to write in instruction-en mode. Default: %(default)s.",
    )
    instruction.add_argument(
        "--out",
        type=Path,
        default=Path("finetune/data/datasets/persona.jsonl"),
        help=(
            "Training JSONL output path. In instruction-en mode, the default is remapped to "
            "finetune/data/datasets/instruction_en_train.jsonl."
        ),
    )
    instruction.add_argument(
        "--eval-out",
        type=Path,
        default=Path("finetune/data/datasets/instruction_en_eval.jsonl"),
        help="Held-out evaluation JSONL output path. Default: %(default)s.",
    )
    instruction.add_argument(
        "--rejected-out",
        type=Path,
        default=Path("finetune/data/datasets/instruction_en_rejected.jsonl"),
        help="Rejected/failed generation records JSONL path. Default: %(default)s.",
    )
    instruction.add_argument(
        "--dry-run-sources",
        action="store_true",
        help="Only sample and filter source prompts; do not call the teacher or write train/eval files.",
    )
    instruction.add_argument(
        "--source-preview-out",
        type=Path,
        default=Path("finetune/data/datasets/instruction_en_source_preview.jsonl"),
        help="JSONL output for --dry-run-sources preview prompts. Default: %(default)s.",
    )

    teacher = parser.add_argument_group("teacher backend")
    teacher.add_argument(
        "--teacher",
        choices=["openai", "gemini", "ollama"],
        default="openai",
        help="Teacher backend used to generate assistant replies. Default: %(default)s.",
    )
    teacher.add_argument(
        "--model",
        default="",
        help="Teacher model name. Required in instruction-en mode unless --dry-run-sources is set.",
    )
    teacher.add_argument(
        "--base-url",
        default="",
        help=(
            "Teacher API base URL. Defaults to https://api.openai.com/v1 for openai, "
            "https://generativelanguage.googleapis.com/v1beta for gemini, "
            "and http://127.0.0.1:11434 for ollama."
        ),
    )
    teacher.add_argument(
        "--api-key",
        default="",
        help=(
            "API key for openai or gemini teacher. If omitted, "
            "OPENAI_API_KEY or GEMINI_API_KEY is used based on --teacher."
        ),
    )
    teacher.add_argument(
        "--json-response-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Request JSON output from OpenAI Responses API or Gemini. Use "
            "--no-json-response-format for models/endpoints that do not support it. Default: %(default)s."
        ),
    )
    teacher.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Teacher sampling temperature. Default: %(default)s.",
    )
    teacher.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Teacher nucleus sampling top_p. Default: %(default)s.",
    )
    teacher.add_argument(
        "--retry-sleep",
        type=float,
        default=0.3,
        help="Seconds to sleep after a teacher/API failure before continuing. Default: %(default)s.",
    )

    filtering = parser.add_argument_group("source sampling and filters")
    filtering.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for source sampling and train/eval ordering. Default: %(default)s.",
    )
    filtering.add_argument(
        "--max-source-rows",
        type=int,
        default=20000,
        help="Maximum rows scanned per source dataset before filtering. Default: %(default)s.",
    )
    filtering.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory used by Hugging Face datasets for downloads/cache. "
            "If omitted, datasets uses its default cache location."
        ),
    )
    filtering.add_argument(
        "--source-oversample",
        type=int,
        default=4,
        help="Collect this many times the target count before final source sampling. Default: %(default)s.",
    )
    filtering.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=8,
        help="Maximum teacher attempts as a multiple of n_train+n_eval. Default: %(default)s.",
    )
    filtering.add_argument(
        "--max-assistant-words",
        type=int,
        default=180,
        help="Reject teacher replies longer than this many English words. Default: %(default)s.",
    )
    filtering.add_argument(
        "--persona-profile-max-chars",
        type=int,
        default=3500,
        help="Maximum characters from the persona profile inserted into the prompt template. Default: %(default)s.",
    )
    args = parser.parse_args()

    if args.mode == "instruction-en":
        if args.out == Path("finetune/data/datasets/persona.jsonl"):
            args.out = Path("finetune/data/datasets/instruction_en_train.jsonl")
        if not args.model and not args.dry_run_sources:
            raise ValueError("--model is required for --mode instruction-en")
        if not args.base_url:
            if args.teacher == "openai":
                args.base_url = "https://api.openai.com/v1"
            elif args.teacher == "gemini":
                args.base_url = "https://generativelanguage.googleapis.com/v1beta"
            else:
                args.base_url = "http://127.0.0.1:11434"
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "instruction-en":
        run_instruction_en(args)
    else:
        run_legacy_persona(args)


if __name__ == "__main__":
    main()
