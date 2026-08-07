"""Shared utilities for teacher-backed data generation scripts."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://127.0.0.1:11434",
}


@dataclass(frozen=True)
class TeacherPrompt:
    system: str
    user: str


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("`\"' ")


def is_mostly_english(text: str) -> bool:
    letters = sum(character.isalpha() for character in text)
    if letters == 0:
        return False
    ascii_letters = sum(
        "a" <= character.lower() <= "z" for character in text
    )
    return ascii_letters / letters >= 0.85


def prompt_template_name(path: Path) -> str:
    return path.stem


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_strict_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text or text.startswith("```"):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def add_teacher_arguments(parser: argparse.ArgumentParser) -> None:
    teacher = parser.add_argument_group("teacher backend")
    teacher.add_argument(
        "--teacher",
        choices=["openai", "openai-compatible", "gemini", "ollama"],
        default="openai",
        help="Teacher backend. Default: %(default)s.",
    )
    teacher.add_argument(
        "--model",
        default="",
        help="Teacher model name. Required unless running a dry run.",
    )
    teacher.add_argument(
        "--base-url",
        default="",
        help=(
            "Teacher API base URL. Required for openai-compatible. Defaults "
            "to the official OpenAI or Gemini endpoint, or the local Ollama "
            "endpoint."
        ),
    )
    teacher.add_argument(
        "--api-key",
        default="",
        help=(
            "API key for OpenAI or Gemini. OpenAI-compatible keys are "
            "optional. If omitted, the matching environment variable is used."
        ),
    )
    teacher.add_argument(
        "--json-response-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request JSON output when supported. Default: %(default)s.",
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


def configure_teacher_args(
    args: argparse.Namespace,
    *,
    require_model: bool,
) -> None:
    if require_model and not args.model:
        raise ValueError("--model is required unless running a dry run")
    if not args.base_url:
        if args.teacher == "openai-compatible":
            raise ValueError(
                "--base-url is required for --teacher openai-compatible"
            )
        args.base_url = DEFAULT_BASE_URLS[args.teacher]


def call_teacher(
    prompt: TeacherPrompt,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.teacher == "openai":
        return _call_openai(prompt, args)
    if args.teacher == "openai-compatible":
        return call_openai_compatible(prompt, args)
    if args.teacher == "gemini":
        return _call_gemini(prompt, args)
    if args.teacher == "ollama":
        return _call_ollama(prompt, args)
    raise ValueError(f"Unknown teacher backend: {args.teacher}")


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} {exc.reason}: {error_body}"
        ) from exc


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


def _call_openai(
    prompt: TeacherPrompt,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --teacher openai")

    payload: dict[str, Any] = {
        "model": args.model,
        "instructions": prompt.system,
        "input": f"{prompt.user}\n\nReturn JSON only.",
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
    return parse_strict_json(_extract_openai_response_text(data))


def _extract_chat_completion_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    chunks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def call_openai_compatible(
    prompt: TeacherPrompt,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Call an OpenAI-compatible Chat Completions endpoint."""
    if not args.base_url:
        raise RuntimeError(
            "--base-url is required for --teacher openai-compatible"
        )

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.json_response_format:
        payload["response_format"] = {"type": "json_object"}

    api_key = (
        args.api_key
        or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    headers = (
        {"Authorization": f"Bearer {api_key}"}
        if api_key
        else None
    )
    data = _post_json(
        f"{args.base_url.rstrip('/')}/chat/completions",
        payload,
        headers=headers,
    )
    return parse_strict_json(_extract_chat_completion_text(data))


def _call_ollama(
    prompt: TeacherPrompt,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    data = _post_json(f"{args.base_url.rstrip('/')}/api/chat", payload)
    content = data.get("message", {}).get("content", "")
    return parse_strict_json(content)


def _call_gemini(
    prompt: TeacherPrompt,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
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

    data = _post_json(
        f"{args.base_url.rstrip('/')}/{model}:generateContent?key={api_key}",
        payload,
    )
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    content = "".join(str(part.get("text", "")) for part in parts)
    return parse_strict_json(content)
