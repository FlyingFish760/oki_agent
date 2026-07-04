"""交互层：CLI（demo 最快）。后续可扩展 GUI / 语音（Whisper ASR + 本地 TTS）。"""

from .cli import run_cli

__all__ = ["run_cli"]
