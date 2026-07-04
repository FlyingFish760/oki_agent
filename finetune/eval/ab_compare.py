"""Day4：A/B 并排对比（向面试官证明"做了真微调"最有力的一张牌）。

同样的 prompt，base vs base+adapter 并排出。
默认走 Ollama：base_model vs persona_model（见 configs/model.yaml 的 ab 段）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本能直接找到 oki 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from oki.config import load_config              # noqa: E402
from oki.inference.ollama_client import OllamaClient  # noqa: E402

PROMPTS_FILE = Path(__file__).parent / "prompts_eval.txt"


def load_prompts(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def run_side_by_side(prompts: list[str]) -> None:
    cfg = load_config()
    ab = cfg.model.get("ab", {})
    base = OllamaClient(ab.get("base_model", "qwen3-35b-a3b"))
    persona = OllamaClient(ab.get("persona_model", "oki-persona"))

    for p in prompts:
        print("=" * 70)
        print(f"PROMPT: {p}\n")
        print("--- base ---")
        print(base.complete([{"role": "user", "content": p}]))
        print("\n--- base + persona adapter ---")
        print(persona.complete([{"role": "user", "content": p}]))
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-file", type=Path, default=PROMPTS_FILE, help="每行一个 prompt，# 为注释")
    args = ap.parse_args()
    prompts = load_prompts(args.prompts_file)
    run_side_by_side(prompts)


if __name__ == "__main__":
    main()
