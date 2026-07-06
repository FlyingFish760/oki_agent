"""Day1：记录"微调前"基线表现。

跑通 base 后一条命令即可：对 prompts_eval.txt 里每个 prompt 调 base 模型，
把 (prompt, response) 连同模型名与时间戳存成 JSONL 快照，留给 Day4 做 A/B 对照。

用法:
    # 默认：注入 persona system prompt，还原"只靠 prompt 撑人格"的基线
    python finetune/eval/baseline.py --model qwen3:latest
    # 纯 qwen base 素颜（不加任何 system prompt）
    python finetune/eval/baseline.py --model qwen3:latest --no-persona-prompt

两种基线各有用：
  - persona 模式 = 微调的直接对照（同样的人格意图，prompt vs 微调谁更稳）。
  - raw 模式    = 模型"素颜"能力，看微调注入了多少、也确认 prompt 本身的贡献。
输出文件名带 _persona / _raw 后缀，记录里也有 "mode" 字段，互不覆盖。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 让脚本能直接找到 oki 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from oki.config import load_config                     # noqa: E402
from oki.inference.ollama_client import OllamaClient    # noqa: E402
from oki.persona.persona import Persona, PersonaKnobs   # noqa: E402


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts


def main() -> None:
    cfg = load_config()
    default_model = cfg.model.get("ab", {}).get("base_model") or cfg.model.get("chat_model", "qwen3-35b-a3b")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=default_model, help="Ollama 中的 base 模型名")
    ap.add_argument("--prompts", type=Path, default=Path(__file__).parent / "prompts_eval.txt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--persona-prompt", action=argparse.BooleanOptionalAction, default=True,
                    help="是否注入 persona system prompt。"
                         "--persona-prompt（默认）=还原'只靠 prompt 撑人格'的基线；"
                         "--no-persona-prompt=纯 qwen base 素颜结果。")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    if not prompts:
        print(f"没有可用 prompt：{args.prompts}")
        return

    client = OllamaClient(model=args.model, host=cfg.model.get("host"))

    system = None
    if args.persona_prompt:
        persona = Persona(cfg.persona.get("card", {"name": "oki"}),
                          PersonaKnobs(**cfg.persona.get("knobs", {})))
        system = persona.system_prompt()

    mode = "persona" if system else "raw"
    safe_model = args.model.replace(":", "_").replace("/", "_")
    out = args.out or Path("finetune/output") / f"baseline_{safe_model}_{mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"基线采集：model={args.model}  mode={mode}  prompts={len(prompts)}  -> {out}\n")
    with out.open("w", encoding="utf-8") as f:
        for i, p in enumerate(prompts, 1):
            messages = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": p}]
            t0 = time.time()
            resp = client.complete(messages)
            dt = time.time() - t0
            rec = {"model": args.model, "mode": mode, "timestamp": stamp, "prompt": p,
                   "response": resp, "latency_s": round(dt, 2)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(prompts)}] ({dt:.1f}s) {p}\n  -> {resp[:120].replace(chr(10), ' ')}\n")

    print(f"完成。基线快照已存到 {out.resolve()}")


if __name__ == "__main__":
    main()
