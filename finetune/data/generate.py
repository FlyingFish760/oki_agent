"""Day2 核心：用更强的模型蒸馏生成人格训练数据。

产出 ChatML/JSONL 多轮对话，每条都受 persona_card.md 约束，覆盖多意图。
关键：
  - 目标量级 300~800 条高质量多轮样本（质量 >> 数量），生成后人工筛。
  - **掺入能力保持样本**（中性语气的正确推理/指令遵循/工具调用），
    对抗 MoE LoRA 上的灾难性遗忘。见 capability_keep.py。
  - **thinking trace 处理**：Qwen3 是带 thinking 的推理模型。训练样本里
    assistant 的 thinking 要么统一保留规范格式、要么统一去掉，
    不能混，否则污染推理链。见 --thinking 开关。

用法（示例，接 API 或本地更强模型自行实现 _call_teacher）:
    python finetune/data/generate.py --n 500 --thinking strip \\
        --out finetune/data/datasets/persona.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PERSONA_CARD = Path(__file__).resolve().parent.parent / "persona_card.md"

# 覆盖的意图/情境（对应 persona card 的评估维度）
INTENTS = [
    "闲聊", "任务执行", "拒绝不合理请求", "情绪安抚",
    "危险操作确认", "承认不知道", "多轮追问", "工具调用",
]

GEN_SYSTEM = """你在为一个名叫 oki 的本地助手生成人格训练对话。
严格遵循下面的人格卡，生成一段自然的多轮对话（2~4 轮）。
输出 JSON: {{"messages": [{{"role": "user|assistant", "content": "..."}}, ...]}}
人格卡：
{card}
本条对话的意图：{intent}
"""


def _call_teacher(system: str) -> dict | None:
    """调用更强的 teacher 模型生成一条样本。

    TODO: 接入你选择的 teacher（本地大模型 or API）。
    返回 {"messages": [...]} 或 None（生成失败）。
    """
    raise NotImplementedError("请实现 _call_teacher：接入 teacher 模型。")


def _strip_thinking(messages: list[dict], mode: str) -> list[dict]:
    """处理 assistant 的 thinking trace。

    mode="strip": 去掉 <think>...</think>
    mode="keep":  保留（需保证格式统一）
    """
    if mode == "keep":
        return messages
    import re
    out = []
    for m in messages:
        c = m["content"]
        if m["role"] == "assistant":
            c = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()
        out.append({**m, "content": c})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--thinking", choices=["strip", "keep"], default="strip")
    ap.add_argument("--out", type=Path, default=Path("finetune/data/datasets/persona.jsonl"))
    args = ap.parse_args()

    card = PERSONA_CARD.read_text(encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for _ in range(args.n):
            intent = random.choice(INTENTS)
            sample = _call_teacher(GEN_SYSTEM.format(card=card, intent=intent))
            if not sample:
                continue
            sample["messages"] = _strip_thinking(sample["messages"], args.thinking)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
    print(f"已生成 {written} 条到 {args.out}。记得人工筛一遍，剔除跑偏样本。")


if __name__ == "__main__":
    main()
