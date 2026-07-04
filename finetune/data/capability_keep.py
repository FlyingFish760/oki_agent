"""能力保持样本：中性语气的正确推理 / 指令遵循 / 工具调用例子。

按一定比例（经验 10%~20%）掺进人格数据，对抗灾难性遗忘——
这是 MoE LoRA 上很容易翻车的点，掺这一手既保命又是面试加分项。

这些样本**不带人格语气**（故意中性），只保证"会做事"的能力不退化。
可来源于：公开指令数据子集、少量自造的推理/格式化任务、工具调用范例。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 少量自造范例占位；实际可混入公开指令集子集。
CAPABILITY_SAMPLES = [
    {"messages": [
        {"role": "user", "content": "把 [3,1,2] 升序排序并解释一步。"},
        {"role": "assistant", "content": "排序结果：[1, 2, 3]。方法：比较相邻元素，小的往前，直到有序。"},
    ]},
    {"messages": [
        {"role": "user", "content": "只输出 JSON：城市北京，温度 30。"},
        {"role": "assistant", "content": "{\"city\": \"北京\", \"temperature\": 30}"},
    ]},
    {"messages": [
        {"role": "user", "content": "帮我读取 notes.txt 的内容。"},
        {"role": "assistant", "content": "调用工具 read_file，参数 {\"path\": \"notes.txt\"}。"},
    ]},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("finetune/data/datasets/capability.jsonl"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for s in CAPABILITY_SAMPLES:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"已写出 {len(CAPABILITY_SAMPLES)} 条能力保持样本到 {args.out}（示例，请扩充）。")


if __name__ == "__main__":
    main()
