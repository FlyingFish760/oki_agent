"""Day4：能力保持评估。

一小组推理 / 指令遵循 / 工具调用题，对比微调前后是否明显退化。
和人格评估分开看：人格要变好，能力不能变差。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 小评测集（占位，请扩充为 20~40 题覆盖推理/格式/工具调用）
CAPABILITY_QUESTIONS = [
    {"q": "12 * 8 = ?", "expect": "96"},
    {"q": "只输出 JSON：名字 A，年龄 3。", "expect_contains": "\"age\""},
    {"q": "要读取文件 a.txt，应调用哪个工具？", "expect_contains": "read_file"},
]


def score_answer(item: dict, answer: str) -> bool:
    if "expect" in item:
        return item["expect"] in answer
    if "expect_contains" in item:
        return item["expect_contains"] in answer
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-call", choices=["base", "adapter"], default="base")
    args = ap.parse_args()

    # TODO: 接入被测模型（base 或 base+adapter），返回回答字符串
    def model_call(q: str) -> str:
        raise NotImplementedError("请接入被测模型")

    passed = 0
    for item in CAPABILITY_QUESTIONS:
        ans = model_call(item["q"])
        ok = score_answer(item, ans)
        passed += ok
        print(f"[{'OK' if ok else 'X'}] {item['q']} -> {ans[:60]}")
    print(json.dumps({"model": args.model_call, "passed": passed,
                      "total": len(CAPABILITY_QUESTIONS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
