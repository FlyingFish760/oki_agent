"""Day4 部署：把 LoRA 合并进 fp16 base，便于后续量化成 GGUF。

两条部署路线：
  A) 合并 -> 转 GGUF -> Ollama（最稳，推理最快）。本脚本做"合并"这步。
  B) 不合并，用 Ollama Modelfile 的 ADAPTER 直接挂 LoRA（见 export/Modelfile）。

用法:
    python finetune/export/merge_lora.py \\
        --base Qwen/Qwen3.6-35B-A3B \\
        --adapter finetune/output/oki-persona-lora \\
        --out finetune/output/oki-persona-merged
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, trust_remote_code=True, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"已合并到 {args.out}")
    print("下一步：用 llama.cpp 的 convert_hf_to_gguf.py 转 GGUF，再 quantize 到 Q4_K_M，")
    print("然后用 export/Modelfile 在 Ollama 里 `ollama create oki-persona -f Modelfile`。")


if __name__ == "__main__":
    main()
