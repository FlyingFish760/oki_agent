"""Day3：QLoRA 训练脚本（trl + peft + bitsandbytes）。

MoE 要点（务必遵守）:
  - LoRA 只挂 attention 投影 q/k/v/o，**不要动 router / 路由专家**，
    否则破坏专家路由稳定性。
  - 数据里 thinking trace 的处理要和 generate.py 一致。

先跑通再调质量。用法:
    python finetune/train_qlora.py --config finetune/configs/qlora.yaml

依赖（见 pyproject 的 [finetune] 可选组）:
    transformers peft trl datasets bitsandbytes accelerate pyyaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="finetune/configs/qlora.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    # 延迟导入重依赖，避免运行时环境未装训练库也能 import 检查
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"])

    lcfg = cfg["lora"]
    lora = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lcfg["target_modules"],   # 只挂 q/k/v/o —— 不碰 router/experts
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=cfg["train"]["data_files"], split="train")

    def format_chat(example):
        return tok.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

    t = cfg["train"]
    sft = SFTConfig(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_epochs"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler"],
        max_seq_length=t["max_seq_len"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=ds,
        processing_class=tok,
        formatting_func=format_chat,
    )
    trainer.train()
    trainer.save_model(t["output_dir"])
    print(f"LoRA adapter 已保存到 {Path(t['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
