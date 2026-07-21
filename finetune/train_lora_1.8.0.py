# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

# LoRA
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```
"""

import argparse


def main(script_args, training_args, model_args, dataset_args):
    from accelerate.logging import get_logger
    from datasets import load_dataset

    from trl import SFTTrainer, get_dataset, get_peft_config, get_quantization_config

    logger = get_logger(__name__)

    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )

    # Load the dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": "finetune/data/datasets/instruction_en_train.jsonl",
            "test": "finetune/data/datasets/instruction_en_eval.jsonl",
        },
    )

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        quantization_config=get_quantization_config(model_args),
        peft_config=get_peft_config(model_args),
    )

    batch = next(iter(trainer.get_train_dataloader()))

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch.get("attention_mask")

    sample_input_ids = input_ids[0].cpu()
    sample_labels = labels[0].cpu()

    masked_ids = [
        token_id.item()
        for token_id, label in zip(sample_input_ids, sample_labels)
        if label == -100
    ]

    trained_ids = [
        token_id.item()
        for token_id, label in zip(sample_input_ids, sample_labels)
        if label != -100
    ]

    print("\n===== 不参与 loss 的 token =====")
    print(
        tokenizer.decode(
            masked_ids,
            skip_special_tokens=False,
        )
    )

    print("\n===== 参与 loss 的 token =====")
    print(
        tokenizer.decode(
            trained_ids,
            skip_special_tokens=False,
        )
    )

    import sys
    sys.exit()

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None, prog: str | None = None):
    from trl import DatasetMixtureConfig, ModelConfig, ScriptArguments, SFTConfig, TrlParser

    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types, prog=prog)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    main(script_args, training_args, model_args, dataset_args)
