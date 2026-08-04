# https://github.com/huggingface/trl/blob/v0.29.1/trl/scripts/sft.py

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
import os
from dataclasses import dataclass, field

from accelerate import logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class OkiScriptArguments(ScriptArguments):
    custom_train_data_path: str = field(
        default="finetune/data/datasets/instruction_en_train.jsonl",
        metadata={
            "help": "Path to the local JSONL training dataset.",
            "aliases": ["--custom-train-data-path"],
        },
    )
    custom_eval_data_path: str = field(
        default="finetune/data/datasets/instruction_en_eval.jsonl",
        metadata={
            "help": "Path to the local JSONL evaluation dataset.",
            "aliases": ["--custom-eval-data-path"],
        },
    )
    non_thinking_training: bool = field(
        default=False,
        metadata={
            "help": (
                "Add chat_template_kwargs={'enable_thinking': False} to every "
                "dataset sample before SFT training."
            ),
            "aliases": ["--non-thinking-training"],
        },
    )


def main(script_args, training_args, model_args, dataset_args)->None:
    ################
    # Model init kwargs
    ################
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Load the dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": script_args.custom_train_data_path,
            "test": script_args.custom_eval_data_path,
        },
    )

    def to_conversational_pc(example)-> dict:
        messages = example["messages"]
        if len(messages) < 2:
            raise ValueError(f"Expected at least 2 messages, got {len(messages)}")
        if messages[-1]["role"] != "assistant":
            raise ValueError(f"Expected final message to be assistant, got {messages[-1]['role']}")

        converted = {
            "prompt": messages[:-1],
            "completion": [messages[-1]],
        }
        if script_args.non_thinking_training:
            converted["chat_template_kwargs"] = {"enable_thinking": False}
        return converted

    dataset = dataset.map(to_conversational_pc, remove_columns=dataset["train"].column_names)

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    tokenizer = processor.tokenizer

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


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (OkiScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
