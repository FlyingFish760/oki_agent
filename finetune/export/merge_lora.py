"""Merge a LoRA adapter into a base causal language model.

Example:
    python finetune/merge_lora.py ^
        --base-path Qwen/Qwen3.6-35B-A3B ^
        --adapter-path finetune/output/oki-persona-lora ^
        --output-path finetune/output/oki-persona-merged
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model and save the merged model.")
    parser.add_argument(
        "--base-path",
        required=True,
        help="Base model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="LoRA adapter directory produced by PEFT/SFT training.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output directory for the merged model and tokenizer/processor.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Torch dtype used when loading the base model. Default: %(default)s.",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        help='Device map passed to from_pretrained. Use "cpu" for low-risk merging. Default: %(default)s.',
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum checkpoint shard size passed to save_pretrained. Default: %(default)s.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "device_map": args.device_map,
    }

    config = AutoConfig.from_pretrained(args.base_path, trust_remote_code=True)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()
    is_vlm = bool(
        config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures)
    )

    if is_vlm:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(args.base_path, trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            args.base_path,
            **model_kwargs,
        )
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_path,
            **model_kwargs,
        )

    merged_model = PeftModel.from_pretrained(base_model, args.adapter_path).merge_and_unload()
    merged_model.save_pretrained(output_path, safe_serialization=True, max_shard_size=args.max_shard_size)
    if is_vlm:
        processor.save_pretrained(output_path)
    else:
        tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
