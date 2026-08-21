export HF_ENDPOINT="https://hf-mirror.com"

accelerate launch --config_file "/root/autodl-tmp/oki-agent/finetune/deepspeed_parallel_training/accelerate_config.yaml" /root/autodl-tmp/oki-agent/finetune/train_lora.py \
    --model_name_or_path /root/autodl-tmp/models/qwen3.6-35b-a3b \
    --custom_train_data_path /root/autodl-tmp/oki-agent/finetune/data/datasets/exp2_gemini_3.5_flash_1250/privacy_security_sft_train.jsonl \
    --custom_eval_data_path /root/autodl-tmp/oki-agent/finetune/data/datasets/exp2_gemini_3.5_flash_1250/privacy_security_sft_valid.jsonl \
    --learning_rate 2.0e-5 \
    --warmup_steps 0.03 \
    --num_train_epochs 2 \
    --packing False \
    --gradient_checkpointing True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eos_token '<|im_end|>' \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 0.2 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules in_proj_qkv in_proj_z out_proj q_proj k_proj v_proj o_proj \
    --output_dir /root/autodl-tmp/models/trl_deepspeed_lora_adapter/exp2 \
    --non_thinking_training True \
    2>&1 | tee /root/autodl-tmp/oki-agent/finetune/run_train_lora_exp2.log