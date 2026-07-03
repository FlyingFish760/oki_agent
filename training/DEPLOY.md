# Deploying the Oki persona adapter

The interview demo separates offline training from local inference.

3. If using an accelerator, run:

```bash
uv run python training/train_qlora.py --config training/qlora_config.json
```

4. Keep the adapter as evidence for the fine-tuning path:

```text
artifacts/oki-persona-lora/
```

5. For local demo inference, either:

- use an Ollama/LM Studio model named `qwen3.6-35b-a3b-oki-lora`, or
- merge the LoRA into the base model and quantize to GGUF Q4_K_M, then register it with that name.

The local CLI also supports `--offline` so memory and tool behavior can be shown without weights.

