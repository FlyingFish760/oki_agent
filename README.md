# Oki Agent

Oki is a one-week local personal AI assistant demo focused on persona fine-tuning.
The core interview claim is: persona is learned with LoRA/QLoRA, user facts stay in
local memory, and tools are gated by user control.

## Baseline

- Baseline model: `qwen3.6-35b-a3b`
- Persona model name used by the demo: `qwen3.6-35b-a3b-oki-lora`
- Local inference target: Ollama or LM Studio with GGUF Q4_K_M
- Training target: the same personal machine during idle windows; cloud/A100 remains an optional accelerator for deadline risk

## Quick demo without model weights

```bash
uv run oki chat --offline
```

Try this flow:

```text
我喜欢简短直接的解释，以后记住
/memory
下次解释 fine-tuning 和 RAG 时应该注意什么？
/search README
/remind 明天上午 10 点排练 Oki 面试 demo
帮我删除所有临时文件
```

## Ollama-backed demo

Run a local Ollama-compatible model named `qwen3.6-35b-a3b` for base mode and
`qwen3.6-35b-a3b-oki-lora` for the persona adapter mode.

```bash
uv run oki chat --model base
uv run oki chat --model persona
```

## A/B persona evaluation

```bash
uv run oki eval-persona --offline
```

The command prints a Markdown table comparing baseline and persona responses for
held-out prompts from `data/eval_prompts.jsonl`.

## Memory controls

```bash
uv run oki memory list
uv run oki memory delete <id>
```

Memories are stored in SQLite at `~/.oki/memory.sqlite3` by default. Use `--db` to
point at a temporary database for demos or tests.

## Local idle training

The product condition is that inference and training can happen on the same personal machine. Oki therefore exposes a guarded idle-training runner. It is opt-in, dry-run by default, and checks the local policy before launching QLoRA.

```bash
uv run oki train-idle
$env:OKI_ALLOW_LOCAL_TRAINING="1"
uv run oki train-idle --execute
```

Policy lives in `training/local_idle_policy.json`. The first implementation checks the quiet-hour window, AC-power preference, and explicit user opt-in. A production build should add GPU temperature, VRAM pressure, foreground app, and user activity checks.

## Fine-tuning path

- Persona rubric: `configs/persona_card.yaml`
- Seed training examples: `data/persona_train.jsonl`
- QLoRA config: `training/qlora_config.json`
- Training script: `training/train_qlora.py`
- Deployment notes: `training/DEPLOY.md`

The seed dataset is intentionally small. For the real interview run, expand it to
300-800 curated multi-turn examples and mix in 10%-20% capability-retention data.

