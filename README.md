# Oki Agent

Oki is a local personal assistant demo built around **Qwen3.6-35B-A3B** as the baseline model. The demo story is intentionally narrow: prove that Oki can have a stable personality through fine-tuning while keeping user facts, preferences, and training data under local control.

## Demo foundation

This repository now contains a minimal local assistant scaffold:

| Layer | Demo implementation |
| --- | --- |
| Model baseline | `qwen3.6-35b-a3b`, configured through `OKI_LLM_MODEL` |
| Persona | `persona/oki_persona.toml`, reused by runtime prompts, seed data, and evaluation |
| Memory | Local SQLite database under `.oki/memory.sqlite3` |
| Tools | `/remember`, `/search`, `/memories`, `/train-persona` |
| Training loop | Local idle-time LoRA/QLoRA job manifest generation |
| Interaction | CLI with offline mode or OpenAI-compatible local endpoint |

## Run the demo

Offline deterministic mode:

```powershell
uv run python main.py --offline --once "我今天事情很多，先做什么？"
```

Interactive mode:

```powershell
uv run python main.py --offline
```

Local model endpoint mode:

```powershell
$env:OKI_LLM_ENDPOINT = "http://localhost:8000/v1"
$env:OKI_LLM_MODEL = "qwen3.6-35b-a3b"
uv run python main.py --once "介绍一下你自己"
```

The endpoint is OpenAI-compatible (`/chat/completions`), so it can be backed by a local vLLM server, LM Studio, llama.cpp server, or another local runtime that exposes the same API.

## Local training story

The target product constraint is that **inference and personalization training happen on the same personal machine**. That means Oki should not assume cloud fine-tuning by default. The intended loop is:

1. User chats and local memories accumulate in SQLite.
2. Oki extracts candidate personalization samples from local data.
3. The user can inspect, delete, or approve samples before training.
4. During idle windows, Oki trains a small LoRA/QLoRA adapter instead of full model weights.
5. Oki evaluates the adapter for persona consistency and ability retention.
6. The user manually activates the adapter, with rollback to the previous adapter.

Generate a local training manifest:

```powershell
uv run python main.py --offline --once "/remember 我喜欢回答先给结论，再解释关键原因"
uv run python main.py --offline --once "/train-persona"
```

This creates `.oki/training/persona_samples_*.jsonl` and `.oki/training/train_lora_*.json`. The manifest is deliberately explicit about local-only training, user approval, and adapter rollback because those are core to the product promise.

## Fine-tuning positioning

Use LoRA/QLoRA to teach Oki **how to speak and behave**:

- stable tone and personality
- refusal and safety style
- tool-call formatting habits
- domain-specific interaction patterns

Do not use fine-tuning as the main store for user facts. Facts, preferences, and changing personal context should live in memory, where they can be viewed, edited, deleted, and conflict-resolved.

## CLI commands

| Command | Purpose |
| --- | --- |
| `/remember <text>` | Save a local semantic memory |
| `/search <text>` | Search text files under the current workspace |
| `/memories` | Show recent saved memories |
| `/train-persona` | Prepare local idle-time persona training artifacts |
| `/exit` | Exit interactive mode |

## Persona baseline testing

Before fine-tuning, run a fixed held-out prompt set so the base model has a measurable baseline:

```powershell
uv run python -m oki_agent.eval_persona --offline
```

With a local Qwen endpoint:

```powershell
$env:OKI_LLM_ENDPOINT = "http://localhost:8000/v1"
$env:OKI_LLM_MODEL = "qwen3.6-35b-a3b"
uv run python -m oki_agent.eval_persona
```

The prompt set lives in `evals/persona_baseline_prompts.jsonl`. Results are written to `.oki/evals/persona_baseline_*.jsonl` with latency, raw response, checklist, and empty manual score fields for later review. Each run uses an isolated eval memory database by default so baseline results are repeatable. Reuse the same prompt file after LoRA/QLoRA training to compare base vs persona adapter.