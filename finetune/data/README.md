# Data Generation

`generate.py` builds SFT JSONL data for oki's persona fine-tuning pipeline.

The current Day2 focus is **English daily conversation**:

- use English user prompts from `OpenAssistant/oasst1`;
- do not use original assistant answers from the dataset;
- do not translate or localize the source text;
- generate oki-style assistant replies with a teacher model;
- write train, eval, and rejected JSONL files.

## Modes

### `daily-en`

Recommended current path. It samples English `prompter` rows from
`OpenAssistant/oasst1`, filters them, then asks a teacher model to produce oki
daily-conversation replies using `finetune/persona_card-daily_en.md`.

### `persona`

Legacy skeleton entrypoint kept for compatibility with the original Day2 plan.
It still requires implementing the legacy teacher hook before use.

## Quick Start

Install finetune dependencies first:

```powershell
uv sync --extra finetune
```

Preview source prompts without calling a teacher:

```powershell
python finetune/data/generate.py `
  --mode daily-en `
  --dry-run-sources `
  --n-train 10 `
  --n-eval 5 `
  --cache-dir data/hf_cache `
  --source-preview-out data/daily_en_source_preview.jsonl
```

Generate the default 300 train / 50 eval samples with an OpenAI-compatible
teacher:

```powershell
python finetune/data/generate.py `
  --mode daily-en `
  --persona-card finetune/persona_card-daily_en.md `
  --cache-dir data/hf_cache `
  --teacher openai-compatible `
  --model <teacher-model-name>
```

Generate with a local Ollama teacher:

```powershell
python finetune/data/generate.py `
  --mode daily-en `
  --cache-dir data/hf_cache `
  --teacher ollama `
  --model <ollama-model-name>
```

Generate with Gemini API:

```powershell
$env:GEMINI_API_KEY = "<your-gemini-api-key>"
python finetune/data/generate.py `
  --mode daily-en `
  --cache-dir data/hf_cache `
  --teacher gemini `
  --model gemini-2.0-flash
```

## Outputs

Default `daily-en` outputs:

- `finetune/data/datasets/daily_en_train.jsonl`
- `finetune/data/datasets/daily_en_eval.jsonl`
- `finetune/data/datasets/daily_en_rejected.jsonl`

Dry-run source preview defaults to:

- `finetune/data/datasets/daily_en_source_preview.jsonl`

All generated dataset files are ignored by git through the repository
`.gitignore`.

## Output Schema

Accepted train/eval records look like:

```json
{
  "id": "daily_en_000001",
  "source_dataset": "OpenAssistant/oasst1",
  "persona_card": "persona_card-daily_en.md",
  "prompt_template": "rolegpt_zero_shot_en",
  "messages": [
    {"role": "user", "content": "I'm so tired today."},
    {"role": "assistant", "content": "Rough day. Take ten minutes first, don't force it."}
  ]
}
```

The training script only consumes `messages`; the metadata is kept for audit and
manual review.

## Source Filtering

For `OpenAssistant/oasst1`, the script keeps rows where:

- `role == "prompter"`;
- `lang` is absent or `"en"`;
- text is mostly English;
- text length is roughly 8 to 220 words;
- text does not look code-heavy, unsafe/sensitive, URL-based, or like a long
  factual essay request.

The source prompt is preserved as the user message. The teacher output is
rejected if it changes that user content.

## Teacher Prompt

`daily-en` uses the RoleGPT-style prompt template:

```text
System Instruction:

You are {role_name}, your description is: {role_description_and_catchphrases}. Now please answer some questions to accurately show your personality traits! Your speaking style should fully imitate the personality role assigned to you! Please do not expose that you are an artificial intelligence model or a language model, you must always remember that you are only assigned one personality role. Don't be verbose or too formal or polite when speaking.

User Prompt:

{user_name}: `{user_instruction}`
```

The script adds a strict JSON-output requirement around this template so the
teacher returns a two-message `user` / `assistant` conversation.

## Post-generation Filtering

Teacher outputs are rejected when:

- JSON is invalid or wrapped in Markdown fences;
- `messages` is not exactly `[user, assistant]`;
- the user content does not exactly match the source prompt;
- `<think>` appears;
- the assistant text is not mostly English;
- the assistant exposes AI/language-model identity;
- customer-service boilerplate appears;
- the assistant reply is longer than `--max-assistant-words`.

Rejected records are written to `daily_en_rejected.jsonl` with a `reason` field.

## Useful Arguments

- `--cache-dir`: Hugging Face dataset download/cache directory.
- `--dry-run-sources`: sample and filter source prompts only; no teacher call.
- `--source-preview-out`: output path for dry-run source preview JSONL.
- `--teacher`: `openai-compatible`, `gemini`, or `ollama`.
- `--model`: teacher model name.
- `--base-url`: override teacher API base URL.
- `--api-key`: explicit OpenAI-compatible or Gemini API key; otherwise
  `OPENAI_API_KEY` or `GEMINI_API_KEY` is used based on `--teacher`.
- `--no-json-response-format`: disable OpenAI `response_format` for compatible
  servers that do not support it. For Gemini, this disables
  `generationConfig.response_mime_type`.
- `--max-source-rows`: maximum OASST rows scanned before filtering.
- `--max-assistant-words`: verbosity guard for teacher replies.

Run the built-in CLI help for the full list:

```powershell
python finetune/data/generate.py --help
```

## Current Limitations

- Only `OpenAssistant/oasst1` is enabled while the data-loading path is being
  stabilized.
- `daily-en` currently generates two-turn samples only.
- No automatic LLM-as-judge scoring is run here; manual review should still
  use the rubric in `finetune/persona_card-daily_en.md`.
