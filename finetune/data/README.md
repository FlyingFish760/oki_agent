# Data Generation

`generate.py` builds SFT JSONL data for oki's persona fine-tuning pipeline.

The current Day2 focus is **English instruction data**:

- use English user prompts from `OpenAssistant/oasst1`;
- do not use original assistant answers from the dataset;
- do not translate or localize the source text;
- generate oki-style assistant replies with a teacher model;
- write train, eval, and rejected JSONL files.

## Modes

### `instruction-en`

Recommended current path. It samples English `prompter` rows from
`OpenAssistant/oasst1`, filters them, then asks a teacher model to produce oki
persona-style replies using `finetune/persona_profile-v2.1-en.md`.
The generation prompt is loaded from
`finetune/data/templates/teacher_prompt_template.txt`.

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
  --mode instruction-en `
  --dry-run-sources `
  --n-train 10 `
  --n-eval 5 `
  --cache-dir data/hf_cache `
  --source-preview-out data/instruction_en_source_preview.jsonl
```

Generate the default 300 train / 50 eval samples with the official OpenAI
Responses API:

```powershell
python finetune/data/generate.py `
  --mode instruction-en `
  --persona-profile finetune/persona_profile-v2.1-en.md `
  --cache-dir data/hf_cache `
  --teacher openai `
  --model <teacher-model-name>
```

Generate with a local Ollama teacher:

```powershell
python finetune/data/generate.py `
  --mode instruction-en `
  --cache-dir data/hf_cache `
  --teacher ollama `
  --model <ollama-model-name>
```

Generate with Gemini API:

```powershell
$env:GEMINI_API_KEY = "<your-gemini-api-key>"
python finetune/data/generate.py `
  --mode instruction-en `
  --cache-dir data/hf_cache `
  --teacher gemini `
  --model gemini-2.0-flash
```

## Outputs

Default `instruction-en` outputs:

- `finetune/data/datasets/instruction_en_train.jsonl`
- `finetune/data/datasets/instruction_en_eval.jsonl`
- `finetune/data/datasets/instruction_en_rejected.jsonl`

Dry-run source preview defaults to:

- `finetune/data/datasets/instruction_en_source_preview.jsonl`

All generated dataset files are ignored by git through the repository
`.gitignore`.

## Output Schema

Accepted train/eval records look like:

```json
{
  "id": "instruction_en_000001",
  "source_dataset": "OpenAssistant/oasst1",
  "persona_profile": "persona_profile-v2.1-en.md",
  "prompt_template": "teacher_prompt_template",
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
- text does not look unsafe/sensitive or URL-based.

The source prompt is preserved as the user message. The teacher output is
used only for the assistant reply.

## Teacher Prompt

`instruction-en` loads its system prompt template from:

```text
finetune/data/templates/teacher_prompt_template.txt
```

The template is designed for persona-style SFT data rewriting rather than
role-playing. It treats the model as a data generation assistant, preserves the
source user message exactly, and asks for only oki's assistant reply.

The full persona card remains the design and evaluation source document. For
generation, the default input is `finetune/persona_profile-v2.1-en.md`, a compact
system-prompt profile extracted from `finetune/persona_card-v2.1-en.md`.

The template must contain these placeholders:

- `{persona_profile}`
- `{user_instruction}`
- `[SOURCE USER MESSAGE]`

The default template already includes the strict JSON-output requirement, so the
teacher returns only an `assistant` field. `generate.py` combines that assistant
reply with the original source prompt to build the final two-message
`user` / `assistant` training example.

At runtime, the template is split at `[SOURCE USER MESSAGE]`. The content before
that marker is sent as the teacher system prompt. The source user message after
that marker is sent as the teacher user prompt. For OpenAI, the system prompt is
sent through the Responses API `instructions` parameter. OpenAI JSON mode also
requires the `input` content to contain the word `JSON`, so the OpenAI backend
adds a minimal `Return JSON only.` line to the teacher input. This does not
change the final training user message, which still comes directly from the
source dataset.

## Post-generation Filtering

Teacher outputs are rejected when:

- JSON is invalid or wrapped in Markdown fences;
- `assistant` is missing or empty;
- `<think>` appears;
- the assistant text is not mostly English;
- the assistant exposes AI/language-model identity;
- customer-service boilerplate appears;
- the assistant reply is longer than `--max-assistant-words`.

Rejected records are written to `instruction_en_rejected.jsonl` with a `reason` field.

## Useful Arguments

- `--cache-dir`: Hugging Face dataset download/cache directory.
- `--dry-run-sources`: sample and filter source prompts only; no teacher call.
- `--source-preview-out`: output path for dry-run source preview JSONL.
- `--prompt-template`: prompt template file for teacher generation.
- `--teacher`: `openai`, `gemini`, or `ollama`.
- `--model`: teacher model name.
- `--base-url`: override teacher API base URL.
- `--api-key`: explicit OpenAI or Gemini API key; otherwise
  `OPENAI_API_KEY` or `GEMINI_API_KEY` is used based on `--teacher`.
- `--no-json-response-format`: disable JSON-output hints for OpenAI or Gemini.
- `--max-source-rows`: maximum OASST rows scanned before filtering.
- `--max-assistant-words`: verbosity guard for teacher replies.

Run the built-in CLI help for the full list:

```powershell
python finetune/data/generate.py --help
```

## Current Limitations

- Only `OpenAssistant/oasst1` is enabled while the data-loading path is being
  stabilized.
- `instruction-en` currently generates two-turn samples only.
- No automatic LLM-as-judge scoring is run here; manual review should still
  use the rubric in the full persona card.
