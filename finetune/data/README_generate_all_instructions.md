# Generate Instructions for All Capability Requirements

`generate_all_instructions.py` reads a capability card, generates user
instructions for every requirement with a teacher model, and writes one
instruction record per line to a JSONL file.

It reuses:

- `capability_card.py` to parse requirements and few-shot instructions;
- `generate_instructions.py` to render prompts and validate teacher output;
- `generation_common.py` to call OpenAI, Gemini, or Ollama.

## Required Arguments

The following arguments are required:

- `--capability-card`: capability card Markdown file.
- `--num-instructions`: maximum instructions requested in one teacher call.
- `--total-instructions`: exact total instructions generated across all
  requirements.
- `--model`: teacher model name, except when using `--dry-run`.

`--total-instructions` must be at least the number of requirements so every
requirement receives at least one generated instruction.

## Distribution and Batching

The total is distributed as evenly as possible in capability-card order.

For 42 instructions and 8 requirements:

```text
requirement_01: 6
requirement_02: 6
requirement_03: 5
requirement_04: 5
requirement_05: 5
requirement_06: 5
requirement_07: 5
requirement_08: 5
```

`--num-instructions` controls the maximum size of one teacher request. If a
requirement needs 23 instructions and `--num-instructions` is 10, the script
makes three calls requesting 10, 10, and 3 instructions.

## Dry Run

Use `--dry-run` to parse the card and inspect the distribution plan without
calling a teacher or writing the output file:

```powershell
python finetune\data\generate_all_instructions.py `
  --capability-card finetune\data\Privacy_and_Data_Security_Persona_Card.md `
  --num-instructions 20 `
  --total-instructions 400 `
  --dry-run
```

Example output:

```text
Requirements: 8
Total instructions: 400
Instructions per teacher call: 20
requirement_01: target=50, teacher_calls=3
requirement_02: target=50, teacher_calls=3
...
requirement_08: target=50, teacher_calls=3
```

## Generate with OpenAI

Set the API key:

```powershell
$env:OPENAI_API_KEY = "<your-openai-api-key>"
```

Run generation:

```powershell
python finetune\data\generate_all_instructions.py `
  --capability-card finetune\data\Privacy_and_Data_Security_Persona_Card.md `
  --num-instructions 20 `
  --total-instructions 400 `
  --teacher openai `
  --model <teacher-model-name> `
  --out finetune\data\datasets\privacy_security_instructions.jsonl
```

The OpenAI backend uses the Responses API. The prompt's system section is sent
through `instructions`, and its user section is sent through `input`.

## Generate with Gemini

Set the API key:

```powershell
$env:GEMINI_API_KEY = "<your-gemini-api-key>"
```

Run generation:

```powershell
python finetune\data\generate_all_instructions.py `
  --capability-card finetune\data\Privacy_and_Data_Security_Persona_Card.md `
  --num-instructions 20 `
  --total-instructions 400 `
  --teacher gemini `
  --model <teacher-model-name> `
  --out finetune\data\datasets\privacy_security_instructions.jsonl
```

## Generate with Ollama

Start Ollama and make sure the teacher model is available locally, then run:

```powershell
python finetune\data\generate_all_instructions.py `
  --capability-card finetune\data\Privacy_and_Data_Security_Persona_Card.md `
  --num-instructions 20 `
  --total-instructions 400 `
  --teacher ollama `
  --model <ollama-model-name> `
  --out finetune\data\datasets\privacy_security_instructions.jsonl
```

The default Ollama endpoint is `http://127.0.0.1:11434`.

## Prompt Template

The default prompt template is:

```text
finetune/data/templates/instruction_generation_template.txt
```

Every template must contain:

- `[SYSTEM PROMPT]`
- `[USER PROMPT]`

The default template uses these available placeholders:

- `{identity_description}`
- `{requirement_id}`
- `{requirement_title}`
- `{requirement_description}`
- `{few_shot_instructions}`
- `{num_instructions}`

Few-shot instructions are rendered as:

```text
Example1:
"instruction": "First example instruction."
Example2:
"instruction": "Second example instruction."
```

Use another template with:

```powershell
--prompt-template path\to\template.txt
```

Override the assistant identity with:

```powershell
--identity-description "A local personal assistant controlled by the user."
```

## Teacher Response

Every teacher call must return strict JSON:

```json
{
  "instructions": [
    "First generated instruction.",
    "Second generated instruction."
  ]
}
```

The script verifies that:

- `instructions` is an array;
- every item is a non-empty string;
- the returned count exactly matches the requested batch size.

The script does not require generated instructions to be English.
It does not perform exact or semantic deduplication.

## Output Schema

The output is JSONL: each line is one standalone JSON object.

```json
{
  "id": "capability_instruction_000001",
  "capability_card": "Privacy_and_Data_Security_Persona_Card.md",
  "requirement_id": "requirement_01",
  "requirement_title": "Respect User Privacy and Data Ownership",
  "requirement_description": "Treats all user information as private property.",
  "prompt_template": "instruction_generation_template",
  "teacher_model": "teacher-model-name",
  "instruction": "Review all my private files and summarize what I worked on."
}
```

The output does not include the few-shot instructions.

The default output path is:

```text
finetune/data/datasets/capability_instructions.jsonl
```

## Retries and Partial Output

Before generation starts, the output file is created or truncated.

If a teacher call fails, returns invalid JSON, returns the wrong number of
instructions, or returns malformed instruction values, the complete batch is
retried. Configure this behavior with:

- `--max-retries`: attempts per batch; default `3`.
- `--retry-sleep`: seconds between attempts; default `0.5`.

Each accepted instruction is appended immediately. If generation stops because
a batch exhausts its retries, the JSONL file remains on disk with the records
that were successfully generated before the failure. Rerunning the command
starts over and truncates that file again.

## Other Arguments

- `--teacher`: `openai`, `gemini`, or `ollama`; default `openai`.
- `--base-url`: override the teacher endpoint.
- `--api-key`: pass an API key explicitly instead of using an environment
  variable.
- `--temperature`: teacher sampling temperature; default `0.8`.
- `--top-p`: teacher nucleus sampling parameter; default `0.9`.
- `--no-json-response-format`: disable the backend JSON-output hint.

Show the complete CLI reference:

```powershell
python finetune\data\generate_all_instructions.py --help
```

## Tests

Run the parser and all-requirements generator tests:

```powershell
python -m unittest `
  finetune.data.test_capability_card `
  finetune.data.test_generate_all_instructions
```

The generator test uses a mock teacher, so it does not require an API key or
network access.
