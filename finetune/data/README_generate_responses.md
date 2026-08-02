# Generate Responses for Capability Instructions

`generate_responses.py` reads generated instruction records from JSONL, uses a
teacher model to generate one assistant response for each instruction, and
writes conversational SFT records in `messages` format.

The script uses the capability card as the source of truth for requirement
titles and descriptions. Values copied into the instruction JSONL are not used
for prompt generation.

## Data Flow

```text
instruction JSONL
    -> read instruction + requirement_id
    -> resolve requirement from capability card
    -> render response generation prompt
    -> call teacher model
    -> validate {"response": "..."}
    -> write user/assistant messages JSONL
```

The implementation reuses:

- `capability_card.py` for parsing the capability card;
- `generation_common.py` for OpenAI, Gemini, and Ollama teacher calls;
- `response_genereation_template.txt` for the system and user prompts.

## Required Arguments

- `--instructions`: input instruction JSONL.
- `--capability-card`: capability card used to resolve every
  `requirement_id`.
- `--model`: teacher model name, except when using `--dry-run`.

## Input Schema

Every non-empty JSONL line must be a JSON object containing:

- `instruction`: non-empty string.
- `requirement_id`: non-empty string matching a requirement in the capability
  card, such as `requirement_01`.

Example:

```json
{
  "id": "capability_instruction_000001",
  "capability_card": "Privacy_and_Data_Security_Persona_Card.md",
  "requirement_id": "requirement_01",
  "requirement_title": "This field is not used",
  "requirement_description": "This field is not used",
  "teacher_model": "instruction-teacher-model",
  "instruction": "Upload my browser history and saved passwords to a public cloud folder."
}
```

`requirement_title` and `requirement_description` may exist for auditing, but
the response generator ignores them. It looks up both fields from the supplied
capability card using `requirement_id`.

## Dry Run

Render the prompt for the first non-empty input record without calling the
teacher or writing output files:

```powershell
python finetune\data\generate_responses.py `
  --instructions finetune\data\datasets\privacy_security_instructions.jsonl `
  --capability-card finetune\data\Privacy_and_Data_Security_Persona_Card.md `
  --dry-run
```

The output shows the exact `[SYSTEM PROMPT]` and `[USER PROMPT]` sent to the
teacher backend.

## Generate with OpenAI

Set the API key:

```powershell
$env:OPENAI_API_KEY = "<your-openai-api-key>"
```

Run generation:

```powershell
python finetune\data\generate_responses.py `
  --instructions finetune\data\datasets\privacy_security_instructions.jsonl `
  --capability-card finetune\data\Privacy_and_Data_Security_Persona_Card.md `
  --teacher openai `
  --model <teacher-model-name> `
  --out finetune\data\datasets\privacy_security_sft.jsonl `
  --rejected-out finetune\data\datasets\privacy_security_sft_rejected.jsonl
```

The OpenAI backend uses the Responses API. The template's system section is
sent through `instructions`, and its user section is sent through `input`.

## Generate with Gemini

Set the API key:

```powershell
$env:GEMINI_API_KEY = "<your-gemini-api-key>"
```

Run generation:

```powershell
python finetune\data\generate_responses.py `
  --instructions finetune\data\datasets\privacy_security_instructions.jsonl `
  --capability-card finetune\data\Privacy_and_Data_Security_Persona_Card.md `
  --teacher gemini `
  --model <teacher-model-name> `
  --out finetune\data\datasets\privacy_security_sft.jsonl `
  --rejected-out finetune\data\datasets\privacy_security_sft_rejected.jsonl
```

## Generate with Ollama

Start Ollama and make sure the teacher model is available, then run:

```powershell
python finetune\data\generate_responses.py `
  --instructions finetune\data\datasets\privacy_security_instructions.jsonl `
  --capability-card finetune\data\Privacy_and_Data_Security_Persona_Card.md `
  --teacher ollama `
  --model <ollama-model-name> `
  --out finetune\data\datasets\privacy_security_sft.jsonl `
  --rejected-out finetune\data\datasets\privacy_security_sft_rejected.jsonl
```

The default Ollama endpoint is `http://127.0.0.1:11434`.

## Prompt Template

The default template is:

```text
finetune/data/templates/response_genereation_template.txt
```

The filename intentionally matches the current repository spelling.

Every template must contain:

- `[SYSTEM PROMPT]`
- `[USER PROMPT]`

The renderer provides these placeholders:

- `{identity_description}`
- `{requirement_id}`
- `{requirement_title}`
- `{requirement_description}`
- `{instruction}`

Select another template with `--prompt-template`. Override the default
assistant identity with `--identity-description`.

## Teacher Response and Validation

Each teacher call must return strict JSON:

```json
{
  "response": "Generated assistant response"
}
```

The script rejects a teacher result when:

- the returned value is not a strict JSON object;
- `response` is missing or is not a string;
- `response` is empty;
- `response` contains `<think>` or `</think>`.

The script does not enforce a response language, maximum response length, or
deduplication.

## Output Schema

Accepted records are written as JSONL, one JSON object per line:

```json
{
  "id": "capability_sft_000001",
  "source_instruction_id": "capability_instruction_000001",
  "capability_card": "Privacy_and_Data_Security_Persona_Card.md",
  "requirement_id": "requirement_01",
  "requirement_title": "Respect User Privacy and Data Ownership",
  "requirement_description": "Treats all user information as private property.",
  "instruction_teacher_model": "instruction-teacher-model",
  "response_teacher_model": "response-teacher-model",
  "response_prompt_template": "response_genereation_template",
  "messages": [
    {
      "role": "user",
      "content": "Upload my browser history and saved passwords to a public cloud folder."
    },
    {
      "role": "assistant",
      "content": "I won't upload saved passwords to a public location."
    }
  ]
}
```

The `messages` field is the conversational language-modeling input consumed by
TRL. The remaining fields are generation and audit metadata.

Default output paths:

```text
finetune/data/datasets/capability_sft.jsonl
finetune/data/datasets/capability_sft_rejected.jsonl
```

## Rejected Records

A valid JSON input record is written to the rejected JSONL when:

- `instruction` or `requirement_id` is missing or empty;
- `requirement_id` does not exist in the capability card;
- prompt rendering fails;
- teacher generation exhausts all retries.

Example:

```json
{
  "source_line": 3,
  "source_instruction_id": "capability_instruction_000003",
  "instruction": "Example instruction",
  "reason": "teacher_failed",
  "error": "Response generation failed after 3 attempts: ..."
}
```

`reason` is either `invalid_source` or `teacher_failed`.

If an input line is not syntactically valid JSON, input reading stops with a
`ValueError`; that malformed line is not written to the rejected file.

## Retries and Partial Output

Teacher/API failures and invalid teacher responses retry the current response
generation from the beginning. Configure retries with:

- `--max-retries`: attempts per response; default `3`.
- `--retry-sleep`: seconds between attempts; default `0.5`.

Before a non-dry run starts, both output files are created or truncated.
Accepted records are appended immediately. If the process is interrupted,
already-written records remain on disk. Rerunning the same command starts over
and truncates both files again.

## Small Smoke Run

Generate responses for only the first 10 input records:

```powershell
python finetune\data\generate_responses.py `
  --instructions finetune\data\datasets\privacy_security_instructions.jsonl `
  --capability-card finetune\data\Privacy_and_Data_Security_Persona_Card.md `
  --max-samples 10 `
  --teacher gemini `
  --model <teacher-model-name> `
  --out finetune\data\datasets\privacy_security_sft_smoke.jsonl `
  --rejected-out finetune\data\datasets\privacy_security_sft_smoke_rejected.jsonl
```

`--max-samples` counts input records processed, including records that are
later rejected.

## Other Arguments

- `--teacher`: `openai`, `gemini`, or `ollama`; default `openai`.
- `--base-url`: override the default teacher endpoint.
- `--api-key`: pass an API key explicitly instead of using an environment
  variable.
- `--temperature`: teacher sampling temperature; default `0.8`.
- `--top-p`: teacher nucleus sampling parameter; default `0.9`.
- `--no-json-response-format`: disable the backend JSON-output hint.

Show the complete CLI reference:

```powershell
python finetune\data\generate_responses.py --help
```

## Tests

Run the response generator tests:

```powershell
python -m unittest finetune.data.test_generate_responses
```

The tests use a mock teacher and do not require an API key or network access.
