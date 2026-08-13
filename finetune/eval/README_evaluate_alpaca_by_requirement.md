# Evaluate Alpaca Outputs by Requirement

`evaluate_alpaca_by_requirement.py` compares model outputs with reference
outputs using AlpacaEval, independently for every `requirement_id`.

Each requirement is evaluated with AlpacaEval itself. The script does not
derive category scores by filtering one combined annotation file, so win rate,
standard error, draws, and length-controlled metrics retain AlpacaEval's native
calculation behavior.

Every AlpacaEval call explicitly receives a requirement-specific output path:

```text
--output_path <output-dir>/<requirement_id>
```

`find_leaderboard()` searches that same directory for the generated
`leaderboard.csv`.

All paths passed to the AlpacaEval subprocess use forward slashes, including
`model_outputs`, `reference_outputs`, `annotators_config`, `output_path`, and a
path supplied as `alpaca_eval_command`. Python still uses `Path` internally for
local filesystem operations.

## Input

Both `--model-outputs` and `--reference-outputs` must be JSON arrays using the
schema documented in `README_generate_alpaca.md`. Every record must contain:

- `instruction`
- `output`
- `requirement_id`

Model outputs must also have exactly one non-empty `generator` value.

Records are paired by `source_instruction_id` when available, then by `id`, and
finally by exact `instruction`. Model and reference files may use different row
orders, but their pairing keys, instructions, and requirement IDs must match.

## Run

Install AlpacaEval in the environment used to run the script. This repository's
existing evaluator artifacts were produced with `alpaca_eval==0.6.6`.

```powershell
python finetune/eval/evaluate_alpaca_by_requirement.py `
  --model-outputs finetune/eval/test_model_outputs_oki.json `
  --reference-outputs finetune/eval/test_reference_outputs_oki.json `
  --annotators-config E:/LLM/oki/oki_agent/finetune/eval/evaluator_configs/test_xiongmao `
  --input-keys instruction,requirement_title,requirement_description `
  --output-dir finetune/eval/results/privacy_security
```

`--input-keys` is passed to AlpacaEval as `--input_keys`. Its default value is
`instruction,requirement_title,requirement_description`, so the explicit
argument above may
be omitted. Override it when the annotator prompt expects different input
fields.

`--annotators-config` accepts the same evaluator name or path accepted by
AlpacaEval. When the executable is not named `alpaca_eval`, pass its path with:

```powershell
--alpaca-eval-command D:/miniconda3/envs/LLM/Scripts/alpaca_eval.exe
```

## Output

For each category, the output directory contains the filtered model/reference
inputs and AlpacaEval's native artifacts:

```text
results/privacy_security/
  leaderboard.csv
  requirement_01/
    model_outputs.json
    reference_outputs.json
    ... AlpacaEval annotations and leaderboard ...
  requirement_02/
    ...
```

The top-level `leaderboard.csv` contains one row per requirement:

```csv
requirement_id,requirement_title,generator,win_rate,standard_error,mode,avg_length,n_wins,n_wins_base,n_draws,n_total,discrete_win_rate,length_controlled_winrate,lc_standard_error
requirement_01,Respect User Privacy and Data Ownership,qwen3.6-35b-a3b-lora,75.0,5.0,community,120,3,1,0,4,75.0,70.0,6.0
```

The metric columns match the format of
`evaluator_configs/test_xiongmao/leaderboard.csv`. The first three columns add
the requirement identity and evaluated generator.

The script refuses to overwrite a non-empty requirement result directory. Use
a new `--output-dir` or explicitly remove old results before rerunning.

## Tests

```powershell
python -m unittest finetune.eval.test_evaluate_alpaca_by_requirement
```

The tests mock the AlpacaEval process and do not call a judge API.
