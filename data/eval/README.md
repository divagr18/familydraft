# MoE Evaluation Datasets

Evaluation datasets for FamilyDraftMoE project.

## Datasets

| Name | Count | Description |
|------|-------|-------------|
| mtbench | 80 | MT-Bench canonical questions |
| humaneval | 164 | OpenAI HumanEval code generation |
| mbpp_sanitized | 974 | EvalPlus sanitized MBPP |
| gsm8k | 1319 | GSM8K math reasoning (test split) |
| structured | 100 | Custom JSON-schema generation tasks |

## Sources

- **mtbench**: FastChat llm_judge (https://github.com/lm-sys/FastChat)
- **humaneval**: HuggingFace `openai_humaneval` dataset
- **mbpp_sanitized**: HuggingFace `evalplus/mbppplus` dataset
- **gsm8k**: HuggingFace `gsm8k` dataset (main config, test split)
- **structured**: Generated from custom JSON schemas

## Licenses

- MT-Bench: Apache 2.0
- HumanEval: MIT
- MBPP: Apache 2.0
- GSM8K: MIT
- Structured: Custom schemas (no external data)

## Manifest

See `MANIFEST.json` for SHA-256 hashes, item counts, and source metadata.

## Usage

```bash
# Build manifest (downloads datasets)
uv run python scripts/build_eval_manifest.py

# Verify manifest integrity
uv run python scripts/verify_manifest.py
```
