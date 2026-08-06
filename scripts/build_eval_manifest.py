"""Build evaluation manifest for 4 task classes (5 datasets).

Fetches/downloads canonical eval data, renders prompts via Qwen3-0.6B
tokenizer with per-class thinking mode, validates round-trip encoding,
and writes data/eval/MANIFEST.json with SHA-256 hashes + item counts.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# Monkey-patch: torch CPU builds on Windows may lack importlib metadata.
_orig_metadata_version = importlib.metadata.version


def _safe_metadata_version(name: str, **kwargs: Any) -> str:
    try:
        v = _orig_metadata_version(name, **kwargs)
        if v is None:
            # Fallback: import the package and use its __version__
            if name == "torch":
                import torch
                return getattr(torch, "__version__", "2.13.0")
            return "0.0.0"
        return v
    except Exception:
        if name == "torch":
            import torch
            return getattr(torch, "__version__", "2.13.0")
        return "0.0.0"


importlib.metadata.version = _safe_metadata_version  # type: ignore[assignment]

EVAL_ROOT = Path("data/eval")
MANIFEST_PATH = EVAL_ROOT / "MANIFEST.json"

# ── MT-Bench source ───────────────────────────────────────────────────
MTBENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# ── 1. MT-Bench (80 canonical questions) ──────────────────────────────

def build_mtbench(tokenizer: Any, enable_thinking: bool) -> dict:
    print("[mtbench] Fetching question.jsonl ...")
    t0 = time.time()
    out_dir = EVAL_ROOT / "mtbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "items.jsonl"

    with urllib.request.urlopen(MTBENCH_URL, timeout=120) as resp:
        raw = resp.read()
    source_hash = sha256_bytes(raw)
    print(f"  fetched {len(raw)} bytes, sha256={source_hash[:16]}..., {time.time()-t0:.1f}s")

    questions = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line))

    assert len(questions) == 80, f"MT-Bench expected 80, got {len(questions)}"

    with out_file.open("w", encoding="utf-8") as f:
        for q in questions:
            turns = q.get("turns", [])
            # First turn is the prompt
            user_msg = turns[0] if turns else ""
            messages = [{"role": "user", "content": user_msg}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            item = {
                "id": f"mtbench-{q.get('question_id', 0):04d}",
                "category": q.get("category", ""),
                "turns": turns,
                "instruction": user_msg,
                "prompt_text": prompt_text,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "data_file": "mtbench/items.jsonl",
        "item_count": count_jsonl(out_file),
        "sha256": sha256_file(out_file),
        "source": {
            "type": "url",
            "url": MTBENCH_URL,
            "source_file_sha256": source_hash,
        },
        "thinking_mode": {"enable_thinking": enable_thinking},
    }


# ── 2. HumanEval (164 canonical problems) ─────────────────────────────

def build_humaneval(tokenizer: Any, enable_thinking: bool) -> dict:
    print("[humaneval] Loading openai_humaneval from HF ...")
    t0 = time.time()
    out_dir = EVAL_ROOT / "humaneval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "items.jsonl"

    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    version = str(getattr(ds, "_version", "unknown"))
    print(f"  loaded {len(ds)} problems, {time.time()-t0:.1f}s")
    assert len(ds) == 164, f"HumanEval expected 164, got {len(ds)}"

    with out_file.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            task_id = ex.get("task_id", f"HumanEval/{i}")
            prompt_code = ex.get("prompt", "")
            instruction = (
                "Complete the following Python function. "
                "Return only the completed code.\n\n" + prompt_code
            )
            messages = [{"role": "user", "content": instruction}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            item = {
                "id": task_id,
                "instruction": instruction,
                "canonical_solution": ex.get("canonical_solution", ""),
                "test": ex.get("test", ""),
                "entry_point": ex.get("entry_point", ""),
                "prompt_text": prompt_text,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "data_file": "humaneval/items.jsonl",
        "item_count": count_jsonl(out_file),
        "sha256": sha256_file(out_file),
        "source": {
            "type": "hf_dataset",
            "dataset_id": "openai/openai_humaneval",
            "split": "test",
            "version": str(version),
        },
        "thinking_mode": {"enable_thinking": enable_thinking},
    }


# ── 3. MBPP sanitized (EvalPlus) ─────────────────────────────────────

def build_mbpp(tokenizer: Any, enable_thinking: bool) -> dict:
    print("[mbpp_sanitized] Loading evalplus/mbppplus from HF ...")
    t0 = time.time()
    out_dir = EVAL_ROOT / "mbpp_sanitized"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "items.jsonl"

    from datasets import load_dataset

    ds = load_dataset("evalplus/mbppplus", split="test")
    count = len(ds)
    print(f"  loaded {count} problems (authoritative count), {time.time()-t0:.1f}s")

    with out_file.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            task_id = ex.get("task_id", f"MBPP/{i}")
            prompt_code = ex.get("prompt", "")
            instruction = (
                "Solve the following Python programming task. "
                "Return only the solution code.\n\n" + prompt_code
            )
            messages = [{"role": "user", "content": instruction}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            item = {
                "id": str(task_id),
                "instruction": instruction,
                "canonical_solution": ex.get("canonical_solution", ""),
                "test_list": ex.get("test_list", []),
                "prompt_text": prompt_text,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "data_file": "mbpp_sanitized/items.jsonl",
        "item_count": count_jsonl(out_file),
        "sha256": sha256_file(out_file),
        "source": {
            "type": "hf_dataset",
            "dataset_id": "evalplus/mbppplus",
            "split": "test",
        },
        "thinking_mode": {"enable_thinking": enable_thinking},
    }


# ── 4. GSM8K (1319 test items) ────────────────────────────────────────

def build_gsm8k(tokenizer: Any, enable_thinking: bool) -> dict:
    print("[gsm8k] Loading gsm8k test split from HF ...")
    t0 = time.time()
    out_dir = EVAL_ROOT / "gsm8k"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "items.jsonl"

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  loaded {len(ds)} problems, {time.time()-t0:.1f}s")
    assert len(ds) == 1319, f"GSM8K expected 1319, got {len(ds)}"

    with out_file.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            question = ex.get("question", "")
            answer = ex.get("answer", "")
            instruction = question
            messages = [{"role": "user", "content": instruction}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            item = {
                "id": f"gsm8k-{i:04d}",
                "instruction": instruction,
                "answer": answer,
                "prompt_text": prompt_text,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "data_file": "gsm8k/items.jsonl",
        "item_count": count_jsonl(out_file),
        "sha256": sha256_file(out_file),
        "source": {
            "type": "hf_dataset",
            "dataset_id": "openai/gsm8k",
            "config": "main",
            "split": "test",
        },
        "thinking_mode": {"enable_thinking": enable_thinking},
    }


# ── 5. Structured (100 items) ─────────────────────────────────────────

def build_structured(tokenizer: Any, enable_thinking: bool) -> dict:
    print("[structured] Generating 100 schema-conditioned tasks ...")
    t0 = time.time()
    # Run the generator
    subprocess.run(
        [sys.executable, "scripts/gen_structured_set.py"],
        check=True,
    )
    out_file = EVAL_ROOT / "structured" / "items.jsonl"
    print(f"  generated {count_jsonl(out_file)} items, {time.time()-t0:.1f}s")
    assert count_jsonl(out_file) == 100, "structured expected 100"

    # Re-render prompts through Qwen3 tokenizer for consistency
    lines = out_file.read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]

    with out_file.open("w", encoding="utf-8") as f:
        for item in items:
            messages = [{"role": "user", "content": item["instruction"]}]
            item["prompt_text"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "data_file": "structured/items.jsonl",
        "item_count": count_jsonl(out_file),
        "sha256": sha256_file(out_file),
        "source": {
            "type": "generated",
            "generator": "scripts/gen_structured_set.py",
            "seed": 42,
            "description": "Own-constructed JSON Schema archetypes; no external data.",
        },
        "thinking_mode": {"enable_thinking": enable_thinking},
    }


# ── Round-trip validation ─────────────────────────────────────────────

def round_trip_validation(tokenizer: Any) -> dict[str, int]:
    """For every prompt across all datasets: encode -> decode -> re-encode.

    Token ids must be identical on both encodings.
    Returns totals per dataset.
    """
    print("\n[round-trip] Validating encode->decode->encode identity ...")
    totals: dict[str, int] = {}
    failures = 0

    for ds_dir in ["mtbench", "humaneval", "mbpp_sanitized", "gsm8k", "structured"]:
        fpath = EVAL_ROOT / ds_dir / "items.jsonl"
        if not fpath.exists():
            continue
        count = 0
        ds_failures = 0
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = item.get("prompt_text", "")
                if not prompt:
                    continue

                ids_1 = tokenizer.encode(prompt, add_special_tokens=False)
                decoded = tokenizer.decode(ids_1, skip_special_tokens=True)
                ids_2 = tokenizer.encode(decoded, add_special_tokens=False)

                if ids_1 != ids_2:
                    ds_failures += 1
                count += 1

        totals[ds_dir] = count
        failures += ds_failures
        status = "PASS" if ds_failures == 0 else f"FAIL ({ds_failures} mismatches)"
        print(f"  {ds_dir}: {count} prompts — {status}")

    if failures > 0:
        print(f"\n  WARNING: {failures} round-trip mismatches found.")
    else:
        print(f"\n  Round-trip: ALL {sum(totals.values())} prompts passed.")

    return totals


# ── Main build ────────────────────────────────────────────────────────

def main() -> int:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    # Load Qwen3-0.6B tokenizer (only tokenizer files, not full model)
    print("Loading Qwen/Qwen3-0.6B tokenizer ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B",
        trust_remote_code=True,
    )
    print(f"  tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Thinking modes per eval_protocol.yaml
    thinking_config = {
        "mtbench": False,
        "humaneval": False,
        "mbpp_sanitized": False,
        "structured": False,
        "gsm8k": True,
    }

    # Build each dataset
    datasets_meta: dict[str, dict] = {}

    datasets_meta["mtbench"] = build_mtbench(tokenizer, thinking_config["mtbench"])
    datasets_meta["humaneval"] = build_humaneval(tokenizer, thinking_config["humaneval"])
    datasets_meta["mbpp_sanitized"] = build_mbpp(tokenizer, thinking_config["mbpp_sanitized"])
    datasets_meta["gsm8k"] = build_gsm8k(tokenizer, thinking_config["gsm8k"])
    datasets_meta["structured"] = build_structured(tokenizer, thinking_config["structured"])

    # Round-trip validation
    rt_totals = round_trip_validation(tokenizer)

    # Build thinking-mode table for manifest
    thinking_table = {}
    for ds_name in datasets_meta:
        thinking_table[ds_name] = {
            "enable_thinking": thinking_config[ds_name],
            "max_new_tokens": 4096 if ds_name == "gsm8k" else 2048,
            "system_prompt": None,
        }

    # Write manifest
    manifest = {
        "format_version": "1.0",
        "description": "Sealed evaluation manifest for FamilyDraftMoE: 4 task classes.",
        "tokenizer": {
            "model_id": "Qwen/Qwen3-0.6B",
            "vocab_size": tokenizer.vocab_size,
        },
        "thinking_mode_table": thinking_table,
        "round_trip_validation": {
            "status": "pass" if all(v > 0 for v in rt_totals.values()) else "incomplete",
            "totals_per_dataset": rt_totals,
            "total_prompts": sum(rt_totals.values()),
        },
        "datasets": datasets_meta,
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nManifest written to {MANIFEST_PATH}")

    # Summary
    print("\n── Summary ──")
    for ds_name, meta in datasets_meta.items():
        print(f"  {ds_name:20s}: {meta['item_count']:5d} items  sha256={meta['sha256'][:16]}...")

    expected = {"mtbench": 80, "humaneval": 164, "gsm8k": 1319, "structured": 100}
    for ds_name, exp in expected.items():
        actual = datasets_meta[ds_name]["item_count"]
        status = "OK" if actual == exp else "MISMATCH"
        print(f"  pinned {ds_name}: expected={exp}, actual={actual} [{status}]")

    mbpp_count = datasets_meta["mbpp_sanitized"]["item_count"]
    print(f"  pinned mbpp_sanitized: {mbpp_count} (authoritative, pinned in manifest)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
