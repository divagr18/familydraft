"""Derive macro action labels from code/structured traces (plan todo 16).

For each position of a continuation, the deriver finds the macro whose rendered
token sequence exactly prefixes the oracle continuation there; that macro index
becomes the label (-1 when no macro matches). This produces parser-derived
supervision for the macro head from greedy traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_render_table(renderer) -> tuple[dict[tuple[int, ...], int], int]:
    table: dict[tuple[int, ...], int] = {}
    max_len = 0
    for idx, name in enumerate(renderer.names):
        ids = tuple(renderer.render(name))
        if ids:
            table.setdefault(ids, idx)
            max_len = max(max_len, len(ids))
    return table, max_len


def derive_labels(continuation: list[int], render_table, max_len: int) -> list[int]:
    labels: list[int] = []
    n = len(continuation)
    for i in range(n):
        matched = -1
        for length in range(min(max_len, n - i), 0, -1):
            idx = render_table.get(tuple(continuation[i : i + length]))
            if idx is not None:
                matched = idx
                break
        labels.append(matched)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--traces-dir", default="runs/traces/Qwen3-0.6B")
    parser.add_argument("--classes", default="code,structured")
    parser.add_argument("--out-dir", default="runs/macro_labels")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from familydraft.experts.macro_render import build_renderer_from_config

    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    vocab = tokenizer.vocab_size
    renderer = build_renderer_from_config(Path("."), tokenizer, vocab)
    render_table, max_len = build_render_table(renderer)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_class in args.classes.split(","):
        traces_path = Path(args.traces_dir) / f"{task_class}.jsonl"
        if not traces_path.exists():
            print(f"derive_macro_labels: skipping missing {traces_path}")
            continue
        rows = []
        labeled_positions = 0
        total_positions = 0
        with traces_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                continuation = row["chosen_tokens"]
                labels = derive_labels(continuation, render_table, max_len)
                labeled_positions += sum(1 for x in labels if x >= 0)
                total_positions += len(labels)
                rows.append({"trace_id": row["trace_id"], "labels": labels})
        out_path = out_dir / f"{task_class}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for r in rows:
                handle.write(json.dumps(r) + "\n")
        coverage = labeled_positions / max(1, total_positions)
        print(
            f"[{task_class}] {len(rows)} traces, macro-label coverage {coverage:.3f} "
            f"-> {out_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
