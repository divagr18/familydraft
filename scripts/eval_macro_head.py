"""Macro head dev accuracy on the structured class (plan todo 16).

Runs one trunk forward per trace to get all hidden states, pairs each
macro-labeled continuation position with parser features, and measures the
(randomly initialized) macro head's argmax accuracy. Recorded to
runs/experts/macro_dev.json. The head is untrained in Phase 1, so this is a
baseline recording, not a performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--traces", default="runs/traces/Qwen3-0.6B/structured.jsonl")
    parser.add_argument("--task-class", default="structured")
    parser.add_argument("--max-positions", type=int, default=200)
    parser.add_argument("--out", default="runs/experts/macro_dev.json")
    args = parser.parse_args()

    import importlib.util
    import sys

    from transformers import AutoTokenizer

    from familydraft.draft.trunk import build_trunk_from_config
    from familydraft.experts.macro import MacroHead, parser_features_from_text
    from familydraft.experts.macro_render import build_renderer_from_config

    spec = importlib.util.spec_from_file_location(
        "derive_macro_labels", Path("scripts/derive_macro_labels.py")
    )
    dml = importlib.util.module_from_spec(spec)
    sys.modules["derive_macro_labels"] = dml
    spec.loader.exec_module(dml)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    renderer = build_renderer_from_config(Path("."), tokenizer, tokenizer.vocab_size)
    render_table, max_len = dml.build_render_table(renderer)

    trunk = build_trunk_from_config(Path("."))
    trunk = trunk.to(device).eval()
    head = MacroHead(trunk.hidden_size, renderer.num_macros).to(device).eval()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"eval_macro_head: missing {traces_path}", flush=True)
        return 2

    rows = []
    with traces_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    correct = 0
    evaluated = 0
    label_count = 0
    for row in rows:
        if evaluated >= args.max_positions:
            break
        prompt_ids = row["prompt_ids"]
        continuation = row["chosen_tokens"]
        labels = dml.derive_labels(continuation, render_table, max_len)
        label_count += sum(1 for x in labels if x >= 0)
        full_ids = torch.tensor(prompt_ids + continuation, dtype=torch.long).unsqueeze(0)
        with torch.inference_mode():
            hidden = trunk(full_ids.to(device), target_id=0)[0]
        prefix_len = len(prompt_ids)
        for i, label in enumerate(labels):
            if evaluated >= args.max_positions:
                break
            if label < 0:
                continue
            pos = prefix_len + i
            h = hidden[pos - 1].float()
            prefix_up_to = tokenizer.decode(
                full_ids[0, : pos + 1].tolist(), skip_special_tokens=True
            )
            feats = torch.tensor(parser_features_from_text(prefix_up_to), device=device)
            with torch.inference_mode():
                logits = head(h.unsqueeze(0), feats.unsqueeze(0))[0]
            pred = int(torch.argmax(logits))
            if pred == label:
                correct += 1
            evaluated += 1

    record = {
        "schema": "familydraft.macro_dev.v1",
        "repo": args.repo,
        "task_class": args.task_class,
        "macro_head_trained": False,
        "positions_evaluated": evaluated,
        "labeled_positions_in_traces": label_count,
        "accuracy": correct / max(1, evaluated),
        "random_baseline": 1.0 / renderer.num_macros,
        "num_macros": renderer.num_macros,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
