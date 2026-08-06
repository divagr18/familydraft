"""Oracle predictability analysis + go/no-go gate (plan todo 11). MILESTONE 2.

For every position of every trace, with oracle knowledge of the target's
continuation, measures how many upcoming tokens a cheap mechanism class
could have recovered:

  copy/repeat  - longest token run (>= copy_suffix_min_match) appearing in
                 prompt or preceding output (repeat = match inside the recent
                 generated window);
  macro        - parse_state v0 structural candidates matched against the
                 decoded oracle continuation (v0 per-token decode
                 approximation, documented).

recovered(i) = max over mechanisms, capped at the speedup horizon.
Gate (pre-registered in configs/oracle_thresholds.yaml, committed before any
analysis run): required classes pass coverage-at-1-token AND overall
oracle-best expected speedup >= threshold. Exit 0 = GO, exit 77 = NO-GO.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _contains_seq(haystack: tuple[int, ...], needle: tuple[int, ...]) -> bool:
    n = len(needle)
    if n == 0 or len(haystack) < n:
        return False
    for j in range(len(haystack) - n + 1):
        if haystack[j : j + n] == needle:
            return True
    return False


def _copy_length(tokens: list[int], prompt: list[int], i: int, max_len: int, min_match: int) -> int:
    remaining = min(max_len, len(tokens) - i)
    if remaining < min_match:
        return 0
    context = tuple(prompt) + tuple(tokens[:i])
    for length in range(remaining, min_match - 1, -1):
        needle = tuple(tokens[i : i + length])
        if _contains_seq(context, needle):
            return length
    return 0


def _macro_length(
    prefix_text: str, token_texts: list[str], i: int, max_len: int
) -> int:
    from familydraft.experts.parse_state import parse_scan

    state = parse_scan(prefix_text)
    if not state.candidates:
        return 0
    best = 0
    for candidate in state.candidates:
        matched = 0
        built = ""
        for length in range(1, min(max_len, len(token_texts) - i) + 1):
            built += token_texts[i + length - 1]
            if candidate.startswith(built):
                matched = length
            else:
                break
        best = max(best, matched)
    return best


def analyze_traces(rows: list[dict], thresholds: dict, tokenizer=None) -> dict:
    definitions = thresholds["definitions"]
    horizon = thresholds["gate"]["speedup_horizon"]
    min_match = definitions["copy_suffix_min_match"]
    depths = definitions["depths"]

    per_class: dict[str, dict[str, float | int]] = {}
    all_recovered: list[int] = []
    for row in rows:
        task_class = row["task_class"]
        prompt = row["prompt_ids"]
        tokens = row["chosen_tokens"]
        stats = per_class.setdefault(
            task_class,
            {"positions": 0, "recovered_sum": 0, **{f"cov_{d}": 0 for d in depths}},
        )
        if tokenizer is not None:
            prompt_text = tokenizer.decode(prompt, skip_special_tokens=True)
            token_texts = [tokenizer.decode([t]) for t in tokens]
        else:
            prompt_text = ""
            token_texts = []
        prefix_text = prompt_text
        for i in range(len(tokens)):
            copy_len = _copy_length(tokens, prompt, i, horizon, min_match)
            if tokenizer is not None:
                macro_len = _macro_length(prefix_text, token_texts, i, horizon)
                prefix_text += token_texts[i]
            else:
                macro_len = 0
            recovered = min(max(copy_len, macro_len), horizon)
            stats["positions"] += 1
            stats["recovered_sum"] += recovered
            for d in depths:
                if recovered >= d:
                    stats[f"cov_{d}"] += 1
            all_recovered.append(recovered)

    result: dict[str, dict] = {}
    for task_class, stats in per_class.items():
        n = max(1, stats["positions"])
        result[task_class] = {
            "positions": stats["positions"],
            "mean_recovered": stats["recovered_sum"] / n,
            "coverage": {d: stats[f"cov_{d}"] / n for d in depths},
            "speedup_upper_bound": 1.0 + min(horizon, stats["recovered_sum"] / n),
        }
    total = max(1, len(all_recovered))
    overall = {
        "positions": len(all_recovered),
        "mean_recovered": sum(all_recovered) / total,
        "speedup_upper_bound": 1.0 + min(horizon, sum(all_recovered) / total),
    }
    return {"classes": result, "overall": overall}


def decide(metrics: dict, thresholds: dict) -> dict:
    gate = thresholds["gate"]
    min_cov = gate["min_position_coverage_at_1_token"]
    passing = [
        c
        for c, m in metrics["classes"].items()
        if m["coverage"][1] >= min_cov
    ]
    required_ok = all(c in passing for c in gate["required_classes"])
    count_ok = len(passing) >= gate["min_classes_passing"]
    speedup_ok = metrics["overall"]["speedup_upper_bound"] >= gate["min_oracle_expected_speedup"]
    go = required_ok and count_ok and speedup_ok
    return {
        "verdict": "GO" if go else "NO-GO",
        "exit_code": 0 if go else 77,
        "passing_classes": sorted(passing),
        "required_classes_ok": required_ok,
        "class_count_ok": count_ok,
        "speedup_ok": speedup_ok,
        "gate": gate,
    }


def _synthetic_rows(kind: str, thresholds: dict) -> list[dict]:
    horizon = thresholds["gate"]["speedup_horizon"]
    rng = random.Random(7 if kind == "above" else 13)
    rows = []
    for task_class in ("code", "structured", "chat", "math"):
        for k in range(3):
            if kind == "above":
                phrase = [rng.randrange(100_000, 150_000) for _ in range(horizon)]
                prompt = list(phrase) + [rng.randrange(100_000, 150_000) for _ in range(4)]
                tokens = phrase * 6
            else:
                seen: set[int] = set()

                def fresh() -> int:
                    while True:
                        t = rng.randrange(100_000, 150_000)
                        if t not in seen:
                            seen.add(t)
                            return t

                prompt = [fresh() for _ in range(12)]
                tokens = [fresh() for _ in range(48)]
            rows.append(
                {
                    "trace_id": f"synthetic-{kind}-{task_class}-{k}",
                    "task_class": task_class,
                    "prompt_ids": prompt,
                    "chosen_tokens": tokens,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", default=None)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    thresholds = _load_yaml(Path("configs/oracle_thresholds.yaml"))

    if args.selftest:
        for kind, expected in (("above", "GO"), ("below", "NO-GO")):
            rows = _synthetic_rows(kind, thresholds)
            metrics = analyze_traces(rows, thresholds, tokenizer=None)
            decision = decide(metrics, thresholds)
            status = "PASS" if decision["verdict"] == expected else "FAIL"
            print(
                f"[selftest:{kind}] expected={expected} got={decision['verdict']} "
                f"speedup_ub={metrics['overall']['speedup_upper_bound']:.3f} {status}"
            )
            if decision["verdict"] != expected:
                return 1
        print("oracle_analysis selftest: PASS (both directions)")
        return 0

    traces_dir = Path(args.traces_dir) if args.traces_dir else None
    if traces_dir is None:
        model_name = args.repo.split("/")[-1]
        traces_dir = Path("runs/traces") / model_name
    files = sorted(traces_dir.glob("*.jsonl"))
    if not files:
        print(f"oracle_analysis: no traces found in {traces_dir}", file=sys.stderr)
        return 2
    rows = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.repo)
    except Exception as exc:  # macro class degrades gracefully to copy-only
        print(f"oracle_analysis: tokenizer unavailable ({exc}); macro class disabled")

    metrics = analyze_traces(rows, thresholds, tokenizer=tokenizer)
    decision = decide(metrics, thresholds)

    out_dir = Path("runs/oracle")
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "familydraft.oracle_verdict.v1",
        "produced_utc": datetime.now(timezone.utc).isoformat(),
        "traces_dir": str(traces_dir),
        "target_id": args.repo,
        "trace_count": len(rows),
        "metrics": metrics,
        "decision": decision,
    }
    (out_dir / "verdict.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    report = ["# Oracle predictability report (M2)", ""]
    report.append(f"- traces: {len(rows)} from `{traces_dir}` (target `{args.repo}`)")
    report.append(
        f"- overall speedup upper bound: {metrics['overall']['speedup_upper_bound']:.3f}"
    )
    report.append(f"- **verdict: {decision['verdict']}**")
    report.append("")
    report.append("| class | positions | mean recovered | cov@d>=1 | cov@d>=2 | cov@d>=4 |")
    report.append("| --- | --- | --- | --- | --- | --- |")
    for c, m in sorted(metrics["classes"].items()):
        cov = m["coverage"]
        report.append(
            f"| {c} | {m['positions']} | {m['mean_recovered']:.3f} "
            f"| {cov[1]:.3f} | {cov.get(2, 0):.3f} | {cov.get(4, 0):.3f} |"
        )
    report.append("")
    report.append(f"Gate: {json.dumps(decision, indent=2)}")
    report_path = Path("docs/reports/oracle_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps(decision, indent=2))
    print(f"verdict.json -> {out_dir / 'verdict.json'}; report -> {report_path}")
    return decision["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
