# Trace format (JSONL)

Campaign traces (todo 10) record one JSONL line per generated step of a
greedy target decode, captured through `familydraft.targets.wrapper.TargetModel`
(todo 4). One shard file = one `<target>/<task_class>` shard; traces live on
the RunPod network volume under `/workspace/traces/<target>/<shard>.jsonl`
and are checksummed into `MANIFEST.traces.json`.

## Line schema

| Field            | Type                    | Description                                                                                     |
| ---------------- | ----------------------- | ----------------------------------------------------------------------------------------------- |
| `step`           | int, ≥ 0                | 0-based index of the generation step within this sequence.                                       |
| `chosen_token`   | int in `[0, 151936)`    | The token the target actually picked at this step (greedy argmax).                               |
| `topk`           | object `{ids, logits, ranks}` | Top-k snapshot of the distribution AFTER consuming the prefix, captured by `TargetModel.topk_logits`. |
| `topk.ids`       | array[int] length `k`   | Vocab ids in descending-logit order; every id in `[0, 151936)`.                                  |
| `topk.logits`    | array[float] length `k` | Logits of `topk.ids`, descending. Full-precision floats at capture time.                          |
| `topk.ranks`     | array[int] length `k`   | Dense rank of each id over the full vocab: count of tokens with a strictly greater logit (ties share a rank; argmax ⇒ rank 0). |
| `latency_ms`     | float, ≥ 0              | Wall time of this decode step in milliseconds, from CUDA-event timing (todo 5) with explicit `torch.cuda.synchronize`. |
| `target_id`      | string                  | Repo id of the target that produced the trace, e.g. `Qwen/Qwen3-8B` (from `configs/targets.yaml`). |
| `config_sha256`  | string (64 hex chars)   | Fingerprint of the campaign config (todo 5 canonical-JSON sha256) that pinned this run.          |
| `seed`           | int                     | Global seed for the run (greedy traces: the seed set at start; sampling traces: per-sequence).    |

Campaign default is `k = 64` (ids + logits + rank per step, per todo 10's
pinned capture set: argmax token + entropy derivable from the full snapshot +
top-64). JSON numbers; no NaN/Inf allowed (emit `null` for undefined values,
which validators treat as a corruption signal).

## Example line (illustrative values)

```json
{"step": 0, "chosen_token": 151644, "topk": {"ids": [151644, 8423, 311], "logits": [12.875, 9.125, 8.6875], "ranks": [0, 1, 2]}, "latency_ms": 11.42, "target_id": "Qwen/Qwen3-8B", "config_sha256": "b5bb9d8014a0f9b1d61e21e796d78dccdf1352f23cd32812f4850b878ae4944c", "seed": 0}
```

## Invariants

- One JSON object per line; shards are concatenation-safe.
- `topk.ids[0] == chosen_token` and `topk.ranks[0] == 0` for greedy traces.
- `ids`, `logits`, `ranks` all have the same length `k`; arrays are ordered
  by descending logit.
- Vocab ids are always in `[0, 151936)` (shared Qwen3 tokenizer vocab).
- Determinism contract: same `target_id` + `config_sha256` + `seed` + prompt
  ⇒ bit-identical `chosen_token` sequence on identical hardware; the harness
  self-checks this and fails loudly on divergence.
