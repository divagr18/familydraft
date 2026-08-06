# M2 pivot options — oracle go/no-go decision package

**Status:** pre-registered M2 gate returned **NO-GO** (exit 77) on the local
dev-box diagnostic. Per the plan's kill branch, Waves D-G are held and this
decision package is surfaced to the user. Decision authority: **user**.

## 1. What the gate measured

Local campaign: 100 greedy traces (25 per class) from `Qwen/Qwen3-0.6B` on the
RTX 4060 dev box, analyzed by `scripts/oracle_analysis.py` against thresholds
pre-registered in `configs/oracle_thresholds.yaml` (committed before any run;
`scripts/m2_order_check.py` PASS).

| class      | positions | mean recovered | cov ≥1 tok | cov ≥2 tok | cov ≥4 tok |
| ---------- | --------- | -------------- | ---------- | ---------- | ---------- |
| chat       | 2935      | 0.320          | 0.058      | 0.056      | 0.055      |
| code       | 3433      | 4.134          | **0.569**  | 0.543      | 0.540      |
| math       | 4800      | 1.240          | 0.201      | 0.201      | 0.201      |
| structured | 2664      | 0.211          | 0.053      | 0.033      | 0.033      |

- Overall oracle-best expected speedup upper bound: **2.565** (gate requires ≥1.5) → speedup_ok PASS.
- Gate requires **both** `code` AND `structured` to reach ≥0.25 coverage-at-1-token, and ≥2 classes passing overall.
- Only `code` passes (0.569). `structured` is 0.053. → required_classes_ok FAIL, class_count_ok FAIL → **NO-GO**.

## 2. Critical scope caveat (read before deciding)

This verdict comes from a **dev-box diagnostic on Qwen3-0.6B**, the smallest
weakest member, run at capped scale (`local_run`: 25 prompts/class, 192 tokens).
The plan's registered trace campaign (todo 10) targets **4B / 8B / 14B on
RunPod** at full horizons (2048-4096 tokens). Stronger targets produce far more
regular structure. So this NO-GO is evidence, **not** the definitive campaign
verdict the gate was registered against.

## 3. Diagnostic insight: mechanisms under-capture JSON

Decoding structured traces shows the 0.6B model emits **clean, well-formed JSON**
(quoted keys, colons, commas, nested brackets, fenced code blocks). The structure
exists; my **v0 mechanism union fails to credit it**:

- Copy requires ≥4-token runs (`copy_suffix_min_match: 4`), but JSON repeats
  short token patterns (`"title"`, `":`, `","`, `"`), each typically 1-3 tokens.
- Macro ruleset v0 (`parse_state.py`) models bracket-stacks, code fences, bullets,
  numbered lists and colon-indent — **not** a JSON punctuation grammar, so the
  dense quote/colon/comma/closer regularity of JSON scores near zero.

Therefore the `structured` failure is substantially a **measurement/mechanism
artifact**, not proof that structured continuations are unpredictable.

## 4. Options

### Option A — Run the registered campaign on real targets before final judgment
Run todo 10 on RunPod against 4B / 8B / 14B (and keep the oracle analysis
identical). The gate thresholds were registered for that campaign. This is the
scientifically clean path: do not kill the direction on a weaker-than-registered
target. Requires RunPod credits (already available) and the existing scripts.
**Cost:** GPU-hours on A100-class pods; no new code.

### Option B — Improve the cheap mechanisms, then re-run the oracle (recommended)
The oracle is only as strong as the mechanism union it measures. Strengthen them,
re-run the same oracle, and let the improved measurement decide:
- Add a JSON-grammar macro expert state (quote/colon/comma/closer transitions) to
  `parse_state.py` — this is precisely the "code and syntax expert" the plan wants.
- Lower copy tolerance for structural tokens (allow 2-3-token structural runs) or
  add normalized structural retrieval.
- Both are already in-scope deliverables (todos 16-17); doing them first de-risks
  the oracle before spending big-target GPU-hours.
**Cost:** modest code, runs on the dev box, then re-run `oracle_analysis.py`.

### Option C — Narrow to a code-specialized drafter
`code` already passes strongly (0.569 coverage, 4.13 mean recovered). Pivot the
research to a code-focused heterogeneous drafter (macro + copy for code, drop the
structured/chat/math ambitions). This is a defensible narrower paper and matches
the strongest measured signal.
**Cost:** scope reduction; abandons the "4 task classes" breadth.

### Option D — Halt and archive
If the direction is no longer worth pursuing, archive the repo at the current
commit with the evidence package. Everything built (verification core, M1
equivalence gate, cost curve, eval manifest, trace+oracle pipeline) remains as
reusable artifacts.
**Cost:** none; forfeits the research direction.

## 5. Recommendation

**B then A.** Strengthen the v0 mechanisms (they are required deliverables anyway),
re-run the oracle locally to confirm `structured` coverage is a measurement
artifact, then run the registered campaign on real RunPod targets for the
definitive verdict. This spends the least GPU before the most and directly tests
the hypothesis that the NO-GO is a mechanism artifact rather than an intrinsic
ceiling. If, after stronger mechanisms AND stronger targets, `structured` still
fails the gate, fall back to Option C (code-specialized) or Option D.

## 6. Exact reproduction

```bash
uv run python scripts/oracle_analysis.py --selftest          # gate logic sanity
uv run python scripts/m2_order_check.py                      # pre-registration proof
uv run python scripts/gen_traces.py                          # regenerate local traces
uv run python scripts/oracle_analysis.py                     # recompute verdict
```

Artifacts: `runs/oracle/verdict.json`, `docs/reports/oracle_report.md`,
`runs/traces/Qwen3-0.6B/*.jsonl`, evidence log
`.omo/evidence/trace-campaign-local.log`.
