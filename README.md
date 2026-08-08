# FamilyDraftMoE

Same-family heterogeneous speculative-drafting mixture-of-experts for the Qwen3
family: a shared truncated trunk, four heterogeneous experts (copy / general /
macro / rejection-memory), a utility router, and a DAG-fused speculative
decoder verified against the target. Full research plan:
`.omo/plans/familydraft-moe.md`.

## Repo map

```
src/familydraft/
  draft/trunk.py          shared truncated trunk + target-variant conditioning
  draft/                   (distillation / trace helpers)
  experts/copy.py         copy/retrieval expert (cheap, repetition-friendly)
  experts/general.py      neural continuation expert over trunk hidden states
  experts/macro.py        structural macro expert + parser features
  experts/parse_state.py  lightweight structural parse state machine
  experts/reject_memory.py online rejection-memory expert
  verify/dag.py           candidate-DAG trie with budget pruning
  eval/draft_loop.py      chain speculative loop (KV-cache incremental drafter)
  eval/draft_dag.py       router-driven multi-expert DAG speculator (thesis)
  eval/flops.py           FLOP accounting for equal-FLOP baseline comparisons
  router/router.py        contextual-bandit utility router
  targets/wrapper.py      target-model wrapper (greedy decode, batch traces)
scripts/
  run_baselines.py        Phase-1 baseline harness (--all, --ablation)
  run_phase1.py           aggregate baselines -> runs/results/phase1.csv (+ --row)
  m3_verdict.py           pre-registered M3 verdict computation
  m3_order_check.py       pre-registration ordering proof
  f1_audit.py             F1 plan-compliance audit (local subset)
  f2_f4_audit.py          F2/F4 code-quality + scope-fidelity audit
  check_determinism.py    baseline determinism self-check
  build_evidence_index.py / check_evidence_index.py   evidence package (todo 30)
  setup_eagle3.sh / eval_eagle3.py / resume_baseline.sh   EAGLE-3 pod pipeline
  run_phase1_8b.sh        one-command pod campaign orchestrator
  pod_setup.sh            idempotent RunPod bootstrap
configs/                  verdict protocol, baseline schema, ablations/, gates
data/eval/                sealed eval manifest (SHA-256 verifiable)
docs/reports/             oracle + Phase-1 verdict reports
docs/paper/               paper skeleton (claims.csv -> evidence index)
EVIDENCE_INDEX.json       sha256 + git-sha index of every run/report artifact
```

## Reproduce

Setup (local dev box): `uv sync` then `uv run pytest tests/`.

One command per milestone:

- **M1 — verification equivalence:** `uv run pytest tests/test_reference_verifier.py tests/test_verify_equivalence.py`
- **M2 — oracle gate:** `uv run python scripts/m2_order_check.py` (pre-registration proof; oracle report in `docs/reports/oracle_report.md`)
- **M3 — Phase-1 campaign (local 0.6B smoke):** `uv run python scripts/run_baselines.py --all --general-checkpoint <ckpt> --router-weights configs/router_weights.json && uv run python scripts/run_phase1.py && uv run python scripts/m3_verdict.py`
- **M3 — pod 8B campaign (thesis measurement):** `bash scripts/run_phase1_8b.sh` (RunPod A100-80GB; trains general expert, runs 6 baselines + 12-config ablation matrix, then aggregates and computes the verdict)
- **EAGLE-3 baseline:** `bash scripts/setup_eagle3.sh` then the printed SpecForge training command, then `python scripts/eval_eagle3.py`
- **Evidence package:** `uv run python scripts/build_evidence_index.py && uv run python scripts/check_evidence_index.py`

## Status

M3 infrastructure is shipped with the local 0.6B campaign run under the
pre-registered protocol: honest FAIL (the DAG loses to vanilla and to the
equal-FLOP dense baseline at 0.6B scale — see `docs/reports/phase1_verdict.md`).
The 8B pod campaign + EAGLE-3 training are pod-deferred.
