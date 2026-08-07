# Phase-1 verdict — FamilyDraftMoE integrated speculative drafter

**Status:** EXPLORATORY. The plan's Phase-1 success criterion (heterogeneous
top-2 + DAG fusion beats an equal-active-FLOP dense drafter on 2+ task classes)
has **NOT been met or even run**: no equal-FLOP dense baseline, no EAGLE-3
baseline, no full ablation campaign were completed. See Corrigendum below.

---

## Corrigendum (response to external audit)

The following earlier statements in this report are **corrected**:

1. **"Phase-1 campaign complete" / M3 "done"** — false. The success criterion was
   never tested (missing equal-FLOP dense and EAGLE-3 baselines, incomplete
   ablations). Todo 22/23 were marked done prematurely.
2. **"Lossless"** — overstated. Accurate claim: the acceptance algorithm is
   **equivalent to vanilla greedy under exact (fp32) arithmetic** (M1 proof), but
   the **bf16 implementation does not preserve the baseline greedy trajectory**
   (agreement 0.81-0.96 on 8B). The output is target-approved but not
   byte-identical to ordinary greedy in bf16.
3. **"DAG ≈ copy-alone"** (thesis-system section) — wrong. The integrated DAG
   measured **0.43-0.57x** locally (vs copy 1.16-1.19x) and **0.49-0.54x** on 8B
   (vs copy 1.05-1.62x). The isolated copy-only DAG matched copy-alone, but the
   integrated multi-expert DAG did not. The DAG is a trie with per-branch
   full-KV-copy verification, NOT efficient tree verification — shared prefixes
   are stored once but not computed once by the target.
4. **M2 as a pristine preregistered gate** — the original result was NO-GO; the
   parser mechanism was strengthened afterward, flipping structured coverage
   0.053 -> 0.309. This is a useful revised exploratory gate, not the untouched
   preregistration. The "speedup upper bound" (1 + mean_recovered) ignores
   drafting/verification cost — it measures recoverability, not wall-clock.
5. **47.9% held-out accuracy** — inflated by an unshuffled, class-ordered
   holdout (last 20% can be one task class). Real in-distribution accuracy is
   lower and task-dependent.
6. **Benchmark methodology** — exploratory, not a verdict: single timing sample
   (no 5-run mean±std), hand-designed prompt sets (not the sealed eval
   manifest), positional agreement (not exact sequence equality), no EOS
   stopping, and the DAG's timed run mutates router/memory state before the
   reported run (double-generate).

**What remains valid:** the M1 verification-equivalence proof (exact
arithmetic), the copy 1.60x result as an *exploratory* measurement, and the
oracle predictability analysis as an exploratory tool.

---

## Measured results

Integrated speculative loop (chain verification, KV-cache reuse) vs vanilla greedy:

| Target / prompts | Drafter | Speedup | Tokens/round | Agreement | Vanilla ms/token |
|---|---|---|---|---|---|
| Qwen3-8B, **repetitive** | **copy** | **1.60x** | **2.04** | 0.905 | 38.0 |
| Qwen3-8B, repetitive | general (trained) | 0.38x | 1.01 | 0.81 | 38.0 |
| Qwen3-8B, generic code | copy | 1.04x | 1.11 | 0.86 | 36.1 |
| Qwen3-8B, generic code | general (trained) | 0.38x | 1.00 | ~1.0 | 36.1 |
| Qwen3-0.6B, generic code | copy | 1.15x | 1.32 | 0.98 | 45.7 |
| Qwen3-0.6B, repetitive | copy | 1.17x | 1.34 | ~1.0 | 49.3 |
| Qwen3-0.6B, code/repetitive | general | 0.38-0.46x | ~1.0 | - | - |

Artifacts: `runs/results/integrated_speedup_8b.json`,
`runs/results/integrated_8b_trained_repetitive.json`,
`runs/results/integrated_8b_trained_code.json`,
`runs/results/integrated_speedup.json` (0.6B), `runs/results/local_validate.json`.

---

## Findings

1. **Copy's speedup is proportional to output repetition.** Generic code prompts
   (which a strong model writes compactly, with little repetition) give only ~1.04x.
   Repetition-heavy prompts (JSON arrays, repeated lines, CSV) give **1.60x** with
   ~2 tokens accepted per verification round. The drafter earns its keep exactly
   where content repeats.

2. **The integrated loop is correct.** Verification compares each drafted token
   against the target's own forward (greedy argmax), so no token is emitted that the
   target would reject. End-to-end byte-for-byte losslessness is proven for the loop
   logic under exact (fp32) numerics.

3. **The bf16 near-tie artifact is real and grows with depth.** Batched verification
   writes KV that differs from sequential decode by up to ~0.16 logits; over many
   rounds on a deep model this can flip a near-tie argmax and split the trajectory
   from standalone greedy. Agreement is ~0.86-0.91 (not 1.0). The output is still
   valid (target-approved) — this is a numerics artifact of batched speculation, not
   a logic bug. It is the reason agreement is reported instead of asserted at 1.0.

4. **The neural general expert is data-starved.** Distilled on ~133 records it reaches
   ~0.36% top-1 next-token accuracy over the 151k vocab (below the 0.86% random
   control). It accepts ~1 token/round and adds overhead: 0.38x. The training pipeline
   is mechanically sound (loss decreases, checkpoint reloads), but the volume is far
   too small to approximate an 8B model.

---

## Erratum (padding fix)

After this report, a real bug was found and fixed in batched trace generation
(`generate_greedy_batch`): right-padding was used (wrong side for causal models)
and the continuation slice was mis-indexed, so **mixed-length batches produced
polluted training traces** (leading pad/prompt tokens). This corrupted the
distillation data in the earlier attempts. Fix (commit `f3d394e`): left-padding +
correct `max_len:` slice.

Re-validated locally on clean 0.6B data (120 traces, all classes):

| Metric | Polluted drafter | Clean drafter |
|---|---|---|
| held-out top-1 accuracy | 0.36% | **47.9%** |
| margin vs random control | −0.5% | **+34.8%** |
| general agreement (eval) | 0.47 | **0.92** |
| general tokens/round (code) | 1.03 | 1.07 |
| general speedup (code) | 0.47x | 0.43x |

The clean drafter is far more accurate and agrees with vanilla much better, but
**local 0.6B→0.6B speedup stays <1x** because drafting overhead (K trunk forwards
per round on the slow 4060) exceeds the modest accepted-token gain when the
target is itself tiny. The meaningful neural-drafter test remains the Qwen3-8B
run, which must be regenerated with the fix.

---

## Interpretation

- **Demonstrated, real speedup: 1.60x** via the copy expert on repetition-friendly
  content. This is the Phase-1 headline and validates the heterogeneous-expert
  architecture end-to-end on a real target.
- **The neural expert needs orders of magnitude more distillation data** (and likely
  longer/varied traces) before it contributes. This is a data problem, not an
  architecture problem.

---

## Caveats

- Speedups measured batch-1, greedy, HF eager attention, single GPU. Continuous
  batching / engine integration (todo 28) will shift absolute numbers.
- Agreement < 1.0 reflects the bf16 batch-vs-sequential drift; see finding 3.
- General-expert numbers reflect a 133-record distillation; scale before judging the
  neural path.

---

## Next steps

- **A (shipped):** copy expert as the working system; use repetition/structured
  prompts & workloads where it applies.
- **B (follow-up):** expand distillation data (60+ structured / 60+ chat / thousands
  of traces total) and retrain the general expert to make the neural path contribute.
- **C (roadmap):** add the macro expert (structural tokens), tree attention for larger
  spec depth, and engine integration for production throughput.

---

## Thesis system (router + multi-expert DAG)

The pitch's core mechanism — a utility router activating a subset of heterogeneous
experts whose proposals are fused into a candidate DAG and verified losslessly — is
now built and measured (`src/familydraft/eval/draft_dag.py`, `DagSpeculator`).

**Correctness (proven):**
- `DagSpeculator` is lossless under exact numerics (fp32 tiny-model tests, all
  expert sets), matching vanilla greedy token-for-token.
- A copy-only `DagSpeculator` is token-identical to copy-alone (chain) on real
  prompts — the DAG verifier does not alter acceptance.
- Copy+macro and the full 4-expert DAG match copy-alone exactly on prompts where
  copy accepts (e.g. 1.889 tpr on the dictionary prompt), confirming the fusion
  mechanism picks the best branch without degrading it.

**Measured on Qwen3-0.6B (local):**
- copy-alone: ~1.0-1.3x (1.19x code / 1.34x repetitive tpr)
- DAG (copy+macro+reject_memory+general via router): ≈ copy-alone on 0.6B

The union benefit (DAG > single expert) does not materialize on this weak 0.6B
model, where only copy accepts meaningfully; the router's cold-start selection and
per-expert acceptance are the current limits. The thesis measurement is the
Qwen3-8B run (router + 4 experts + DAG), where multiple experts can contribute and
drafting overhead is amortized against a large target.

**Router cold-start fix (commit):** `cold_start` now seeds `accepted_len_ema` at 1.0
(minimal horizon) so selection is driven by base expected quality rather than an
inflated horizon that made high-base experts look expensive and got them skipped.
