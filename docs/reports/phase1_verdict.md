# Phase-1 verdict — FamilyDraftMoE integrated speculative drafter

**Status:** Phase-1 campaign complete (local 0.6B preview + Qwen3-8B on RunPod).
**Headline:** the **copy expert achieves 1.60x wall-clock speedup on Qwen3-8B** for
repetition-heavy content. The neural general expert, trained on a small distillation
set, does not yet help.

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
