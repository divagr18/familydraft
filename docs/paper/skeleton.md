# FamilyDraftMoE — paper skeleton (plan todo 30)

> Skeleton only: section outlines and a claims→evidence table. No manuscript
> prose. Every claim row in claims.csv must resolve to an indexed artifact
> (checked by scripts/check_evidence_index.py).

## Sections

### 1. Introduction
- **Claim:** C1, C3 — same-family heterogeneous speculative drafting is
  correctness-preserving (fp32 lossless) and beats single-expert drafting on
  repetition-friendly workloads.
- **Claim:** C6 — at small (0.6B) scale the integrated DAG does not yet beat
  vanilla or an equal-FLOP dense drafter; the thesis measurement is the 8B pod
  campaign.

### 2. Related work
- **Claim:** C10 — EAGLE-3 (SpecForge) is the generalist bar; pod-deferred.
- References seeded in refs.bib: Jakiro, SpecForge, EAGLE 1–3, MetaSD,
  Not-a-Bandit, BanditSpec, Cascade, EVICT, EcoSpec, DraftExpert, Medusa,
  Sequoia/UMbreLLa, SpecInfer.

### 3. Method
- **Claim:** C2 — tree attention verifies all DAG nodes in one forward over a
  cached context.
- **Claim:** C7 — each single ablation switch changes throughput ≤4% except
  fixed-top1 and no-target-embedding.
- **Claim:** C8 — seeded baseline reruns are bit-identical; injected
  nondeterminism is detected.
- **Claim:** C9 — the verdict protocol was pre-registered before the campaign.

### 4. Experiments
- **Claim:** C5 — bf16 batch-vs-sequential KV drift flips near-tie argmaxes
  (agreement < 1.0); fp32 exactness proven.
- **Claim:** C11 — the sealed eval manifest is byte-verifiable (leak-proof).
- **Claim:** C6 — full Phase-1 campaign + ablation matrix in runs/results/phase1.csv.

## Evidence index
All artifacts are indexed with sha256 + git sha + config hash in
`EVIDENCE_INDEX.json` (built by scripts/build_evidence_index.py).
