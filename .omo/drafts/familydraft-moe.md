---
slug: familydraft-moe
status: plan-complete
intent: clear
review_required: false
pending-action: delivered plan summary; user chooses start-work vs dual high-accuracy review
approach: >
  Full-scope plan over all four concept-note phases with Phase 1 decision-complete.
  Phase 0 oracle predictability go/no-go -> Phase 1 four-expert prototype on standalone
  PyTorch harness proving the routing thesis -> Phase 2 specialization + target-variant
  conditioning (gated experts) -> Phase 3 online adaptation -> Phase 4 engine/production
  eval. Standalone harness first, SGLang/vLLM integration only in final phase.
  Plan file: .omo/plans/familydraft-moe.md (30 todos, waves A-G, milestones M1-M3).
---

# Draft: familydraft-moe

## Components (topology ledger)
| id | outcome (one line) | status | evidence |
| --- | --- | --- | --- |
| C1 | Oracle predictability analysis produces go/no-go evidence with pre-registered kill threshold | active | planned Wave C |
| C2 | Multi-target distillation pipeline (Qwen3 traces/logits, target-ID tagged) | active | planned Wave C |
| C3 | Drafter core: trunk + general/macro/copy/rejection-memory experts | active | planned Wave D |
| C4 | Variable-k utility router with abstention + per-expert horizons | active | planned Wave D |
| C5 | Candidate DAG builder + exact verification (equiv.-tested) | active | planned Wave B |
| C6 | Evaluation & baselines harness (dense, equal-FLOP, EAGLE-3, single-expert) | active | planned Wave E |

## Open assumptions (announced defaults)
| assumption | adopted default | rationale | reversible? |
| --- | --- | --- | --- |
| Plan scope | all 4 concept-note phases, Phase 1 decision-complete | full scope is the default | yes (user veto) |
| Stack | Python 3.11 + PyTorch + HF transformers, uv project | ecosystem standard for this work | yes |
| Venue/pacing | milestone-driven, systems venue (MLSys-class), no calendar date | no deadline given | yes |
| Execution model | single AI-executor, sequenced waves | environment is agent-driven | yes |
| Trunk architecture | truncated Qwen3-0.6B (~6 layers) + target-variant embedding | family-native, vocab-aligned, cheap; options per concept note §3.1 | yes |
| Router v1 | contextual bandit + abstention threshold (Not-a-Bandit machinery), neural router deferred | router overhead risk (§11.3); static routing known-fragile | yes |

## Findings (cited)
- Qwen3-8B config: vocab 151936, no MTP fields — huggingface.co/Qwen/Qwen3-8B/raw/main/config.json
- QwQ-32B config: vocab 152064, model_type qwen2 -> EXCLUDED — huggingface.co/Qwen/QwQ-32B/raw/main/config.json
- Qwen3-Coder-480B-A35B-Instruct: vocab 151936, qwen3_moe, 262k ctx — huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/raw/main/config.json
- Qwen3-Next also vocab 151936 (248k jump only at Qwen3.5); out of scope (hybrid attention branch)
- Reasoning coverage within tokenizer: unified thinking mode + Thinking-2507 checkpoints (Qwen3-30B-A3B-Thinking-2507, Qwen3-235B-A22B-Thinking-2507)
- SpecForge: sgl-project/SpecForge, MIT, arXiv:2603.18567; dense-only draft registry; §7.3 dense > MoE drafters (Same Params / Same FLOPs / Shared Experts); 13-task eval harness + simulated_acc_len evaluator; SHA 7d5a693
- EAGLE: SafeAILab/EAGLE Apache-2.0; traineagle3 = ShareGPT jsonl + in-loop frozen target, 3-layer feature fusion, 7-step TTT, 32k draft vocab; EAGLE-1/2 = offline hidden states; README directs users to SpecForge for EAGLE-3 training; EAGLE-2 3.05-4.26x; EAGLE-3 up to 6.5x bs1 / 1.38x SGLang bs64; SHA cb7e084
- Jakiro: haiduo/Jakiro MIT incl. training; arXiv:2502.06282; ACL 2026 main (2026.acl-long.487); SHA f8b018d
- Tree code: FasterDecoding/Medusa Apache-2.0 (stale 2024-06); Infini-AI-Lab/Sequoia NO LICENSE (avoid; successor UMbreLLa Apache-2.0)
- Eval protocol: EAGLE papers temp=0 + temp=1, speedup + mean accepted length tau, MT-Bench/HumanEval/GSM8K/Alpaca/CNN-DM/NQ, bs1 latency, engine throughput for batches
- Cascade arXiv:2506.20675: speculation can HURT MoE targets (2-3x verification cost growth; 18/35 pairs degrade; up to 1.5x slowdown)
- Not-a-Bandit arXiv:2510.20064: specialized drafter pools on average UNDERPERFORM generalist EAGLE; static routing misroutes ~98% of MedQA under prompt variation -> router must be online-adaptive; bar = beat generalist
- MetaSD = Findings of ACL 2026 (2026.findings-acl.1629) -> concept-note ref 3 needs correction
- BanditSpec ICML 2025 (arXiv:2505.15141); EVICT 2605.00342, EcoSpec 2607.12696, DraftExpert 2607.24434 = MoE-verification-cost cluster
- Local hardware: RTX 4060 8GB / 16GB RAM / Win10 (nvidia-smi probe) = dev box; RunPod (user credits) = experiment rig

## Decisions (with rationale)
- D1 Baseline family = Qwen3 2025 line (no MTP heads; shared 151936 tokenizer; variant spread incl. Coder + Thinking-2507). Qwen3.5 rejected: ships MTP fleet-wide (vendor-erosion trap proven by Qwen3->Qwen3-Next->Qwen3.5 timeline).
- D2 Positioning = adaptive heterogeneous drafting vs shallow built-in MTP + family-level amortization; NOT "family lacks MTP".
- D3 Contribution triad = heterogeneous proposal interface (sequence-level DAG fusion) + utility-calibrated variable-k routing w/ abstention + online rejection memory. Reasoning-transition & logit-dynamics experts deferred behind Phase-0 evidence gate.
- D4 Cost-tiered experts: cheap structural experts (copy/macro/memory) always-on; router gates neural budget only.
- D5 First experiment = oracle predictability analysis before any drafter training; kill threshold pre-registered.
- D6 Compute = local 4060 dev + RunPod A100/H100 pods + network volumes.
- D7 Engineering surface = standalone PyTorch harness Phases 1-3; SGLang/vLLM only Phase 4. (User chose.)
- D8 Tests = backfilled after each milestone EXCEPT verification-equivalence property gate hard at M1. (User chose.)
- D9 Success bar = beat equal-active-FLOP dense drafter on end-to-end tok/s (Phase-1 criterion) AND report gap vs SpecForge-trained EAGLE-3 generalist (per Not-a-Bandit lesson).

## Scope IN
- Qwen3 targets: 4B, 8B, 14B, 32B (dense) + 30B-A3B (MoE, verification-cost instrumented) + Qwen3-Coder-30B-A3B; bf16 primary.
- Four-expert prototype (general neural, macro, copy/retrieval, rejection memory), variable-k bandit router, DAG fusion, exact verification.
- Full baseline suite + §10.4 ablation matrix + results report (markdown + figures).
- Phases 2-3 gated continuations; Phase 4 SGLang integration + batch matrix + paper skeleton.

## Scope OUT (Must NOT have)
- Cross-family transfer; QwQ-32B (vocab 152064); Qwen3-235B-A22B (multi-node); Qwen3.5/3.6/Next branches.
- Engine integration before Phase 4 gate; continuous batching before Phase 4.
- Sampling-regime theory work beyond standard speculative-sampling correction.
- Reasoning-transition / logit-dynamics experts unless Phase-0 gate passes.
- Quantized-target serving as default; paper prose beyond skeleton + evidence package.
- Sequoia code reuse (unlicensed).

## Open questions
- None (all forks resolved: compute, surface, tests).

## Approval gate
status: approved
- Brief presented; user reply "go" = approval to write the plan file.
- Approval authorizes plan writing ONLY, never implementation.

## Metis gap analysis (mandatory) - RECEIVED AND FOLDED
- Session: ses_027c8fc88ffeNayDJcBYGQ3M00 (bg_265ac232), 24 gaps (5 critical, 13 major, 6 minor). All folded into .omo/plans/familydraft-moe.md:
  - #1 train/eval leak -> todo 12 content-hash leak-proof assertion + aux-corpus-only training shards
  - #2 verdict measurability -> todo 22 pre-registered configs/verdict_protocol.yaml (hardware pin, 5-run mean±std, ≥5% margin, decode-only formula, all-4-class reporting, {code,structured} pair)
  - #3 rollout handwave -> todo 20 policy-only scope, stochastic draft exploration, net-tok/s reward, pinned budget, fallback that never blocks the verdict
  - #4/#7 output equivalence + temp-1 -> verification strategy invariant + todo 8 sampling-mode statistical gate before any temp-1 numbers
  - #5/#12 kill branch + metric definitions -> todo 11 exact pre-registered definitions, oracle-best speedup bound ≥1.5x, NO-GO pivot deliverable, decision authority = user
  - #6 SpecForge Qwen3 unvalidated -> todo 13 dry-run validation gate before spending credits + corpus-fairness + baseline-quality guard
  - #8 parser ordering -> parse_state.py v0 moved INTO todo 11, consumed by todo 16
  - #9 target matrix -> Execution strategy target-by-wave matrix
  - #10 thinking mode -> todo 3 per-class pins (GSM8K thinking=on, others off)
  - #11 router cold start -> todo 19 offline simulated-acceptance init
  - #13 trace params irreversibility -> todo 10 pinned k=64/lengths/greedy in committed config
  - #14 Jakiro contradiction -> todo 21 explicit descope w/ rationale (Phase-2 optional)
  - #15 ablation compute -> todo 22 config-switch-only ablations, hidden-state-mixing descope to Phase 2
  - #16 Windows/Linux -> platform pin in verification strategy (SDPA/eager only, no flash-attn)
  - #17 run-order reproducibility -> cold-start verdict pin in verification strategy + todo 22
  - #18 unseen-target embedding -> todo 14 attribute-conditioned embedding (config-vector MLP + per-id residual)
  - #19-22 minors -> todos 3/5/9/16/17 (structured-set provenance, CUDA-event sync, latency budgets)
  - #23 empty draft file -> resolved: this file restored with full state before plan writing
