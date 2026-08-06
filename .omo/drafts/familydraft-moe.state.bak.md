# FamilyDraftMoE - planning draft (durable state)

- slug: familydraft-moe
- intent: clear
- review_required: false
- classify: architecture (greenfield research system)
- status: interviewing (research wave in flight)
- pending action: after forks answered -> approval gate -> write .omo/plans/familydraft-moe.md

## Source truth
- Concept note: D:\MoE\family_draft_moe_concept.md (807 lines, fully read)
- Workspace is otherwise empty (greenfield). No git repo yet.

## Strategic decisions already agreed with user (prior turns)
1. Baseline family: Qwen3 dense line (no MTP heads, shared tokenizer, rich variant spread incl. QwQ + Coder). NOT Qwen3.5 — verified via web: entire Qwen3.5 lineup ships MTP ("trained with multi-steps", vLLM/SGLang qwen3_next_mtp support).
2. Positioning: not "family lacks MTP" (vendor erosion — Qwen3->Qwen3-Next->Qwen3.5 added MTP within months) but "adaptive heterogeneous drafting beats shallow fixed-horizon built-in drafters on stretches they can't reach" + family-level amortization.
3. EAGLE remains a mandatory baseline (external feature drafter trains against any open-weights target).
4. Defensible contribution triad: heterogeneous proposal interface (sequence-level DAG fusion), utility-calibrated variable-k routing with abstention, online rejection memory. Cut reasoning-transition + logit-dynamics experts to Phase 2+, justify by oracle analysis first.
5. Cost-tiered expertise: cheap structural experts (copy, macro, memory) always-on; router gates neural budget only.
6. First experiment: oracle predictability analysis on real Qwen3 traces (which mechanism class recovers target continuations, with oracle) BEFORE building the drafter.

## Adopted defaults (announced, not asked)
- Full-scope plan: all 4 phases from concept note §13, deepest decision-completeness in Phase 1. (Full scope is the default; no invented MVP.)
- Language/stack: Python + PyTorch (ecosystem standard for this work).
- Venue/pacing: milestone-driven, target systems venue (MLSys-class); no hard calendar date encoded.
- Execution model: single-executor plan (agent-driven), sequenced accordingly.

## Components ledger (topology lock - presented to user for confirmation)
- C1 Oracle predictability analysis (go/no-go evidence; can fail => kill/pivot)
- C2 Data & distillation pipeline (Qwen3 target traces/logits collection)
- C3 Drafter core (shared trunk + general-neural, macro, copy/retrieval, rejection-memory experts)
- C4 Router (variable k in {0,1,2}, utility/bandit-based, abstention)
- C5 Candidate DAG builder + verification loop (trie fusion, exact acceptance)
- C6 Evaluation & baselines harness (throughput instrumentation, dense/EAGLE/equal-FLOP baselines, ablations)

## Discovered facts
- Local machine: RTX 4060 8GB, 16GB RAM, Windows 10 = dev box only (serves Qwen3-0.6B/1.7B, 4B quantized).
- User has ample RunPod credits -> cloud rig: A100/H100-class nodes + network volumes for datasets/checkpoints; Linux pods (HF/vLLM/SGLang stack is Linux-first).

## Forks - ALL RESOLVED
- F1 Compute: RESOLVED = local dev box (RTX 4060 8GB, Windows 10) + RunPod experiment rig (ample credits). No target-size constraints; Linux pods for HF/vLLM/SGLang stack.
- F2 Engineering surface: RESOLVED = standalone PyTorch harness for Phases 1-3 (target via HF transformers, custom draft-verify loop, DAG verification, full instrumentation); engine (vLLM/SGLang) integration deferred to Phase 4. User chose recommended option.
- F3 Test strategy: RESOLVED = tests AFTER each milestone (user override of tests-first default). Constraint encoded in plan: verification-equivalence property test (DAG verifier vs brute-force sequential) is the hard gate of the first milestone regardless - silent verification corruption invalidates all metrics.

## Verified facts - direct primary sources (config.json fetches)
- Qwen3-8B: vocab_size=151936, model_type=qwen3, NO MTP fields, max_position_embeddings=40960 (32K native; YaRN ext per docs).
- QwQ-32B: vocab_size=152064, model_type=qwen2 (Qwen2.5-based) -> DIFFERENT tokenizer -> EXCLUDED from same-family target matrix.
- Qwen3-Coder-480B-A35B-Instruct: vocab_size=151936, model_type=qwen3_moe, 262144 ctx, 160 experts/8 active -> IN scope.
- Qwen3-Next-80B-A3B-Instruct: vocab_size=151936 too (the 248k jump happens at Qwen3.5), but OUT of scope (hybrid linear attention; separate branch).
- Reasoning variants within tokenizer: Qwen3 unified thinking/non-thinking mode + Qwen3-30B-A3B-Thinking-2507 / Qwen3-235B-A22B-Thinking-2507 (Jul 2025).
- Librarian A session died mid-analysis; replaced by direct fetches above (sufficient for plan).

## Librarian B findings (bg_ded85785; near-complete, final fact sheet pending - treat as sourced-claims until sheet lands)
- SpecForge: sgl-project/SpecForge, MIT, arXiv:2603.18567 confirmed. Draft registry = dense only (llama3_eagle/dflash/domino/dspark/peagle); NO MoE drafter shipped. Paper 7.3: dense > MoE drafters under Same-Params, Same-FLOPs, With-Shared-Experts. Ships 13-task benchmark harness + eval/evaluator.py (simulated_acc_len).
- EAGLE: SafeAILab/EAGLE, Apache-2.0. EAGLE-3 training: ShareGPT jsonl + frozen target in-loop, 3-layer (low/mid/high) feature fusion, 7-step TTT rollout, 32k draft vocab; EAGLE-1/2 need offline hidden-state dumps. EAGLE-2 3.05-4.26x; EAGLE-3 up to 6.5x bs1, 1.38x SGLang throughput bs64.
- Jakiro: haiduo/Jakiro, MIT, training code included (EAGLE1-style), arXiv:2502.06282, ACL 2026 main.
- Tree attention sources: FasterDecoding/Medusa Apache-2.0 (stale Jun 2024); Infini-AI-Lab/Sequoia NO LICENSE FILE (avoid); EAGLE repo Apache-2.0 tree code.
- Eval protocol (EAGLE papers): temp=0 AND temp=1, speedup + mean accepted length tau, MT-Bench/HumanEval/GSM8K/Alpaca/CNN-DM/NQ; bs1 latency, engine throughput for larger batches.
- SHAs pinned by librarian: SpecForge 7d5a693, EAGLE cb7e084, Jakiro f8b018d.
- PLAN-SHAPING new findings:
  1. Cascade (arXiv:2506.20675): speculation can HURT MoE targets (draft tokens activate more experts; verification 2-3x costlier; 18/35 pairs degrade; up to 1.5x slowdown) -> risk for Qwen3-30B-A3B/235B targets; must instrument verification-cost-vs-DAG-size per target.
  2. Not-a-Bandit (2510.20064): specialized drafter pools on average UNDERPERFORM generalist EAGLE; static routing misroutes ~98% of MedQA under prompt variation -> router must be online-adaptive; must beat generalist EAGLE baseline, not strawmen.
  3. MetaSD = Findings of ACL 2026 (2026.findings-acl.1629) -> concept-note ref 3 correction.
  4. BanditSpec ICML 2025 (arXiv:2505.15141); EVICT/EcoSpec/DraftExpert = MoE-verification-cost literature cluster.

## Status
- Exploration exhausted. Librarian B complete (fact sheet received; sources + pinned SHAs above).

## GATE RECORD
- status: awaiting-approval
- pending action: write .omo/plans/familydraft-moe.md (scaffold via scaffold-plan.mjs, then Metis review, then todo waves)
- approach: 4-phase plan per concept note §13 re-anchored on agreed strategy: Phase 0 oracle predictability go/no-go -> Phase 1 four-expert prototype on standalone PyTorch harness (proves routing thesis) -> Phase 2 specialization + target-variant conditioning -> Phase 3 online adaptation -> Phase 4 engine/production eval. Targets: Qwen3 dense 4B/8B/14B/32B + 30B-A3B (Cascade-risk instrumented) + Coder-30B-A3B; QwQ-32B excluded (vocab 152064). Baselines: vanilla, dense drafter, equal-FLOP dense, EAGLE-3 (trained via SpecForge), Jakiro (MIT code), + concept-note ablations. Protocol: temp 0 + temp 1, MT-Bench/HumanEval/GSM8K, wall-clock bs1 + acceptance length; primary metric end-to-end tokens/sec. Tests backfilled per milestone; verification-equivalence property test is M1 hard gate. Infra: local 4060 dev + RunPod A100/H100 pods + network volumes.

## Research dispatched (background)
- Librarian A: Qwen3 family lineup facts (sizes, tokenizer, variant spread, serving footprints)
- Librarian B: spec-decoding tooling/baselines (EAGLE-3 repo+training, SpecForge repo, tree-verification impls, standard eval benchmarks/protocols, Jakiro code availability)
- Local: nvidia-smi probe for discoverable compute facts

## Gate record
(approval gate fields recorded here when reached)
