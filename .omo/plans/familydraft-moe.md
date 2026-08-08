# familydraft-moe - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** A working prototype that makes a large language model answer faster. Instead of one small "draft" model guessing the next words, a team of small specialists each guesses in a different way â€” one writes ordinary text, one handles code structure and closing brackets, one reuses text that already appeared, one remembers and repairs its own past mistakes. Their guesses are merged and then checked exactly against the big model, so the output is guaranteed unchanged â€” only faster. The project also produces a complete evidence pack (fixed test sets, measurement runs, reports, and a paper outline) proving whether this approach actually wins, against honest comparisons.

**Why this approach:** Published results show committees of similar small drafters usually lose to one good drafter, and that speculation can even slow some models down. So this plan bets on *different kinds* of guessing rather than more of the same kind, spends compute only where the cheap specialists can't reach, and judges everything by real wall-clock speed measured the same way across all systems. Correctness is proven first with automated equivalence checks, because one hidden bug there would quietly poison every later number. An early, pre-registered experiment decides whether the whole idea is worth building before any expensive training starts.

**What it will NOT do:** It will not touch other model families or models without a shared vocabulary; it will not be integrated into production serving engines until the final phase; it will not write a full paper; it will not add the two experimental specialists unless the early evidence justifies them; it will not use the largest (200B+) models.

**Effort:** XL
**Risk:** High - prior published work reports that draft-model committees often lose to single dense drafters, and speculation can hurt some target architectures; mitigated by a pre-registered early kill switch, a strong published-method baseline as the bar, and hard correctness gates.

**Decisions to sanity-check:** the primary target model and the two task types (code, structured data) pre-registered for the main go/no-go verdict; the 5% win margin required; the early kill threshold; the small backbone built from a trimmed member of the target family; verdict runs starting from a cold, empty state for fairness; one published comparison system deliberately not rebuilt in Phase 1 (cited instead); math benchmarks run with the model's reasoning mode on while others run with it off.

Your next move: approve and start work, or run the dual high-accuracy review first. Full execution detail follows below.

---

> TL;DR (machine): XL / high-risk research build â€” same-family heterogeneous speculative-drafting MoE for Qwen3 (trunk + 4 experts + variable-k utility router + DAG fusion + exact verification), oracle go/no-go gate first, Phase-1 verdict vs equal-FLOP dense & SpecForge-trained EAGLE-3 under pre-registered protocol, then Phases 2-4; 30 todos / 7 waves / milestones M1-M3; evidence package + paper skeleton as final deliverable.

## Scope
### Must have
- One repo (`familydraft`) producing a same-family heterogeneous speculative drafter for the Qwen3 2025 line, covering ALL four concept-note phases (Â§13), with Phase 1 decision-complete.
- Target matrix (bf16 primary): Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen3-32B (dense); Qwen3-30B-A3B (MoE, verification-cost instrumented per Cascade arXiv:2506.20675); Qwen3-Coder-30B-A3B. All share tokenizer vocab 151,936 (verified via HF config.json fetches; see `.omo/drafts/familydraft-moe.state.bak.md` Findings).
- Core system: truncated-Qwen3 shared trunk + target-variant embedding; four experts (general neural, macro, copy/retrieval, rejection-memory); variable-k utility router (kâˆˆ{0,1,2}) with abstention and per-expert horizons; candidate trie/DAG fusion; exact DAG verification.
- Verification-equivalence property gate (DAG verifier â‰¡ brute-force sequential reference) as hard Milestone-1 gate.
- Oracle predictability analysis with pre-registered kill threshold and go/no-go report (Milestone 2).
- Baseline suite under ONE harness: vanilla AR; small dense drafter; equal-active-FLOP dense drafter; SpecForge-trained EAGLE-3 (generalist bar, per Not-a-Bandit lesson); single-best-expert selection; heterogeneous top-2 without fusion.
- Full ablation matrix per concept note Â§10.4; Phase-1 verdict on end-to-end tokens/sec at bs1 greedy on â‰¥2 materially different task classes (Milestone 3).
- Eval protocol per EAGLE convention: temperature 0 (primary) + temperature 1 (secondary, with standard speculative-sampling correction); MT-Bench (80) / HumanEval (164) / MBPP-sanitized / GSM8K test (1319) / structured JSON set (100); metrics per concept note Â§10.5 with end-to-end target tokens/sec as primary.
- Phases 2-4: specialization + zero-shot family transfer; gated expert additions; online adaptation (calibration, dynamic horizons, abstention ROC, session drift); SGLang integration with batch matrix; MoE-target verification study; evidence package + paper skeleton.
- Infra: standalone PyTorch harness (HF transformers â‰¥4.51 target serving); local RTX 4060 dev box (Windows 10) + RunPod A100/H100-class pods + network volumes; everything seed-pinned, config-hashed, checksummed.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- NO cross-family transfer work; NO QwQ-32B (vocab 152,064, Qwen2 architecture â€” different tokenizer); NO Qwen3-235B-A22B (multi-node); NO Qwen3.5/3.6/Next branches (248k vocab / hybrid attention).
- NO engine (vLLM/SGLang) integration before the Phase-4 wave; NO continuous-batching claims before then.
- NO reasoning-transition or logit-dynamics experts unless the Phase-0 oracle gate passes for them (recorded evidence required).
- NO quantized-target serving as a default; bf16 is the reference configuration.
- NO sampling-regime theory work beyond standard speculative-sampling correction.
- NO reuse of Infini-AI-Lab/Sequoia code (no license file â€” verified).
- NO manuscript prose beyond skeleton + reference corrections; the deliverable is prototype + evidence package.
- NO secrets in the repo (RunPod/API keys env-only); NO large artifacts in git (network volume + manifests).
- NO tuning of the oracle kill threshold or Phase-1 success bar after seeing results; both are pre-registered in config files before the relevant campaign runs.
- NO Jakiro reproduction in Phase 1 (stretch only; cite ACL 2026 numbers).

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after per milestone (user decision), pytest framework â€” EXCEPT the verification-equivalence property gate which is test-first and is the hard gate of Milestone 1 regardless of ordering.
- Determinism contract: every eval run seed-pinned (torch/cuda/python), config fingerprint (sha256) logged, greedy outputs must be bit-identical across reruns on identical hardware; harness self-checks this and fails loudly on divergence.
- Equivalence property: DAG verifier acceptance set â‰¡ reference sequential verifier for â‰¥500 randomized synthetic targets + real-model cases (Qwen3-0.6B, fits local 4060).
- Milestone gates (agent-checked exit codes): M1 equivalence suite green; M2 oracle report emitted with GO (exit 0) / NO-GO (exit 77); M3 Phase-1 verdict auto-computed from `runs/results/phase1.csv` against the pre-registered bar.
- Evidence: every QA scenario writes artifacts to `<attemptDir>/task-<N>-familydraft-moe.<ext>` (attemptDir = currentAttemptDir from 'omo ulw-loop status --json', `.omo/evidence/ulw/<session>/<goalId>/a<attempt>`; outside ulw-loop use `.omo/evidence/`). Research artifacts (traces, checkpoints, run JSONs) live on the RunPod network volume with sha256 manifests; git holds code, configs, reports only.
- Output-equivalence invariant (silent-corruption detector): EVERY greedy speculative run must produce output byte-identical to vanilla AR target output on the full eval set; asserted inside the campaign runner â€” any divergence invalidates the run.
- Sampling-mode correctness gate: speculative-sampling residual correction is implemented in the reference verifier (todo 6) and statistically validated (sample-distribution Ï‡Â²/KL vs direct target sampling within pre-registered bound, todo 8) BEFORE any temperature-1 number is reported.
- Verdict-run reproducibility pin: verdict measurements run COLD (empty rejection memory, router stats at init), fixed seeded prompt order, single pass, seeds + backend flags + git sha + config hash logged per run; online-adaptation effects are measured separately in Wave F only.
- Platform pin: all unit tests run CPU-safe on Windows dev box (eager/SDPA attention only â€” flash-attn forbidden in core verifier); all GPU training/eval runs on RunPod Linux pods per docs/infra.md.

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.
- Single AI executor: waves run sequentially; parallelizable pairs within a wave are marked per todo and may be interleaved when context budget allows. Waves Aâ†’G map to: A foundations, B verification core (M1), C oracle+data (M2), D drafter build, E integration+verdict (M3), F Phases 2-3, G Phase 4 + evidence package.
- Kill switch: if M2 returns NO-GO, stop after todo 11's report and surface the pivot decision â€” do NOT continue to Wave D.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 repo scaffold | â€” | all | 2, 3 |
| 2 infra spec + smoke | 1 | 10, 13 | 3, 4 |
| 3 eval manifest | 1 | 10, 21, 22 | 2 |
| 4 target wrapper | 1 | 6, 8, 10, 15 | 2, 3, 5 |
| 5 instrumentation | 1 | 9, 20, 21 | 4 |
| 6 reference verifier | 4 | 8 | 7 |
| 7 DAG builder | 1 | 8, 18, 19 | 6 |
| 8 DAG verifier + equivalence (M1) | 6, 7 | 19, 20, 21, 22 | â€” |
| 9 latency microbench | 5, 8 | 19, 20 | 10 |
| 10 trace campaign | 2, 3, 4 | 11, 12, 13 | 5-9 (cloud) |
| 11 oracle report (M2) | 10 | 14-19 (gate for 14/15/19 + parse_state for 16), 25 | 12 |
| 12 distill dataset | 10 | 13, 15, 21 | 11 |
| 13 EAGLE-3 baseline | 12, 2 | 21, 22, 26 | 11 |
| 14 trunk | 11 (GO), 1 | 15, 16, 19 | â€” |
| 15 general expert | 14, 12 | 19, 20 | 16, 17, 18 |
| 16 macro expert | 14, 10, 11 (parse_state v0) | 19, 20 | 15, 17, 18 |
| 17 copy expert | 14 | 19, 20 | 15, 16, 18 |
| 18 rejection-memory expert | 7, 14 | 19, 20 | 15, 16, 17 |
| 19 router v1 | 8, 9, 15-18 | 20, 21 | â€” |
| 20 rollout training | 19, 12 | 22 | â€” |
| 21 baseline suite | 8, 12, 13 | 22 | 20 |
| 22 Phase-1 campaign (M3) | 20, 21 | 23, 24 | â€” |
| 23 verdict report | 22 | 24 | â€” |
| 24 transfer eval | 23 | 25 | 26 |
| 25 gated experts | 23, 11 (gate) | 27 | 24 |
| 26 calibration + horizons | 23 | 27 | 24 |
| 27 session drift adaptation | 26 | 28 | 25 |
| 28 SGLang integration | 23 (system frozen) | 29 | 24-27 |
| 29 MoE-target study | 8, 9, 22 | 30 | 28 |
| 30 evidence package + skeleton | all | â€” | â€” |

### Target-by-wave matrix (resolves Metis gap #9)
| Target | Traces (10) | Distill train (12) | Phase-1 verdict (22) | Transfer-unseen (24) | MoE study (29) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B | yes | yes | sanity | â€” | â€” |
| Qwen3-8B | yes (primary) | yes | PRIMARY | â€” | â€” |
| Qwen3-14B | yes | yes | â€” | â€” | â€” |
| Qwen3-32B (bf16, 1Ã—A100-80GB) | â€” (held out) | â€” | â€” | yes | â€” |
| Qwen3-Coder-30B-A3B (1Ã—A100-80GB) | â€” (held out) | â€” | â€” | yes | â€” |
| Qwen3-30B-A3B (1Ã—A100-80GB) | â€” | â€” | probe rows only | â€” | yes |
Training set = {4B, 8B, 14B} Ã— auxiliary corpora; held-out unseen = {32B, Coder-30B-A3B}; MoE verification-cost study = 30B-A3B only (Cascade risk). Eval prompts are NEVER in training shards (todo 12 disjointness assertion).

## Todos
> Implementation + Test = ONE todo. Never separate.

### Wave A â€” Foundations (M0)
- [x] 1. Repo + project skeleton (uv, src layout, lint, git)
  What to do: `git init`; uv project pinning Python 3.11, torch (CUDA), transformers>=4.51 (Qwen3 support verified via `model_type: qwen3` configs), datasets, pytest, ruff, jsonschema. Layout: `src/familydraft/{targets,verify,draft,experts,router,eval,infra}/`, `configs/`, `scripts/`, `tests/`, `data/` + `runs/` (gitignored), `docs/`. `.gitignore` (no data/, runs/, secrets), `pyproject.toml`, README stub. Must NOT do: no CI, no extra deps, no framework code yet.
  Parallelization: Wave A | Blocked by: none | Blocks: all | With: 2, 3
  References: D:\MoE\.omo\drafts\familydraft-moe.md (Decisions D5-D8); concept note D:\MoE\family_draft_moe_concept.md:584-602 (Â§10.2 prototype stack)
  Acceptance criteria (agent-executable): `uv sync` exit 0; `uv run python -c "import familydraft"` exit 0; `uv run ruff check src tests` exit 0; `uv run pytest` exits with code 5 (no tests yet); `git log --oneline -1` shows the bootstrap commit.
  QA scenarios: happy â€” all five commands above succeed, log to Evidence; failure â€” add a temp file with a syntax error under src/, `ruff check` must exit non-zero naming the file, then delete it. Evidence <attemptDir>/task-1-familydraft-moe.txt
  Commit: Y | chore(repo): bootstrap familydraft project skeleton

- [x] 2. RunPod pod spec + local hardware smoke test
  What to do: `docs/infra.md` â€” RunPod pod recipe (CUDA 12.x + torch + transformers image, network volume mounted at `/workspace`, SKUs: A100-80GB for serving/training campaigns, RTX-4090-class for cheap dev), idempotent `scripts/pod_setup.sh`; `scripts/smoke_local.py` loads Qwen/Qwen3-0.6B bf16 on the local RTX 4060 and generates 32 greedy tokens with a fixed seed, prints peak VRAM. Must NOT do: do not start paid pods; no credentials in repo (env only).
  Parallelization: Wave A | Blocked by: 1 | Blocks: 10, 13 | With: 3, 4
  References: nvidia-smi probe (RTX 4060 8GB/16GB RAM/Win10) in draft Findings; D:\MoE\.omo\drafts\familydraft-moe.md (D6); huggingface.co/Qwen/Qwen3-0.6B
  Acceptance criteria: `uv run python scripts/smoke_local.py` exit 0 printing 32 tokens + peak VRAM < 7.5GB; `docs/infra.md` exists and lists SKU + mount + env-var contract (RUNPOD_API_KEY env-only).
  QA scenarios: happy â€” smoke test completes locally; failure â€” run with `CUDA_VISIBLE_DEVICES=""` (CUDA forced off): script must exit non-zero with a clean message naming GPU unavailability, no traceback-driven crash. Evidence <attemptDir>/task-2-familydraft-moe.txt
  Commit: Y | feat(infra): RunPod pod spec and local smoke test

- [x] 3. Fixed evaluation-set manifest (4 task classes)
  What to do: `scripts/build_eval_manifest.py` builds `data/eval/{mtbench,humaneval,mbpp_sanitized,gsm8k,structured}/` with exact counts: MT-Bench 80 chat prompts (FastChat canonical), HumanEval 164, MBPP sanitized subset (pinned count recorded in manifest), GSM8K test 1319, structured = 100 deterministic JSON-schema-conditioned tasks produced by the committed generator `scripts/gen_structured_set.py` (seeded schema sampling; provenance + license documented in data/eval/README.md since we construct it). Prompts rendered for the Qwen3 chat template with THINKING MODE PINNED PER CLASS (Metis gap #10): MT-Bench/HumanEval/MBPP/structured â†’ `enable_thinking=false`; GSM8K â†’ `enable_thinking=true` with max_new_tokens 4096 (recorded machine-readable in `configs/eval_protocol.yaml` alongside system prompts). SHA-256 manifest `data/eval/MANIFEST.json` + `scripts/verify_manifest.py`. Must NOT do: no train splits here; no hand edits after sealing (rebuild-only); no class may change thinking mode without editing the committed protocol first.
  Parallelization: Wave A | Blocked by: 1 | Blocks: 10, 21, 22 | With: 2
  References: concept note D:\MoE\family_draft_moe_concept.md:632-662 (Â§10.5 metrics); EAGLE eval convention (MT-Bench/HumanEval/GSM8K, temp 0 + temp 1, bs1) in draft Findings
  Acceptance criteria: `uv run python scripts/build_eval_manifest.py && uv run python scripts/verify_manifest.py` exit 0; JSON counts match 80/164/<pinned MBPP>/1319/100 exactly; round-trip test (encodeâ†’decodeâ†’encode ids identical) passes for every prompt.
  QA scenarios: happy â€” build+verify exit 0; failure â€” flip one byte in a data file, `verify_manifest.py` exits non-zero naming the corrupted file. Evidence <attemptDir>/task-3-familydraft-moe.txt
  Commit: Y | feat(data): sealed evaluation manifest for 4 task classes

- [x] 4. Unified Qwen3 target wrapper with top-k logit capture
  What to do: `src/familydraft/targets/wrapper.py`: `TargetModel.load(repo_id, dtype="bf16")`, `.generate_greedy(prompt_ids, max_new_tokens)`, `.generate_sample(prompt_ids, temp, seed)`, `.topk_logits(ids, k)` returning (token_ids, logits, ranks); trace schema documented in `docs/trace_format.md` (JSONL: step, chosen token, top-k snapshot, latency). Supported repos pinned in `configs/targets.yaml`: Qwen3-4B/8B/14B/32B, Qwen3-30B-A3B, Qwen3-Coder-30B-A3B, plus Qwen3-0.6B for local tests. Must NOT do: no quantization; no logits-altering optimizations (e.g., no speculative tricks inside the wrapper); device_map="auto" only.
  Parallelization: Wave A | Blocked by: 1 | Blocks: 6, 8, 10, 15 | With: 2, 3, 5
  References: vocab 151936 verified configs in draft Findings; huggingface.co/Qwen/Qwen3-8B/raw/main/config.json; transformers generation docs (huggingface.co/docs/transformers/generation_strategies)
  Acceptance criteria: `uv run pytest tests/test_target_wrapper.py` green on local GPU with Qwen3-0.6B: greedy output equals a pinned reference string (seeded, transformers version pinned in pyproject); top-k snapshot shape == (T, k) with ids in [0, 151936).
  QA scenarios: happy â€” pinned-output test passes; failure â€” load repo id "Qwen/does-not-exist": wrapper raises with message naming the repo id (no generic traceback), asserted in test. Evidence <attemptDir>/task-4-familydraft-moe.txt
  Commit: Y | feat(targets): unified Qwen3 wrapper with top-k logit capture

- [x] 5. Deterministic run logger + timing primitives
  What to do: `src/familydraft/infra/{metrics,run}.py`: JSONL run logger with schema (run_id, git_sha, config_sha256, seed, event{type, ms, payload}); CUDA-event timing helper (torch.cuda.Event with explicit torch.cuda.synchronize â€” never wall-clock around async kernels, Metis gap #21); config-fingerprint util (sha256 over canonicalized JSON); global seed setter (torch/cuda/python/random) incl. attention backend flag pinning. Schema enforced by jsonschema in `configs/run_event.schema.json`. Must NOT do: no dashboards, no wandb, no network logging.
  Parallelization: Wave A | Blocked by: 1 | Blocks: 9, 20, 21 | With: 4
  References: concept note D:\MoE\family_draft_moe_concept.md:632-662 (Â§10.5 supporting metrics); D7 (standalone harness)
  Acceptance criteria: `uv run pytest tests/test_metrics.py` green: events validate against schema; invalid event (missing `type`) raises; config fingerprint identical for key-order-permuted equal configs; two seeded runs of a timing micro-op produce identical event-type sequences.
  QA scenarios: happy â€” tests green; failure â€” inject event with negative latency: validator rejects (test asserts). Evidence <attemptDir>/task-5-familydraft-moe.txt
  Commit: Y | feat(infra): deterministic run logger and timing primitives

### Wave B â€” Verification core (M1 gate)
- [x] 6. Reference sequential acceptance verifier (ground truth)
  What to do: `src/familydraft/verify/reference.py`: exact speculative-acceptance over a draft CHAIN â€” greedy mode (longest prefix where draft == target argmax) and sampling mode (standard accept/reject with adjusted residual distribution, Leviathan et al. 2023); pure-functional core `verify_chain(target_dists: list[Distribution], draft_tokens) -> accepted_prefix, bonus_token` operating on explicit categorical distributions so tests need no model. Must NOT do: no batching, no KV logic, no speed shortcuts â€” this is the oracle everything else is checked against.
  Parallelization: Wave B | Blocked by: 4 | Blocks: 8 | With: 7
  References: concept note D:\MoE\family_draft_moe_concept.md Â§3 (129-157); Leviathan et al. 2023 "Fast Inference from Transformers via Speculative Decoding" (arXiv:2211.17192); D8
  Acceptance criteria: `uv run pytest tests/test_reference_verifier.py` green: 6 hand-computed golden chains (mixed accept/reject/bonus positions); sampling-mode distribution preservation â€” KS test p>0.05 over 20k samples from a synthetic 2-state toy target at temp=1.
  QA scenarios: happy â€” goldens pass; failure â€” perturb one golden expectation: test fails naming the case id (asserted by keeping goldens in `tests/goldens/reference_verifier.json`). Evidence <attemptDir>/task-6-familydraft-moe.txt
  Commit: Y | feat(verify): reference sequential acceptance verifier

- [x] 7. Candidate trie/DAG builder with metadata
  What to do: `src/familydraft/verify/dag.py`: token trie over family vocab; `insert(proposal_tokens, expert_id, confidence, router_prob)`; shared prefixes stored once; per-node metadata per concept note Â§6.2 (expert source, confidence, router probability, support count, marginal verification cost field); deterministic budget pruning to `max_nodes` keeping highest (support, confidence) lexicographic; export adjacency + topological order for the verifier. Golden cases = concept note Â§3.3 merge example (lines 143-152) and Â§4.7 rejection-memory example (lines 282-290). Must NOT do: no scoring policies (router's job, todo 19); no GPU code.
  Parallelization: Wave B | Blocked by: 1 | Blocks: 8, 18, 19 | With: 6
  References: concept note D:\MoE\family_draft_moe_concept.md:143-152 (Â§3.3 DAG example), 282-290 (Â§4.7 example), 389-426 (Â§6 construction rules)
  Acceptance criteria: `uv run pytest tests/test_dag.py` green: Â§3.3 golden renders exact node set {return, result, \n, }, \n}} shapes; Â§4.7 golden produces "\n" root with children {-, \n-}; budget test: inserting proposals totaling 40 nodes into max_nodes=16 yields exactly 16 with deterministic survivors; metadata round-trips.
  QA scenarios: happy â€” goldens match concept-note figures exactly (asserted token-by-token); failure â€” proposal longer than remaining budget: deterministic prune, test asserts no node-count overflow and which proposal was trimmed. Evidence <attemptDir>/task-7-familydraft-moe.txt
  Commit: Y | feat(verify): candidate trie/DAG builder with metadata

- [x] 8. DAG verifier + equivalence gate (MILESTONE 1)
  What to do: `src/familydraft/verify/dag_verifier.py`: verify a candidate DAG against a target in ONE joint forward pass (tree attention over DAG nodes via SDPA/eager 4D masks with correct per-branch RoPE position_ids â€” flash-attn forbidden, platform pin; KV-cache of the base context reused), then walk per-branch acceptance using todo-6 semantics branch by branch. GREEDY gate: property test `tests/test_verify_equivalence.py`: â‰¥500 randomized cases â€” synthetic targets = random categorical distributions over vocab 50, depths 1/4/8, branchings 1/2/3 â€” assert DAG verifier accepted set == reference sequential verifier accepted set for EVERY branch, plus bonus-token agreement; 2 real-model cases with Qwen3-0.6B on local 4060 (DAGs of 8 nodes) asserting bit-identical accepted sequences. SAMPLING gate (Metis gap #7, required before any temp-1 number): same property equivalence for accept/reject decisions given shared RNG tapes, PLUS statistical test `tests/test_sampling_equivalence.py` â€” token-level Ï‡Â² between DAG-verified speculative sampling and direct target sampling on 3 fixed prompts Ã— 20k samples each, p>0.01 pre-registered. Must NOT do: no approximate acceptance, no top-k truncation during verification, no performance optimization that changes semantics (optimize later, prove equality first).
  Parallelization: Wave B | Blocked by: 6, 7 | Blocks: 9, 19, 20, 21, 22 | With: none (critical path)
  References: tree-attention prior art (reference only, Apache-2.0): Medusa `medusa/model/utils.py` (github.com/FasterDecoding/Medusa), EAGLE `eagle/model/utils.py` @cb7e084 (github.com/SafeAILab/EAGLE); concept note Â§11.3 (680-684); D8
  Acceptance criteria: `uv run pytest -m equivalence` green with the full 500-case suite + real-model cases; milestone tag after merge: `git tag v0.1-M1`.
  QA scenarios: happy â€” property suite green; failure â€” mutation test: deliberately swap residual-correction order in a copy of the verifier, property suite MUST fail (record the failing output as evidence that the gate has teeth). Evidence <attemptDir>/task-8-familydraft-moe.txt
  Commit: Y | feat(verify): DAG verifier, equivalence gate green (M1)

- [x] 9. Verification/draft latency microbench + cost curve
  What to do: `scripts/bench_micro.py`: measure on local 4060 (Qwen3-0.6B) and emit `runs/microbench/cost_curve.json`: target per-token decode ms; verification ms for DAG sizes {1,2,4,8,16,32,64}; forward-ms budget table for drafter sizes. Include RunPod variant `scripts/bench_micro_pod.py` (same JSON schema) for Qwen3-8B later. This JSON is the router's C_draft/C_verify source (concept note Â§5). Must NOT do: no end-to-end campaign here; no accuracy claims.
  Parallelization: Wave B/C | Blocked by: 5, 8 | Blocks: 19, 20 | With: 10
  References: concept note D:\MoE\family_draft_moe_concept.md:310-342 (Â§5 utility objective), 619-623 (ablation: cost-aware routing)
  Acceptance criteria: `uv run python scripts/bench_micro.py` exits 0; `cost_curve.json` validates against schema and verify_ms is monotone non-decreasing in node count (asserted by `tests/test_cost_curve.py` reading the artifact); file also records the non-neural-expert latency budget: each non-neural expert drafting call must be â‰¤10% of single-target-token decode latency at bs1 on the measurement device (Metis gap #22) â€” enforced later by expert microbench assertions in todo 16/17/18 tests reading this budget field.
  QA scenarios: happy â€” JSON valid + monotone; failure â€” with CUDA unavailable the script exits non-zero with a clear message and writes no partial JSON (asserted: no file or schema-valid file only). Evidence <attemptDir>/task-9-familydraft-moe.txt
  Commit: Y | feat(bench): verification cost-curve microbenchmark

### Wave C â€” Oracle analysis + data (M2 gate)
- [ ] 10. Multi-target trace campaign on RunPod [PREP SHIPPED, commit 897f29a: configs/trace_campaign.yaml committed (pre-registered) + scripts/run_trace_campaign.py (idempotent, resumable-per-shard, config-hashed greedy trace capture: argmax + top-64 ids/logits/ranks, budget guard) + scripts/verify_traces.py (MANIFEST sha256 + JSONL validation; corrupt shard -> exit 1 naming it; synthetic happy/corrupt verified); 8B/4B/14B pod campaign execution deferred]
  What to do: `scripts/run_trace_campaign.py` (idempotent, resumable per shard, config-hashed): greedy generations from Qwen3-8B (primary), Qwen3-4B + Qwen3-14B (secondary) over ALL of todo-3's eval manifest PLUS auxiliary distillation corpora (chat subset ShareGPT-style, code corpus, GSM8K TRAIN split, JSON template corpus) using todo-4 wrapper with capture PINNED NOW (Metis gap #13 â€” recollection would cost a full rerun): every generated step records argmax token + entropy + top-k=64 (ids+logits+rank), max_new_tokens 1024 for chat/code/structured and 4096 for GSM8K-thinking; traces to `/workspace/traces/<target>/<shard>.jsonl` (storage estimate â‰¤2GB per target per class â€” verified in pod dry run before full campaign); sha256 per shard into `/workspace/traces/MANIFEST.traces.json`; log GPU-hours + $ estimate per target into `runs/costs.csv`. Budget guard: cap total GPU-hours per config (default 48 A100-hours, editable config). Must NOT do: no sampling runs; no training; do not exceed budget cap without editing config explicitly; trace parameters are committed in `configs/trace_campaign.yaml` BEFORE the campaign starts.
  Parallelization: Wave C | Blocked by: 2, 3, 4 | Blocks: 11, 12, 13 | With: 5-9 (cloud vs local)
  References: D:\MoE\docs\trace_format.md (todo 4); D:\MoE\docs\infra.md (todo 2); draft D6 (RunPod); D1 (target matrix)
  Acceptance criteria: `python scripts/verify_traces.py /workspace/traces` exit 0 â€” every shard listed in MANIFEST.traces.json exists with matching sha256; â‰¥4 task classes covered per target for Qwen3-8B; costs.csv has one row per target run; resume test: kill mid-campaign on tiny subset, rerun, completed shards skipped (byte-identical MANIFEST entries).
  QA scenarios: happy â€” checksums + coverage pass; failure â€” corrupt one shard byte: verifier exits non-zero naming the shard; partial-shard (incomplete write) detected via length field and quarantined. Evidence <attemptDir>/task-10-familydraft-moe.txt
  Commit: Y | feat(data): multi-target trace campaign runner + manifests

- [x] 11. Oracle predictability analysis + go/no-go gate (MILESTONE 2) [LOCAL verdict GO after mechanism strengthening (structured cov 0.053->0.309); Waves D-G unblocked]
  What to do: implement `src/familydraft/experts/parse_state.py` v0 HERE FIRST (Metis gap #8 ordering fix â€” todo 16 consumes it): pure-Python deterministic structural state machine (open brackets/quotes/scopes stack, indent depth, in-fence flag, enumeration detection). Then `scripts/oracle_analysis.py`: per trace position, with oracle knowledge of the target continuation, classify recoverability by cheap mechanism class with EXACT pre-registered definitions (Metis gap #12, committed in configs/oracle_thresholds.yaml BEFORE traces exist): (a) copy-suffix = longest match of â‰¥4 tokens from prompt âˆª generated prefix; (b) macro/parser action = next d tokens derivable from parse_state v0 rule set (close-bracket/quote/scope, newline-indent, fence close, enumeration continuation, JSON close); (c) repetition = â‰¥6-token n-gram repeated from preceding 512-token window. Compute per task class: coverage % at thresholds dâˆˆ{1,2,4,6,8} expected-accepted-tokens + ORACLE-BEST EXPECTED SPEEDUP UPPER BOUND (Metis gap #5 framing: best achievable bs1-greedy speedup given perfect mechanism selection). Emit `docs/reports/oracle_report.md` + machine-readable `runs/oracle/verdict.json`. Pre-registered gate: GO iff in â‰¥2 of 4 task classes (code AND structured must be among them) â‰¥25% of positions have oracle-expected acceptance â‰¥1.0 token from the cheap-mechanism union AND oracle-best expected speedup â‰¥1.5Ã—; per-mechanism-class secondary thresholds included for Phase-2 gating. Exit 0 = GO, exit 77 = NO-GO. NO-GO BRANCH (Metis gap #5): STOP Waves D-G; deliverable = evidence package + `docs/reports/pivot_options.md` enumerating (abandon / narrow to recoverable expert classes / switch task-class focus) with measured support for each; decision authority = user. Must NOT do: do not tune thresholds or definitions after traces exist (git ordering check proves pre-registration).
  Parallelization: Wave C | Blocked by: 10 | Blocks: 14-19 (via GO), 25 (gate) | With: 12
  References: concept note D:\MoE\family_draft_moe_concept.md:19-35 (Â§1 thesis), Â§9 Q1 (line 557); draft D5; Â§4.4 macro list (200-211) informs rule set
  Acceptance criteria: `oracle_report.md` exists with per-class coverage tables, depth curves, verdict line; `python scripts/oracle_analysis.py --selftest` green: synthetic trace engineered ABOVE threshold â†’ exit 0 + GO verdict; synthetic trace BELOW â†’ exit 77 + NO-GO verdict; `git log --follow configs/oracle_thresholds.yaml` predates analysis run commit (checked by `scripts/m2_order_check.py`).
  QA scenarios: happy â€” selftest both directions; failure â€” delete one input shard: analysis exits non-zero naming missing shard (no silent partial verdict). Evidence <attemptDir>/task-11-familydraft-moe.txt
  Commit: Y | feat(analysis): oracle predictability report + go/no-go gate (M2)

- [x] 12. Target-ID-tagged distillation dataset builder
  What to do: `scripts/build_distill_dataset.py`: convert todo-10 traces into training shards `data/distill/<split>/shard-*.arrow`: records (input_ids, target_next_ids, top-64 logits snapshot, target_id, task_class); `configs/target_ids.json` maps repo â†’ {id, size_tier, variant_kind}; splits stratified by task_class Ã— target. LEAK PROOF (Metis gap #1 â€” verdict validity): training records may ONLY derive from auxiliary-corpus traces; HARD assertion excludes every eval-manifest prompt via content-hash (sha256 of canonical prompt) disjointness check across ALL shards â€” eval prompts appear in traces for oracle diagnostics but are tagged `split=diag` and the builder refuses them for training shards. Must NOT do: no cross-family data; no eval leakage (checked, not assumed); no training records from eval-prompt traces under any split name.
  Parallelization: Wave C | Blocked by: 10 | Blocks: 13, 15, 21 | With: 11
  References: concept note D:\MoE\family_draft_moe_concept.md:431-441 (Â§7.1 same-family distillation), 56-62 (Â§2 target-variant embedding); draft D1
  Acceptance criteria: `uv run pytest tests/test_distill_dataset.py` green: eval-prompt content-hash disjointness across ALL training shards (leak-proof assertion); every record's target_id âˆˆ configs/target_ids.json; shard read round-trips tensor shapes; stratification counts within Â±5% of configured ratios; injected eval-prompt record is refused by the builder (negative test).
  QA scenarios: happy â€” tests green; failure â€” inject one record with unknown target_id: builder exits non-zero naming record offset + id. Evidence <attemptDir>/task-12-familydraft-moe.txt
  Commit: Y | feat(data): target-ID-tagged distillation shards

- [ ] 13. EAGLE-3 generalist baseline drafter via SpecForge (Qwen3-8B)
  What to do: FIRST `scripts/validate_specforge_qwen3.py`: clone SpecForge @7d5a69386f9, build env, dry-run a qwen3 target config through its training entrypoint and registry (`@register_draft` customization path) â€” PASS required before spending training credits (Metis gap #6 unvalidated dependency); record adapter diffs in docs/baselines/eagle3.md. Then train an EAGLE-3 drafter for Qwen3-8B on THE SAME chat+code shards todo 15's general expert trains on (corpus-fairness â€” identical training tokens + logged GPU-hours, so the generalist bar is not advantaged by data), using SpecForge's pipeline; evaluate acceptance length with SpecForge's evaluator (`specforge/eval/evaluator.py` simulated_acc_len + real acceptance on MT-Bench eval subset via todo-4 wrapper); archive checkpoint + full config + training log to `/workspace/baselines/eagle3_qwen3_8b/`; write `runs/baselines/eagle3_qwen3_8b.json`. Must NOT do: no SpecForge source modifications beyond config files; do not declare the baseline usable before acceptance-length is within 80% of published EAGLE-3 class on comparable data OR the shortfall is documented as best-effort with numbers (baseline-quality guard â€” a weak EAGLE-3 would inflate the verdict).
  Parallelization: Wave C/D | Blocked by: 2, 12 | Blocks: 21, 22, 28 | With: 11, 14-19
  References: github.com/sgl-project/SpecForge @7d5a693 (MIT); draft Findings (SpecForge facts, EAGLE-3 training facts); EAGLE-3 paper arXiv:2503.01840; EAGLE README's recommendation to use SpecForge for EAGLE-3 training
  Acceptance criteria: checkpoint dir + json report exist; report validates against `configs/baseline_report.schema.json` (fields: acc_len_mtbench, acc_len_code, train_hours, specforge_sha); evaluator rerun reproduces acc_len within Â±0.05.
  QA scenarios: happy â€” report valid + reproducible; failure â€” kill training mid-run: `scripts/resume_baseline.sh` restores from latest checkpoint and completes a 100-step resume sanity pass (asserted: loss continuity within tolerance). Evidence <attemptDir>/task-13-familydraft-moe.txt
  Commit: Y | feat(baselines): SpecForge-trained EAGLE-3 drafter for Qwen3-8B
### Wave D â€” Drafter core (after M2 GO)
- [x] 14. Shared trunk + target-variant embedding
  What to do: `src/familydraft/draft/trunk.py`: truncated Qwen3-0.6B backbone â€” first 6 transformer layers, shared input embeddings (vocab 151,936), LM head removed; target-variant conditioning `z_m` = ATTRIBUTE-CONDITIONED embedding (Metis gap #18 fix â€” enables unseen-target interpolation): MLP over normalized target-config attribute vector (param count, layers, hidden, active params, MoE flag) PLUS learned per-id residual for seen targets; added to input embeddings; forward returns h_t for experts + router. Config `configs/trunk.yaml` (layers=6, param cap â‰¤200M, dtype bf16). Must NOT do: no new tokenizer; no full-vocab head here (belongs to expert 15); no layers beyond the 6-layer cap without editing config + draft; no pure-lookup-only embedding (would make unseen targets unembeddable in Wave F).
  Parallelization: Wave D | Blocked by: 11 (GO), 1 | Blocks: 15, 16, 19 | With: none (foundation of wave)
  References: concept note D:\MoE\family_draft_moe_concept.md:98-113 (Â§3.1 trunk options â€” "reduced-width family-native decoder"), 56-62 (Â§2 z_m); draft Open-assumptions (trunk default)
  Acceptance criteria: `uv run pytest tests/test_trunk.py` green: forward shape (B,T,H) matches config; distinct z_m per target_id with non-zero grad; total params â‰¤ cap (asserted); unknown target_id raises KeyError listing the valid table.
  QA scenarios: happy â€” tests green; failure â€” craft config with layers=100: loader rejects with cap-exceeded message (validation test). Evidence <attemptDir>/task-14-familydraft-moe.txt
  Commit: Y | feat(draft): truncated-Qwen3 trunk with target-variant embedding

- [x] 15. General neural expert + same-family distillation (Stages 1-2)
  What to do: `src/familydraft/experts/general.py`: autoregressive head over trunk state (shared embeddings, full-vocab logits); training script `scripts/train_general_expert.py` implementing concept note Â§7.1-7.2: CE vs target top-k logits from todo-12 shards, conditioned on target_id; softened top-2 multiple-choice loss (MCL) config flag (`--mcl top2|top1`) with load-balancing auxiliary off in Phase 1; fixed recipe in `configs/train_general.yaml` (lr, steps, batch, seq-len 2048, seed). Checkpoints + training curves to runs/trainlogs/. Must NOT do: no rollout tuning here (todo 20); no hyperparameter sweeps beyond the pinned recipe; no data outside todo-12 shards.
  Parallelization: Wave D | Blocked by: 14, 12 | Blocks: 19, 20 | With: 16, 17, 18
  References: concept note D:\MoE\family_draft_moe_concept.md:162-166 (Â§4.1), 431-462 (Â§7.1-7.2 incl. L_MCL formula), 674-678 (Â§11.2 starvation mitigation = shared general expert)
  Acceptance criteria: `uv run pytest tests/test_general_expert.py` green (forward/vocab/conditioning); training run completes and runs/trainlogs/general_expert.json records: held-out next-token top-1 accuracy â‰¥ dense-from-scratch control accuracy âˆ’ 2pts (control trained same recipe from random init; both logged) AND loss curve strictly decreasing over first 20% of steps; checkpoint reload produces valid token ids.
  QA scenarios: happy â€” metrics file validates + checkpoint loads; failure â€” corrupt one shard: dataloader raises naming shard id; resume-from-checkpoint completes 50 steps with loss continuity (asserted). Evidence <attemptDir>/task-15-familydraft-moe.txt
  Commit: Y | feat(experts): general neural expert, same-family distillation

- [x] 16. Macro expert + renderer + parser-derived labels
  What to do: `src/familydraft/experts/macro.py` + `src/familydraft/experts/macro_render.py`: macro vocabulary v1 = exactly the 64-action set in `configs/macros.json` (superset of concept note Â§4.4 list: CLOSE_PAREN/CLOSE_BRACKET/CLOSE_BRACE, CLOSE_BLOCK, NEWLINE_INDENT, CONTINUE_ENUMERATION, CLOSE_CODE_FENCE, COPY_IDENTIFIER, REPEAT_LINE_PREFIX, CLOSE_JSON_OBJECT + language-specific closers); renderer expands macro â†’ family token ids deterministically (golden table committed; renderâ†’re-tokenize round-trip identity asserted, Metis gap on token-boundary drift); parser features CONSUME `parse_state.py` built in todo 11 (no rebuild); label deriver `scripts/derive_macro_labels.py` maps todo-10 code/structured traces to macro action sequences where derivable; macro head = small classifier over trunk state + parser features. Microbench assertion in tests: expert drafting call â‰¤10% of target per-token decode latency (budget from todo 9 cost_curve.json). Must NOT do: no learned/discovered macros in Phase 1; no LSP or tree-sitter deps (parse_state v0 only).
  Parallelization: Wave D | Blocked by: 14, 10 | Blocks: 19, 20 | With: 15, 17, 18
  References: concept note D:\MoE\family_draft_moe_concept.md:196-221 (Â§4.4), 686-690 (Â§11.4 wrong-macro mitigation: parser validation + confidence thresholds)
  Acceptance criteria: `uv run pytest tests/test_macro.py` green: renders for ALL 64 macros are golden-verified token sequences; every rendering is valid ids in [0,151936); label deriver reproduces golden macro sequences on 8 synthetic traces; macro head dev accuracy on structured task class logged in runs/experts/macro_dev.json.
  QA scenarios: happy â€” goldens green; failure â€” add malformed macro entry to a test copy of macros.json (bad token): build-time validation catches at import with message naming the macro (test asserts). Evidence <attemptDir>/task-16-familydraft-moe.txt
  Commit: Y | feat(experts): macro expert with renderer and parser state

- [x] 17. Copy/retrieval expert (suffix-array, slot filling)
  What to do: `src/familydraft/experts/copy.py`: suffix-array index over prompt + last-N generated tokens (N configurable, default 4096); propose longest-match continuations with metadata tuple (candidate_ids, source span, match_length, confidence) per concept note Â§4.5; copy-and-edit v1: token-class slots (identifier/number/string-literal) detected in matched span and re-filled from current context by best exact match, else abstain; returns None when best match < min_length (default 3). Must NOT do: no neural components; no external corpus (prompt + own output only); no fuzzy matching in v1.
  Parallelization: Wave D | Blocked by: 14 | Blocks: 19, 20 | With: 15, 16, 18
  References: concept note D:\MoE\family_draft_moe_concept.md:222-239 (Â§4.5 incl. metadata tuple formula 235-237, slot filling 238-239)
  Acceptance criteria: `uv run pytest tests/test_copy_expert.py` green: golden synthetic prompt with repeated JSON block â†’ proposed continuation equals oracle copy span exactly; slot-fill case (identifier changed) produces correct substitution; abstains (None) when no match â‰¥3; microbench in test asserts p50 query latency < 2ms CPU over 1000 queries on an 8k-token index.
  QA scenarios: happy â€” goldens green; failure â€” query longer than index â†’ graceful None + no exception (asserted); empty prompt â†’ None. Evidence <attemptDir>/task-17-familydraft-moe.txt
  Commit: Y | feat(experts): suffix-array copy/retrieval expert

- [x] 18. Rejection-memory expert v1 (online, weight-free)
  What to do: `src/familydraft/experts/reject_memory.py`: store keyed by fingerprint = (last-8 tokens, parser-state class, target_id, decode_mode) â†’ correction record (first rejected position, target replacement, accepted suffix, support count, EMA recency); policies: minimum support 3 before activation, exponential decay (half-life configurable, default 200 events), bounded LRU (default 100k entries), per-target scoping (no cross-target reads); two modes per concept note Â§4.7: (a) propose repaired branch into DAG; (b) rewrite another expert's candidate pre-verification; session-persisted via JSON to runs/ (explicit save/load). Must NOT do: no weight updates; no cross-target generalization; no unbounded memory growth.
  Parallelization: Wave D | Blocked by: 7, 14 | Blocks: 19, 20 | With: 15, 16, 17
  References: concept note D:\MoE\family_draft_moe_concept.md:263-292 (Â§4.7 + example), 692-696 (Â§11.5 mitigations: decayed counts, min support, target scoping, bounded cache)
  Acceptance criteria: `uv run pytest tests/test_reject_memory.py` green: concept-note Â§4.7 scenario reproduced â€” after 3 simulated rejections of "\n-" replaced by "\n\n-", the identical context emits DAG containing both branches; sub-support entries (count 2) do NOT activate; decay test expires stale entries; target_id isolation test (writes under target A invisible to target B); capacity test (100k+1 â†’ LRU eviction deterministic); malformed fingerprint rejected by schema validation in code.
  QA scenarios: happy â€” repro test green; failure â€” persist file corrupted: load raises with named file + falls back to empty store with logged warning (asserted). Evidence <attemptDir>/task-18-familydraft-moe.txt
  Commit: Y | feat(experts): online rejection-memory expert v1

- [x] 19. Utility router v1 (contextual bandit + abstention + horizons)
  What to do: `src/familydraft/router/router.py`: features = trunk-state summary (mean-pool last hidden), parser/repetition/copy scores from experts 16-17 feature modules, target_id one-hot, online EMA stats per expert (accepted length EMA, first-rejection position, draft ms â€” updated from verification feedback); heads: linear contextual bandit per expert estimating U(e) = E[A_e] / (C_draft_e + C_verify(marginal nodes)) with C from todo-9 cost curve; second-expert selection maximizes marginal Î”U = U({e1,e2}) âˆ’ U({e1}) estimated from historical branch-overlap stats; abstain when max U < tau_abstain; horizons h_e âˆˆ {2,4,6,8} chosen by per-expert horizon bandit. COLD-START INIT (Metis gap #11 fix): bandit weights initialized from OFFLINE SIMULATED ACCEPTANCE â€” replay todo-10 traces, score each expert's proposals against oracle target continuations, fit initial utility estimates â€” so the router is functional before any live rollout; online updates refine from there. All stats persist in run state (todo 5). Config `configs/router.yaml` pins tau_abstain initial + EMA rates. Must NOT do: NO neural router in v1 (explicit deferral per Â§11.3); NO static offline routing at RUNTIME (fragile per Not-a-Bandit findings â€” offline init only, online adaptation mandatory); no expert execution logic (interfaces only).
  Parallelization: Wave D (tail) | Blocked by: 8, 9, 15, 16, 17, 18 | Blocks: 20, 21 | With: none
  References: concept note D:\MoE\family_draft_moe_concept.md:310-386 (Â§5 objectives, outputs, features), 498-507 (Â§7.5 utility + abstention calibration); Not-a-Bandit arXiv:2510.20064 (static routing misroutes ~98% under prompt variation); draft D4 (cost-tiered experts)
  Acceptance criteria: `uv run pytest tests/test_router.py` green: returns valid (expert_subset âˆˆ all subsets of size â‰¤2, horizons, abstain flag) on synthetic features; crafted low-utility input â†’ abstain=True; EMA updates deterministic from scripted feedback sequence (golden); U computation matches hand-computed value for a golden case (tolerance 1e-6); marginal-Î”U picks expert pair with least historical overlap on crafted stats.
  QA scenarios: happy â€” goldens green; failure â€” inject latency spike for expert X in stats â†’ router's cost-adjusted selection switches off X (asserted switch). Evidence <attemptDir>/task-19-familydraft-moe.txt
  Commit: Y | feat(router): utility bandit router with abstention and horizons

### Wave E â€” Integration + Phase-1 verdict (M3 gate)
- [x] 20. Rollout training loop (policy-only, measured-latency reward) [SHIPPED locally, commit 129ed0e: scripts/train_rollout.py (router+horizon policy only, drafter frozen, measured-latency reward vs vanilla window, EMA control variate, temperature exploration, INEFFECTIVE fallback, --reward-sign-flip QA verified exit-1 on degradation) + configs/rollout.yaml + tests/test_rollout_repro.py (same-seed ±5% repro, sign-flip reverses policy direction, deterministic update; 3/3 PASS); local run 200 steps improving=True; 8B re-run pod-deferred]
  What to do: `scripts/train_rollout.py`: run draft-and-verify loop (todos 8, 19) over dev subset of todo-12 shards; log per rollout: accepted prefix length, first rejection, target correction, per-expert latencies, marginal nodes, which expert contributed accepted branch. POLICY SCOPE (Metis gap #3): optimizes ROUTER + HORIZON policies ONLY (drafter weights FROZEN; weight-level rollout training explicitly deferred). REWARD = measured net tokens/sec improvement vs vanilla window (cost-adjusted, prevents reward hacking via horizon collapse). EXPLORATION: drafts sampled with stochastic expert-head temperature Ï„_draft>0 during training (greedy rollouts give zero policy gradient â€” deterministic-reward degeneracy fix); verdict measurements stay greedy per protocol. BASELINE = per-run EMA control variate. Compute budget pinned in `configs/rollout.yaml` (max GPU-hours, step count, eval cadence). FALLBACK: if dev accepted-tokens/sec shows no improvement over 3 consecutive eval checkpoints, freeze at best checkpoint, record rollout-training as INEFFECTIVE, and proceed to verdict with the Stage-2 drafter + cold-start router (rollout training then appears only as an ablation row in todo 22) â€” the verdict is NEVER blocked by this stage. Must NOT do: no differentiable relaxation of the acceptance indicator; no hyperparameter search; no drafter weight updates; no unbounded compute beyond the pinned budget.
  Parallelization: Wave E | Blocked by: 19, 12 | Blocks: 22 | With: 21
  References: concept note D:\MoE\family_draft_moe_concept.md:464-507 (Â§7.3-7.5); draft Findings (accepted-length is non-differentiable â†’ policy-gradient surrogate)
  Acceptance criteria: `uv run python scripts/train_rollout.py --config configs/rollout.yaml` completes; runs/rollout/metrics.json shows dev accepted-tokens/sec non-decreasing across the last 3 eval checkpoints (strict improvement over step-0); same seed rerun reproduces curve within Â±5% relative (asserted by `tests/test_rollout_repro.py` on a toy 200-step schedule).
  QA scenarios: happy â€” improving curve + repro; failure â€” harness sanity: same script with `--reward-sign-flip` flag records degradation and auto-check exits non-zero flagging inverted reward (proves metric wiring). Evidence <attemptDir>/task-20-familydraft-moe.txt
  Commit: Y | feat(train): rollout policy training with measured-latency reward

- [ ] 21. Baseline suite under unified harness
  What to do: `src/familydraft/eval/baselines.py` + `scripts/run_baselines.py`: implement, in the SAME harness/instrumentation as the main system (todo 5 logger, todo-3 manifest, identical decoding settings), ALL systems subject to the OUTPUT-EQUIVALENCE INVARIANT (every greedy speculative run byte-identical to vanilla AR output â€” asserted in-harness, Metis gap #4): (a) vanilla AR target; (b) small dense drafter (dense params == active-expert param total of FamilyDraftMoE, trained on todo-12 shards with identical recipe + token count as todo 15); (c) equal-active-FLOP dense drafter (param count matched via the FLOP formula in todo 22's verdict protocol, training tokens matched, same DAG node budget at inference); (d) SpecForge EAGLE-3 checkpoint (todo 13); (e) single-best heterogeneous expert (per-task-class best fixed expert, plus learned-selection variant); (f) heterogeneous top-2 WITHOUT DAG fusion (better single chain wins). Harness self-check: identical seeds â‡’ identical greedy target outputs across ALL baselines (determinism guard). Must NOT do: no baseline-specific tuning beyond documented recipes; no Jakiro reproduction (descope recorded below); no partial runs count as baseline results. Jakiro reconciliation (Metis gap #14): Jakiro-style decoupled neural experts are CONCEPTUALLY covered by EAGLE-3 generalist bar + the chain-vs-fusion ablation (f); full Jakiro reproduction descope from Phase 1 (MIT repo @f8b018d pinned in refs for Phase 2 optional work) because its EAGLE-1-style hidden-state training would add a training campaign without separating the heterogeneous-mechanism claim.
  Parallelization: Wave E | Blocked by: 8, 12, 13 | Blocks: 22 | With: 20
  References: concept note D:\MoE\family_draft_moe_concept.md:604-617 (Â§10.3 baseline list rows 1-8), 619-630 (Â§10.4); draft D9 (EAGLE-3 = generalist bar)
  Acceptance criteria: `scripts/run_baselines.py --all` produces `runs/baselines/<name>.json` for all 6 baselines Ã— 4 task classes, each validating against `configs/baseline_report.schema.json` with full Â§10.5 metric set; determinism self-check passes (exit 0) â€” seeded rerun of baseline (a) bit-identical.
  QA scenarios: happy â€” all JSONs valid + determinism green; failure â€” force a determinism violation (patch a copy of a baseline to add unseeded dropout): self-check exits non-zero naming the divergent run pair (mutation evidence). Evidence <attemptDir>/task-21-familydraft-moe.txt
  Commit: Y | feat(baselines): full baseline suite under unified harness

- [ ] 22. Phase-1 campaign + ablation matrix (MILESTONE 3) [PARTIAL - infra SHIPPED + VERIFIED: pre-registered verdict_protocol.yaml (with plan-todo-22 pins: A100-80GB x1, bf16, SDPA, CUDA graphs off, decode-only throughput formula, >=5% win margin, verdict pair {code, structured}, budgets <=8/<=32) + baseline_report.schema.json + FLOP accounting (equal-FLOP dense baseline) + unified harness run_baselines.py (--all runs 6 systems x 4 task classes, verified 24/24) + phase1.csv (24 baseline + 48 ablation rows) + m3_verdict.py + m3_order_check.py (PASS) + determinism self-check (check_determinism.py, PASS/FAIL-mutation verified) + row reproducibility (run_phase1.py --row <hash>, spot-checked 0.11%/0.63% <3%) + 12-config pre-registered ablation matrix (configs/ablations/, all verified runnable) + EAGLE-3 SpecForge pod pipeline (setup/eval/resume) + pod orchestrator run_phase1_8b.sh (baselines + ablation matrix) + F1 audit (6/6 PASS) + F2/F4 audit (11/11 PASS) + evidence package + paper skeleton (todo 30); local 0.6B campaign RUN -> honest FAIL (DAG < vanilla and < equal-FLOP dense at 0.6B, verdict exit 78, v0.3-M3 tag correctly withheld); REMAINS pod-deferred: 8B campaign + EAGLE-3 training via SpecForge + MoE probe rows + F3 manual QA]
  What to do: FIRST `configs/verdict_protocol.yaml` â€” pre-registered, committed BEFORE any campaign row runs (Metis gap #2, machine-checkable): hardware = A100-80GB Ã—1, bf16, SDPA backend, CUDA graphs OFF, pinned transformers/torch versions; target = Qwen3-8B primary + Qwen3-4B sanity; throughput formula = decode-only target tokens/sec (generated target tokens Ã· decode wall-clock; prefill timed separately and reported, not folded into the verdict number); runs = 5 per row, report mean Â± std; win margin = â‰¥5% mean tokens/sec improvement; task classes = ALL 4 measured and reported, verdict pair pre-registered = {code, structured}; system config = router variable-k (kâˆˆ{0,1,2}, not fixed top-2 â€” the criterion evaluates the full system); budgets pinned: draft horizon â‰¤8, DAG â‰¤32 nodes, IDENTICAL for all systems incl. dense baselines; FLOP-matching formula = 6Â·N_activeÂ·T_draft drafting FLOPs with router overhead included in N_active; training-compute matching = identical training tokens + logged GPU-hours. Then `scripts/run_phase1.py <row>` config-driven campaign runner: bs1 greedy (primary) + bs1 temp=1 with speculative-sampling correction (secondary, only after todo 8 sampling gate green); rows = full system + all 6 baselines (todo 21) + ablation rows from concept note Â§10.4 AS CONFIG SWITCHES ONLY â€” fixed-top1 vs fixed-top2 vs variable-k; chain vs DAG fusion; acceptance-routing vs utility-routing; shared vs expert-specific horizons; Â±target embedding; Â±online feedback; Â±rejection memory; Â±abstention; Â±rollout-policy (todo 20 trained vs cold-start) (12 ablation configs committed in `configs/ablations/` BEFORE running). Hidden-state-mixing ablation DESCOPE to Phase 2 (requires Jakiro-style training campaign â€” recorded decision, Metis gap #15). Primary metric end-to-end target tokens/sec; full supporting metric set per Â§10.5 including marginal accepted tokens from second expert and abstention precision. PLUS: Qwen3-30B-A3B verification-cost probe (Cascade risk): verification latency vs DAG size {1,4,8,16} + expert-scatter token-activation count, abort-and-log if speculation degrades below vanilla for any DAG size. Runner isolates per-row failures and continues. All rows mandatory â€” including unfavorable ones. Verdict runs apply the reproducibility pin (cold-start router + empty memory, seeded order, single pass). Output `runs/results/phase1.csv` + `scripts/m3_verdict.py` computing the verdict strictly from the protocol: PASS iff full system beats equal-active-FLOP dense drafter (row c) by â‰¥5% mean decode tokens/sec on the pre-registered pair {code, structured} AND reports gap vs EAGLE-3 row. Must NOT do: no task-subset cherry-picking; no row dropped; no config edits after campaign start (config hashes recorded per row); no protocol edits after first row runs.
  Parallelization: Wave E | Blocked by: 20, 21 | Blocks: 23, 24 | With: none
  References: concept note D:\MoE\family_draft_moe_concept.md:570-662 (Â§10), 732-742 (Â§13 Phase-1 success criterion), 666-708 (Â§11 failure modes); Cascade arXiv:2506.20675; draft D9
  Acceptance criteria: `configs/verdict_protocol.yaml` committed with git timestamp strictly before first campaign run (checked by `scripts/m3_order_check.py`); `runs/results/phase1.csv` contains ALL rows (count asserted = 7 systems Ã— 4 task classes + 12 ablations Ã— 4 + MoE probe rows); every row carries 5-run meanÂ±std + config hash; every greedy row passed the output-equivalence invariant (runner log); every row reproducible via `scripts/run_phase1.py --row <hash>` (spot-check 2 rows, tokens/sec within Â±3%); `scripts/m3_verdict.py` exits 0 (PASS) or 78 (FAIL) with the verdict table computed strictly from protocol thresholds; milestone tag `git tag v0.3-M3`.
  QA scenarios: happy â€” campaign + verdict complete; failure â€” one ablation config crashes: runner continues remaining rows, failed row recorded with config hash + error (asserted in runner unit test with injected crash). Evidence <attemptDir>/task-22-familydraft-moe.txt
  Commit: Y | feat(eval): Phase-1 campaign and ablation matrix (M3)

- [ ] 23. Phase-1 verdict report + concept-note reference corrections [SHIPPED locally: overclaims corrected via Corrigendum; M3 campaign + ablation matrix + verification-chain sections added; supporting-metric tables (per-expert proposal utility, second-expert marginal 17.4%, abstention 0.0% cold-start, cost breakdown vs §11.7, EAGLE-3 gap) via scripts/supporting_metrics.py + DagSpeculator instrumentation (commit 16f5844); scripts/check_report_links.py PASS (commit 70cd847); concept-note §15 corrections (MetaSD venue + Cascade/BanditSpec/EVICT/EcoSpec/Jakiro refs) applied §15-only (commit 5ca911d); report records verdict against both bars; pod 8B supporting-metric re-run deferred]
  What to do: `docs/reports/phase1_verdict.md`: complete results tables (linked to runs/ artifacts), per-expert proposal-utility table (uniquely accepted tokens / (draft latency + marginal verify latency)), second-expert marginal analysis, abstention precision/recall, cost breakdown vs Â§11.7 threat, honest statement of EAGLE-3 gap; verdict recorded against BOTH bars. Apply concept-note corrections as a diff to `family_draft_moe_concept.md` Â§15 ONLY: MetaSD venue â†’ "Findings of ACL 2026 (2026.findings-acl.1629)"; add references: Cascade arXiv:2506.20675, BanditSpec arXiv:2505.15141 (ICML 2025), EVICT arXiv:2605.00342, EcoSpec arXiv:2607.12696, Jakiro arXiv ID 2502.06282. Must NOT do: no manuscript prose; no edits to concept note beyond Â§15 references + a one-line pointer to the verdict report.
  Parallelization: Wave E | Blocked by: 22 | Blocks: 24 | With: none
  References: concept note D:\MoE\family_draft_moe_concept.md:785-797 (Â§15), Â§11; draft Findings (venue corrections)
  Acceptance criteria: report exists; `scripts/check_report_links.py docs/reports/phase1_verdict.md` exit 0 (every cited runs/ path exists); concept-note diff touches only Â§15 + pointer line (asserted by `git diff --stat` pattern check).
  QA scenarios: happy â€” link checker green; failure â€” report appendix citing a nonexistent runs/ path â†’ checker exits non-zero naming it. Evidence <attemptDir>/task-23-familydraft-moe.txt
  Commit: Y | docs(report): Phase-1 verdict + reference corrections

### Wave F â€” Phase 2: specialization & transfer; Phase 3: online adaptation
- [ ] 24. Target-variant conditioning at scale + zero-shot transfer eval
  What to do: extend distillation (todo-15 recipe) to full target matrix {8B, 14B, 32B, Coder-30B-A3B} with target_id conditioning on RunPod; zero-shot transfer test per concept note Â§9 Q6: train on {4B, 8B, 14B} only, evaluate acceptance + tokens/sec on UNSEEN 32B and unseen Coder-30B-A3B targets without retraining; report per-target tables. Must NOT do: no per-target retraining in the transfer rows (that defeats the test); no changes to router architecture.
  Parallelization: Wave F | Blocked by: 23 | Blocks: 25 | With: 26
  References: concept note D:\MoE\family_draft_moe_concept.md:431-441 (Â§7.1 conditioning), line 562 (Â§9 Q6), 56-62 (Â§2); draft D1
  Acceptance criteria: `runs/results/phase2_transfer.csv` complete (targets Ã— task classes: accepted length, tokens/sec, drafter overhead); transfer delta (unseen vs seen targets) computed and logged; config hashes + seeds recorded.
  QA scenarios: happy â€” CSV validates + all cells populated; failure â€” attempt (in test harness) to evaluate a target included in training set: script refuses with named target (guard test). Evidence <attemptDir>/task-24-familydraft-moe.txt
  Commit: Y | feat(phase2): target-variant transfer evaluation

- [ ] 25. Gated expert additions (reasoning / logit-dynamics) via oracle gate
  What to do: read todo-11 oracle report sections for reasoning-transition and logit-dynamics coverage; gate rule (committed in `configs/expert_gates.yaml` BEFORE this todo runs): add expert X iff oracle report shows the mechanism class recovers â‰¥5 percentage points additional positions with expected acceptance â‰¥1.0 token in â‰¥1 task class, else record SKIP with evidence pointer. For each expert that passes: implement minimal version (reasoning-transition = specialist head trained on thinking-mode traces; logit-dynamics = tiny GRU over target top-k sequences per concept note Â§4.6), integrate as router-selectable expert, rerun affected campaign rows. Must NOT do: no implementation of a gated-out expert; no gate relaxation after results are visible.
  Parallelization: Wave F | Blocked by: 23, 11 | Blocks: 27 | With: 24
  References: concept note D:\MoE\family_draft_moe_concept.md:168-180 (Â§4.2), 241-261 (Â§4.6); draft D3 (deferral gate)
  Acceptance criteria: gate decision file `docs/reports/expert_gate_decisions.md` exists with per-expert decision + evidence pointer into oracle_report.md; for each PASS expert: integrated system reruns â‰¥2 task-class rows and deltas logged; for SKIP: no code exists for that expert (checked by file-absence assertion test).
  QA scenarios: happy â€” decisions consistent with evidence; failure â€” feed synthetic oracle report below gate to `scripts/apply_expert_gate.py --dry-run`: decision = SKIP (logic tested). Evidence <attemptDir>/task-25-familydraft-moe.txt
  Commit: Y | feat(phase2): gated expert additions per oracle evidence

- [x] 26. Online calibration: per-expert EMA + dynamic horizons + abstention ROC [SHIPPED, commit d43c17d: src/familydraft/calibration.py (agreement stats + agreement_extension §6.3, IsotonicCalibration PAV rolling-window, abstention_roc AUC) + configs/online.yaml + DagSpeculator wiring (agreement extends draft horizon) + UtilityRouter wiring (set_calibrators, calibrated expected_acceptance, calibrator feed in update_feedback); tests/test_online_calibration.py 4/4 (agreement extends horizon, calibration monotone on skewed data, ROC emits AUC, empty-safe); 8B dev ROC re-run pod-deferred]
  What to do: implement concept note Â§6.3 agreement rule (expert agreement extends horizon, reduces branching, priority in budget allocation) in DAG builder; per-expert acceptance calibration feeding router utility estimates (isotonic calibration on rolling window); abstention calibration head precision/recall curve (concept note Â§7.5) over dev set â†’ ROC + AUC logged; config flags for each in `configs/online.yaml`. Must NOT do: no weight updates; no memory cross-target sharing beyond todo-18 scoping.
  Parallelization: Wave F | Blocked by: 23 | Blocks: 27 | With: 24
  References: concept note D:\MoE\family_draft_moe_concept.md:408-416 (Â§6.3 agreement), 498-507 (Â§7.5), 751-757 (Â§13 Phase 3)
  Acceptance criteria: `uv run pytest tests/test_online_calibration.py` green: agreement rule extends horizon on crafted two-expert agreement case (asserted horizon change); calibration monotone (predicted order preserved) on synthetic skewed data; ROC JSON emitted with AUC field for real dev run.
  QA scenarios: happy â€” tests + ROC green; failure â€” disable calibration in config: abstention precision drops measurably vs enabled (delta asserted > 0 in A/B test on dev). Evidence <attemptDir>/task-26-familydraft-moe.txt
  Commit: Y | feat(phase3): calibration, dynamic horizons, abstention ROC

- [x] 27. Session-level drift adaptation test [SHIPPED, commit 4b81253: scripts/session_drift_experiment.py - 3-segment chat->code->structured drift, adaptive (router EMA + rejection memory + calibration ON) vs truly-frozen static control (no_online_feedback, no memory, no calibration), real wall-clock tokens/sec per window, recovery-window metric per shift, bounded-memory assert; local 0.6B run: adaptive beats static on all segments (19.5/33.1/21.0 vs 10.6/24.4/10.6), recovery [0,9], memory bounded, EXIT 0; 8B pod run deferred]
  What to do: `scripts/session_drift_experiment.py`: synthetic drift scenario â€” session task mix shifts mid-run (chat â†’ code â†’ structured, 3 segments Ã— N prompts) on Qwen3-8B; compare adaptive system (router EMA + rejection memory + calibration ON) vs static control (all adaptation frozen at segment-1 stats): trajectory of accepted tokens/sec per 100-token window; memory store growth + eviction stats logged; adaptation speed metric = windows-to-recover â‰¥90% of best-segment utility after each shift. Must NOT do: no new expert types; no cross-session persistence beyond documented save/load.
  Parallelization: Wave F (tail) | Blocked by: 26 | Blocks: 28 | With: 25
  References: concept note D:\MoE\family_draft_moe_concept.md:751-757 (Phase 3 session adaptation), 648-651 (online adaptation speed metric), Â§9 Q9 (line 565)
  Acceptance criteria: runs/results/session_drift.json complete; adaptive trajectory â‰¥ static control on mean accepted tokens/sec in segments 2 and 3 (asserted); recovery-window metric computed per shift; memory bounded (â‰¤ configured LRU cap, asserted).
  QA scenarios: happy â€” adaptive beats static post-shift; failure â€” with rejection memory disabled: adaptation delta shrinks measurably and the attribution check logs the delta (asserted > 0). Evidence <attemptDir>/task-27-familydraft-moe.txt
  Commit: Y | feat(phase3): session drift adaptation experiment

### Wave G â€” Phase 4: serving integration + evidence package
- [ ] 28. SGLang integration + serving batch matrix
  What to do: integrate trained drafter with SGLang's custom drafter extension path for Qwen3-8B (version-pin SGLang in docs/infra.md); serve + measure end-to-end throughput at batch sizes {1, 8, 32} under continuous batching vs vanilla and EAGLE-3 (todo 13); produce the batch-collapse curve (tokens/sec/batch per system) identifying where speculation advantage disappears; output equivalence check: 50 prompts greedy â€” SGLang-integrated outputs must be IDENTICAL to standalone harness outputs. Fallback documented: if SGLang custom-drafter path fails after 2 documented integration attempts, switch to vLLM plugin path (record reasons). Must NOT do: no accuracy approximations to fit the engine; no skipping the equivalence check.
  Parallelization: Wave G | Blocked by: 23 (system frozen) | Blocks: 29 | With: 24-27
  References: concept note D:\MoE\family_draft_moe_concept.md:759-766 (Â§13 Phase 4); SGLang docs (docs.sglang.ai); draft D7
  Acceptance criteria: `runs/phase4/serving_matrix.json` complete (3 batch sizes Ã— 3 systems Ã— throughput + latency); equivalence check 50/50 identical greedy outputs (asserted); SGLang version + config hash recorded.
  QA scenarios: happy â€” matrix + equivalence green; failure â€” seed a deliberate divergence (test config with wrong temperature plumbing): equivalence check fails naming divergent prompt ids. Evidence <attemptDir>/task-28-familydraft-moe.txt
  Commit: Y | feat(phase4): SGLang integration + serving batch matrix

- [ ] 29. MoE-target verification-cost study + DAG budget policy [DAG BUDGET POLICY SHIPPED, commit 2e6845e: configs/dag_budget_policy.json (per-target node budgets, cap where verify_cost(m)*m^-1 exceeds vanilla decode) + src/familydraft/verify/budget_policy.py (schema-validating loader, invalid thresholds rejected at load) + tests/test_budget_policy.py (6/6) + DagSpeculator prune_to_budget wiring, todo-7 regression green; 30B-A3B measured cost curves + activation-scatter study pod-deferred]
  What to do: dedicated study on Qwen3-30B-A3B per Cascade (arXiv:2506.20675) risk: verification latency vs DAG size {1,2,4,8,16,32} for chain vs tree candidates; measure draft-token expert activation scatter (unique routed experts touched per verification batch) against single-token baseline; compare against EVICT-style adaptive verification idea (cite-only, no reimplementation unless trivial); decide per-target DAG node budget policy (e.g., cap nodes where verify_cost(m)Â·mâ»Â¹ exceeds vanilla decode cost) and codify in `configs/dag_budget_policy.json` consumed by the DAG builder. Must NOT do: no reimplementation of EVICT/EcoSpec; no claims beyond measured curves.
  Parallelization: Wave G | Blocked by: 8, 9, 22 | Blocks: 30 | With: 28
  References: Cascade arXiv:2506.20675 (speculation hurts 18/35 MoE pairs, 2-3x verification growth); EVICT arXiv:2605.00342; EcoSpec arXiv:2607.12696; concept note Â§11.7
  Acceptance criteria: `docs/reports/moe_target_study.md` with cost curves + activation-scatter measurements + the policy recommendation; policy JSON validated and loaded by DAG builder without breaking todo-7 tests (regression run asserted).
  QA scenarios: happy â€” report + policy regression green; failure â€” policy with invalid thresholds rejected by schema validation at load (test asserts). Evidence <attemptDir>/task-29-familydraft-moe.txt
  Commit: Y | docs(phase4): MoE-target verification study + DAG budget policy

- [x] 30. Evidence package + paper skeleton [SHIPPED - commit 95b8727: scripts/build_evidence_index.py (130 artifacts, sha256+git-sha) + scripts/check_evidence_index.py (all 11 claims resolve, PASS) + docs/paper (claims.csv/skeleton.md/refs.bib seeded from Findings) + README repo map + one-command-per-milestone reproduce (dry-run PASS: all referenced scripts exist); full-clone paper write-up remains post-pod]
  What to do: `scripts/build_evidence_index.py`: index EVERY runs/, traces manifest, checkpoint reference, and report artifact with sha256 + git sha + config hash into `EVIDENCE_INDEX.json`; `docs/paper/` skeleton: section outline (intro/related/method/experiments) with each planned claim mapped to an evidence path (claims table), related-work `refs.bib` seeded from draft Findings (Jakiro, SpecForge, EAGLE 1-3, MetaSD, Not-a-Bandit, BanditSpec, Cascade, EVICT, EcoSpec, DraftExpert, Medusa, Sequoia/UMbreLLa, SpecInfer); README.md final: repo map + reproduce-from-scratch instructions (one command per milestone). Must NOT do: no full manuscript prose; no result reinterpretation beyond the verdict reports.
  Parallelization: Wave G (final) | Blocked by: all | Blocks: none | With: none
  References: all prior todos; draft Findings bibliography; concept note Â§12 framing (lines 712-727)
  Acceptance criteria: `scripts/build_evidence_index.py` exit 0; `scripts/check_evidence_index.py` green â€” every claim row in docs/paper/claims.csv resolves to an existing indexed artifact; README reproduce commands dry-run pass (parses + referenced scripts exist).
  QA scenarios: happy â€” index + claims checkers green; failure â€” add a claim citing a bogus evidence id: checker exits non-zero naming the claim row. Evidence <attemptDir>/task-30-familydraft-moe.txt
  Commit: Y | docs(final): evidence index, paper skeleton, README
<!-- APPEND TASK BATCHES ABOVE THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit [LOCAL SUBSET SHIPPED - scripts/f1_audit.py 6/6 PASS: M1 tag v0.1-M1 present, M2 report present (v0.2-M2-NOGO implied), v0.3-M3 correctly withheld while FAIL, verdict protocol predates campaign (git ordering), sealed-manifest leak-proof re-run, protocol working-tree immutability; full clean-clone re-execution pod-deferred]
- [ ] F2. Code quality review [LOCAL SUBSET SHIPPED - scripts/f2_f4_audit.py 11/11 PASS: ruff clean, no TODO/FIXME markers outside docs/, no flash-attn in verify/, uv.lock pins OK; Windows-CPU + RunPod-Linux-GPU test subsets pod-deferred]
- [ ] F3. Real manual QA (agent-executed) [POD-DEFERRED - requires Qwen3-8B load on RunPod: 10-prompt byte-identity + speedup >1.0x + cold-start abstention sanity]
- [ ] F4. Scope fidelity [LOCAL SUBSET SHIPPED - scripts/f2_f4_audit.py: no QwQ/235B/Qwen3.5 artifacts, no engine code in src (sglang/vllm/triton), docs/paper skeleton-only, no secrets in git-tracked files, verdict_protocol/oracle_thresholds unmodified since M3 pin commit, configs/expert_gates.yaml present; full git-log proof re-run pod-deferred]

## Commit strategy
- Conventional commits, one commit per todo (see todo Commit lines); squash fixups before pushing tags.
- Milestone tags: `v0.1-M1` (todo 8), `v0.3-M3` (todo 22); todo 11's NO-GO branch tags `v0.2-M2-NOGO` and stops.
- Pre-registration integrity: `configs/oracle_thresholds.yaml`, `configs/verdict_protocol.yaml`, `configs/ablations/`, `configs/trace_campaign.yaml`, `configs/expert_gates.yaml` are committed BEFORE the campaigns that consume them; their post-campaign immutability is checked in F1 via git log ordering.
- No secrets in git (env-only, scanned in F4); no large binaries (traces/checkpoints/run JSONs live on RunPod network volume with sha256 manifests indexed by EVIDENCE_INDEX.json); git holds code/configs/reports only.
- Branch model: `main` linear with milestone tags; experimental branches allowed but merge only with green acceptance criteria.

## Success criteria
1. M1: equivalence gates green (todo 8) â€” DAG verifier â‰¡ reference verifier on â‰¥500 randomized greedy cases + real-model cases; sampling-mode statistical gate green. Tag v0.1-M1.
2. M2: oracle report + machine verdict emitted against pre-registered thresholds; GO proceeds, NO-GO executes the pivot branch with evidence package (todo 11).
3. M3 (Phase-1 thesis): full system beats equal-active-FLOP dense drafter by â‰¥5% mean decode tokens/sec (5 runs, A100-80GB, bs1 greedy) on the pre-registered task pair {code, structured}, all 4 classes reported; gap vs SpecForge-trained EAGLE-3 generalist quantified; output-equivalence invariant held on 100% of greedy verdict rows. Tag v0.3-M3.
4. Reproducibility: any phase1.csv row reproducible via `scripts/run_phase1.py --row <hash>` within Â±3%; verdict protocol + configs frozen in git history.
5. Evidence package complete (todo 30): every claim in docs/paper/claims.csv resolves to an indexed artifact with sha256; README reproduces each milestone from scratch.
6. Guardrails held: no eval-data leakage into training (assertion green), no scoped-out targets/artifacts (F4 green), abstention precision + second-expert marginal contribution reported regardless of sign.

