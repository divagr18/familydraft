# RunPod runbook — FamilyDraftMoE on Qwen3-8B

Run the integrated speculative-decoding speedup eval on **Qwen3-8B** on RunPod.
Everything below is copy-paste. Two code-transfer paths are given: **(A) git**
(recommended) and **(B) bundle upload** (no git).

---

## 1. Pod sizing & template

- **Model footprint:** Qwen3-8B in bf16 ≈ **16 GB** + KV cache/activations + the
  small 0.6B trunk (~1.2 GB) ≈ **18–20 GB** working set.
- **GPU choice:**
  - **A100-40GB / A100-80GB** — recommended (headroom + fastest).
  - **RTX 4090 (24 GB)** — fits and is cheap; slightly slower.
- **Template:** any RunPod **PyTorch 2.x / CUDA** template (ships torch + CUDA).
  `pod_setup.sh` keeps the template's CUDA torch and only adds the other deps.

Cost sanity: Qwen3-8B downloads ~16 GB the first run; subsequent runs reuse the
Hugging Face cache on the network volume. Set `HF_HOME=/workspace/hf-cache` so the
weights persist across pod restarts.

---

## 2. Path A — push to GitHub, clone on the pod (recommended)

### 2a. (One-time, on your Windows machine) push the repo

Create an empty repo on GitHub (web or `gh`), then in `D:\MoE` (PowerShell):

```powershell
cd D:\MoE
git add .gitignore pyproject.toml
git commit -m "chore: ignore codegraph index"
git remote add origin https://github.com/<YOUR_USER>/familydraft.git
git branch -M main
git push -u origin main
```

(`.gitignore` already excludes `data/`, `runs/`, `.venv/`, `.codegraph/`, secrets —
so only code/configs/docs are pushed.)

### 2b. On the pod terminal

```bash
# optional: keep HF weights on the persistent volume
export HF_HOME=/workspace/hf-cache

cd /workspace
git clone https://github.com/<YOUR_USER>/familydraft.git
cd familydraft
bash scripts/pod_setup.sh     # installs deps (keeps template torch), verifies GPU
bash scripts/run_8b.sh        # runs the Qwen3-8B speedup eval
```

Results land in `runs/results/integrated_speedup_8b.json` and are printed at the end.

---

## 3. Path B — bundle upload (no git)

### 3a. (On your Windows machine) make the bundle

If you have WSL/Git-Bash:
```bash
cd /d/MoE
bash scripts/make_bundle.sh
# -> ../familydraft_bundle.tar.gz
```
Otherwise zip the repo excluding `.git/`, `.venv/`, `data/`, `runs/`, `.codegraph/`.

### 3b. Upload + run on the pod

Upload the bundle to the pod's `/workspace` (RunPod web UI "Upload" or `scp`), then:
```bash
export HF_HOME=/workspace/hf-cache
cd /workspace
tar xzf familydraft_bundle.tar.gz
cd familydraft
bash scripts/pod_setup.sh
bash scripts/run_8b.sh
```

---

## 4. Tuning the run

`run_8b.sh` reads env overrides:

```bash
# copy expert only (fastest headline number, skips the 0.6B trunk load)
DRAFTERS=copy bash scripts/run_8b.sh

# copy + general experts (default) with deeper speculation
REPO=Qwen/Qwen3-8B MAX_NEW=128 SPEC_LEN=8 DRAFTERS=copy,general bash scripts/run_8b.sh
```

- `SPEC_LEN=8` drafts up to 8 tokens/round (more upside on a faster GPU).
- `MAX_NEW` = tokens generated per prompt (128 gives stable timing).

---

## 5. Reading the output

`run_speculative_eval.py` prints a JSON summary:

- `mean_speedup` — vanilla_time / speculative_time (>1 means faster).
- `mean_tokens_per_round` — accepted tokens per verification round (>1 means the
  drafter is earning its keep).
- `mean_agreement` — token-for-token match with standalone greedy (should be ~1.0;
  a tiny shortfall is the documented bf16 near-tie artifact, not wrong tokens).

Local 0.6B reference (RTX 4060): copy ≈ **1.15×** (peak 1.57×). On A100 with 8B and
`SPEC_LEN=8`, expect a **higher** number because per-token overhead is what
speculation amortizes. Report what you get — that is the Phase-1 data point.

---

## 6. Retrieving results

```bash
cat /workspace/familydraft/runs/results/integrated_speedup_8b.json
```
Copy it back via the RunPod web UI download, or `scp` from the pod, or `git add`
the JSON + push if you want it versioned.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `No CUDA GPU visible` in pod_setup | You picked a CPU pod or non-NVIDIA template. Redeploy with a GPU template. |
| `torch.cuda.is_available()` False though driver present | Template torch/CUDA mismatch. Use a standard RunPod PyTorch template; do not force-reinstall torch. |
| OOM loading Qwen3-8B | Use A100-40/80GB, or close other GPU processes; 24GB cards are tight. |
| Slow first run | HF weight download (~16GB). Reuses cache after. Ensure `HF_HOME=/workspace/hf-cache`. |
| `import yaml` / module errors | pod_setup.sh was skipped or failed. Re-run `bash scripts/pod_setup.sh`. |
| git clone needs auth (private repo) | Use a deploy key / PAT, or use Path B (bundle). |

---

## 8. What this produces for the project

This run is the **Phase-1 campaign at scale** (plan todo 22 on a real target). It
yields the wall-clock speedup of the heterogeneous drafter vs vanilla AR on
Qwen3-8B — the headline number the local 0.6B result was a small-scale preview of.
Feed `integrated_speedup_8b.json` into `docs/reports/` when you're back.
