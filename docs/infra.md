# Infrastructure — RunPod pods + local dev box

This document is the single source of truth for where familydraft code runs.
Plan reference: `.omo/plans/familydraft-moe.md` (todo 2, platform pin in
"Verification strategy").

## 1. Where things run

| Environment | Role | Allowed workloads |
|---|---|---|
| Local Windows box (RTX 4060 8GB, Win 10/11) | dev-only | unit tests (CPU-safe), `scripts/smoke_local.py` (Qwen3-0.6B bf16), lint, small debugging runs |
| RunPod Linux pod, A100-80GB class | campaigns | target serving/training: trace campaign (todo 10), expert training (todos 15-18), rollout (todo 20), baselines (todo 21), Phase-1 (todo 22), MoE study (todo 29) |
| RunPod Linux pod, RTX-4090-class (24GB) | cheap dev | harness debugging, microbenches (todo 9), 4B bf16 dry runs, anything that does not need 80GB |

All GPU training/eval campaigns run on RunPod Linux pods. The local box never
runs campaigns. flash-attn is forbidden in the core verifier path (platform
pin: eager/SDPA only), so no pod image may hard-depend on it.

## 2. Base image (RunPod)

Pinned software versions, transcribed from `uv.lock` (do not hand-edit these;
bump the lock, then update this table):

| Component | Version (from uv.lock) |
|---|---|
| Python | >=3.11,<3.13 (pods standardize on 3.11; lock also resolves 3.12) |
| torch | 2.13.0 |
| transformers | 5.14.1 (pyproject floor: >=4.51) |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.26.1 |
| numpy | 2.4.6 (py3.11) / 2.5.1 (py3.12) |
| triton (Linux only) | 3.7.1 |
| nvidia-cuda-runtime (Linux torch dep) | 13.0.96 |
| nvidia-cudnn-cu13 (Linux torch dep) | 9.20.0.48 |

### CUDA version note (spec deviation, recorded honestly)

The work plan sketched "CUDA 12.x". The locked torch 2.13.0 manylinux wheels
actually pull the **CUDA 13.x** runtime stack via pip
(`nvidia-cuda-runtime 13.0.96`, `nvidia-cudnn-cu13 9.20.0.48`,
`nvidia-nccl-cu13`, `cuda-toolkit` extras — all Linux-markered in `uv.lock`).
Therefore:

- Pod images must provide a CUDA 13-capable NVIDIA driver (>=580). Any recent
  RunPod CUDA template satisfies this.
- Recommended template class: `runpod/pytorch:2.13.0-py3.11-cuda13.0.0-devel`
  if offered; otherwise any CUDA 13.x devel image with Python 3.11 + curl.
  Because the pip wheels bundle the CUDA runtime libraries, a driver-only
  image also works — `scripts/pod_setup.sh` does not need nvcc present.
- A CUDA 12.x-only image is NOT compatible with the locked wheel set; do not
  pick one.

### Windows local box caveat (recorded honestly)

PyPI's torch wheels for Windows are CPU-only (`torch==2.13.0` installed as
`2.13.0+cpu` by a plain `uv sync`). For local GPU work, install the CUDA
build into the venv without touching `pyproject.toml`/`uv.lock`:

```powershell
uv pip install "torch==2.13.0+cu130" --extra-index-url https://download.pytorch.org/whl/cu130
```

(`uv sync` will revert this; re-run the command after any sync on the dev box.
This machine currently runs driver 610.88 / CUDA UMD 13.3, which supports
cu130.)

## 3. Network volume

One RunPod network volume, mounted at `/workspace` on every pod. Layout:

```
/workspace/
  hf-cache/          # HF_HOME=/workspace/hf-cache (models survive pod restarts)
  traces/            # todo 10 trace shards + MANIFEST.traces.json
  baselines/         # todo 13 EAGLE-3 checkpoints
  checkpoints/       # expert/trunk/rollout checkpoints
  manifests/         # sha256 manifests for all of the above
```

Rules:

- Large artifacts (models, traces, checkpoints) live ONLY on the volume —
  never in git. Git holds code, configs, reports only.
- Every campaign writes a sha256 manifest next to its output.
- Set `HF_HOME=/workspace/hf-cache` in the pod environment so downloads are
  cached across restarts and never consume root-disk quota.

## 4. SKU list

| SKU | VRAM | Use | Notes |
|---|---|---|---|
| A100-80GB (SXM/PCIe) | 80GB | serving + training campaigns | Qwen3-8B/14B bf16 with room for DAG verification; Qwen3-32B, Qwen3-30B-A3B and Qwen3-Coder-30B-A3B fit on 1x80GB bf16 |
| H100-80GB | 80GB | same as A100, opportunistic | swap in when cheaper/faster; record SKU actually used in `runs/costs.csv` |
| RTX 4090 | 24GB | cheap dev | Qwen3-0.6B/4B bf16, harness debugging, todo 9 microbenches |
| (local) RTX 4060 | 8GB | dev-only | unit tests + 0.6B smoke; nothing larger |

Budget guard: campaign scripts cap GPU-hours per config (todo 10, default 48
A100-hours). Pods are stopped between sessions; the volume persists.

## 5. Env-var contract

Env-only. NEVER committed, NEVER written into repo files, NEVER echoed into
committed artifacts. `.env` is gitignored; these variables are set in the
RunPod pod template / shell profile.

| Variable | Required | Purpose |
|---|---|---|
| `RUNPOD_API_KEY` | yes (for pod automation) | RunPod API auth for start/stop/status scripts |
| `HF_TOKEN` | only for gated HF repos | Hugging Face auth; Qwen3 models are public, so optional |
| `HF_HOME` | yes on pods | must be `/workspace/hf-cache` |

Any new secret follows the same rule: env var + `.gitignore`d local `.env`,
nothing else. If a credential ever lands in git, rotate it immediately; the
commit history is considered compromised.

## 6. Pod bootstrap

On a fresh pod (or after a restart), run:

```bash
bash scripts/pod_setup.sh
```

The script is idempotent (safe to re-run): installs `uv` if missing, runs
`uv sync --frozen` against the committed `uv.lock`, verifies a GPU is visible
(`nvidia-smi` + `torch.cuda.is_available()`), and prints the pinned versions
for the evidence log. Exit 0 = pod ready; non-zero = do not start work.

## 7. Local smoke test

```powershell
uv run python scripts/smoke_local.py
```

Loads `Qwen/Qwen3-0.6B` bf16 on the local GPU, generates exactly 32 greedy
tokens from the fixed prompt "Speculative decoding works by" with
`torch.manual_seed(0)` and `do_sample=False`, prints token count, decoded
text, and peak VRAM (`torch.cuda.max_memory_allocated`). Exit codes:
0 = success, 1 = wrong token count / runtime fault, 2 = no CUDA GPU visible
(clean one-line message, no traceback).
