"""Determinism utilities: config fingerprinting and global seed pinning."""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import torch

# numpy is optional: runs must work in environments without it.
try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without numpy
    np = None  # type: ignore[assignment]

_NUMPY_SEED_MODULUS = 2**32


def config_fingerprint(config: dict[str, Any]) -> str:
    """Return the sha256 hex digest of a config over canonicalized JSON.

    Canonical form: sorted keys (recursively), fixed separators, ASCII.
    Equal configs that differ only in key insertion order fingerprint identically.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def set_global_seeds(seed: int) -> dict[str, Any]:
    """Pin python/numpy/torch/cuda RNGs and deterministic attention backends.

    Returns the flag values actually set so callers can record them in the run
    record (RunLogger ``backend_flags``). Platform pin: eager/SDPA math
    attention only - flash attention is forbidden (no flash-attn in the core).
    """
    seed = int(seed)
    rngs_seeded = ["python_random"]
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % _NUMPY_SEED_MODULUS)
        rngs_seeded.append("numpy")
    torch.manual_seed(seed)
    rngs_seeded.append("torch")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        rngs_seeded.append("torch_cuda")

    # Required for bit-exact cuBLAS reductions when deterministic algorithms
    # are later enabled; harmless to set unconditionally.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    return {
        "seed": seed,
        "rngs_seeded": rngs_seeded,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "attention_backend": {
            "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient_sdp": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math_sdp": bool(torch.backends.cuda.math_sdp_enabled()),
        },
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
