"""Unified Qwen3-family target wrapper with top-k logit capture.

Single-device, bf16-reference serving of Hugging Face target repos. No
quantization, no speculative tricks, no logits-altering optimizations: this
wrapper is the ground-truth oracle that all verification and trace campaigns
are measured against, so its logits must be exactly the model's logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = [
    "TargetLoadError",
    "TargetModel",
    "TopKSnapshot",
    "parse_dtype",
    "topk_snapshot",
]

_DTYPE_ALIASES: Final[dict[str, torch.dtype]] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


class TargetLoadError(RuntimeError):
    """A target repo failed to load. The message always names the repo_id.

    This is the only error type :meth:`TargetModel.load` raises for a bad or
    unreachable repo, so callers never see raw hub/HTTP tracebacks.
    """


def parse_dtype(name: str) -> torch.dtype:
    """Map a pinned dtype name to a torch dtype; reject everything else."""
    try:
        return _DTYPE_ALIASES[name]
    except KeyError:
        allowed = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(f"unknown dtype {name!r}; allowed: {allowed}") from None


@dataclass(frozen=True)
class TopKSnapshot:
    """Per-position top-k logit snapshot with dense ranks."""

    token_ids: torch.Tensor  # (T, k) int64 vocab ids, descending logit order
    logits: torch.Tensor  # (T, k) logits of those ids, descending
    ranks: torch.Tensor  # (T, k) int32 dense rank over the full vocab


def topk_snapshot(logits: torch.Tensor, k: int) -> TopKSnapshot:
    """Snapshot the top-k of a (T, vocab) logit tensor.

    Ranks use dense-rank semantics: the rank of a selected token is the
    number of vocabulary tokens with a STRICTLY greater logit. Ties share
    a rank and the argmax always carries rank 0, regardless of
    tie-breaking order. (torch.topk and torch.argsort break ties
    differently on CUDA, so positional-in-sort ranks are not well-defined;
    dense ranks are.)
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (T, vocab); got {tuple(logits.shape)}")
    vocab = logits.shape[-1]
    if k > vocab:
        raise ValueError(f"k={k} exceeds vocab size {vocab}")
    values, token_ids = torch.topk(logits, k, dim=-1)
    # rank = #{v : logits[v] > value} = first insertion index of -value in
    # the ascending sort of -logits (equal logits are NOT strictly greater).
    sorted_neg, _ = torch.sort(-logits, dim=-1)
    ranks = torch.searchsorted(sorted_neg, -values).to(torch.int32)
    return TopKSnapshot(token_ids=token_ids, logits=values, ranks=ranks)


class TargetModel:
    """One Hugging Face Qwen3-family target on one device (no sharding)."""

    def __init__(
        self,
        repo_id: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dtype_name: str,
        device: torch.device,
    ) -> None:
        self.repo_id = repo_id
        self.model = model
        self.tokenizer = tokenizer
        self.dtype_name = dtype_name
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    @classmethod
    def load(
        cls,
        repo_id: str,
        dtype: str = "bf16",
        device: str | torch.device | None = None,
    ) -> TargetModel:
        """Load repo at the pinned dtype onto one device.

        ``device`` defaults to cuda when available, else cpu. Unknown or
        failed repos raise :class:`TargetLoadError` naming ``repo_id``.
        """
        torch_dtype = parse_dtype(dtype)
        if device is None:
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch_dtype)
        except Exception as exc:  # boundary: any hub/load failure -> typed error
            raise TargetLoadError(
                f"failed to load target model '{repo_id}': {exc}"
            ) from exc
        model = model.to(resolved)
        model.eval()
        return cls(
            repo_id=repo_id,
            model=model,
            tokenizer=tokenizer,
            dtype_name=dtype,
            device=resolved,
        )

    def generate_greedy(self, prompt_ids: object, max_new_tokens: int) -> torch.Tensor:
        """Deterministic greedy decode. Returns the FULL sequence (1, P+N)."""
        ids = self._prepare_ids(prompt_ids)
        with torch.inference_mode():
            return self.model.generate(
                ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._pad_token_id(),
            )

    def generate_greedy_batch(
        self, prompts: list[list[int]], max_new_tokens: int
    ) -> list[list[int]]:
        """Greedy decode a batch of prompts in ONE model call.

        Returns per-prompt NEW tokens only (continuation), each trimmed at the
        first EOS. Padding is attention-masked, so per-sequence outputs are
        identical to calling generate_greedy individually.
        """
        if not prompts:
            return []
        pad = self._pad_token_id()
        eos = self._eos_id()
        max_len = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), max_len), pad, dtype=torch.long, device=self._device)
        mask = torch.zeros((len(prompts), max_len), dtype=torch.long, device=self._device)
        for i, p in enumerate(prompts):
            # LEFT-pad: canonical side for decoder-only (causal) generation.
            # Right-padding is a known footgun for causal attention (HF warns),
            # so shorter prompts sit at the END of the row with pad on the left.
            start = max_len - len(p)
            ids[i, start:] = torch.tensor(p, dtype=torch.long, device=self._device)
            mask[i, start:] = 1
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad,
            )
        results: list[list[int]] = []
        for i, p in enumerate(prompts):
            # Rows are left-padded: the generated continuation is everything
            # AFTER position max_len (prompt occupies [max_len-len(p), max_len)).
            new = out[i, max_len:].tolist()
            if eos is not None and eos in new:
                new = new[: new.index(eos)]
            results.append(new)
        return results

    def _eos_id(self) -> int | None:
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        return None if eos is None else int(eos)

    def generate_sample(
        self,
        prompt_ids: object,
        temp: float,
        seed: int,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """Seeded temperature sampling (top_k/top_p disabled: pure temperature).

        Returns the FULL sequence (1, P+N). Identical (prompt, temp, seed)
        triples reproduce identical token sequences.
        """
        if temp <= 0:
            raise ValueError("temperature must be positive; use generate_greedy")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        ids = self._prepare_ids(prompt_ids)
        with torch.inference_mode():
            return self.model.generate(
                ids,
                do_sample=True,
                temperature=float(temp),
                top_k=0,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._pad_token_id(),
            )

    def topk_logits(self, ids: object, k: int) -> TopKSnapshot:
        """One joint forward pass over the full sequence; (T, k) snapshots.

        Position t captures the distribution over the token that follows
        prefix ids[0 : t + 1]. Logits are exactly the model's — no cache
        mutation, no approximation.
        """
        batch = self._prepare_ids(ids)
        with torch.inference_mode():
            logits = self.model(input_ids=batch).logits[0]
        return topk_snapshot(logits, k)

    def _prepare_ids(self, ids: object) -> torch.Tensor:
        if isinstance(ids, torch.Tensor):
            tensor = ids.to(dtype=torch.long)
        else:
            tensor = torch.as_tensor(ids, dtype=torch.long)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[0] != 1:
            raise ValueError(
                f"prompt_ids must be 1-D or (1, T); got {tuple(tensor.shape)}"
            )
        return tensor.to(self._device)

    def _pad_token_id(self) -> int | None:
        config = self.model.generation_config
        pad = getattr(config, "pad_token_id", None)
        if pad is not None:
            return int(pad)
        eos = getattr(config, "eos_token_id", None)
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        return None if eos is None else int(eos)
