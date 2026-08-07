"""Integrated draft->verify loop (plan Wave E / todo 20-22 glue).

Incremental greedy chain speculative decoding with KV-cache reuse. Lossless by
construction: a drafted token is accepted only when it equals the target's own
greedy argmax, and the bonus token is always the target's argmax, so the emitted
sequence is identical to vanilla greedy decoding (asserted in tests + eval).
"""

from __future__ import annotations

import torch


def _crop_past(past, length: int):
    if hasattr(past, "crop"):
        past.crop(length)
        return past
    raise TypeError(f"unsupported cache type for cropping: {type(past).__name__}")


class IntegratedSpeculator:
    def __init__(self, target_model, draft_fn, spec_len: int = 4, target_id: int = 0) -> None:
        self.model = target_model.model
        self.tokenizer = target_model.tokenizer
        self.draft_fn = draft_fn
        self.spec_len = spec_len
        self.target_id = target_id
        self.device = next(self.model.parameters()).device
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        self._eos = None if eos is None else int(eos)

    def _feed(self, past, input_ids: torch.Tensor):
        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids, past_key_values=past, use_cache=True
            )
        return out.past_key_values, out.logits[0]

    def generate(
        self, prompt_ids: list[int], max_new_tokens: int
    ) -> dict:
        x = torch.tensor([prompt_ids], device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=x, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        seq_len = len(prompt_ids)

        generated: list[int] = []
        rounds = 0
        accepted_tokens = 0
        while len(generated) < max_new_tokens:
            rounds += 1
            t_next = int(torch.argmax(logits, dim=-1))
            context_ids = prompt_ids + generated
            draft = list(self.draft_fn(context_ids))[: self.spec_len]

            if not draft or draft[0] != t_next:
                generated.append(t_next)
                if t_next == self._eos:
                    break
                past, logits_next = self._feed(
                    past, torch.tensor([[t_next]], device=self.device)
                )
                logits = logits_next[-1]
                seq_len += 1
                continue

            d = torch.tensor([draft], device=self.device)
            draft_past, draft_logits = self._feed(past, d)
            accepted = [draft[0]]
            bonus = None
            for j in range(len(draft) - 1):
                t = int(torch.argmax(draft_logits[j], dim=-1))
                if draft[j + 1] == t:
                    accepted.append(draft[j + 1])
                else:
                    bonus = t
                    break
            if bonus is None:
                bonus = int(torch.argmax(draft_logits[len(draft) - 1], dim=-1))
            m = len(accepted)
            accepted_tokens += m
            cropped = _crop_past(draft_past, seq_len + m)
            past, logits_next = self._feed(
                cropped, torch.tensor([[bonus]], device=self.device)
            )
            generated.extend(accepted)
            generated.append(bonus)
            if bonus == self._eos:
                break
            logits = logits_next[-1]
            seq_len += m + 1

        return {
            "tokens": generated[:max_new_tokens],
            "rounds": rounds,
            "accepted_tokens": accepted_tokens,
            "tokens_per_round": len(generated) / max(1, rounds),
        }


class GeneralDrafter:
    """KV-cache-aware general drafter (P3).

    Keeps the trunk's KV cache across rounds and resyncs it to the current
    context via longest-prefix matching: the cache is truncated to the shared
    prefix and only the delta is re-forwarded, so drafting costs one forward
    per draft token instead of a full context re-encode per draft token.
    """

    def __init__(
        self, expert, spec_len: int, target_id: int, device
    ) -> None:
        self.expert = expert
        self.trunk = expert.trunk
        self.spec_len = spec_len
        self.target_id = target_id
        self.device = device
        self.past = None
        self.seq: list[int] = []
        self._logits = None
        self._logits_len = 0
        self._base_len = 0

    def _forward_cached(self, token_list: list[int], past):
        ids = torch.tensor([token_list], device=self.device)
        inputs_embeds = self.trunk.backbone.embed_tokens(ids)
        z = self.trunk.target_variant(self.target_id).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + z
        out = self.trunk.backbone(
            inputs_embeds=inputs_embeds, past_key_values=past, use_cache=True
        )
        h = out.last_hidden_state
        self._logits = self.expert.lm_head(h)[0, -1]
        return out.past_key_values

    def _resync(self, context_ids: list[int]) -> None:
        # Fast path: the common case is context_ids extending the previous
        # context (seq = prev_context + draft). Verify the boundary first so a
        # fully divergent context (new prompt in train_router) still resyncs
        # from scratch.
        L = 0
        if (
            self._base_len > 0
            and len(context_ids) >= self._base_len
            and self.seq[: self._base_len] == context_ids[: self._base_len]
        ):
            L = self._base_len
        while L < len(self.seq) and L < len(context_ids) and self.seq[L] == context_ids[L]:
            L += 1
        if L == 0 or self.past is None:
            self.past = self._forward_cached(context_ids, None)
            self.seq = list(context_ids)
            self._logits_len = len(self.seq)
            self._base_len = len(context_ids)
            return
        self.past = _crop_past(self.past, L)
        self.seq = self.seq[:L]
        delta = context_ids[L:]
        if delta:
            self.past = self._forward_cached(delta, self.past)
            self.seq = list(context_ids)
            self._logits_len = len(self.seq)
        else:
            self._logits_len = 0
        self._base_len = len(context_ids)

    def __call__(self, context_ids: list[int]) -> list[int]:
        self._resync(context_ids)
        if self._logits_len < len(self.seq):
            if len(self.seq) == 1:
                self.past = self._forward_cached([self.seq[-1]], None)
            else:
                self.past = _crop_past(self.past, len(self.seq) - 1)
                self.past = self._forward_cached([self.seq[-1]], self.past)
            self._logits_len = len(self.seq)
        draft: list[int] = []
        with torch.inference_mode():
            for _ in range(self.spec_len):
                tok = int(torch.argmax(self._logits, dim=-1))
                draft.append(tok)
                self.past = self._forward_cached([tok], self.past)
                self.seq.append(tok)
                self._logits_len = len(self.seq)
        return draft


def make_general_drafter(expert, spec_len: int, target_id: int, device) -> GeneralDrafter:
    return GeneralDrafter(expert, spec_len, target_id, device)


def make_copy_drafter(copy_expert, spec_len: int):
    def draft_fn(context_ids: list[int]) -> list[int]:
        copy_expert.build_index(context_ids)
        proposal = copy_expert.propose(context_ids, max_new=spec_len)
        if proposal is None:
            return []
        return list(proposal.candidate_ids)

    return draft_fn
