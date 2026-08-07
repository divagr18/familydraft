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


def make_general_drafter(expert, spec_len: int, target_id: int, device):
    def draft_fn(context_ids: list[int]) -> list[int]:
        cur = torch.tensor([context_ids], device=device)
        draft: list[int] = []
        with torch.inference_mode():
            for _ in range(spec_len):
                logits = expert(cur, target_id)[0, -1]
                tok = int(torch.argmax(logits, dim=-1))
                draft.append(tok)
                cur = torch.cat([cur, torch.tensor([[tok]], device=device)], dim=1)
        return draft

    return draft_fn


def make_copy_drafter(copy_expert, spec_len: int):
    def draft_fn(context_ids: list[int]) -> list[int]:
        copy_expert.build_index(context_ids)
        proposal = copy_expert.propose(context_ids, max_new=spec_len)
        if proposal is None:
            return []
        return list(proposal.candidate_ids)

    return draft_fn
