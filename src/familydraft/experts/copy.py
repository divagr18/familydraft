"""Copy/retrieval expert (plan todo 17).

Non-neural suffix-index retrieval over the prompt plus recent generated tokens.
Proposes the longest continuation that repeats an earlier span, with metadata
(candidate_ids, source span, match_length, confidence). A v1 copy-and-edit
pass re-fills token-class slots (identifier/number/string) from the most recent
same-class token in the current context, abstaining when no such token exists.
No fuzzy matching, no external corpus, no learned components.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

SLOT_CLASSES = frozenset({"identifier", "number", "string"})


@dataclass(frozen=True)
class CopyProposal:
    candidate_ids: tuple[int, ...]
    source_start: int
    source_end: int
    match_length: int
    confidence: float


class CopyExpert:
    def __init__(self, seed: int = 4, min_length: int = 3, max_index: int = 4096) -> None:
        if seed < 1:
            raise ValueError("seed must be >= 1")
        self.seed = seed
        self.min_length = min_length
        self.max_index = max_index
        self.tokens: list[int] = []
        self.seed_index: dict[tuple[int, ...], list[int]] = defaultdict(list)

    def build_index(self, tokens: list[int]) -> None:
        self.tokens = list(tokens)[-self.max_index :]
        self.seed_index = defaultdict(list)
        n = len(self.tokens)
        for i in range(n - self.seed + 1):
            self.seed_index[tuple(self.tokens[i : i + self.seed])].append(i)

    def update(self, new_tokens: list[int]) -> None:
        self.build_index(self.tokens + list(new_tokens))

    def propose(self, context: list[int], max_new: int = 8) -> CopyProposal | None:
        if len(context) < self.seed or not self.tokens:
            return None
        anchor = tuple(context[-self.seed :])
        positions = self.seed_index.get(anchor)
        if not positions:
            return None
        n_ctx = len(context)
        n_idx = len(self.tokens)
        best_len = -1
        best_start = -1
        for q in positions:
            if q >= n_idx - self.seed:
                continue  # the trivial tail self-match
            if q + self.seed > n_idx:
                continue
            k = self.seed
            start = q
            while (
                start - 1 >= 0
                and n_ctx - k - 1 >= 0
                and self.tokens[start - 1] == context[n_ctx - k - 1]
            ):
                start -= 1
                k += 1
            if k > best_len:
                best_len = k
                best_start = start
        if best_len < self.min_length or best_start < 0:
            return None
        cont_start = best_start + best_len
        cont_end = min(cont_start + max_new, n_idx)
        candidate = tuple(self.tokens[cont_start:cont_end])
        if not candidate:
            return None
        confidence = min(1.0, best_len / max(self.min_length * 2, 1))
        return CopyProposal(
            candidate_ids=candidate,
            source_start=best_start,
            source_end=cont_end,
            match_length=best_len,
            confidence=confidence,
        )

    @staticmethod
    def fill_slots(
        candidate_ids: list[int],
        context: list[int],
        token_class_fn,
    ) -> list[int] | None:
        recent_by_class: dict[str, int] = {}
        for tok in reversed(context):
            cls = token_class_fn(tok)
            if cls in SLOT_CLASSES and cls not in recent_by_class:
                recent_by_class[cls] = tok
        result: list[int] = []
        for tok in candidate_ids:
            cls = token_class_fn(tok)
            if cls in SLOT_CLASSES:
                replacement = recent_by_class.get(cls)
                if replacement is None:
                    return None
                result.append(replacement)
            else:
                result.append(tok)
        return result
