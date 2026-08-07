"""Rejection-memory expert v1 (plan todo 18).

Online, weight-free store mapping context fingerprints to corrections learned
from failed speculative verifications. Policies: minimum support before
activation, exponential decay (half-life), bounded LRU eviction, and per-target
scoping (target_id is part of the key, so there are no cross-target reads).

Two consumption modes:
  (a) propose(fingerprint)  -> repaired branch tokens to add to the DAG
  (b) rewrite(candidate, fingerprint) -> rewrite another expert's candidate

Session persistence is explicit save/load of JSON under runs/.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

MIN_SUPPORT_DEFAULT = 3
HALF_LIFE_DEFAULT = 200.0
MAX_ENTRIES_DEFAULT = 100_000


class MemoryLoadError(RuntimeError):
    pass


@dataclass
class CorrectionEntry:
    replacement: tuple[int, ...]
    accepted_suffix: tuple[int, ...]
    support: float
    last_event: int


def validate_fingerprint(fingerprint) -> None:
    if not isinstance(fingerprint, (tuple, list)) or len(fingerprint) != 4:
        raise ValueError(f"malformed fingerprint: expected 4 fields, got {fingerprint!r}")
    last8, parser_class, target_id, decode_mode = fingerprint
    if not isinstance(last8, (tuple, list)) or len(last8) > 8:
        raise ValueError(f"malformed fingerprint: last8 must be <=8 tokens, got {last8!r}")
    if not all(isinstance(t, int) for t in last8):
        raise ValueError(f"malformed fingerprint: last8 tokens must be ints, got {last8!r}")
    if not isinstance(parser_class, str) or not isinstance(decode_mode, str):
        raise ValueError("malformed fingerprint: parser_class and decode_mode must be str")
    if not isinstance(target_id, (str, int)):
        raise ValueError("malformed fingerprint: target_id must be str or int")


class RejectionMemory:
    def __init__(
        self,
        min_support: int = MIN_SUPPORT_DEFAULT,
        half_life: float = HALF_LIFE_DEFAULT,
        max_entries: int = MAX_ENTRIES_DEFAULT,
    ) -> None:
        self.min_support = min_support
        self.half_life = half_life
        self.max_entries = max_entries
        self._store: "OrderedDict[str, CorrectionEntry]" = OrderedDict()
        self._event = 0

    @property
    def size(self) -> int:
        return len(self._store)

    def advance_events(self, n: int = 1) -> None:
        self._event += n

    @staticmethod
    def _key(fingerprint) -> str:
        last8, parser_class, target_id, decode_mode = fingerprint
        return json.dumps([list(last8), parser_class, str(target_id), decode_mode])

    def _effective_support(self, entry: CorrectionEntry) -> float:
        dt = self._event - entry.last_event
        if dt <= 0:
            return entry.support
        return entry.support * (0.5 ** (dt / self.half_life))

    def record_rejection(
        self,
        fingerprint,
        replacement: list[int],
        accepted_suffix: list[int] | None = None,
    ) -> None:
        validate_fingerprint(fingerprint)
        self._event += 1
        key = self._key(fingerprint)
        entry = self._store.get(key)
        if entry is None:
            entry = CorrectionEntry(
                tuple(replacement), tuple(accepted_suffix or []), 0.0, self._event
            )
        entry.support += 1.0
        entry.last_event = self._event
        entry.replacement = tuple(replacement)
        entry.accepted_suffix = tuple(accepted_suffix or [])
        self._store[key] = entry
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def get_correction(self, fingerprint) -> CorrectionEntry | None:
        validate_fingerprint(fingerprint)
        entry = self._store.get(self._key(fingerprint))
        if entry is None:
            return None
        if self._effective_support(entry) < self.min_support:
            return None
        return entry

    def propose(self, fingerprint) -> list[int] | None:
        entry = self.get_correction(fingerprint)
        if entry is None:
            return None
        return list(entry.replacement)

    def rewrite(self, candidate_ids: list[int], fingerprint) -> list[int]:
        entry = self.get_correction(fingerprint)
        if entry is None:
            return list(candidate_ids)
        replaced = list(entry.replacement)
        tail = list(candidate_ids)[len(replaced) :]
        return replaced + tail

    def save(self, path: Path) -> None:
        payload = {
            "schema": "familydraft.rejection_memory.v1",
            "event": self._event,
            "min_support": self.min_support,
            "half_life": self.half_life,
            "entries": {
                key: {
                    "replacement": list(e.replacement),
                    "accepted_suffix": list(e.accepted_suffix),
                    "support": e.support,
                    "last_event": e.last_event,
                }
                for key, e in self._store.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RejectionMemory":
        raw = Path(path).read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
            entries = payload["entries"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise MemoryLoadError(f"corrupted rejection-memory file: {path}") from exc
        mem = cls(
            min_support=payload.get("min_support", MIN_SUPPORT_DEFAULT),
            half_life=payload.get("half_life", HALF_LIFE_DEFAULT),
        )
        mem._event = payload.get("event", 0)
        for key, e in entries.items():
            mem._store[key] = CorrectionEntry(
                tuple(e["replacement"]),
                tuple(e["accepted_suffix"]),
                float(e["support"]),
                int(e["last_event"]),
            )
        return mem

    @classmethod
    def load_or_empty(cls, path: Path) -> "RejectionMemory":
        try:
            return cls.load(path)
        except (MemoryLoadError, OSError):
            return cls()
