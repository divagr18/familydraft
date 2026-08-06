"""Schema-validated JSONL run logging and GPU-safe timing primitives."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
import torch

RUN_EVENT_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "configs" / "run_event.schema.json"


@lru_cache(maxsize=1)
def load_run_event_schema() -> dict[str, Any]:
    """Load (and cache) the run-event JSON Schema."""
    with RUN_EVENT_SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate_run_record(record: Mapping[str, Any]) -> None:
    """Validate one run record against the schema; raise ValueError naming the field."""
    try:
        jsonschema.validate(instance=dict(record), schema=load_run_event_schema())
    except jsonschema.ValidationError as exc:
        where = "/".join(str(p) for p in exc.absolute_path)
        raise ValueError(
            f"run record failed schema validation at {where!r}: {exc.message}"
        ) from exc


@dataclass
class TimingResult:
    """Outcome of one timed block: backend used plus elapsed milliseconds."""

    backend: str
    elapsed_ms: float | None = None


@contextmanager
def cuda_timer() -> Iterator[TimingResult]:
    """Time a block with CUDA events when a GPU is available, else perf_counter.

    On CUDA, start/stop events bookend the block and torch.cuda.synchronize()
    runs before the elapsed time is read, so async kernels are never timed by
    wall clock. ``TimingResult.backend`` records which backend measured the block.
    """
    if torch.cuda.is_available():
        result = TimingResult(backend="cuda_event")
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield result
        finally:
            stop.record()
            torch.cuda.synchronize()
            result.elapsed_ms = start.elapsed_time(stop)
    else:
        result = TimingResult(backend="perf_counter")
        start_perf = time.perf_counter()
        try:
            yield result
        finally:
            result.elapsed_ms = (time.perf_counter() - start_perf) * 1000.0


class RunLogger:
    """Append-only JSONL run logger; every line is a schema-validated run record."""

    def __init__(
        self,
        path: Path | str,
        *,
        run_id: str,
        git_sha: str,
        config_sha256: str | None = None,
        seed: int | None = None,
        backend_flags: Mapping[str, Any] | None = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._base: dict[str, Any] = {
            "run_id": run_id,
            "git_sha": git_sha,
            "config_sha256": config_sha256,
            "seed": seed,
            "backend_flags": dict(backend_flags) if backend_flags is not None else None,
            "events": [],
        }
        validate_run_record(self._base)  # fail fast on bad identity fields

    @property
    def path(self) -> Path:
        return self._path

    def log(
        self, event_type: str, ms: float, payload: Mapping[str, Any] | None = None
    ) -> None:
        """Append one event as a schema-validated run record line."""
        self._check_event(event_type, ms)
        event = {
            "type": event_type,
            "ms": ms,
            "payload": dict(payload) if payload is not None else None,
        }
        record = {**self._base, "events": [event]}
        validate_run_record(record)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    @staticmethod
    def _check_event(event_type: str, ms: float) -> None:
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event 'type' must be a non-empty string")
        if isinstance(ms, bool) or not isinstance(ms, (int, float)):
            raise ValueError(f"event 'ms' must be a number, got {type(ms).__name__}")
        if not math.isfinite(ms) or ms < 0:
            raise ValueError(f"event 'ms' must be a finite number >= 0, got {ms!r}")
