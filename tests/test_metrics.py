"""CPU-only tests for familydraft.infra.metrics and familydraft.infra.run."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import torch

from familydraft.infra.metrics import (
    RunLogger,
    cuda_timer,
    load_run_event_schema,
    validate_run_record,
)
from familydraft.infra.run import config_fingerprint, set_global_seeds

GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


def _valid_record() -> dict:
    return {
        "run_id": "unit-test-run",
        "git_sha": GIT_SHA,
        "config_sha256": "a" * 64,
        "seed": 7,
        "backend_flags": None,
        "events": [{"type": "step", "ms": 1.5, "payload": None}],
    }


class TestRunEventSchema:
    def test_valid_record_passes(self) -> None:
        validate_run_record(_valid_record())
        jsonschema.validate(_valid_record(), load_run_event_schema())

    def test_rejects_event_missing_type(self) -> None:
        record = _valid_record()
        record["events"] = [{"ms": 1.0, "payload": None}]
        with pytest.raises(ValueError, match="type"):
            validate_run_record(record)

    def test_rejects_negative_ms(self) -> None:
        record = _valid_record()
        record["events"] = [{"type": "step", "ms": -0.1, "payload": None}]
        with pytest.raises(ValueError, match="ms"):
            validate_run_record(record)

    def test_rejects_missing_run_id(self) -> None:
        record = _valid_record()
        del record["run_id"]
        with pytest.raises(ValueError, match="run_id"):
            validate_run_record(record)

    def test_rejects_missing_git_sha(self) -> None:
        record = _valid_record()
        del record["git_sha"]
        with pytest.raises(ValueError, match="git_sha"):
            validate_run_record(record)


class TestConfigFingerprint:
    def test_identical_for_key_order_permuted_equal_configs(self) -> None:
        a = {"model": "qwen3-8b", "seed": 7, "gen": {"max_new_tokens": 128, "temp": 0.0}}
        b = {"gen": {"temp": 0.0, "max_new_tokens": 128}, "seed": 7, "model": "qwen3-8b"}
        assert a == b
        assert config_fingerprint(a) == config_fingerprint(b)

    def test_sensitive_to_value_change(self) -> None:
        assert config_fingerprint({"seed": 7}) != config_fingerprint({"seed": 8})
        assert config_fingerprint({"a": {"x": 1}}) != config_fingerprint({"a": {"x": 2}})

    def test_is_sha256_hex(self) -> None:
        fp = config_fingerprint({"k": "v"})
        assert len(fp) == 64
        int(fp, 16)  # raises if not hex


class TestSetGlobalSeeds:
    def test_torch_and_numpy_draws_reproducible(self) -> None:
        np = pytest.importorskip("numpy")
        set_global_seeds(123)
        t1, n1 = torch.rand(8), np.random.rand(4)
        set_global_seeds(123)
        assert torch.equal(t1, torch.rand(8))
        assert (n1 == np.random.rand(4)).all()

    def test_different_seeds_differ(self) -> None:
        set_global_seeds(1)
        t1 = torch.rand(8)
        set_global_seeds(2)
        assert not torch.equal(t1, torch.rand(8))

    def test_returns_pinned_attention_backend_flags(self) -> None:
        flags = set_global_seeds(5)
        assert flags["seed"] == 5
        assert flags["attention_backend"] == {
            "flash_sdp": False,
            "mem_efficient_sdp": False,
            "math_sdp": True,
        }
        assert flags["cudnn_deterministic"] is True
        assert flags["cudnn_benchmark"] is False
        assert "torch" in flags["rngs_seeded"]
        # flags reflect real torch state
        assert torch.backends.cuda.flash_sdp_enabled() is False
        assert torch.backends.cuda.math_sdp_enabled() is True


class TestCudaTimer:
    def test_measures_and_reports_backend(self) -> None:
        with cuda_timer() as timing:
            torch.arange(1000).sum()
        expected = "cuda_event" if torch.cuda.is_available() else "perf_counter"
        assert timing.backend == expected
        assert timing.elapsed_ms is not None
        assert timing.elapsed_ms >= 0.0


def _seeded_timed_loop(seed: int, path: Path) -> list[dict]:
    flags = set_global_seeds(seed)
    logger = RunLogger(
        path,
        run_id="loop-run",
        git_sha=GIT_SHA,
        config_sha256=config_fingerprint({"op": "matmul"}),
        seed=seed,
        backend_flags=flags,
    )
    logger.log("init", 0.0, {"rngs": flags["rngs_seeded"]})
    for i in range(4):
        with cuda_timer() as timing:
            draw = float(torch.rand(1)[0])
            _ = torch.rand(16, 16) @ torch.rand(16, 16)
        logger.log("matmul_step", timing.elapsed_ms, {"step": i, "draw": draw})
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [event for rec in records for event in rec["events"]]


class TestSeededDeterminism:
    def test_two_seeded_runs_identical_event_type_sequences(self, tmp_path: Path) -> None:
        events_a = _seeded_timed_loop(2024, tmp_path / "a.jsonl")
        events_b = _seeded_timed_loop(2024, tmp_path / "b.jsonl")
        assert [e["type"] for e in events_a] == [e["type"] for e in events_b]
        assert [e["type"] for e in events_a] == ["init", *["matmul_step"] * 4]
        # seeded RNG draws inside the timed loop reproduce exactly (ms excluded: nondeterministic)
        signatures_a = [(e["type"], e["payload"]) for e in events_a]
        signatures_b = [(e["type"], e["payload"]) for e in events_b]
        assert signatures_a == signatures_b


class TestRunLogger:
    def test_round_trip_jsonl_validates(self, tmp_path: Path) -> None:
        path = tmp_path / "run.jsonl"
        logger = RunLogger(
            path, run_id="rt", git_sha="ab12cd34ef", config_sha256="b" * 64, seed=3
        )
        logger.log("warmup", 0.5, {"device": "cpu"})
        logger.log("decode_step", 1.25)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        for rec in records:
            jsonschema.validate(rec, load_run_event_schema())
        assert [r["events"][0]["type"] for r in records] == ["warmup", "decode_step"]
        assert records[0]["events"][0]["payload"] == {"device": "cpu"}
        assert records[0]["events"][0]["ms"] == 0.5
        assert records[1]["events"][0]["payload"] is None
        assert {r["run_id"] for r in records} == {"rt"}
        assert {r["config_sha256"] for r in records} == {"b" * 64}

    def test_rejects_missing_type(self, tmp_path: Path) -> None:
        logger = RunLogger(tmp_path / "x.jsonl", run_id="r", git_sha="c" * 40)
        with pytest.raises(ValueError, match="type"):
            logger.log("", 1.0)

    def test_rejects_negative_ms(self, tmp_path: Path) -> None:
        logger = RunLogger(tmp_path / "x.jsonl", run_id="r", git_sha="c" * 40)
        with pytest.raises(ValueError, match="ms"):
            logger.log("step", -1.0)

    def test_rejects_bool_ms(self, tmp_path: Path) -> None:
        logger = RunLogger(tmp_path / "x.jsonl", run_id="r", git_sha="c" * 40)
        with pytest.raises(ValueError, match="ms"):
            logger.log("step", True)

    def test_invalid_event_writes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        logger = RunLogger(path, run_id="r", git_sha="c" * 40)
        with pytest.raises(ValueError):
            logger.log("step", -2.0)
        assert not path.exists()

    def test_constructor_rejects_non_hex_git_sha(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            RunLogger(tmp_path / "x.jsonl", run_id="r", git_sha="NOT-A-SHA")
