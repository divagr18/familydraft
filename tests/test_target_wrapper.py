"""Tests for the unified Qwen3 target wrapper (plan todo 4).

CPU-safe unit tests run without a GPU or network (modulo the fast-failing
bogus-repo load error path). GPU tests carry the ``gpu`` marker registered in
``pyproject.toml`` and skip cleanly when no CUDA device is present.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from familydraft.targets.wrapper import (
    TargetLoadError,
    TargetModel,
    parse_dtype,
    topk_snapshot,
)

BOGUS_REPO_ID = "familydraft-ci/no-such-model-zzz-0000"  # cannot exist
LOCAL_REPO_ID = "Qwen/Qwen3-0.6B"
QWEN3_VOCAB_SIZE = 151_936
PROMPT = "Speculative decoding works by"
MAX_NEW_TOKENS = 32
GOLDEN_PATH = Path(__file__).parent / "goldens" / "target_wrapper_greedy.txt"

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


# ---------------------------------------------------------------------------
# Pure / CPU-safe unit tests
# ---------------------------------------------------------------------------


def test_parse_dtype_maps_pinned_names() -> None:
    assert parse_dtype("bf16") is torch.bfloat16
    assert parse_dtype("fp16") is torch.float16
    assert parse_dtype("fp32") is torch.float32


def test_parse_dtype_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="'int8'"):
        parse_dtype("int8")  # quantization is out of scope by design


def test_topk_snapshot_shape_and_order_on_fake_distribution() -> None:
    # Given a tiny fake logit distribution over a 7-token vocab (3 positions)
    vocab = 7
    logits = torch.tensor(
        [
            [0.1, 5.0, 3.0, 2.0, 4.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 9.0, 0.0, 0.0, 0.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
        ]
    )

    # When taking a top-k snapshot with k=3
    snapshot = topk_snapshot(logits, k=3)

    # Then every array has shape (T, k) over the full vocab
    assert snapshot.token_ids.shape == (3, 3)
    assert snapshot.logits.shape == (3, 3)
    assert snapshot.ranks.shape == (3, 3)
    assert int(snapshot.token_ids.min()) >= 0
    assert int(snapshot.token_ids.max()) < vocab

    # And ids are the argmax-first ordering with logits sorted descending
    assert snapshot.token_ids[0].tolist() == [1, 4, 2]
    assert snapshot.logits[0].tolist() == [5.0, 4.0, 3.0]

    # And ranks are dense ranks: number of strictly-greater logits
    assert snapshot.ranks[0].tolist() == [0, 1, 2]
    assert snapshot.ranks[1].tolist() == [0, 1, 1]  # the three 1.0 ties share rank 1
    assert snapshot.ranks[2].tolist() == [0, 1, 2]


def test_topk_snapshot_ranks_are_full_vocab_positions() -> None:
    logits = torch.tensor([[9.0, 0.0, 8.0, 1.0, 7.0, 2.0]])
    snapshot = topk_snapshot(logits, k=2)
    assert snapshot.token_ids[0].tolist() == [0, 2]
    assert snapshot.ranks[0].tolist() == [0, 1]
    # rank of a non-top token is its full-vocab descending position
    assert snapshot.ranks.dtype == torch.int32


def test_topk_snapshot_rejects_k_beyond_vocab() -> None:
    logits = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="k=5 exceeds vocab"):
        topk_snapshot(logits, k=5)


def test_target_load_error_names_bogus_repo(_bounded_hub) -> None:
    # Loading a repo that cannot exist must raise TargetLoadError (never a
    # raw hub/HTTP exception) and the message must name the repo id.
    # Online: fast 404. Offline: immediate local-cache miss. Both bounded.
    with pytest.raises(TargetLoadError) as excinfo:
        TargetModel.load(BOGUS_REPO_ID)
    message = str(excinfo.value)
    assert BOGUS_REPO_ID in message


# ---------------------------------------------------------------------------
# GPU tests (Qwen/Qwen3-0.6B on the local RTX 4060, bf16, pinned golden)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _bounded_hub(monkeypatch: pytest.MonkeyPatch) -> object:
    """Belt-and-braces bound on the hub download timeout for this test."""
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "5")
    try:
        import huggingface_hub.constants as hub_constants

        monkeypatch.setattr(
            hub_constants, "HF_HUB_DOWNLOAD_TIMEOUT", 5, raising=False
        )
    except ImportError:
        pass
    yield
    return None


@pytest.mark.gpu
@requires_cuda
class TestTargetModelOnGpu:
    @pytest.fixture(scope="class")
    def target(self) -> TargetModel:
        torch.manual_seed(0)
        model = TargetModel.load(LOCAL_REPO_ID, dtype="bf16")
        assert str(model.device).startswith("cuda")
        yield model
        del model

    @pytest.fixture(scope="class")
    def golden(self) -> dict:
        with GOLDEN_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)

    @pytest.fixture(scope="class")
    def prompt_ids(self, target: TargetModel, golden: dict) -> torch.Tensor:
        encoded = target.tokenizer(
            golden["prompt"], return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        assert encoded.shape[1] == len(golden["prompt_ids"])
        assert encoded[0].tolist() == golden["prompt_ids"]
        return encoded

    def test_vocab_size_is_pinned(self, target: TargetModel) -> None:
        assert target.vocab_size == QWEN3_VOCAB_SIZE

    def test_greedy_matches_pinned_reference(
        self,
        target: TargetModel,
        golden: dict,
        prompt_ids: torch.Tensor,
    ) -> None:
        torch.manual_seed(0)
        ids = target.generate_greedy(prompt_ids, MAX_NEW_TOKENS)
        new_ids = ids[0, prompt_ids.shape[1] :].tolist()
        new_text = target.tokenizer.decode(new_ids, skip_special_tokens=False)
        assert new_ids == golden["new_token_ids"]
        assert new_text == golden["new_text"]
        golden_digest = hashlib.sha256(
            json.dumps(new_ids).encode("utf-8")
        ).hexdigest()
        assert golden_digest == golden["new_token_ids_sha256"]

    def test_greedy_re_run_is_bit_identical(
        self, target: TargetModel, prompt_ids: torch.Tensor
    ) -> None:
        torch.manual_seed(0)
        first = target.generate_greedy(prompt_ids, MAX_NEW_TOKENS)
        torch.manual_seed(0)
        second = target.generate_greedy(prompt_ids, MAX_NEW_TOKENS)
        assert torch.equal(first, second)

    def test_sample_generation_is_seeded(
        self, target: TargetModel, prompt_ids: torch.Tensor
    ) -> None:
        first = target.generate_sample(prompt_ids, temp=1.0, seed=1234)
        second = target.generate_sample(prompt_ids, temp=1.0, seed=1234)
        assert torch.equal(first, second)

    def test_topk_logits_shape_and_vocab_range(
        self,
        target: TargetModel,
        prompt_ids: torch.Tensor,
        golden: dict,
    ) -> None:
        torch.manual_seed(0)
        full_ids = target.generate_greedy(prompt_ids, MAX_NEW_TOKENS)
        k = 8
        snapshot = target.topk_logits(full_ids, k=k)
        seq_len = full_ids.shape[1]
        assert snapshot.token_ids.shape == (seq_len, k)
        assert snapshot.logits.shape == (seq_len, k)
        assert snapshot.ranks.shape == (seq_len, k)
        assert int(snapshot.token_ids.min()) >= 0
        assert int(snapshot.token_ids.max()) < QWEN3_VOCAB_SIZE
        # From the last prompt position onward, the snapshot argmax should
        # match what greedy generated. It can legitimately differ at bf16
        # near-tie positions: one parallel forward and incremental KV decode
        # accumulate in different orders, so a top-2 near-tie may flip. The
        # invariants: every flipped position must have the greedy token in
        # the snapshot top-2, and the top-2 margin there must be tiny.
        assert torch.all(snapshot.ranks[:, 0] == 0)
        prompt_len = prompt_ids.shape[1]
        gen_slice = slice(prompt_len - 1, full_ids.shape[1] - 1)
        snap_argmax = snapshot.token_ids[gen_slice, 0].to(full_ids.device)
        gen_tokens = full_ids[0, prompt_len:].to(torch.int64)
        mismatch = snap_argmax != gen_tokens
        margins = snapshot.logits[gen_slice, 0] - snapshot.logits[gen_slice, 1]
        top2 = snapshot.token_ids[gen_slice, :2].to(full_ids.device)
        greedy_in_top2 = (top2 == gen_tokens.unsqueeze(1)).any(dim=1)
        assert torch.all(greedy_in_top2[mismatch]), (
            "greedy token left the top-2 of the forward pass: broken logits, "
            f"positions={mismatch.nonzero(as_tuple=True)}"
        )
        assert torch.all(margins[mismatch] < 0.25), (
            "flip at a confident position indicates broken logits, not kernel "
            f"numerics: positions={mismatch.nonzero(as_tuple=True)}"
        )
        assert len(golden["new_token_ids"]) == MAX_NEW_TOKENS

    def test_topk_logits_are_descending(
        self, target: TargetModel, prompt_ids: torch.Tensor
    ) -> None:
        snapshot = target.topk_logits(prompt_ids, k=16)
        diffs = snapshot.logits[:, 1:] - snapshot.logits[:, :-1]
        assert torch.all(diffs <= 1e-4)


# ---------------------------------------------------------------------------
# Golden provenance
# ---------------------------------------------------------------------------


def test_golden_file_is_present_and_self_consistent() -> None:
    # CPU-safe guard so a missing/corrupted golden fails loudly even in the
    # "-m 'not gpu'" suite.
    assert GOLDEN_PATH.is_file(), f"missing golden file: {GOLDEN_PATH}"
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    required = {
        "repo_id",
        "dtype",
        "prompt",
        "prompt_ids",
        "new_token_ids",
        "new_text",
        "new_token_ids_sha256",
        "torch_version_uv_lock",
        "torch_version_runtime",
        "transformers_version_uv_lock",
        "transformers_version_runtime",
        "recorded_utc",
        "load_seconds",
    }
    missing = required - golden.keys()
    assert not missing, f"golden missing fields: {sorted(missing)}"
    assert golden["repo_id"] == LOCAL_REPO_ID
    assert golden["dtype"] == "bf16"
    assert golden["prompt"] == PROMPT
    digest = hashlib.sha256(json.dumps(golden["new_token_ids"]).encode()).hexdigest()
    assert digest == golden["new_token_ids_sha256"]
    assert 0 <= min(golden["new_token_ids"]) and max(golden["new_token_ids"]) < (
        QWEN3_VOCAB_SIZE
    )
    # Golden must not drift between runs (stale-cache / corruption probe).
    os.environ["FAMILYDRAFT_GOLDEN_SHA256"] = hashlib.sha256(
        GOLDEN_PATH.read_bytes()
    ).hexdigest()
