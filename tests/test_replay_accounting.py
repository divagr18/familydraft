"""Replay-accounting tests for the spec-loop FLOP ledger (protocol v1.1 amendment).

The DagSpeculator rebuilds the continuation cache by replaying accepted + bonus
(m + 1 = emitted) tokens through the TARGET after every verification round;
chain loops forward only the bonus. The ledger must count that replay: omitting
it understates DAG FLOP spend and sizes the equal-FLOP dense baseline too small.
"""

from __future__ import annotations

from familydraft.eval.flops import FlopBudget, spec_loop_flops_per_emitted_token

TARGET = FlopBudget.from_config(1024, 28, 3072, 8, 16, 151936, "target")
TRUNK = FlopBudget.from_config(1024, 6, 3072, 8, 16, 151936, "trunk")


def test_chain_accounting_counts_single_bonus_decode() -> None:
    flops = spec_loop_flops_per_emitted_token(
        TARGET, TRUNK, draft_tokens_per_round=8, verify_nodes_per_round=8,
        emitted_tokens_per_round=2.0, replays_accepted_path=False,
    )
    expected = (8 * TRUNK.flops_per_token + 8 * TARGET.flops_per_token
                + 1 * TARGET.flops_per_token) / 2.0
    assert flops == expected


def test_replay_accounting_counts_emitted_target_forwards() -> None:
    flops = spec_loop_flops_per_emitted_token(
        TARGET, TRUNK, draft_tokens_per_round=8, verify_nodes_per_round=12,
        emitted_tokens_per_round=3.0, replays_accepted_path=True,
    )
    expected = (8 * TRUNK.flops_per_token + 12 * TARGET.flops_per_token
                + 3 * TARGET.flops_per_token) / 3.0
    assert flops == expected


def test_replay_accounting_is_strictly_higher() -> None:
    """The amendment only ever raises the DAG's counted cost (anti-thesis fix)."""
    for tpr in (1.2, 2.0, 3.5):
        chain = spec_loop_flops_per_emitted_token(
            TARGET, TRUNK, 8, 8, tpr, replays_accepted_path=False)
        replay = spec_loop_flops_per_emitted_token(
            TARGET, TRUNK, 8, 8, tpr, replays_accepted_path=True)
        delta = replay - chain
        assert delta > 0
        assert abs(delta - TARGET.flops_per_token * (tpr - 1) / tpr) < 1e-6


def test_flops_ledger_flags_replay_for_dag_systems() -> None:
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "run_baselines", repo / "scripts" / "run_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from familydraft.eval.flops import FlopBudget, spec_loop_flops_per_emitted_token

    target_budget = FlopBudget.from_config(
        mod.TARGET_HIDDEN, mod.TARGET_LAYERS, mod.TARGET_INTER,
        mod.TARGET_KV, mod.TARGET_HEADS, mod.VOCAB, "target")
    trunk_budget = FlopBudget.from_config(
        mod.TARGET_HIDDEN, mod.TRUNK_LAYERS, mod.TARGET_INTER,
        mod.TARGET_KV, mod.TARGET_HEADS, mod.VOCAB, "trunk")
    old_ledger = spec_loop_flops_per_emitted_token(
        target_budget, trunk_budget, 12, 12, 3.0, replays_accepted_path=False)

    dag = mod._flops_ledger("full_proposal_moe", 8, 3.0, 12, 12)
    chain = mod._flops_ledger("small_dense_drafter", 8, 3.0, 8, None)
    assert dag["flops_per_emitted_token"] > old_ledger
    assert chain["flops_per_emitted_token"] < dag["flops_per_emitted_token"]
