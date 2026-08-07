"""P3 correctness: KV-cache-aware GeneralDrafter must draft the SAME tokens as
the old full-re-encode drafter (and stay lossless inside the chain loop).

The old drafter re-encoded the whole context per draft token; the new one
resyncs a persistent KV cache via longest-prefix matching. Draft equality is
the correctness contract: any divergence would change acceptance.
"""

from pathlib import Path

import torch

from familydraft.draft.trunk import build_trunk_from_config
from familydraft.eval.draft_loop import GeneralDrafter, IntegratedSpeculator
from familydraft.experts.general import GeneralExpert
from familydraft.targets.wrapper import TargetModel


def _old_drafter(expert, spec_len, target_id, device):
    def draft_fn(context_ids):
        cur = torch.tensor([context_ids], device=device)
        draft = []
        with torch.inference_mode():
            for _ in range(spec_len):
                logits = expert(cur, target_id)[0, -1]
                tok = int(torch.argmax(logits, dim=-1))
                draft.append(tok)
                cur = torch.cat([cur, torch.tensor([[tok]], device=device)], dim=1)
        return draft

    return draft_fn


def test_draft_equality():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk = build_trunk_from_config(Path(".")).to(device)
    trunk.eval()
    expert = GeneralExpert(trunk).to(device)
    target_id = 0

    old = _old_drafter(expert, 4, target_id, device)
    new = GeneralDrafter(expert, 4, target_id, device)

    contexts = [
        [101, 245, 3012, 40, 556, 988],
        [101, 245, 3012, 40, 556, 988, 1024],
        [101, 245, 3012, 40, 556, 988, 1024, 77],
        [999, 888, 777],  # divergent context: cache must resync from scratch
        [101, 245, 3012, 40, 556, 988, 1024, 77, 33],
        [101, 245],  # context is a strict prefix of the cache: crop + re-draft
    ]

    for i, ctx in enumerate(contexts):
        a = old(ctx)
        b = new(ctx)
        assert a == b, f"context #{i} {ctx}: old={a} new={b}"


def test_chain_equivalence_lossless():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk = build_trunk_from_config(Path(".")).to(device)
    trunk.eval()
    expert = GeneralExpert(trunk).to(device)
    target = TargetModel.load("Qwen/Qwen3-0.6B", dtype="bf16")
    target_id = 0

    old_spec = IntegratedSpeculator(
        target, _old_drafter(expert, 4, target_id, device), 4, target_id
    )
    new_spec = IntegratedSpeculator(
        target, GeneralDrafter(expert, 4, target_id, device), 4, target_id
    )

    prompt = target.tokenizer(
        "def f(): return", return_tensors="pt", add_special_tokens=False
    )["input_ids"][0].tolist()

    res_old = old_spec.generate(prompt, 24)
    res_new = new_spec.generate(prompt, 24)

    assert res_old["tokens"] == res_new["tokens"], (
        f"tokens differ: old={res_old['tokens']} new={res_new['tokens']}"
    )
    assert res_old["tokens_per_round"] == res_new["tokens_per_round"], (
        f"tpr differ: old={res_old['tokens_per_round']} new={res_new['tokens_per_round']}"
    )

    vanilla = target.generate_greedy(
        torch.tensor([prompt], device=device), 24
    )[0, len(prompt):].tolist()
    assert res_new["tokens"] == vanilla, "chain not lossless vs vanilla greedy"


if __name__ == "__main__":
    test_draft_equality()
    print("draft equality: PASS")
    test_chain_equivalence_lossless()
    print("chain equivalence + lossless: PASS")
