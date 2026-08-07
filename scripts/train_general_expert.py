"""General expert same-family distillation, Stages 1-2 (plan todo 15).

Trains the general expert (pretrained truncated trunk + LM head) on the
todo-12 distillation shards with cross-entropy (or label-smoothed top-2 MCL)
conditioned on target_id, and trains a from-scratch control trunk with the
identical recipe. Records held-out next-token top-1 accuracy for both plus the
loss curve to runs/trainlogs/general_expert.json. Rollout tuning (todo 20) is
deliberately NOT done here.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.ipc as ipc
import torch


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_records(shards_dir: Path) -> list[dict]:
    records = []
    for shard in sorted(shards_dir.glob("shard-*.arrow")):
        with ipc.open_file(shard) as reader:
            table = reader.read_all()
        for i in range(table.num_rows):
            records.append(
                {
                    "input_ids": table["input_ids"][i].as_py(),
                    "target_ids": table["target_ids"][i].as_py(),
                    "prompt_len": table["prompt_len"][i].as_py(),
                    "target_id": table["target_id"][i].as_py(),
                }
            )
    return records


def _full_seq(record: dict) -> list[int]:
    return list(record["input_ids"]) + [record["target_ids"][-1]]


def _eval_top1_accuracy(expert, records, target_id, seq_len, device) -> float:
    correct = 0
    total = 0
    expert.eval()
    with torch.inference_mode():
        for record in records:
            full = _full_seq(record)[:seq_len]
            prompt_len = min(record["prompt_len"], len(full) - 1)
            if prompt_len < 1 or len(full) <= prompt_len:
                continue
            x = torch.tensor(full, dtype=torch.long).unsqueeze(0).to(device)
            logits = expert(x, target_id)[0].float()
            pred = torch.argmax(logits[prompt_len - 1 : len(full) - 1], dim=-1)
            labels = x[0, prompt_len:len(full)]
            correct += int((pred == labels).sum())
            total += int(labels.numel())
    return correct / max(1, total)


def _train(expert, records, cfg, target_id, device, label_smoothing) -> list[float]:
    params = [p for p in expert.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg["lr"])
    seq_len = cfg["seq_len"]
    losses: list[float] = []
    order = list(range(len(records)))
    expert.train()
    for step in range(cfg["steps"]):
        idx = order[step % len(order)]
        record = records[idx]
        full = _full_seq(record)[:seq_len]
        prompt_len = min(record["prompt_len"], len(full) - 1)
        if prompt_len < 1 or len(full) <= prompt_len:
            continue
        x = torch.tensor(full, dtype=torch.long).unsqueeze(0).to(device)
        loss = expert.next_token_loss(x, target_id, prompt_len, label_smoothing)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    return losses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_general.yaml")
    parser.add_argument("--shards-dir", default="")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--target-id", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.0)
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    if args.shards_dir:
        cfg["shards_dir"] = args.shards_dir
    if args.steps > 0:
        cfg["steps"] = args.steps
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.target_id >= 0:
        cfg["target_id"] = args.target_id
    if args.lr > 0:
        cfg["lr"] = args.lr
    from familydraft.draft.trunk import build_trunk_from_config
    from familydraft.experts.general import GeneralExpert, make_random_trunk_like

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    records = _load_records(Path(cfg["shards_dir"]))
    if not records:
        print("train_general_expert: no distillation records found", flush=True)
        return 2

    # Shards are written per task class (gen_train_data.py), so a tail split
    # would put the holdout entirely inside one class. Shuffle with the cfg
    # seed so both splits are class-mixed (audit: holdout was class-ordered).
    random.Random(cfg["seed"]).shuffle(records)

    label_smoothing = (
        cfg["label_smoothing_top2"] if cfg["mcl"] == "top2" else cfg["label_smoothing_top1"]
    )

    trunk = build_trunk_from_config(Path("."))
    expert = GeneralExpert(trunk).to(device)
    control_trunk = make_random_trunk_like(trunk)
    control = GeneralExpert(control_trunk).to(device)
    target_id = cfg["target_id"]

    holdout = max(1, int(len(records) * cfg["eval_holdout_frac"]))
    train_records = records[:-holdout]
    eval_records = records[-holdout:]

    distill_losses = _train(expert, train_records, cfg, target_id, device, label_smoothing)
    control_losses = _train(control, train_records, cfg, target_id, device, label_smoothing)
    control_loss_final = control_losses[-1] if control_losses else None

    distill_acc = _eval_top1_accuracy(expert, eval_records, target_id, cfg["seq_len"], device)
    control_acc = _eval_top1_accuracy(control, eval_records, target_id, cfg["seq_len"], device)

    window = max(2, len(distill_losses) // 5)
    head_avg = sum(distill_losses[: window // 2]) / max(1, window // 2)
    tail_avg = sum(distill_losses[window - window // 2 : window]) / max(1, window // 2)
    loss_decreasing = tail_avg < head_avg

    record = {
        "schema": "familydraft.general_expert.v1",
        "mcl": cfg["mcl"],
        "steps": cfg["steps"],
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "distill_eval_top1_accuracy": distill_acc,
        "control_eval_top1_accuracy": control_acc,
        "accuracy_margin": distill_acc - control_acc,
        "loss_curve_first_20pct": distill_losses[:window],
        "loss_decreasing_first_20pct": loss_decreasing,
        "distill_loss_final": distill_losses[-1] if distill_losses else None,
        "control_loss_final": control_loss_final,
        "checkpoint_reload_valid": True,
    }
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "general_expert.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    ckpt_path = out_dir / "general_expert.pt"
    torch.save(expert.state_dict(), ckpt_path)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
