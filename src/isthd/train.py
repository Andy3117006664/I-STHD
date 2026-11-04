"""Train I-STHD on source windows and evaluate frozen rolling forecasts."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from .data import load_csv, make_demo_data, sample_training_batch, training_queries
from .evaluate import evaluate_model
from .loss import LossWeights, compute_loss
from .model import ISTHD, ISTHDConfig


def _json_arguments(namespace: argparse.Namespace) -> Dict[str, Any]:
    return {key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(namespace).items()}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="synthetic execution check, not a benchmark")
    source.add_argument("--data", type=Path, help="CSV with user_id,item_id,timestamp")
    parser.add_argument("--config", type=Path, help="JSON model configuration, optionally under a model key")
    parser.add_argument("--window-seconds", type=float, default=86400.0)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--anchors", type=int)
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_negatives < 1:
        parser.error("epochs, batch-size, and num-negatives must be positive")
    if args.lr <= 0:
        parser.error("lr must be positive")
    if not args.ks or any(k <= 0 for k in args.ks):
        parser.error("ks must contain positive cutoffs")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    data = (make_demo_data(args.seed) if args.demo else load_csv(
        args.data, window_seconds=args.window_seconds,
        train_fraction=args.train_fraction, val_fraction=args.val_fraction))
    options: Dict[str, Any] = {"dim": 16, "score_dim": 16, "num_anchors": 3,
                               "num_factors": 3, "num_layers": 2}
    if args.config:
        config_text = json.loads(args.config.read_text(encoding="utf-8"))
        options.update(config_text.get("model", config_text))
    if args.dim is not None:
        options["dim"] = args.dim
        options["score_dim"] = args.dim
    if args.anchors is not None:
        options["num_anchors"] = args.anchors
    options.update(num_users=data.num_users, num_items=data.num_items, seed=args.seed)
    config = ISTHDConfig(**options)
    model = ISTHD(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = LossWeights()
    queries = training_queries(data)
    if not queries:
        raise ValueError("no eligible source queries; choose wider windows or more initial catalog coverage")
    if not any(data.forecast_queries(target)[0]
               for target in data.target_windows("validation")):
        raise ValueError(
            "validation has no eligible queries after vocabulary/history filtering; "
            "adjust window size or chronological split before selecting a checkpoint"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    # Every epoch visits each eligible user--origin query once. Only queries with
    # the same origin share a batch, since they share a visible graph prefix.
    history = []
    best_value, best_state, best_epoch = -float("inf"), None, 0
    best_validation: Dict[str, Any] = {}
    selection_key = f"ndcg@{min(args.ks)}"
    for epoch in range(1, args.epochs + 1):
        model.train()
        origins = list(queries)
        rng.shuffle(origins)
        accumulated: Dict[str, float] = {}
        query_count = 0
        for cutoff in origins:
            origin_queries = list(queries[cutoff])
            rng.shuffle(origin_queries)
            prefix = [snapshot.to(device) for snapshot in data.snapshots[:cutoff + 1]]
            for start in range(0, len(origin_queries), args.batch_size):
                batch = origin_queries[start:start + args.batch_size]
                users, items = sample_training_batch(batch, rng, args.num_negatives)
                optimizer.zero_grad(set_to_none=True)
                output = model(users.to(device), items.to(device), prefix, cutoff)
                loss, components = compute_loss(model, output, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite training loss at epoch {epoch}, origin {cutoff}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                model.project_anchors_()
                query_count += len(batch)
                for name, value in components.items():
                    if name == "total":
                        continue
                    accumulated[name] = accumulated.get(name, 0.0) + float(value.detach()) * len(batch)
                accumulated["total"] = accumulated.get("total", 0.0) + float(loss.detach()) * len(batch)
        validation = evaluate_model(model, data, split="validation", ks=args.ks, device=device)
        row = {"epoch": epoch, "train_queries": query_count,
               "train_loss": {key: value / query_count for key, value in accumulated.items()},
               "validation": validation}
        history.append(row)
        if validation[selection_key] > best_value:
            best_value = validation[selection_key]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch, best_validation = epoch, copy.deepcopy(validation)
        print(json.dumps({"epoch": epoch, "train_loss": row["train_loss"]["total"],
                          f"validation_{selection_key}": validation[selection_key]}, allow_nan=False))
    assert best_state is not None
    model.load_state_dict(best_state)
    # Test labels are consulted only after source-validation checkpoint choice.
    test = evaluate_model(model, data, split="test", ks=args.ks, device=device)
    checkpoint = {
        "format_version": 1,
        "model_config": dataclasses.asdict(config),
        "model_state": best_state,
        "optimizer_config": {"name": "Adam", "lr": args.lr, "gradient_clip_norm": 1.0},
        "loss_weights": dataclasses.asdict(weights),
        "training_args": _json_arguments(args),
        "data_metadata": data.metadata(),
        "best_epoch": best_epoch,
        "selection_metric": selection_key,
    }
    torch.save(checkpoint, args.output / "checkpoint.pt")
    report = {"kind": "synthetic_execution_check" if args.demo else "chronological_reference_run",
              "best_epoch": best_epoch, "selection_metric": selection_key,
              "data": data.metadata(), "epochs": history,
              "validation": best_validation, "test": test}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output / "checkpoint.pt"),
                      "metrics": str(args.output / "metrics.json"),
                      "test": {key: value for key, value in test.items() if key != "windows"}}, allow_nan=False))


if __name__ == "__main__":
    main()
