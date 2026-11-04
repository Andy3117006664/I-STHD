"""Frozen-parameter rolling, full-visible-catalog recommendation evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch

from .data import TemporalData, load_csv, make_demo_data
from .model import ISTHD, ISTHDConfig


@torch.no_grad()
def evaluate_model(
    model: ISTHD,
    data: TemporalData,
    *,
    split: str = "test",
    targets: Optional[Iterable[int]] = None,
    ks: Sequence[int] = (10, 20),
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Score every eligible visible item; update inputs, never parameters.

    Metrics average eligible user--origin queries. Novel IDs, items not yet
    observed at the prediction origin, and repeat interactions are reported as
    exclusions rather than silently counted as failed warm-start predictions.
    Graphs passed to the model stop at target-1. Target labels are used solely
    to compute metrics after the candidate universe has been constructed.
    """
    ks = tuple(sorted(set(int(k) for k in ks)))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("evaluation cutoffs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    metric_names = [f"{name}@{k}" for k in ks for name in ("recall", "ndcg")]
    sums = {name: 0.0 for name in metric_names}
    totals: Dict[str, int] = {
        "num_queries": 0,
        "raw_events": 0,
        "eligible_positive_pairs": 0,
        "excluded_unknown_source_id_events": 0,
        "excluded_not_yet_visible_pairs": 0,
        "excluded_repeat_pairs": 0,
        "users_without_eligible_positive": 0,
    }
    windows = []
    target_list = list(data.target_windows(split) if targets is None else targets)
    try:
        for target in target_list:
            queries, counts = data.forecast_queries(target)
            window_sums = {name: 0.0 for name in metric_names}
            prefix = [snapshot.to(device) for snapshot in data.snapshots[:target]]
            for start in range(0, len(queries), batch_size):
                batch = queries[start:start + batch_size]
                width = max(len(query.candidates) for query in batch)
                item_rows = [query.candidates + [query.candidates[0]] * (width - len(query.candidates))
                             for query in batch]
                user_tensor = torch.tensor([query.user for query in batch], dtype=torch.long, device=device)
                item_tensor = torch.tensor(item_rows, dtype=torch.long, device=device)
                output = model(user_tensor, item_tensor, prefix, target - 1)
                scores = output["scores"]
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("nonfinite recommendation scores during evaluation")
                lengths = torch.tensor([len(query.candidates) for query in batch], device=device)
                valid = torch.arange(width, device=device)[None, :] < lengths[:, None]
                scores = scores.masked_fill(~valid, -float("inf"))
                # Candidate lists are item-ID sorted; stable sorting gives a
                # deterministic ID tie-break rather than depending on topk ties.
                order = torch.argsort(scores, dim=-1, descending=True, stable=True).cpu().tolist()
                for query, indices in zip(batch, order):
                    positive_set = set(query.positives)
                    ranking = [query.candidates[index]
                               for index in indices[:min(max(ks), len(query.candidates))]]
                    for k in ks:
                        hits = [int(item in positive_set) for item in ranking[:k]]
                        recall = sum(hits) / len(positive_set)
                        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
                        idcg = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(positive_set))))
                        window_sums[f"recall@{k}"] += recall
                        window_sums[f"ndcg@{k}"] += dcg / idcg
            number = len(queries)
            totals["num_queries"] += number
            for name, value in counts.items():
                totals[name] += value
            for name in metric_names:
                sums[name] += window_sums[name]
            windows.append({
                "target_window": target,
                "num_queries": number,
                **counts,
                **{name: value / max(1, number) for name, value in window_sums.items()},
            })
    finally:
        model.train(was_training)
    return {
        "split": split,
        "parameter_updates": 0,
        "candidate_protocol": "full_visible_source_catalog_excluding_user_history",
        **totals,
        **{name: value / max(1, totals["num_queries"]) for name, value in sums.items()},
        "windows": windows,
    }


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    """Checkpoints contain tensors and primitive metadata only."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("checkpoint lacks model_config or model_state")
    return checkpoint


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", type=Path, help="CSV with user_id,item_id,timestamp")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    metadata = checkpoint["data_metadata"]
    training_args = checkpoint.get("training_args", {})
    demo = args.demo or (args.data is None and str(metadata.get("provenance", "")).startswith("synthetic-demo:"))
    if demo:
        seed = int(training_args.get("seed", 42))
        data = make_demo_data(seed, fixed_metadata=metadata)
    else:
        source = args.data or training_args.get("data")
        if source is None:
            parser.error("provide --data for this checkpoint")
        data = load_csv(source, fixed_metadata=metadata)
    model = ISTHD(ISTHDConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    report = evaluate_model(model, data, split=args.split, ks=args.ks,
                            batch_size=args.batch_size, device=device)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
