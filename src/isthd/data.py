"""Chronological interaction data with an explicit prediction-time boundary.

IDs are fitted on source training windows only. A forecast candidate is a
source-vocabulary item already observed by that origin, excluding the user's
previously observed items. Future labels never define the candidate universe.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch


Record = Tuple[str, str, float]


@dataclass
class ForecastQuery:
    user: int
    candidates: List[int]
    positives: List[int]


@dataclass
class TemporalData:
    """Window edges use local, zero-based user and item indices."""

    snapshots: List[torch.Tensor]
    user_ids: List[str]
    item_ids: List[str]
    train_end: int
    val_end: int
    window_seconds: float
    start_timestamp: float
    raw_event_counts: List[int]
    unknown_id_event_counts: List[int]
    provenance: str = ""

    @property
    def num_users(self) -> int:
        return len(self.user_ids)

    @property
    def num_items(self) -> int:
        return len(self.item_ids)

    @property
    def num_windows(self) -> int:
        return len(self.snapshots)

    def target_windows(self, split: str) -> List[int]:
        if split == "train":
            return list(range(1, self.train_end + 1))
        if split in {"val", "validation"}:
            return list(range(self.train_end + 1, self.val_end + 1))
        if split == "test":
            return list(range(self.val_end + 1, self.num_windows))
        raise ValueError("split must be train, validation, or test")

    def metadata(self) -> Dict[str, Any]:
        return {
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "train_end": self.train_end,
            "val_end": self.val_end,
            "window_seconds": self.window_seconds,
            "start_timestamp": self.start_timestamp,
            "num_windows": self.num_windows,
            "provenance": self.provenance,
            "id_vocabulary": "source_training_only",
            "candidate_protocol": "observed_source_items_excluding_user_history",
        }

    def forecast_queries(self, target: int) -> Tuple[List[ForecastQuery], Dict[str, int]]:
        """Build the origin catalog first; only then intersect target labels.

        ``target`` is the window being predicted, so no edge in snapshots[target]
        enters a candidate set, user history, or graph input.
        """
        if not 1 <= target < self.num_windows:
            raise ValueError("a target window must have an earlier observed window")
        histories: Dict[int, Set[int]] = {}
        catalog: Set[int] = set()
        for snapshot in self.snapshots[:target]:
            for user, item in snapshot.tolist():
                catalog.add(item)
                histories.setdefault(user, set()).add(item)
        labels: Dict[int, Set[int]] = {}
        for user, item in self.snapshots[target].tolist():
            labels.setdefault(user, set()).add(item)
        statistics = {
            "raw_events": self.raw_event_counts[target],
            "excluded_unknown_source_id_events": self.unknown_id_event_counts[target],
            "excluded_not_yet_visible_pairs": 0,
            "excluded_repeat_pairs": 0,
            "eligible_positive_pairs": 0,
            "users_without_eligible_positive": 0,
        }
        queries: List[ForecastQuery] = []
        for user in sorted(labels):
            seen = histories.get(user, set())
            # This expression deliberately has no dependence on target labels.
            candidates = sorted(catalog.difference(seen))
            candidate_set = set(candidates)
            positives = sorted(labels[user].intersection(candidate_set))
            statistics["excluded_not_yet_visible_pairs"] += len(labels[user] - catalog)
            statistics["excluded_repeat_pairs"] += len(labels[user].intersection(seen))
            if positives:
                queries.append(ForecastQuery(user, candidates, positives))
                statistics["eligible_positive_pairs"] += len(positives)
            else:
                statistics["users_without_eligible_positive"] += 1
        return queries, statistics


def parse_timestamp(value: str) -> float:
    """Accept Unix seconds or ISO-8601 timestamps; naive dates mean UTC."""
    try:
        result = float(value)
    except ValueError:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        result = parsed.timestamp()
    if not math.isfinite(result):
        raise ValueError("timestamp must be finite")
    return result


def read_interaction_csv(path: str | Path) -> List[Record]:
    """Read a headered CSV with user_id,item_id,timestamp columns."""
    records: List[Record] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "item_id", "timestamp"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain user_id,item_id,timestamp columns")
        for line_number, row in enumerate(reader, start=2):
            user, item = row["user_id"].strip(), row["item_id"].strip()
            if not user or not item:
                raise ValueError(f"empty user or item ID on CSV line {line_number}")
            try:
                timestamp = parse_timestamp(row["timestamp"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid timestamp on CSV line {line_number}") from error
            records.append((user, item, timestamp))
    if not records:
        raise ValueError("interaction CSV is empty")
    return records


def temporal_data_from_records(
    records: Iterable[Record],
    *,
    window_seconds: float = 86400.0,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    fixed_metadata: Optional[Dict[str, Any]] = None,
    provenance: str = "",
) -> TemporalData:
    """Fit source IDs or reuse an existing checkpoint's exact vocabulary.

    Fractions partition *forecast target windows*: window zero supplies initial
    history, then the first train_fraction targets are source supervision.
    Empty intermediate windows are retained.
    """
    rows = sorted(records, key=lambda row: (row[2], row[0], row[1]))
    if not rows:
        raise ValueError("at least one interaction is required")
    if any(not math.isfinite(float(row[2])) for row in rows):
        raise ValueError("all timestamps must be finite")
    if fixed_metadata is not None:
        window_seconds = float(fixed_metadata["window_seconds"])
        start = float(fixed_metadata["start_timestamp"])
    else:
        start = float(rows[0][2])
    if window_seconds <= 0 or not math.isfinite(window_seconds):
        raise ValueError("window_seconds must be positive and finite")
    if rows[0][2] < start:
        raise ValueError("evaluation data starts before the checkpoint's time origin")
    assignments = [int(math.floor((float(row[2]) - start) / window_seconds)) for row in rows]
    num_windows = max(assignments) + 1
    if fixed_metadata is None:
        if not (0 < train_fraction < 1 and 0 < val_fraction < 1
                and train_fraction + val_fraction < 1):
            raise ValueError("train/validation fractions must be positive and sum to less than one")
        if num_windows < 4:
            raise ValueError("need at least four windows: history, train, validation, test")
        n_targets = num_windows - 1
        train_end = max(1, min(n_targets - 2, int(n_targets * train_fraction)))
        val_count = max(1, int(n_targets * val_fraction))
        val_end = min(n_targets - 1, train_end + val_count)
        source = [row for row, window in zip(rows, assignments) if window <= train_end]
        # Ordered first appearance is reproducible and contains no target-only ID.
        user_ids = list(dict.fromkeys(row[0] for row in source))
        item_ids = list(dict.fromkeys(row[1] for row in source))
    else:
        train_end, val_end = int(fixed_metadata["train_end"]), int(fixed_metadata["val_end"])
        user_ids, item_ids = list(fixed_metadata["user_ids"]), list(fixed_metadata["item_ids"])
    if not user_ids or not item_ids:
        raise ValueError("source training vocabulary is empty")
    user_map = {key: index for index, key in enumerate(user_ids)}
    item_map = {key: index for index, key in enumerate(item_ids)}
    edge_sets: List[Set[Tuple[int, int]]] = [set() for _ in range(num_windows)]
    raw_counts, unknown_counts = [0] * num_windows, [0] * num_windows
    for (user, item, _), window in zip(rows, assignments):
        raw_counts[window] += 1
        if user not in user_map or item not in item_map:
            unknown_counts[window] += 1
            continue
        edge_sets[window].add((user_map[user], item_map[item]))
    snapshots = [torch.tensor(sorted(edges), dtype=torch.long).reshape(-1, 2) for edges in edge_sets]
    return TemporalData(snapshots, user_ids, item_ids, train_end, val_end,
                        float(window_seconds), start, raw_counts, unknown_counts, provenance)


def load_csv(
    path: str | Path,
    *,
    window_seconds: float = 86400.0,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    fixed_metadata: Optional[Dict[str, Any]] = None,
) -> TemporalData:
    return temporal_data_from_records(
        read_interaction_csv(path), window_seconds=window_seconds,
        train_fraction=train_fraction, val_fraction=val_fraction,
        fixed_metadata=fixed_metadata, provenance=str(Path(path).resolve()),
    )


def make_demo_data(seed: int = 42, *, fixed_metadata: Optional[Dict[str, Any]] = None) -> TemporalData:
    """Small synthetic data for execution checks, not an OOD benchmark."""
    item_permutation = list(range(36))
    random.Random(seed).shuffle(item_permutation)
    records: List[Record] = []
    for window in range(10):
        for user in range(12):
            # A few inactive users demonstrate graph changes without a new edge.
            if window == 3 and user % 4 == 0:
                continue
            for offset in range(3 if window == 0 else 2):
                item = item_permutation[(3 * user + 3 * window + offset) % 36]
                records.append((str(user), str(item), float(window * 86400 + user * 10 + offset)))
    return temporal_data_from_records(records, window_seconds=86400.0,
                                      fixed_metadata=fixed_metadata,
                                      provenance=f"synthetic-demo:seed={seed}")


def training_queries(data: TemporalData) -> Dict[int, List[ForecastQuery]]:
    """Queries indexed by forecast origin, each with a positive and negative."""
    result: Dict[int, List[ForecastQuery]] = {}
    for target in data.target_windows("train"):
        queries, _ = data.forecast_queries(target)
        eligible = [query for query in queries if len(query.candidates) > len(query.positives)]
        if eligible:
            result[target - 1] = eligible
    return result


def sample_training_batch(
    queries: Sequence[ForecastQuery], rng: random.Random, num_negatives: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample only the next source window's labels; no later exclusions.

    Column zero is one uniformly selected positive. Negative IDs are sampled
    without replacement when possible, and with replacement for smaller pools.
    """
    if num_negatives < 1:
        raise ValueError("num_negatives must be at least one")
    users, candidates = [], []
    for query in queries:
        positive_set = set(query.positives)
        pool = [item for item in query.candidates if item not in positive_set]
        if not pool:
            raise ValueError("training query has no eligible negative")
        negatives = (rng.sample(pool, num_negatives) if len(pool) >= num_negatives
                     else rng.choices(pool, k=num_negatives))
        users.append(query.user)
        candidates.append([rng.choice(query.positives)] + negatives)
    return torch.tensor(users, dtype=torch.long), torch.tensor(candidates, dtype=torch.long)
