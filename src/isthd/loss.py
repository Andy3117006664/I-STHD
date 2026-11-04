"""Ranking supervision and shared-coordinate regularization."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

if TYPE_CHECKING:
    from .model import ISTHD


@dataclass(frozen=True)
class LossWeights:
    common: float = 0.1
    seed: float = 0.1
    separation: float = 0.01
    balance: float = 0.01
    consistency: float = 0.1
    orthogonality: float = 0.001
    l2: float = 1e-6

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")


def pairwise_ranking_loss(scores: Tensor) -> Tensor:
    """Column zero is a positive; every remaining column is a sampled negative."""
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("ranking scores must have shape [batch, 1 + negatives]")
    return F.softplus(scores[:, 1:] - scores[:, :1]).mean()


def compute_loss(
    model: ISTHD, output: dict[str, Tensor], weights: LossWeights | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Optimize exactly the serving score, plus common/seed auxiliary objectives.

    The seed loss supplies supervision to the query whose router connection is
    detached. Neither auxiliary head replaces the deployed gated serving score.
    """
    weights = LossWeights() if weights is None else weights
    components = {
        "ranking": pairwise_ranking_loss(output["scores"]),
        "common": pairwise_ranking_loss(output["scores_common"]),
        "seed": pairwise_ranking_loss(output["scores_seed"]),
        **model.regularization(output),
    }
    total = components["ranking"]
    for field in fields(weights):
        total = total + getattr(weights, field.name) * components[field.name]
    components["total"] = total
    return total, components

