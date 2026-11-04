"""I-STHD: shared-interest spatio-temporal routing for recommendation.

The encoder uses fixed-origin tangent aggregation. In exact arithmetic
Log_o(Exp_o(F(x))) = F(x),
so its tangent output is equivalent to the corresponding Euclidean GCN. The
actual hyperbolic metric enters through the anchor-separation regularizer; the
router, reconstruction, and scoring operate in tangent coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry import clip_radius, exp_map_origin, hyperbolic_distance, log_map_origin


@dataclass
class ISTHDConfig:
    num_users: int
    num_items: int
    dim: int = 32
    num_anchors: int = 4
    num_factors: int = 4
    num_layers: int = 2
    context_window: int = 3
    routing_horizon: int = 3
    neighbor_cap: Optional[int] = 32
    score_dim: int = 32
    radius: float = 2.0
    separation_margin: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "num_users", "num_items", "dim", "num_factors", "context_window",
            "routing_horizon", "score_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.num_anchors, bool) or not isinstance(self.num_anchors, int) or self.num_anchors < 2:
            raise ValueError("num_anchors must be an integer >= 2")
        if isinstance(self.num_layers, bool) or not isinstance(self.num_layers, int) or self.num_layers < 0:
            raise ValueError("num_layers must be a nonnegative integer")
        if self.neighbor_cap is not None and (
            isinstance(self.neighbor_cap, bool)
            or not isinstance(self.neighbor_cap, int)
            or self.neighbor_cap <= 0
        ):
            raise ValueError("neighbor_cap must be a positive integer or None")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be finite and positive")
        if not math.isfinite(self.separation_margin) or self.separation_margin <= 0:
            raise ValueError("separation_margin must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


class ISTHD(nn.Module):
    """Shared-anchor historical routing and adaptive next-window item scoring.

    Snapshots contain raw ``[user_id, item_id]`` edges in individual windows.
    Windows and IDs are zero-based. Pass exactly ``snapshots[:cutoff + 1]``;
    rejecting a longer sequence prevents accidental access to future graphs.
    No graph states are persisted across calls, so source optimization never
    reuses representations produced by an earlier parameter version.

    The optional neighbor cap uses a deterministic, seeded uniform permutation
    for each user/window. This makes repeated evaluation reproducible without
    changing or consuming the caller's global random generator.
    """

    def __init__(self, config: ISTHDConfig) -> None:
        super().__init__()
        self.config = config
        d = config.dim
        self.node_embeddings = nn.Embedding(config.num_users + config.num_items, d)
        self.anchors = nn.Parameter(torch.empty(config.num_anchors, d))
        self.self_transforms = nn.ModuleList(
            nn.Linear(d, d, bias=False) for _ in range(config.num_layers)
        )
        self.neighbor_transforms = nn.ModuleList(
            nn.Linear(d, d, bias=False) for _ in range(config.num_layers)
        )
        self.base_network = nn.Linear(2 * d, d)
        self.key = nn.Linear(d, d, bias=False)
        self.user_compatibility = nn.Linear(d, d, bias=False)
        self.item_compatibility = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.time_coefficients = nn.Parameter(torch.empty(config.num_anchors, 2))
        self.profile_gate = nn.Linear(2 * d, d)
        self.factor_heads = nn.ModuleList(nn.Linear(d, d) for _ in range(config.num_factors))
        self.common_map = nn.Linear(d, config.score_dim, bias=False)
        self.residual_map = nn.Linear(d, config.score_dim, bias=False)
        self.score_gate = nn.Linear(2 * d + 1, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.node_embeddings.weight, std=0.05)
        nn.init.normal_(self.anchors, std=0.15)
        nn.init.zeros_(self.time_coefficients)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.project_anchors_()

    @property
    def device(self) -> torch.device:
        return self.node_embeddings.weight.device

    @torch.no_grad()
    def project_anchors_(self) -> "ISTHD":
        """Apply the draft's tangent-radius constraint after an optimizer step."""
        self.anchors.copy_(clip_radius(self.anchors, self.config.radius))
        return self

    def _validate_inputs(
        self, user_ids: Tensor, candidate_items: Tensor,
        snapshots: Sequence[Tensor], cutoff: int,
    ) -> None:
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 0:
            raise ValueError("cutoff must be a nonnegative zero-based window index")
        if len(snapshots) != cutoff + 1:
            raise ValueError(
                "pass only visible snapshots: len(snapshots) must equal cutoff + 1"
            )
        if user_ids.dtype != torch.long or user_ids.ndim != 1 or user_ids.numel() == 0:
            raise ValueError("user_ids must be a nonempty LongTensor of shape [B]")
        if (
            candidate_items.dtype != torch.long or candidate_items.ndim != 2
            or candidate_items.shape[0] != user_ids.shape[0]
            or candidate_items.shape[1] == 0
        ):
            raise ValueError("candidate_items must be a nonempty LongTensor of shape [B,C]")
        if bool(((user_ids < 0) | (user_ids >= self.config.num_users)).any()):
            raise ValueError("user ID outside configured user vocabulary")
        if bool(((candidate_items < 0) | (candidate_items >= self.config.num_items)).any()):
            raise ValueError("candidate item ID outside configured item vocabulary")
        for edges in snapshots:
            if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[1] != 2:
                raise ValueError("each snapshot must be a LongTensor with shape [E,2]")
            if edges.numel() and (
                bool(((edges[:, 0] < 0) | (edges[:, 0] >= self.config.num_users)).any())
                or bool(((edges[:, 1] < 0) | (edges[:, 1] >= self.config.num_items)).any())
            ):
                raise ValueError("snapshot contains an ID outside the configured vocabulary")

    def _encode_context(self, snapshots: Sequence[Tensor], tau: int) -> Tensor:
        """Encode the edge union in the W-window context ending at tau."""
        start = max(0, tau - self.config.context_window + 1)
        context = torch.cat(
            [snapshot.to(self.device) for snapshot in snapshots[start : tau + 1]], dim=0
        )
        # A repeated interaction is one edge in a context's unweighted union.
        if context.numel():
            context = torch.unique(context, dim=0)
            user_nodes = context[:, 0]
            item_nodes = context[:, 1] + self.config.num_users
            src = torch.cat((user_nodes, item_nodes))
            dst = torch.cat((item_nodes, user_nodes))
            degrees = torch.bincount(
                src, minlength=self.config.num_users + self.config.num_items
            ).to(self.node_embeddings.weight.dtype)
            weights = (degrees[src] * degrees[dst]).rsqrt()
        else:
            src = dst = torch.empty(0, device=self.device, dtype=torch.long)
            weights = self.node_embeddings.weight.new_empty(0)
        h = exp_map_origin(clip_radius(self.node_embeddings.weight, self.config.radius))
        for self_transform, neighbor_transform in zip(
            self.self_transforms, self.neighbor_transforms
        ):
            x = log_map_origin(h)
            aggregate = torch.zeros_like(x).index_add(
                0, dst, x[src] * weights.unsqueeze(-1)
            )
            h = exp_map_origin(torch.tanh(self_transform(x) + neighbor_transform(aggregate)))
        return log_map_origin(h)

    def _histories(
        self, snapshots: Sequence[Tensor], users: List[int], first_cutoff: int,
    ) -> Dict[Tuple[int, int], Dict[int, int]]:
        """Last direct-interaction windows, stored only for the queried users."""
        latest = {u: {} for u in users}  # type: Dict[int, Dict[int, int]]
        saved = {}  # type: Dict[Tuple[int, int], Dict[int, int]]
        for tau, snapshot in enumerate(snapshots):
            for u, i in snapshot.detach().cpu().tolist():
                if u in latest:
                    latest[u][i] = tau
            if tau >= first_cutoff:
                for u in users:
                    saved[(u, tau)] = latest[u].copy()
        return saved

    def _sample_neighbors(self, user: int, tau: int, history: Dict[int, int]) -> List[int]:
        identities = sorted(history)
        cap = self.config.neighbor_cap
        if cap is None or len(identities) <= cap:
            return identities
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            (self.config.seed + 1_000_003 * user + 9_176 * tau) % (2**63 - 1)
        )
        selected = torch.randperm(len(identities), generator=generator)[:cap].tolist()
        return [identities[index] for index in selected]

    def forward(
        self, user_ids: Tensor, candidate_items: Tensor,
        snapshots: Sequence[Tensor], cutoff: int,
    ) -> Dict[str, Tensor]:
        """Score candidates using only graph inputs visible through ``cutoff``.

        Required score outputs have shape [B,C]; mixtures have shape [B,M].
        Routing masses are zero for empty message sets and are excluded from
        consistency regularization. Diagnostics retain the forward graph so
        tests can inspect the exact seed-detachment boundary.
        """
        self._validate_inputs(user_ids, candidate_items, snapshots, cutoff)
        user_ids = user_ids.to(self.device)
        candidate_items = candidate_items.to(self.device)
        first_cutoff = max(0, cutoff - self.config.routing_horizon + 1)
        cutoffs = range(first_cutoff, cutoff + 1)
        states = {tau: self._encode_context(snapshots, tau) for tau in cutoffs}
        users = user_ids.detach().cpu().tolist()
        histories = self._histories(snapshots, sorted(set(users)), first_cutoff)
        d_scale = math.sqrt(self.config.dim)

        summaries = []
        for user in users:
            items = sorted(histories[(user, cutoff)])
            if items:
                indices = torch.tensor(items, device=self.device, dtype=torch.long)
                summaries.append(self.node_embeddings(indices + self.config.num_users).mean(0))
            else:
                summaries.append(self.anchors.new_zeros(self.config.dim))
        user_embeddings = self.node_embeddings(user_ids)
        base = torch.tanh(self.base_network(torch.cat((user_embeddings, torch.stack(summaries)), -1)))
        initial_logits = base @ self.anchors.T / d_scale
        initial_log_mixture = F.log_softmax(initial_logits, dim=-1)
        initial_mixture = initial_log_mixture.exp()
        detached_log_prior = initial_log_mixture.detach()

        routed, masses, counts = [], [], []
        for row, user in enumerate(users):
            message_states, message_items, ages = [], [], []
            for tau in cutoffs:
                history = histories[(user, tau)]
                items = self._sample_neighbors(user, tau, history)
                if not items:
                    continue
                indices = torch.tensor(items, device=self.device, dtype=torch.long)
                message_states.append(states[tau][indices + self.config.num_users])
                message_items.extend(items)
                ages.extend(
                    [[math.log1p(cutoff - tau), math.log1p(cutoff - history[i])] for i in items]
                )
            count = len(message_items)
            counts.append(count)
            if not count:
                routed.append(self.anchors.new_zeros(self.config.dim))
                masses.append(self.anchors.new_zeros(self.config.num_anchors))
                continue
            x = torch.cat(message_states, dim=0)
            indices = torch.tensor(message_items, device=self.device, dtype=torch.long)
            item_id_vectors = self.node_embeddings(indices + self.config.num_users)
            identity_match = (
                self.user_compatibility(user_embeddings[row]).unsqueeze(0)
                * self.item_compatibility(item_id_vectors)
            ).sum(-1) / d_scale
            time_features = self.anchors.new_tensor(ages)
            logits = (
                detached_log_prior[row].unsqueeze(0)
                + self.key(x) @ self.anchors.T / d_scale
                + identity_match.unsqueeze(-1)
                + time_features @ self.time_coefficients.T
            )
            # Joint Q x M normalization lets evidence revise the seed's masses.
            rho = F.softmax(logits.reshape(-1), dim=0).reshape_as(logits)
            masses.append(rho.sum(dim=0))
            routed.append((rho.sum(dim=-1, keepdim=True) * self.value(x)).sum(dim=0))

        routed_evidence = torch.stack(routed)
        routing_masses = torch.stack(masses)
        message_counts = torch.tensor(counts, device=self.device, dtype=torch.long)
        gate = torch.sigmoid(self.profile_gate(torch.cat((base, routed_evidence), -1)))
        updated = base + gate * routed_evidence
        final_mixture = F.softmax(updated @ self.anchors.T / d_scale, dim=-1)
        common = final_mixture @ self.anchors
        residual = updated - common

        candidate_nodes = candidate_items + self.config.num_users
        latest_items = states[cutoff][candidate_nodes]
        factors = torch.stack([torch.tanh(head(latest_items)) for head in self.factor_heads], dim=-2)
        factor_weights = F.softmax(
            (factors * common[:, None, None, :]).sum(-1) / d_scale, dim=-1
        )
        item_profile = (factor_weights.unsqueeze(-1) * factors).sum(dim=-2)
        score_scale = math.sqrt(self.config.score_dim)
        common_scores = (
            self.common_map(common).unsqueeze(1) * self.common_map(item_profile)
        ).sum(-1) / score_scale
        residual_scores = (
            self.residual_map(residual).unsqueeze(1) * self.residual_map(item_profile)
        ).sum(-1) / score_scale
        entropy = -(final_mixture * final_mixture.clamp_min(torch.finfo(final_mixture.dtype).tiny).log()).sum(-1, keepdim=True)
        gamma = torch.sigmoid(self.score_gate(torch.cat((updated, residual, entropy), -1))).squeeze(-1)
        scores = (1.0 - gamma[:, None]) * common_scores + gamma[:, None] * residual_scores
        seed_profile = initial_mixture @ self.anchors
        seed_scores = (
            seed_profile[:, None, :] * self.node_embeddings(candidate_nodes)
        ).sum(-1) / d_scale
        return {
            "scores": scores,
            "scores_common": common_scores,
            "scores_residual": residual_scores,
            "scores_seed": seed_scores,
            "initial_mixture": initial_mixture,
            "final_mixture": final_mixture,
            "routing_masses": routing_masses,
            "has_messages": message_counts > 0,
            "message_counts": message_counts,
            "gamma": gamma,
            "base_profile": base,
            "routed_evidence": routed_evidence,
            "updated_profile": updated,
            "common_component": common,
            "residual_component": residual,
            "profile_gate": gate,
            "initial_logits": initial_logits,
            "detached_log_prior": detached_log_prior,
            "item_profile": item_profile,
        }

    def regularization(self, output: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Return unweighted objective terms; ranking losses are caller-owned."""
        manifold_anchors = exp_map_origin(self.anchors)
        pair_indices = torch.triu_indices(
            self.config.num_anchors, self.config.num_anchors, offset=1, device=self.device
        )
        distance = hyperbolic_distance(
            manifold_anchors[pair_indices[0]], manifold_anchors[pair_indices[1]]
        )
        separation = F.relu(self.config.separation_margin - distance).square().mean()
        usage = 0.5 * (output["initial_mixture"] + output["final_mixture"]).mean(dim=0)
        tiny = torch.finfo(usage.dtype).tiny
        balance = (usage * (usage.clamp_min(tiny) * self.config.num_anchors).log()).sum()
        active = output["has_messages"]
        if bool(active.any()):
            p = output["routing_masses"][active]
            q = output["final_mixture"][active]
            midpoint = 0.5 * (p + q)
            log_midpoint = midpoint.clamp_min(tiny).log()
            consistency = 0.5 * (
                (p * (p.clamp_min(tiny).log() - log_midpoint)).sum(-1)
                + (q * (q.clamp_min(tiny).log() - log_midpoint)).sum(-1)
            ).mean()
        else:
            consistency = self.anchors.sum() * 0.0
        orthogonality = (self.common_map.weight.T @ self.residual_map.weight).square().sum()
        l2 = sum(parameter.square().sum() for parameter in self.parameters())
        return {
            "separation": separation,
            "balance": balance,
            "consistency": consistency,
            "orthogonality": orthogonality,
            "l2": l2,
        }
