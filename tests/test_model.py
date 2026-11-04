import math

import pytest
import torch

from isthd import ISTHD, ISTHDConfig
from isthd.loss import LossWeights, compute_loss


@pytest.fixture
def example():
    torch.manual_seed(17)
    model = ISTHD(ISTHDConfig(
        num_users=4, num_items=8, dim=8, score_dim=6,
        num_anchors=3, num_factors=2, neighbor_cap=None,
    ))
    snapshots = [
        torch.tensor([[0, 0], [0, 3], [1, 1], [2, 2]]),
        torch.tensor([[1, 0], [1, 4], [2, 3]]),
    ]
    users = torch.tensor([0, 1, 3])
    candidates = torch.tensor([[1, 2, 4], [2, 3, 5], [0, 2, 6]])
    return model, snapshots, users, candidates


def test_reconstruction_and_serving_score(example):
    model, snapshots, users, candidates = example
    out = model(users, candidates, snapshots, 1)
    assert out["scores"].shape == (3, 3)
    torch.testing.assert_close(
        out["common_component"] + out["residual_component"], out["updated_profile"])
    torch.testing.assert_close(out["scores"],
        (1 - out["gamma"][:, None]) * out["scores_common"]
        + out["gamma"][:, None] * out["scores_residual"])
    for key in ("initial_mixture", "final_mixture"):
        torch.testing.assert_close(out[key].sum(-1), torch.ones(3))
    torch.testing.assert_close(out["routing_masses"].sum(-1), torch.tensor([1., 1., 0.]))


def test_inactive_user_keeps_history_but_receives_new_context(example):
    model, snapshots, _, candidates = example
    user = torch.tensor([0])
    before = model(user, candidates[:1], snapshots[:1], 0)
    after = model(user, candidates[:1], snapshots, 1)
    torch.testing.assert_close(before["base_profile"], after["base_profile"])
    torch.testing.assert_close(before["initial_mixture"], after["initial_mixture"])
    assert after["message_counts"].item() == 4
    assert not torch.allclose(before["routed_evidence"], after["routed_evidence"])


def test_empty_history_falls_back_to_base_profile(example):
    model, snapshots, users, candidates = example
    out = model(users[-1:], candidates[-1:], snapshots, 1)
    assert not out["has_messages"].item()
    assert out["message_counts"].item() == 0
    torch.testing.assert_close(out["updated_profile"], out["base_profile"])
    assert model.regularization(out)["consistency"].item() == 0
    assert torch.isfinite(out["scores"]).all()


def test_only_query_router_connection_is_detached(example):
    model, snapshots, users, candidates = example
    out = model(users, candidates, snapshots, 1)
    main_to_seed = torch.autograd.grad(out["scores"].sum(), out["initial_logits"],
                                       retain_graph=True, allow_unused=True)[0]
    assert main_to_seed is None
    auxiliary_to_seed = torch.autograd.grad(out["scores_seed"].sum(), out["initial_logits"],
                                           retain_graph=True)[0]
    assert auxiliary_to_seed.abs().sum() > 0
    main_to_anchors = torch.autograd.grad(out["scores"].sum(), model.anchors)[0]
    assert torch.isfinite(main_to_anchors).all()
    assert main_to_anchors.abs().sum() > 0


def test_complete_loss_has_finite_backward_and_bounded_anchors(example):
    model, snapshots, users, candidates = example
    out = model(users, candidates, snapshots, 1)
    loss, parts = compute_loss(model, out)
    assert torch.isfinite(loss)
    assert 0 <= parts["consistency"].item() <= math.log(2) + 1e-6
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.step()
    with torch.no_grad():
        model.anchors.mul_(100)
    model.project_anchors_()
    assert model.anchors.norm(dim=-1).max() <= model.config.radius + 1e-6


def test_future_graphs_are_rejected(example):
    model, snapshots, users, candidates = example
    with pytest.raises(ValueError, match="visible snapshots"):
        model(users, candidates, snapshots, 0)


def test_neighbor_sampling_is_deterministic_without_global_rng_mutation(example):
    model, snapshots, users, candidates = example
    model.config.neighbor_cap = 1
    model.eval()
    rng = torch.random.get_rng_state().clone()
    first = model(users, candidates, snapshots, 1)
    second = model(users, candidates, snapshots, 1)
    torch.testing.assert_close(first["scores"], second["scores"], rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert first["message_counts"].max() <= 2


def test_repeated_edges_do_not_change_unweighted_graph(example):
    model, snapshots, users, candidates = example
    duplicated = [torch.cat([edges, edges]) for edges in snapshots]
    original = model(users, candidates, snapshots, 1)
    repeated = model(users, candidates, duplicated, 1)
    torch.testing.assert_close(original["scores"], repeated["scores"])


def test_loss_weights_reject_invalid_values():
    with pytest.raises(ValueError):
        LossWeights(consistency=-1)
    with pytest.raises(ValueError):
        LossWeights(seed=float("nan"))

