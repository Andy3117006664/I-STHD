import random

import torch
from torch import nn

from isthd import ISTHD, ISTHDConfig
from isthd.data import TemporalData, make_demo_data, temporal_data_from_records, sample_training_batch
from isthd.evaluate import evaluate_model, load_checkpoint


def records():
    return [
        ("u0", "i0", 0.), ("u1", "i1", 0.), ("u2", "i2", 0.),
        ("u0", "i1", 10.), ("u1", "i2", 10.),
        ("u0", "i2", 20.), ("u1", "i0", 20.),
        ("u2", "i1", 30.), ("u0", "i0", 30.),
        ("u2", "i0", 40.), ("new-user", "new-item", 50.),
    ]


def test_source_vocabulary_excludes_target_only_ids():
    data = temporal_data_from_records(records(), window_seconds=10.)
    assert "new-user" not in data.user_ids
    assert "new-item" not in data.item_ids
    assert data.unknown_id_event_counts[-1] == 1


def test_negatives_do_not_exclude_later_future_positives():
    data = temporal_data_from_records(records(), window_seconds=10.)
    queries, _ = data.forecast_queries(1)
    query = next(query for query in queries if data.user_ids[query.user] == "u0")
    _, candidates = sample_training_batch([query], random.Random(7), 3)
    assert data.item_ids[candidates[0, 0]] == "i1"
    # i2 is a positive in a later window, but is still a legal negative now.
    assert all(data.item_ids[i] == "i2" for i in candidates[0, 1:].tolist())


def test_adding_future_edges_does_not_change_earlier_candidates():
    original = temporal_data_from_records(records(), window_seconds=10.)
    changed = temporal_data_from_records(records() + [("u0", "i1", 50.)],
                                          fixed_metadata=original.metadata())
    assert original.forecast_queries(1) == changed.forecast_queries(1)


def test_evaluation_freezes_parameters_and_restores_training_mode():
    torch.manual_seed(4)
    data = make_demo_data()
    model = ISTHD(ISTHDConfig(data.num_users, data.num_items, dim=6, score_dim=5,
                             num_anchors=3, num_factors=2, num_layers=1))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.train()
    report = evaluate_model(model, data, split="test", ks=[1, 5])
    assert report["parameter_updates"] == 0
    assert report["num_queries"] > 0
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for k in (1, 5):
        assert 0 <= report[f"recall@{k}"] <= 1
        assert 0 <= report[f"ndcg@{k}"] <= 1


def test_known_perfect_ranking_has_unit_recall_and_ndcg():
    class ItemOrder(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.tensor(0.))

        def forward(self, users, candidates, snapshots, cutoff):
            assert len(snapshots) == cutoff + 1 == 1
            return {"scores": -candidates.float() + self.offset}

    data = TemporalData(
        snapshots=[torch.tensor([[0, 0], [1, 1], [1, 2]]), torch.tensor([[0, 1]])],
        user_ids=["u0", "u1"], item_ids=["i0", "i1", "i2"],
        train_end=0, val_end=0, window_seconds=1., start_timestamp=0.,
        raw_event_counts=[3, 1], unknown_id_event_counts=[0, 0],
    )
    report = evaluate_model(ItemOrder(), data, targets=[1], ks=[1, 2])
    assert report["num_queries"] == 1
    for k in (1, 2):
        assert report[f"recall@{k}"] == 1
        assert report[f"ndcg@{k}"] == 1


def test_tensor_only_checkpoint_roundtrip(tmp_path):
    config = ISTHDConfig(2, 3, dim=4, score_dim=4, num_layers=1)
    model = ISTHD(config)
    path = tmp_path / "checkpoint.pt"
    import dataclasses
    torch.save({"model_config": dataclasses.asdict(config), "model_state": model.state_dict()}, path)
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    restored = ISTHD(ISTHDConfig(**checkpoint["model_config"]))
    restored.load_state_dict(checkpoint["model_state"])
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_training_rejects_empty_validation_instead_of_selecting_first_epoch(tmp_path):
    import csv
    import pytest
    from isthd.train import main

    path = tmp_path / "interactions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "item_id", "timestamp"])
        for window in range(6):
            for user in range(3):
                # Source windows introduce all three items per user; later
                # windows contain only repeats and cannot validate novel items.
                item = (user + min(window, 2)) % 3
                writer.writerow([str(user), str(item), window * 10])
    with pytest.raises(ValueError, match="validation has no eligible queries"):
        main(["--data", str(path), "--window-seconds", "10", "--epochs", "1",
              "--output", str(tmp_path / "run")])
