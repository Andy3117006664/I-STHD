import torch

from isthd.geometry import exp_map_origin, log_map_origin, hyperbolic_distance, lorentz_inner


def test_maps_round_trip_and_stay_on_lorentz_manifold():
    torch.manual_seed(9)
    vectors = torch.cat([torch.zeros(1, 5), torch.randn(6, 5) * 0.3]).double()
    points = exp_map_origin(vectors)
    torch.testing.assert_close(lorentz_inner(points, points), -torch.ones(7, dtype=torch.float64))
    torch.testing.assert_close(log_map_origin(points), vectors, rtol=1e-9, atol=1e-9)


def test_distance_is_symmetric_and_gradients_at_coincident_points_are_finite():
    vectors = torch.tensor([[0., 0.], [0.3, 0.1], [-0.2, 0.2]], requires_grad=True)
    points = exp_map_origin(vectors)
    d = hyperbolic_distance(points[:, None, :], points[None, :, :])
    torch.testing.assert_close(d, d.T)
    assert d.diag().max() < 1e-3
    assert (d >= 0).all()
    d.sum().backward()
    assert torch.isfinite(vectors.grad).all()


def test_origin_maps_do_not_change_tangent_aggregation():
    # Documents the present encoder's exact-arithmetic geometry boundary.
    vectors = torch.tensor([[0.2, -0.4], [0.8, 0.1]], dtype=torch.float64)
    aggregate = torch.tanh(vectors + vectors.flip(0) * 0.5)
    torch.testing.assert_close(log_map_origin(exp_map_origin(aggregate)), aggregate,
                               rtol=1e-10, atol=1e-10)

