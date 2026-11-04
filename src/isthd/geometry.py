"""Curvature-one Lorentz geometry used by the I-STHD model.

Tangent vectors are stored by their ``d`` spatial coordinates at the origin;
manifold points have ``d + 1`` coordinates with the time coordinate first.
The graph encoder's fixed-origin Exp/Log pair does not, by itself, introduce
non-Euclidean message weighting. Hyperbolic distance is used by anchor separation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def clip_radius(v: Tensor, radius: float) -> Tensor:
    """Project vectors onto a Euclidean radius ball without changing direction."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if not v.is_floating_point():
        raise TypeError("geometry inputs must be floating-point tensors")
    eps = torch.finfo(v.dtype).eps
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return v * (radius / norm.clamp_min(eps)).clamp(max=1.0)


def exp_map_origin(v: Tensor) -> Tensor:
    """Map origin-tangent coordinates to the unit Lorentz hyperboloid."""
    if not v.is_floating_point() or v.shape[-1] < 1:
        raise ValueError("expected floating-point tangent vectors with positive width")
    eps = torch.finfo(v.dtype).eps
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    norm2 = norm.square()
    # The clamped denominator keeps both torch.where branches finite at zero.
    ratio = torch.where(
        norm > eps**0.5,
        torch.sinh(norm) / norm.clamp_min(eps),
        1.0 + norm2 / 6.0 + norm2.square() / 120.0,
    )
    return torch.cat((torch.cosh(norm), ratio * v), dim=-1)


def log_map_origin(p: Tensor) -> Tensor:
    """Return origin-tangent coordinates of a Lorentz point.

    Using ``asinh(norm(spatial))`` avoids an ``acosh`` derivative singularity at
    the origin and is exactly equivalent on the upper unit hyperboloid.
    """
    if not p.is_floating_point() or p.shape[-1] < 2:
        raise ValueError("expected floating-point Lorentz points with width >= 2")
    spatial = p[..., 1:]
    eps = torch.finfo(p.dtype).eps
    norm = torch.linalg.vector_norm(spatial, dim=-1, keepdim=True)
    norm2 = norm.square()
    ratio = torch.where(
        norm > eps**0.5,
        torch.asinh(norm) / norm.clamp_min(eps),
        1.0 - norm2 / 6.0 + 3.0 * norm2.square() / 40.0,
    )
    return ratio * spatial


def lorentz_inner(x: Tensor, y: Tensor) -> Tensor:
    """Broadcasted Lorentz inner product, with the time coordinate first."""
    if x.shape[-1] != y.shape[-1] or x.shape[-1] < 2:
        raise ValueError("Lorentz points must have the same width >= 2")
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def hyperbolic_distance(x: Tensor, y: Tensor) -> Tensor:
    """Curvature-one geodesic distance with a finite-gradient numerical floor.

    Distances below the floating-point floor are represented by a small positive
    value, rather than differentiating ``acosh(1)``. For float32, intermediate
    inner products use float64 to reduce cancellation near coincident anchors.
    """
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("distance inputs must be floating-point tensors")
    if x.dtype != y.dtype:
        raise TypeError("distance inputs must have the same dtype")
    dtype = x.dtype
    # MPS does not support float64; retain the original type on that device.
    use_double = dtype == torch.float32 and x.device.type != "mps"
    xx, yy = (x.double(), y.double()) if use_double else (x, y)
    argument = -lorentz_inner(xx, yy)
    floor = 1.0 + torch.finfo(dtype).eps
    return torch.acosh(argument.clamp_min(floor)).to(dtype=dtype)


# Short names mirror the equations without introducing extra geometry objects.
expmap0 = exp_map_origin
logmap0 = log_map_origin
