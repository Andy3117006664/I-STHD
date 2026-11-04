# Computation and geometry

## Shared coordinates

The trainable anchor coordinates are `a_m` in the origin tangent space. Their manifold views are `p_m = Exp_o(a_m)`. There is one parameter bank, reused for the initial mixture, historical routing, and final preference reconstruction.

At cutoff `t`, a direct-history profile `b` yields `pi0 = softmax(b A^T / sqrt(d))`. Only the router's copy of the mixture is detached. The embeddings and anchors still learn through the other differentiable paths; an auxiliary seed ranking objective trains this query.

For a previously observed neighbor `i` at historical cutoff `tau`, the routing logit is:

```text
z[i,tau,m] = log(sg(pi0[m]))
             + a[m]^T W_K x[i,tau] / sqrt(d)
             + (W_U e[u])^T W_I e[i] / sqrt(d)
             + c[m]^T [log(1+t-tau), log(1+t-last_interaction(u,i,tau))]
rho = softmax(z over all neighbor-time-interest triples)
omega[m] = sum_(i,tau) rho[i,tau,m]
r = sum_(i,tau,m) rho[i,tau,m] W_V x[i,tau]
v = b + sigmoid(W_g [b,r] + b_g) * r
pi = softmax(v A^T / sqrt(d))
v_common = sum_m pi[m] a[m]
v_residual = v - v_common
```

Every normalization uses only the sampled accessible message set. A neighbor's state may remain accessible after its latest direct interaction; that state does not assert a new interaction at `tau`. Sampling uses a seeded permutation for each user/window, giving reproducible forecasts.

The final score combines separate common and residual scoring maps, applied to the same interest-conditioned item representation:

```text
score = (1 - gamma) * score_common + gamma * score_residual
gamma = sigmoid(w_gamma^T [v, v_residual, entropy(pi)] + b_gamma)
```

`g` is a coordinate-wise profile gate. `gamma` is a scalar score gate. They are different computations.

## Objectives

The serving-score pairwise ranking loss is primary. Auxiliary common and seed ranking losses retain anchor predictiveness and train the detached query. Regularizers include:

- a bounded-radius hyperbolic-distance margin between anchor points;
- population-average initial/final mixture balance, without requiring each user to be uniform;
- `JS(omega, pi)` within the same user and cutoff, excluding queries with no messages;
- `||W_common^T W_residual||_F^2` on scoring weights;
- parameter L2 regularization.

The weight product limits cross-pathway output overlap; it does not prove statistical independence or identify causal versus spurious preferences.

## Which operations are hyperbolic?

`Exp_o` maps tangent coordinates to curvature-one Lorentz points; `Log_o` returns origin-tangent coordinates. Anchor separation uses `acosh(-<p,q>_L)`. Its argument is numerically clamped slightly above one so gradients at coincident points remain finite; float32 self-distances therefore have a small positive numerical floor.

The encoder stores manifold states, but its actual neighbor aggregation, self transform, and activation occur at the fixed origin's tangent space. Algebraically:

```text
x_next = Log_o(Exp_o(tanh(B x + W sum_neighbors alpha * x_neighbor)))
       = tanh(B x + W sum_neighbors alpha * x_neighbor)
```

Consequently, this encoder's tangent output equals the corresponding Euclidean GCN in exact arithmetic. Geometric distance regularization still changes the anchor optimization. No metric-aware attention or curvature-dependent encoder benefit is assumed by the implementation.

Routing dot products, gated updates, reconstruction, residual subtraction, item-factor attention, and ranking are vector operations. Mixtures and routing masses live on probability simplexes; they are not manifold points.

## Forecast boundary

All windows and integer node indices are zero-based. `forward(users, candidates, snapshots, cutoff)` requires `len(snapshots) == cutoff + 1`. This catches accidental inclusion of future graphs. Window `cutoff+1` is a target, not an input. Source optimization recomputes graph states before each batch; target-period evaluation freezes parameters.
