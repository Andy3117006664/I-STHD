# I-STHD

**Interest-Anchored Spatio-Temporal Hyperbolic Disentanglement** for recommendation on evolving user–item graphs.

I-STHD uses one shared interest-anchor bank to connect historical message selection with preference modeling after aggregation. Users share the interest coordinates while their mixtures change as new observations arrive.

![I-STHD framework](docs/framework.png)

## Framework

1. **Encode observed graph contexts.** Build a finite-window interaction graph at each historical cutoff. A shared graph encoder summarizes its multi-hop collaborative neighborhoods.
2. **Construct an independent interest query.** Pool user identity and directly observed item embeddings, without graph propagation. Match this base profile to the anchors, then detach the initial mixture before passing it to the router.
3. **Route historical evidence.** Jointly weight neighbor–time–interest combinations using the initial mixture, anchor alignment, user–item identity compatibility, and two time ages. A historical item can contribute its updated collaborative context even when the user has no new interaction.
4. **Update and reorganize preferences.** Gate routed evidence into the base profile. Reuse the same anchors to estimate the final mixture, reconstruct its interest-aligned component, and retain the unexplained residual. A within-origin JS objective connects routed interest masses with the final mixture.
5. **Rank the next window.** Both components score the same user-conditioned item-factor representation. A learned gate combines the two scores, so useful information outside the anchor reconstruction can contribute at inference.

Shared anchors provide common coordinates for the two stages; users' interest mixtures can change over time. The residual can retain useful preferences outside the current anchor reconstruction.

Routing, reconstruction, and scoring operate in origin-tangent coordinates. In the fixed-origin encoder, `Log(Exp(...))` cancels in exact arithmetic; the hyperbolic metric enters through anchor separation.

## Code map

| File | Purpose |
|---|---|
| `src/isthd/geometry.py` | Lorentz exponential/log maps, distances, radius constraints |
| `src/isthd/model.py` | Shared encoder, query, joint router, reconstruction, factor heads and score gates |
| `src/isthd/loss.py` | Serving-score BPR, common/seed supervision and regularization |

## Install

Requires Python 3.10+ and PyTorch 2.2+.

```bash
python -m pip install -e .
```

