"""First-stage-only tau selection (anti-leakage invariant, memo Section 6.2-6.3)."""
from __future__ import annotations

import numpy as np

from .canoncorr import CanonicalAnalysis, canonical_analysis
from .preprocess import prepare
from .teststats import jacobi_mu_sigma
from .tw import default_tw1


def select_tau(x, z, w=None, level: float = 0.95,
               canon: CanonicalAnalysis | None = None) -> float:
    """tau_hat = k_hat / p with k_hat the contiguous-from-top TW-outlier count.

    Depends on (X, Z) only. Unit test asserts invariance under permutation of Y.
    """
    if canon is None:
        xs, zs, _yr, _sc = prepare(x, z, None, w)
        canon = canonical_analysis(xs, zs)
    p_dim = len(canon.r)
    n = canon.xi.shape[0]
    q = canon.az.shape[0]
    mu, sig = jacobi_mu_sigma(n, q, min(p_dim, q))
    f_thr = default_tw1().ppf(level)
    r = canon.r[:p_dim]
    l2 = np.log(np.clip(r**2 / (1.0 - r**2), 1e-300, None))
    t_vals = (l2 - mu) / sig
    k_hat = 0
    for flag in t_vals > f_thr:
        if flag:
            k_hat += 1
        else:
            break
    return float(np.clip(k_hat / p_dim, 1.0 / p_dim, 1.0))
