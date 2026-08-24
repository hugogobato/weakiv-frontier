"""Phase-3 gate check: fast shared-pass path == frozen estimator definitions.

The preregistration memo (WP-P3-R0) commits every decisive run to the fast
path in `spectraliv.experiments.fast_rep`; this test enforces that it is
numerically identical to the frozen, identity-tested `ivestimators` /
`select_tau` code paths (max abs diff < 1e-8 on betas and tau; r2max exact to
1e-10). Runs before any decisive run per memo Section 3.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectraliv.dgps import make_single_spike
from spectraliv.experiments import fast_rep
from spectraliv.ivestimators import (
    bekker,
    fuller,
    jive,
    liml,
    pca_2sls,
    prepared_all,
    tsls,
    truncated_2sls,
    whiten_2sls,
)
from spectraliv.select_tau import select_tau

CONFIGS = [
    (300, 40, 1),
    (350, 90, 2),
    (400, 150, 5),
]


@pytest.mark.parametrize("n,q,p", CONFIGS)
def test_fast_matches_frozen(n, q, p):
    rng = np.random.default_rng(1000 + n + q + p)
    dgp = make_single_spike(n, q, 0.28, 0.45, rng, p=p, beta=0.5)

    fr = fast_rep(dgp.y, dgp.x, dgp.z, estimators=True,
                  k_list=sorted({1, 2, p}), pca_rule="tw")

    # canonical statistic
    ys, xs, zs, _ca, _sc = prepared_all(dgp.y, dgp.x, dgp.z, None)
    from spectraliv.canoncorr import canoncorr
    r2_ref = float(canoncorr(xs, zs)[0] ** 2)
    assert fr["r2max"] == pytest.approx(r2_ref, abs=1e-10)

    def vec(f, *a, **kw):
        return np.asarray(f(dgp.y, dgp.x, dgp.z, *a, **kw), float).reshape(-1)

    pairs = [
        ("tsls", vec(tsls)),
        ("liml", vec(liml)),
        ("fuller", vec(fuller)),
        ("bekker", vec(bekker)),
        ("jive", vec(jive)),
        ("pca_l", vec(pca_2sls, ell=fr["l_pc"] if fr["l_pc"] else 1)),
        ("whiten", vec(whiten_2sls, ridge_rel=0.05)),
    ]
    for k in sorted({1, 2, p}):
        kk = min(max(k, 1), p)

        def trunc_ref(kk=kk):
            return np.asarray(truncated_2sls(dgp.y, dgp.x, dgp.z, k=kk),
                              float).reshape(-1)
        pairs.append((f"trunc_k{kk}", trunc_ref()))
    for name, ref in pairs:
        got = fr["beta"][name]
        assert got.shape == ref.shape, name
        assert np.max(np.abs(got - ref)) < 1e-8, (
            f"{name}: max diff {np.max(np.abs(got - ref)):.3e}")

    tau_ref = select_tau(dgp.x, dgp.z)
    assert fr["tau_hat"] == pytest.approx(float(tau_ref), abs=1e-12)


def test_ar_interval_matches_bruteforce():
    """Closed-form p=1 AR interval == grid inversion of the AR statistic."""
    from scipy import stats as st

    from spectraliv.experiments import ar_accepts, ar_interval_p1
    rng = np.random.default_rng(7)
    n, q = 500, 120
    dgp = make_single_spike(n, q, 0.22, 0.6, rng, p=1, beta=0.4)
    lo, hi = ar_interval_p1(dgp.y, dgp.x, dgp.z)
    assert np.isfinite(lo) and np.isfinite(hi)
    from spectraliv.preprocess import prepare
    xs, zs, yr, _sc = prepare(dgp.x, dgp.z, np.asarray(dgp.y), None)
    ys = yr / np.std(yr, ddof=1)
    f_crit = st.f.ppf(0.95, q, n - q)

    def accepts(b0):
        e = ys - xs[:, 0] * b0
        pez = zs @ np.linalg.lstsq(zs, e, rcond=None)[0]
        num = float(e @ pez)
        den = float(e @ e) - num
        return (num / q) / (den / (n - q)) <= f_crit

    grid = np.linspace(lo - 0.3, hi + 0.3, 4001)
    flags = np.array([accepts(b) for b in grid])
    inside = flags[(grid > lo + 1e-9) & (grid < hi - 1e-9)]
    outside_hi = flags[grid > hi + 1e-6]
    outside_lo = flags[grid < lo - 1e-6]
    assert inside.all()
    assert not outside_hi.any() and not outside_lo.any()
    assert accepts((lo + hi) / 2)
