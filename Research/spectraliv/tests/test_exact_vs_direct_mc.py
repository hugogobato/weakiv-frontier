"""KILL-condition check (plan, Phase 2 give-up rules): exact-Jacobi quantile
implementation must agree with direct Monte Carlo of canonical correlations of
Gaussian data beyond MC error. A disagreement is a misformulation witness.
"""
import numpy as np
import pytest
from scipy import stats

from spectraliv.canoncorr import canoncorr
from spectraliv.jacobiquantiles import jacobi_null_roots


@pytest.mark.parametrize("n,q,p,big_b", [
    (250, 50, 2, 4000),
    (500, 250, 3, 2500),
    (1000, 300, 5, 1500),
])
def test_exact_ensemble_vs_direct_mc(n, q, p, big_b):
    rng1 = np.random.default_rng(1000 + n)
    roots_exact = jacobi_null_roots(n, q, p, big_b, rng1)

    rng2 = np.random.default_rng(2000 + n)
    roots_direct = np.empty(big_b)
    for i in range(big_b):
        z = rng2.standard_normal((n, q))
        x = rng2.standard_normal((n, p))
        xs = (x - x.mean(0)) / x.std(0, ddof=1)
        zs = (z - z.mean(0)) / z.std(0, ddof=1)
        roots_direct[i] = canoncorr(xs, zs)[0] ** 2

    ks = stats.ks_2samp(roots_exact, roots_direct)
    assert ks.pvalue > 1e-3, f"KS rejected: stat={ks.statistic}"

    for prob in (0.5, 0.95):
        qa = np.quantile(roots_exact, prob)
        qb = np.quantile(roots_direct, prob)
        assert abs(qa - qb) < 0.012, f"q{prob}: {qa} vs {qb}"
