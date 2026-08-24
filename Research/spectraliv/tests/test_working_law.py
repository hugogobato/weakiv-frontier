"""Light regression guard for the P3 working law (Phase-1 Finding 1):

    E[beta_2SLS] - beta ~= rho / (1 + mu^2/q),  mu^2 = N theta/(1-theta)

Median relative gap must stay small on a compact grid; guards against silent
DGP drift between phases. Not a proof (P3 remains a working hypothesis).
"""
import numpy as np
import pytest

from spectraliv.dgps import make_single_spike, rho_of_kappa
from spectraliv.ivestimators import tsls

N = 1000
ALPHA = 0.3
KAPPAS = [0.5, 1.0]
THETAS = [0.15, 0.3, 0.5, 0.7]
BIG_B = 600


@pytest.mark.slow
def test_working_law_median_gap_small():
    q = int(round(ALPHA * (N - 1)))
    rng_master = np.random.default_rng(20260823)
    gaps = []
    for kappa in KAPPAS:
        rho = rho_of_kappa(kappa)
        for th in THETAS:
            bh = np.empty(BIG_B)
            for b in range(BIG_B):
                dgp = make_single_spike(N, q, float(th), float(rho),
                                        rng=np.random.default_rng(rng_master.integers(1 << 31)),
                                        beta=0.0)
                bh[b] = tsls(dgp.y, dgp.x, dgp.z)[0]
            mu2 = N * th / (1.0 - th)
            formula = rho / (1.0 + mu2 / q)
            mc_bias = bh.mean()
            gaps.append(abs(mc_bias - formula) / abs(formula))
    assert np.median(gaps) < 0.05, f"median rel gap {np.median(gaps):.4f}"
