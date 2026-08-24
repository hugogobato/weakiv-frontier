"""P2 numeric check (formalization memo Section 4, MANDATORY before proof work).

Candidate deterministic-equivalent lift map for a single spike at fixed p:

    g(theta)   = alpha + (1 - alpha) * theta          (affine)
    g^{-1}(r2) = (N r2 - q) / (N - q)

Pass rules preregistered here (Phase-2 gate evidence):
  (a) |median_MC(r_max^2) - g(theta)| <= TOL_LOCATION for theta >= THETA_MIN;
  (b) |median_MC(g^{-1}(r_max^2)) - theta| <= TOL_INVERSION for theta >= THETA_MIN;
  (c) edge upward-bias witness, analytic (Beta skewness at theta = 0; see
      test_edge_upward_bias_witness_analytic).

Correction record: this check originally failed for the first drafted map
g_wrong = alpha/(1-theta(1-alpha)) and forced the affine correction recorded
in formalization_memo.md P2.
"""
import numpy as np
import pytest

from spectraliv.canoncorr import canoncorr
from spectraliv.dgps import make_single_spike
from spectraliv.preprocess import prepare

THETAS = np.array([0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8])
CONFIGS = [(1000, 0.3), (1000, 0.7), (500, 0.5)]
BIG_B = 300
THETA_MIN = 0.15
TOL_LOCATION = 0.02
TOL_INVERSION = 0.03


def g_of_theta(theta, alpha):
    return alpha + (1.0 - alpha) * theta


def g_inv(r2, n, q):
    return (n * r2 - q) / (n - q)


@pytest.mark.slow
@pytest.mark.parametrize("n,alpha", CONFIGS, ids=[f"n{n}_a{a}" for n, a in CONFIGS])
def test_lift_map_location_and_inversion(n, alpha):
    q = int(round(alpha * (n - 1)))
    rng_master = np.random.default_rng(555000 + int(alpha * 100) + n)
    r2 = np.empty((BIG_B, len(THETAS)))
    for b in range(BIG_B):
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        for ti, th in enumerate(THETAS):
            dgp = make_single_spike(n, q, float(th), rho=0.0, rng=rng)
            xs, zs, _yr, _sc = prepare(dgp.x, dgp.z, None, None)
            r2[b, ti] = canoncorr(xs, zs)[0] ** 2

    med = np.median(r2, axis=0)
    pred = np.array([g_of_theta(t, alpha) for t in THETAS])
    loc_err = np.abs(med - pred)[THETAS >= THETA_MIN]
    assert loc_err.max() <= TOL_LOCATION, (
        f"location mismatch: med={med}, pred={pred}")

    inv_med = np.median(g_inv(r2, n, q), axis=0)
    inv_err = np.abs(inv_med - THETAS)[THETAS >= THETA_MIN]
    assert inv_err.max() <= TOL_INVERSION, (
        f"inversion bias: {inv_med} vs {list(THETAS)}")


def test_edge_upward_bias_witness_analytic():
    """Witness (c), analytic version (correction record: the original MC form
    at theta=0.05 was noise-dominated; the bias mechanism lives AT the edge,
    where for p=1 it is provable: r^2 ~ Beta(q/2,(N-q)/2) has skewness
    2(b-a)sqrt(a+b+1)/((a+b+2)sqrt(ab)) > 0 whenever q < N-q, hence
    E[g^{-1}(r^2)] > median[g^{-1}(r^2)] exactly. This right-skew of the
    inverted estimate at the edge is the mechanism behind the conservative
    envelope's false-flag cost (Phase-1 Finding 3)."""
    from scipy import stats as st

    n, q = 500, 150
    a_param, b_param = q / 2.0, (n - q) / 2.0
    skew = st.beta.stats(a_param, b_param, moments="s")
    assert skew > 0
    alpha = q / n
    mean_inv = (st.beta.mean(a_param, b_param) - alpha) / (1 - alpha)
    med_inv = (st.beta.median(a_param, b_param) - alpha) / (1 - alpha)
    assert mean_inv > med_inv
    # light MC confirmation of the same fact
    rng = np.random.default_rng(31337)
    r2 = rng.beta(a_param, b_param, size=40000)
    inv = g_inv(r2, n, q)
    assert inv.mean() > np.median(inv)


def test_map_matches_phase1_thresholds_directionally():
    """Sanity: at alpha=0.9 the map's lift over the edge for theta=0.29 is small
    but positive (consistent with Phase-1 O2 finding: power far below edge)."""
    alpha = 0.9
    edge = g_of_theta(0.0, alpha)          # = alpha exactly
    lifted = g_of_theta(0.29, alpha)
    assert edge == pytest.approx(alpha)
    assert 0.0 < lifted - edge < 0.05
