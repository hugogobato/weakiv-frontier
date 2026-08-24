import numpy as np
import pytest
from scipy import stats as st

from spectraliv.canoncorr import canonical_analysis
from spectraliv.preprocess import prepare
from spectraliv.teststats import (
    jacobi_mu_sigma,
    naive_f_pvalue,
    spec_test,
)
from spectraliv.dgps import make_null, make_single_spike


def test_johnstone_constants_against_hand_computation():
    # hand-computed reference for a small configuration (independent arithmetic)
    n, q, p = 1000, 300, 2
    m = n - q            # error df (uncentered)
    mn1 = m + q - 1      # = 698
    s2g = (p - 0.5) / mn1
    s2f = (q - 0.5) / mn1
    g = 2 * np.arcsin(np.sqrt(s2g))
    f = 2 * np.arcsin(np.sqrt(s2f))
    mu_ref = 2 * np.log(np.tan((f + g) / 2))
    sig3_ref = 16.0 / mn1**2 / (np.sin(f + g) * np.sin(f) * np.sin(g))
    mu, sig = jacobi_mu_sigma(n, q, p)
    assert mu == pytest.approx(mu_ref, rel=1e-12)
    assert sig == pytest.approx(sig3_ref ** (1 / 3), rel=1e-12)


def test_mu_np_centers_empirical_null_logit():
    """Standardized null roots follow TW1: sd matches sigma*sd(TW1), mean
    matches sigma*mean(TW1). A residual location offset of order 0.1 sigma is
    EXPECTED at fixed p (we sit slightly outside Johnstone's strict regime);
    exact-Jacobi MC remains the primary calibration (memo Section 5)."""
    from spectraliv.jacobiquantiles import simulate_jacobi_ensemble
    from spectraliv.tw import TW1_MEAN, TW1_VAR

    n, q, p = 500, 200, 3
    mu, sig = jacobi_mu_sigma(n, q, p)
    rng = np.random.default_rng(2026)
    lam = simulate_jacobi_ensemble(n, q, p, 1500, rng)[:, 0]
    t_vals = (np.log(lam / (1.0 - lam)) - mu) / sig
    assert abs(t_vals.mean() - TW1_MEAN) < 0.25
    assert abs(t_vals.std() - np.sqrt(TW1_VAR)) < 0.15


def test_naive_f_pvalue_matches_scipy():
    r2, n, q = 0.2, 500, 100
    f_stat = (r2 / q) / ((1 - r2) / (n - q))
    assert naive_f_pvalue(r2, n, q) == pytest.approx(st.f.sf(f_stat, q, n - q), rel=1e-12)


def test_spec_test_runs_and_fields_valid():
    rng = np.random.default_rng(3)
    dgp = make_single_spike(600, 180, theta=0.25, rho=0.4, rng=rng)
    res = spec_test(dgp.x, dgp.z, level=0.05, b_cal=800, rng=np.random.default_rng(4))
    assert 0.0 <= res.r2max <= 1.0
    assert 0.0 <= res.p_exact <= 1.0
    assert res.cv_exact > 0.0
    assert isinstance(res.reject_exact, bool)
    assert res.meta["n"] == 600 and res.meta["q"] == 180 and res.meta["p"] == 1
    # TW branch populated in this regime
    assert res.t_tw is not None and res.cv_tw is not None


def test_spec_test_null_size_sanity_small():
    """Rough size check at one null cell (not a substitute for Phase-3 X1)."""
    rng_master = np.random.default_rng(20260823)
    rejects = 0
    big_b = 400
    n, q = 300, 90
    cv = float(st.beta.ppf(0.95, q / 2.0, (n - q) / 2.0))  # p=1 exact CV
    for _ in range(big_b):
        dgp = make_null(n, q, 1, rng_master)
        xs, zs, _yr, _sc = prepare(dgp.x, dgp.z, None, None)
        ca = canonical_analysis(xs, zs)
        rejects += ca.r[0] ** 2 > cv
    size = rejects / big_b
    se = np.sqrt(0.05 * 0.95 / big_b)
    assert abs(size - 0.05) < 4 * se, f"size={size}"


def test_spec_test_power_direction():
    rng = np.random.default_rng(17)
    rejections = 0
    big_b = 200
    for _ in range(big_b):
        dgp = make_single_spike(800, 400, theta=0.7, rho=0.5, rng=rng)
        res = spec_test(dgp.x, dgp.z, b_cal=500,
                        rng=np.random.default_rng(rng.integers(1 << 31)))
        rejections += res.reject_exact
    assert rejections / big_b > 0.8
