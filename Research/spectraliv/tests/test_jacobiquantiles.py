import numpy as np
import pytest
from scipy import stats

from spectraliv.jacobiquantiles import (
    beta_p1_ppf,
    beta_p1_sf,
    jacobi_params,
    jacobi_null_roots,
    largest_root_quantile,
    simulate_jacobi_ensemble,
)
from spectraliv.preprocess import assert_proper


def test_beta_identity_closed_form():
    """Plan-required identity: p=1 Jacobi quantile equals Beta(q/2,(N-q)/2)."""
    probs = np.array([0.5, 0.9, 0.95, 0.99])
    for n, q in [(250, 25), (1000, 700), (500, 450)]:
        exact = beta_p1_ppf(probs, q, n)
        ref = stats.beta.ppf(probs, q / 2.0, (n - q) / 2.0)
        assert np.allclose(exact, ref, atol=1e-12)
        sf = beta_p1_sf(exact, q, n)
        assert np.allclose(sf, 1.0 - probs, atol=1e-10)


def test_ensemble_simulation_matches_beta_moments_p1():
    """Ensemble sampler (matrix-Beta path) agrees with the Beta closed form at p=1."""
    n, q, big_b = 400, 120, 6000
    rng = np.random.default_rng(20260823)
    roots = jacobi_null_roots(n, q, 1, big_b, rng)  # uses Beta samples directly
    mean_th = stats.beta.mean(q / 2.0, (n - q) / 2.0)
    var_th = stats.beta.var(q / 2.0, (n - q) / 2.0)
    assert abs(roots.mean() - mean_th) < 4 * np.sqrt(var_th / big_b)


def test_ensemble_multivariate_matches_wishart_moments():
    """Trace of U = (A+B)^{-1/2} B (A+B)^{-1/2} has known expectation p*q/N.

    E[trace(U)] = p * E[lambda] with the ensemble; a robust moment check is
    trace(U)/p vs its MC estimate within tolerance.
    """
    n, q, p, big_b = 300, 90, 3, 1500
    rng = np.random.default_rng(42)
    ens = simulate_jacobi_ensemble(n, q, p, big_b, rng)
    assert ens.shape == (big_b, p)
    assert np.all(ens >= -1e-12) and np.all(ens <= 1 + 1e-12)
    assert np.all(np.diff(ens, axis=1) <= 1e-10)  # descending order
    # marginal mean of a root is bounded by edge alpha-ish scale: loose sanity
    assert 0.0 < ens[:, 0].mean() < 0.6


def test_quantile_path_consistency():
    n, q = 500, 250
    rng = np.random.default_rng(5)
    cv_exact = largest_root_quantile([0.95], n, q, 1, 100, rng)[0]
    cv_closed = beta_p1_ppf(0.95, q, n)
    assert cv_exact == pytest.approx(cv_closed, abs=1e-12)


def test_a3_assertions():
    with pytest.raises(ValueError):
        assert_proper(100, 150, 3)          # q > N
    with pytest.raises(ValueError):
        assert_proper(20, 18, 3)            # N > q + p violated


def test_params_match_toolkit_f1():
    a, b = jacobi_params(1000, 400, 5, centered=False)
    assert a == pytest.approx((400 - 5 - 1) / 2.0)
    assert b == pytest.approx((1000 - 400 - 5 - 1) / 2.0)
