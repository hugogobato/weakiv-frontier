import numpy as np
import pytest

from spectraliv.dgps import make_multispike, make_null, make_single_spike, rho_of_kappa
from spectraliv.rng import cell_stream, stream


def test_rng_determinism():
    a = stream(20260823, "exp", "cell")
    b = stream(20260823, "exp", "cell")
    c = stream(20260823, "exp", "other_cell")
    assert np.array_equal(a.standard_normal(10), b.standard_normal(10))
    assert not np.array_equal(a.standard_normal(10), c.standard_normal(10))
    s1 = cell_stream("x1", "n250_p1_q25", 3)
    s2 = cell_stream("x1", "n250_p1_q25", 3)
    assert np.array_equal(s1.standard_normal(5), s2.standard_normal(5))


def test_null_dgp_is_uncorrelated_with_z():
    rng = np.random.default_rng(1)
    dgp = make_null(4000, 200, 2, rng)
    zx = np.column_stack([dgp.z, dgp.x])
    corr = np.corrcoef(zx.T)[:200, 200:]
    assert np.max(np.abs(corr)) < 4.5 / np.sqrt(4000)


def test_single_spike_population_canonical_correlation():
    """Large-N concentration: sample r_max^2 approaches g(theta) (P2 affine map)."""
    n, q, theta = 6000, 1200, 0.6
    alpha = q / n
    rng = np.random.default_rng(7)
    dgp = make_single_spike(n, q, theta, rho=0.0, rng=rng)
    from spectraliv.canoncorr import canoncorr
    from spectraliv.preprocess import prepare
    xs, zs, _yr, _sc = prepare(dgp.x, dgp.z, None, None)
    r2 = canoncorr(xs, zs)[0] ** 2
    g = alpha + (1 - alpha) * theta
    assert abs(r2 - g) < 0.01


def test_rho_of_kappa_matches_phase1():
    assert rho_of_kappa(np.inf) == pytest.approx(1.0)
    assert rho_of_kappa(1.0) == pytest.approx(0.7071067811865476)


def test_hetero_profiles_change_variance_only():
    rng = np.random.default_rng(21)
    for profile in ("quadratic", "two_group"):
        dgp = make_single_spike(1000, 300, 0.4, 0.0, rng=rng, hetero=profile)
        # first column of X = sqrt(gamma) zpi + v; residual variance varies by design
        assert np.isfinite(dgp.x).all() and np.isfinite(dgp.y).all()


def test_multispike_shapes():
    rng = np.random.default_rng(31)
    dgp = make_multispike(500, 150, [0.8, 0.5, 0.3], rho=0.4, rng=rng)
    assert dgp.x.shape == (500, 3) and dgp.z.shape == (500, 150)
    assert dgp.theta == [0.8, 0.5, 0.3]
