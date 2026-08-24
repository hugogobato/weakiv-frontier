import numpy as np
import pytest

from spectraliv.tw import TracyWidom1, TW1_MEAN, TW1_VAR, default_tw1


@pytest.fixture(scope="module")
def tw1():
    return default_tw1()


def test_published_moments(tw1):
    s = np.linspace(-8.0, 8.0, 4000)
    pdf = tw1.pdf(s)
    mean = np.trapezoid(s * pdf, s)
    var = np.trapezoid((s - mean) ** 2 * pdf, s)
    assert abs(mean - TW1_MEAN) < 2e-3
    assert abs(var - TW1_VAR) < 5e-3


def test_johnstone_95pct_quantile(tw1):
    # f_0.95 = 0.9793 quoted in Johnstone (2009) eq (6) context / toolkit F3
    assert tw1.ppf(0.95) == pytest.approx(0.9793, abs=2e-3)


def test_cdf_sf_consistency_and_roundtrip(tw1):
    for s in (-3.0, -1.0, 0.0, 1.5, 3.0):
        c = float(tw1.cdf(s))
        sf = float(tw1.sf(s))
        assert abs(c + sf - 1.0) < 1e-9
        assert tw1.ppf(c) == pytest.approx(s, abs=1e-4)


def test_monotone_cdf(tw1):
    s = np.linspace(-6, 6, 200)
    c = tw1.cdf(s)
    assert np.all(np.diff(c) > 0)


@pytest.mark.slow
def test_end_to_end_jacobi_standardized_roots_follow_tw1():
    """Johnstone mapping validated end-to-end: standardized logit largest roots
    of the exact Jacobi null ensemble behave as TW1 (KS tolerance)."""
    from spectraliv.teststats import jacobi_mu_sigma
    from spectraliv.jacobiquantiles import simulate_jacobi_ensemble
    from scipy import stats as st

    n, q, p, big_b = 600, 240, 4, 2500
    mu, sig = jacobi_mu_sigma(n, q, p)
    rng = np.random.default_rng(99)
    ens = simulate_jacobi_ensemble(n, q, p, big_b, rng)
    lam_max = ens[:, 0]
    w = np.log(lam_max / (1.0 - lam_max))
    t_vals = (w - mu) / sig
    ks = st.kstest(t_vals, lambda x: default_tw1().cdf(x))
    assert ks.pvalue > 0.01, f"TW1 end-to-end KS stat={ks.statistic}"
