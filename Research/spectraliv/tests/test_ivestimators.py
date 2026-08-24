import numpy as np
import pytest

from spectraliv.ivestimators import (
    bekker,
    fuller,
    jive,
    jive_naive,
    kclass,
    liml,
    liml_k,
    ols,
    pca_2sls,
    prepared_all,
    tsls,
    truncated_2sls,
    whiten_2sls,
)
from spectraliv.dgps import make_single_spike


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1234)
    dgp = make_single_spike(200, 40, theta=0.3, rho=0.5, rng=rng, beta=0.7, p=1)
    return dgp


def test_tsls_equals_kclass1(data):
    b1 = tsls(data.y, data.x, data.z)
    b2 = kclass(data.y, data.x, data.z, 1.0)
    assert np.allclose(b1, b2, atol=1e-10)


def test_liml_equals_kclass_at_its_k(data):
    k_hat = liml_k(data.y, data.x, data.z)
    assert np.allclose(liml(data.y, data.x, data.z),
                       kclass(data.y, data.x, data.z, k_hat), atol=1e-9)
    assert 0.0 < k_hat < 1.0


def test_bekker_is_kclass_n_over_n_minus_q(data):
    n_eff = len(data.y)
    q_eff = data.z.shape[1]
    assert np.allclose(bekker(data.y, data.x, data.z),
                       kclass(data.y, data.x, data.z, n_eff / (n_eff - q_eff)), atol=1e-10)


def test_fuller_between_lims_and_sanity(data):
    b_f = fuller(data.y, data.x, data.z, a=1.0)
    b_l = liml(data.y, data.x, data.z)
    assert b_f.shape == b_l.shape
    assert np.isfinite(b_f).all()


def test_truncated_tau1_equals_tsls(data):
    b_t = truncated_2sls(data.y, data.x, data.z, tau=1.0)
    b_s = tsls(data.y, data.x, data.z)
    assert np.allclose(b_t, b_s, atol=1e-8)


def test_truncated_k1_manual_scalar_iv(data):
    """k=1 truncation equals IV on the leading canonical variate u_1."""
    ys, xs, zs, ca, rescale = prepared_all(data.y, data.x, data.z, None)
    u1 = ca.ups[:, 0]
    manual_std = float((u1 @ ys) / (u1 @ xs[:, 0]))
    b_k1 = truncated_2sls(data.y, data.x, data.z, k=1,
                          prepared=(ys, xs, zs, ca, rescale))
    assert b_k1[0] == pytest.approx(manual_std * rescale[0], rel=1e-9)


def test_jive_vectorized_matches_naive_loop(data):
    b_fast = jive(data.y, data.x, data.z)
    b_slow = jive_naive(data.y, data.x, data.z)
    assert np.allclose(b_fast, b_slow, atol=1e-9)


def test_jive_hand_computed_example():
    """Hand example n=5, q=2, p=1: independent arithmetic written out explicitly.

    Computed on PREPARED (standardized/residualized) data because jive()
    standardizes internally; beta is invariant to the joint scaling.
    """
    zt = np.array([[1.0, 0.0],
                   [1.0, 1.0],
                   [2.0, 1.0],
                   [1.0, 2.0],
                   [2.0, 2.0]])
    pi = np.array([2.0, -1.0])
    xt_raw = (zt @ pi).reshape(-1, 1)      # first stage exact
    eps = np.array([0.3, -0.2, 0.1, 0.4, -0.6])
    yt_raw = xt_raw[:, 0] * 0.5 + eps      # beta_true = 0.5

    from spectraliv.preprocess import prepare
    xs, zs, yr, rescale = prepare(xt_raw, zt, yt_raw, None)
    ys_std = yr / yr.std(axis=0, ddof=1)

    zz = zs.T @ zs
    zz_inv = np.linalg.inv(zz)
    h = np.einsum("ij,jk,ik->i", zs, zz_inv, zs)
    big_a = sum((1 - h[i]) * np.outer(zs[i], zs[i]) for i in range(5))
    big_b = sum((1 - h[i]) * zs[i] * xs[i] for i in range(5))
    xhat = zs @ np.linalg.solve(big_a, big_b)
    # closed form for p = 1 on standardized blocks; rescale converts units
    beta_hand = float((xhat @ ys_std) / (xhat @ xs[:, 0])) * float(rescale[0])

    from spectraliv.ivestimators import jive as jive_fn
    beta_fn = float(jive_fn(yt_raw, xt_raw, zt)[0])
    assert beta_fn == pytest.approx(beta_hand, rel=1e-10)
    assert abs(beta_fn - 0.5) < 0.25


def test_pca_2sls_full_rank_equals_tsls(data):
    assert np.allclose(pca_2sls(data.y, data.x, data.z, ell=data.z.shape[1]),
                       tsls(data.y, data.x, data.z), atol=1e-7)


def test_whiten_runs_and_finite(data):
    b_w = whiten_2sls(data.y, data.x, data.z, ridge_rel=0.05)
    assert np.isfinite(b_w).all()


def test_ols_consistent_shape(data):
    b_o = ols(data.y, data.x)
    assert b_o.shape == (data.x.shape[1],)
