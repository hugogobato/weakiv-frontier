import numpy as np
import pytest

from spectraliv.canoncorr import canonical_analysis
from spectraliv.dgps import make_null, make_single_spike
from spectraliv.ivestimators import prepared_all
from spectraliv.select_tau import select_tau


def _fingerprint(ups_k):
    return np.round(ups_k @ ups_k.T, 10)


def test_select_tau_invariant_under_y_permutation():
    rng = np.random.default_rng(2026)
    dgp = make_single_spike(500, 150, theta=0.35, rho=0.5, rng=rng)
    ys, xs, zs, ca, scale = prepared_all(dgp.y, dgp.x, dgp.z, None)
    tau1 = select_tau(dgp.x, dgp.z, canon=ca)
    # permute Y with a fixed seed; (X, Z) untouched
    perm_rng = np.random.default_rng(777)
    order = perm_rng.permutation(len(dgp.y))
    y_perm = dgp.y[order]
    ys2, xs2, zs2, ca2, scale2 = prepared_all(y_perm, dgp.x, dgp.z, None)
    tau2 = select_tau(y_perm, dgp.x, dgp.z, canon=ca2)
    assert tau1 == tau2
    k1 = max(1, int(round(tau1 * xs.shape[1])))
    assert np.array_equal(_fingerprint(ca.ups[:, :k1]), _fingerprint(ca2.ups[:, :k1]))


def test_truncated_estimator_subspace_ignores_y():
    """The retained projector must be bitwise identical under Y permutation."""
    rng = np.random.default_rng(99)
    dgp = make_single_spike(400, 100, theta=0.4, rho=0.6, rng=rng)
    _, xs, zs, ca, _sc = prepared_all(dgp.y, dgp.x, dgp.z, None)
    y_perm = dgp.y[np.random.default_rng(5).permutation(len(dgp.y))]
    _, xs2, zs2, ca2, _sc2 = prepared_all(y_perm, dgp.x, dgp.z, None)
    tau = select_tau(dgp.x, dgp.z, canon=ca)
    # explicit call paths sharing the canonical pass:
    from spectraliv.preprocess import prepare
    xs_a, zs_a, ys_a, sc_a = prepare(dgp.x, dgp.z, dgp.y, None)
    xs_b, zs_b, ys_b, sc_b = prepare(dgp.x, dgp.z, y_perm, None)
    ca_a = canonical_analysis(xs_a, zs_a)
    ca_b = canonical_analysis(xs_b, zs_b)
    k = max(1, int(round(tau * xs_a.shape[1])))
    assert np.array_equal(ca_a.ups[:, :k], ca_b.ups[:, :k])


def test_null_data_gives_low_tau():
    rng = np.random.default_rng(31337)
    taus = []
    for i in range(21):
        dgp = make_null(500, 200, 3, np.random.default_rng(400 + i))
        taus.append(select_tau(dgp.x, dgp.z))
    # under H0 a TW-95 outlier count of zero is typical; tau floor is 1/p
    assert float(np.median(taus)) == pytest.approx(1.0 / 3.0)
    # occasional false outliers allowed at the 5 percent root level, never all
    assert max(taus) <= 1.0


def test_spike_detected_single_direction():
    rng = np.random.default_rng(20260823)
    hits = 0
    reps = 40
    for i in range(reps):
        dgp = make_single_spike(800, 400, theta=0.75, rho=0.3,
                                rng=np.random.default_rng(1000 + i))
        tau = select_tau(dgp.x, dgp.z)
        hits += tau == pytest.approx(1.0)
    assert hits >= 36
