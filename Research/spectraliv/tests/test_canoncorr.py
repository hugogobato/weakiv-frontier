import numpy as np
import pytest

from spectraliv.canoncorr import canonical_analysis, canoncorr, manual_canoncorr_reference
from spectraliv.preprocess import prepare


def test_manual_3x2_example():
    # hand-checked 3x2 example: X and Z share exactly one direction
    x = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
    z = np.array([[2.0, 1.0],
                  [1.0, -1.0],
                  [3.0, 0.0]])
    xs, zs, _yr, _sc = prepare(x, z, None, None)
    r = canoncorr(xs, zs)
    ref = manual_canoncorr_reference(xs, zs)
    assert np.allclose(r, ref, atol=1e-10)
    assert r[0] > 0.5  # strong shared direction by construction
    # perfect-correlation sanity: proportional columns after centering
    x2 = np.array([[1.0], [2.0], [3.0]])
    z2 = np.array([[2.0], [4.0], [6.0]])
    assert canoncorr(x2, z2)[0] == pytest.approx(1.0, abs=1e-8)


def test_random_case_matches_reference():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((40, 5))
    z = rng.standard_normal((40, 7))
    xs, zs, _yr, _sc = prepare(x, z, None, None)
    r = canoncorr(xs, zs)
    ref = manual_canoncorr_reference(xs, zs)
    assert np.allclose(r, ref, atol=1e-9)
    assert np.all(np.diff(r) <= 1e-12)  # descending
    assert len(r) == 5


def test_variates_are_orthonormal_and_consistent():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((60, 3))
    z = rng.standard_normal((60, 9))
    w = rng.standard_normal((60, 2))
    xs, zs, _yr, _sc = prepare(x, z, None, w)
    ca = canonical_analysis(xs, zs)
    assert np.allclose(ca.xi.T @ ca.xi, np.eye(3), atol=1e-8)
    assert np.allclose(ca.ups.T @ ca.ups, np.eye(3), atol=1e-8)
    # cross-moments equal correlations times identity (canonical property)
    cross = ca.xi.T @ ca.ups
    assert np.allclose(cross, np.diag(ca.r), atol=1e-8)


def test_row_count_mismatch_raises():
    with pytest.raises(ValueError):
        canoncorr(np.zeros((5, 2)), np.zeros((4, 3)))
