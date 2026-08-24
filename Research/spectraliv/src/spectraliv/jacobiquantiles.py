"""Exact null (Jacobi ensemble) sampling and largest-root quantiles.

Under H0 + A1-A3 the squared sample canonical correlations follow the real
Jacobi ensemble beta=1 (toolkit F1 / P1):

    a = (q - p - 1)/2,  b = (N - q - p - 1)/2          [uncentered]

Sampling method "matrix_beta": draw A ~ W_p(N - q, I), B ~ W_p(q, I) from
Gaussian factors; U = (A+B)^{-1/2} B (A+B)^{-1/2} is matrix-Beta(q/2, (N-q)/2)
and eig(U) has exactly the Jacobi-1 density. At p = 1 the closed form
Beta(q/2, (N-q)/2) is used everywhere (exact, zero MC cost).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def jacobi_params(n: int, q: int, p: int, centered: bool = False) -> tuple[float, float]:
    """Ensemble exponents a, b of F1 (uncentered by default)."""
    shift = 1 if centered else 0
    a = (q - p - 1) / 2.0
    b = ((n - shift) - q - p - 1) / 2.0
    return a, b


def simulate_jacobi_ensemble(n: int, q: int, p: int, big_b: int,
                             rng: np.random.Generator,
                             centered: bool = False) -> np.ndarray:
    """B replicates of the full ordered (descending) eigenvalue set, shape (B, p).

    Vectorization note: per-replicate cost is O(N p^2 + q p^2); for Phase-3
    scale this loop is the dominant kernel and is profiled in WP-P2-I2.
    """
    a, _b = jacobi_params(n, q, p, centered)
    if a <= -1:
        raise ValueError("Jacobi density not proper: need q > p - 1 (A3)")
    out = np.empty((big_b, p))
    n_err = n - q - (1 if centered else 0)
    for i in range(big_b):
        g_a = rng.standard_normal((n_err, p))
        g_b = rng.standard_normal((q, p))
        aa = g_a.T @ g_a
        bb = g_b.T @ g_b
        c = np.linalg.cholesky(aa + bb)
        # U = C^{-1} B C^{-T}, symmetric psd with eigenvalues in [0, 1]
        cinv = np.linalg.inv(c)
        u = cinv @ bb @ cinv.T
        u = (u + u.T) / 2.0
        out[i] = np.sort(np.linalg.eigvalsh(u))[::-1]
    return out


def jacobi_null_roots(n: int, q: int, p: int, big_b: int,
                      rng: np.random.Generator, centered: bool = False) -> np.ndarray:
    """B replicates of the LARGEST squared canonical correlation under H0, shape (B,)."""
    if p == 1:
        return beta_p1_samples(q, n, big_b, rng, centered=centered)
    return simulate_jacobi_ensemble(n, q, p, big_b, rng, centered=centered)[:, 0]


def beta_p1_ppf(probs, q: int, n: int, centered: bool = False):
    """Exact p=1 quantiles of r_max^2 under H0: r^2 ~ Beta(q/2, (n-q)/2)."""
    shift = 1 if centered else 0
    return stats.beta.ppf(probs, q / 2.0, (n - shift - q) / 2.0)


def beta_p1_sf(r2, q: int, n: int, centered: bool = False):
    """Exact p=1 survival function of r_max^2 under H0."""
    shift = 1 if centered else 0
    return stats.beta.sf(r2, q / 2.0, (n - shift - q) / 2.0)


def beta_p1_samples(q: int, n: int, big_b: int, rng: np.random.Generator,
                    centered: bool = False) -> np.ndarray:
    shift = 1 if centered else 0
    return rng.beta(q / 2.0, (n - shift - q) / 2.0, size=big_b)


def largest_root_quantile(probs, n: int, q: int, p: int, big_b: int,
                          rng: np.random.Generator, centered: bool = False) -> np.ndarray:
    """Critical values for r_max^2 at the requested upper-tail probabilities.

    p = 1 uses the exact Beta closed form; p > 1 uses ensemble simulation.
    """
    probs = np.atleast_1d(np.asarray(probs, dtype=float))
    if p == 1:
        return beta_p1_ppf(probs, q, n, centered=centered)
    roots = jacobi_null_roots(n, q, p, big_b, rng, centered=centered)
    return np.quantile(roots, probs, method="linear")


def largest_root_pvalue(r2_obs: float, n: int, q: int, p: int, big_b: int,
                        rng: np.random.Generator, centered: bool = False) -> float:
    """Empirical survival function P(lambda_max >= r2_obs) under H0."""
    if p == 1:
        return float(beta_p1_sf(r2_obs, q, n, centered=centered))
    roots = jacobi_null_roots(n, q, p, big_b, rng, centered=centered)
    return float((np.sum(roots >= r2_obs) + 0.5) / (len(roots) + 1.0))
