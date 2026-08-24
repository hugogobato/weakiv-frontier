"""Sample canonical correlations and canonical variates via QR + SVD.

Single shared pass used by both the test statistic (T_spec step 4) and the
truncated estimators (so the retained subspace is identical in both).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CanonicalAnalysis:
    """Results of a canonical correlation analysis of standardized/residualized data.

    r        descending sample canonical correlations, shape (r,) with r = min(p, q)
    xi       X-side canonical variates, n x r (orthonormal columns)
    ups      Z-side canonical variates, n x r (orthonormal columns)
    ax       X-side coefficients in X-space, p x r
    az       Z-side coefficients in Z-space, q x r
    """

    r: np.ndarray
    xi: np.ndarray
    ups: np.ndarray
    ax: np.ndarray
    az: np.ndarray


def canonical_analysis(x: np.ndarray, z: np.ndarray) -> CanonicalAnalysis:
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    if x.shape[0] != z.shape[0]:
        raise ValueError("canonical_analysis: row counts differ")
    qx, rx = np.linalg.qr(x)
    qz, rz = np.linalg.qr(z)

    def inv_upper(rmat):
        return np.linalg.solve(rmat, np.eye(rmat.shape[0]))

    ax0 = inv_upper(rx)   # p x p
    az0 = inv_upper(rz)   # q x q
    m = qx.T @ qz         # p x q
    u_s, s_vals, vt = np.linalg.svd(m, full_matrices=False)
    r = s_vals
    ax = ax0 @ u_s                       # p x r
    az = az0 @ vt.T                      # q x r
    xi = qx @ u_s                        # n x r
    ups = qz @ vt.T                      # n x r
    order = np.argsort(-r)
    return CanonicalAnalysis(r=r[order], xi=xi[:, order], ups=ups[:, order],
                             ax=ax[:, order], az=az[:, order])


def canoncorr(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Convenience wrapper returning only the descending correlations."""
    return canonical_analysis(x, z).r


def manual_canoncorr_reference(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Independent textbook route used ONLY inside unit tests.

    Squared canonical correlations are the eigenvalues of
    Czz^{-1} Czx Cxx^{-1} Cxz  (q x q, similar to symmetric PSD).
    Deliberately implemented via raw solves + nonsymmetric eig, a different
    path from the QR/SVD production code.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    cxx = x.T @ x
    czz = z.T @ z
    cxz = x.T @ z
    inner = np.linalg.solve(cxx, cxz)               # p x q
    prod = np.linalg.solve(czz, cxz.T @ inner)      # q x q
    lam = np.sort(np.linalg.eigvals(prod).real)[::-1]
    lam = np.clip(lam, 0.0, None)
    return np.sqrt(lam)[: min(x.shape[1], z.shape[1])]
