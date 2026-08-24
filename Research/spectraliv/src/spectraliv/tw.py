"""Tracy-Widom beta=1 distribution via stable Fredholm determinants.

Method (Bornemann, "On the Numerical Evaluation of Distributions in Random
Matrix Theory: A Review", arXiv:0904.1581 [VERIFIED-FETCHED 2026-08-23],
Section 6; kernel from his eq. (6.4)):
for the operator V on L^2(s, inf) with kernel

    V(x, y) = (1/2) * Ai((x + y) / 2),

Bornemann's Theorem 6.1 / Ferrari-Spohn factorization gives

    F1(s) = det(I - V),      F2(s) = det(I - V) det(I + V),
    F4(s) = (det(I - V) + det(I + V)) / 2.

The determinants are evaluated by Gauss-Legendre Nystrom quadrature on
(s, inf) under the exponential map x = s + (1+u)/(1-u). This avoids the
exponential instability of Painleve-II initial-value integration (Bornemann
Table 1: IVP breaks down at x ~ -5.57 in IEEE double precision; the Fredholm
route reaches ~1e-15 absolute error).

We only use the beta = 1 branch (real data; the beta trap tagged in
toolkit_formulas.md).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import airy

_S_MIN = -8.2
_S_MAX = 8.0
_N_GRID = 328
_N_QUAD = 96

_QUAD_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _quad(n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    if n_quad not in _QUAD_CACHE:
        _QUAD_CACHE[n_quad] = np.polynomial.legendre.leggauss(n_quad)
    return _QUAD_CACHE[n_quad]


def _kernel_matrix(s: float, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes/weights under x = s + (1+u)/(1-u) and kernel V(x,y)."""
    u, w = _quad(n_quad)
    x = s + (1.0 + u) / (1.0 - u)
    wgt = w * 2.0 / (1.0 - u) ** 2
    k_mat = 0.5 * airy((x[:, None] + x[None, :]) / 2.0)[0]
    return k_mat, wgt


def _fredholm_logdet(s: float, n_quad: int = _N_QUAD, sign_plus: bool = False,
                     return_sign: bool = False):
    """log det(I -/+ V) on L^2(s, inf), V(x,y) = (1/2) Ai((x+y)/2)."""
    k_mat, wgt = _kernel_matrix(s, n_quad)
    eye = np.eye(k_mat.shape[0])
    m_mat = eye + (wgt[None, :] * k_mat if sign_plus else -wgt[None, :] * k_mat)
    sg, logdet = np.linalg.slogdet(m_mat)
    if return_sign:
        return float(logdet), float(sg)
    return float(logdet)


class TracyWidom1:
    """F_1 CDF/SF/PDF and quantiles; instantiate once and reuse."""

    def __init__(self, s_min: float = _S_MIN, s_max: float = _S_MAX,
                 n_grid: int = _N_GRID, n_quad: int = _N_QUAD):
        self.s_min, self.s_max = s_min, s_max
        grid = np.linspace(s_min, s_max, n_grid)
        # batch the Airy evaluations across all grid points
        u, w = _quad(n_quad)
        logcdf = np.empty(n_grid)
        chunk = 40
        for c0 in range(0, n_grid, chunk):
            s_chunk = grid[c0:c0 + chunk]
            x2 = s_chunk[:, None] + (1.0 + u) / (1.0 - u)          # (C, n)
            wgt = w * 2.0 / (1.0 - u) ** 2
            arg = (x2[:, :, None] + x2[:, None, :]) / 2.0           # (C, n, n)
            km = 0.5 * airy(arg)[0]
            for j, sj in enumerate(s_chunk):
                m_mat = np.eye(n_quad) - wgt[None, :] * km[j]
                logcdf[c0 + j] = min(np.linalg.slogdet(m_mat)[1], 0.0)
        if not np.all(np.diff(logcdf) > -1e-10):
            raise RuntimeError("TW1 log-CDF not increasing; check quadrature")
        self.grid = grid
        self._pchip_logcdf = PchipInterpolator(grid, logcdf)
        pdf = np.exp(logcdf) * PchipInterpolator(grid, np.gradient(logcdf, grid))(grid)
        self._pchip_pdf = PchipInterpolator(grid, np.clip(pdf, 0.0, None))
        self._logcdf_floor = logcdf[0]

    def log_cdf(self, s):
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        out = np.where(s_arr < self.s_min, self._logcdf_floor, self._pchip_logcdf(np.clip(s_arr, self.s_min, self.s_max)))
        return float(out[0]) if out.size == 1 else out

    def cdf(self, s):
        lc = self.log_cdf(s)
        return np.exp(lc)

    def sf(self, s):
        lc = self.log_cdf(s)
        return -np.expm1(lc)  # 1 - e^{lc}, stable at both ends

    def pdf(self, s):
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        inside = np.clip(s_arr, self.s_min, self.s_max)
        out = np.where(s_arr > self.s_max, 0.0, self._pchip_pdf(inside))
        return float(out[0]) if out.size == 1 else out

    def ppf(self, p):
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        out = np.empty_like(p_arr)
        for i, pi in enumerate(p_arr.flat):
            target = np.log(pi)
            a = self.s_min - 20.0
            while self.log_cdf(a) > target and a > -200.0:
                a *= 1.5
            out.flat[i] = brentq(lambda x: self.log_cdf(x) - target,
                                 max(a, self.s_min - 60.0), self.s_max,
                                 xtol=1e-7, rtol=1e-10)
        return float(out[0]) if out.size == 1 else out


_DEFAULT: TracyWidom1 | None = None


def default_tw1() -> TracyWidom1:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TracyWidom1()
    return _DEFAULT


# Reference constants (Tracy-Widom 1994 / standard tables)
TW1_MEAN = -1.2065335745820
TW1_VAR = 1.607781034581


def fredholm_f2(s: float, n_quad: int = _N_QUAD) -> float:
    """F2(s) = det(I-V) det(I+V); exposed for cross-validation in tests."""
    lm, sm = _fredholm_logdet(s, n_quad, sign_plus=False, return_sign=True)
    lp, sp = _fredholm_logdet(s, n_quad, sign_plus=True, return_sign=True)
    assert sm > 0 and sp > 0
    return float(np.exp(min(lp + lm, 0.0)))
