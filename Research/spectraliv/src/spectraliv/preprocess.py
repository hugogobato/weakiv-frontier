"""Explicit preprocessing: standardization and residualization.

Assumption A2 (formalization memo Section 3) requires standardized designs.
Standardization is an explicit, tested step, never a silent side effect.
"""
from __future__ import annotations

import numpy as np


def standardize_columns(a: np.ndarray) -> np.ndarray:
    """Center columns and divide by sample sd (ddof=1). Zero-sd columns raise."""
    a = np.asarray(a, dtype=float)
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, ddof=1, keepdims=True)
    if np.any(sd <= 0):
        raise ValueError("standardize_columns: zero-variance column")
    return (a - mu) / sd


def residualize(a: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    """Project columns of a off the column space of w (lstsq; handles rank deficiency)."""
    if w is None:
        return np.asarray(a, dtype=float)
    w = np.asarray(w, dtype=float)
    coef, *_ = np.linalg.lstsq(w, a, rcond=None)
    return a - w @ coef


def prepare(x: np.ndarray, z: np.ndarray, y: np.ndarray | None = None,
            w: np.ndarray | None = None):
    """Full preprocessing pipeline shared by tests and estimators.

    Returns (xs, zs, yr, rescale):
      xs, zs   w-residualized, then standardized blocks (A2),
      yr       centered + w-residualized y in ORIGINAL units,
      rescale  vector sy/sx. An estimator computed as the regression of
               (yr / sy) on xs yields gamma_hat with beta_original =
               gamma_hat * rescale (verified against the exact just-identified
               closed form in tests). With w given, all sds are taken after
               residualization on w.
    """
    x_arr = np.asarray(x, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    xr = residualize(x_arr - x_arr.mean(axis=0), w)
    zr = residualize(z_arr - z_arr.mean(axis=0), w)
    sx = xr.std(axis=0, ddof=1)
    sz = zr.std(axis=0, ddof=1)
    if np.any(sx <= 0) or np.any(sz <= 0):
        raise ValueError("prepare: zero-variance column after residualization")
    xs = xr / sx
    zs = zr / sz
    if y is None:
        return xs, zs, None, np.ones_like(sx)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    yr = residualize(y_arr - y_arr.mean(), w)
    sy = yr.std(axis=0, ddof=1)
    return xs, zs, yr, sy / sx


def assert_proper(n: int, q: int, p: int, centered: bool = False) -> None:
    """Assumption A3: Jacobi density properness (toolkit F1).

    Uncentered baseline requires q > p - 1 and N > q + p.
    Centered case costs one df: q > p - 1 and N - 1 > q + p + 1 - 1 i.e. N > q + p + 1.
    """
    shift = 1 if centered else 0
    if not (q > p - 1):
        raise ValueError(f"A3 violated: need q > p - 1, got q={q}, p={p}")
    if not (n - shift > q + p + (1 if centered else 0)):
        raise ValueError(f"A3 violated: need N > q + p (centered: N > q + p + 1), got N={n}, q={q}, p={p}")
