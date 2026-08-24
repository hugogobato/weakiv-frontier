"""IV estimator family (formalization memo Section 6).

All estimators accept optional included-exogenous W (partialled out first),
standardize blocks explicitly (preprocess.prepare), and report coefficients in
the ORIGINAL input units via the `scale` factor (correction record: an earlier
version silently returned standardized-unit betas; caught by the preregistered
working-law regression test). The truncated-2SLS retained subspace depends on
(X, Z) only.
"""
from __future__ import annotations

import numpy as np

from .canoncorr import CanonicalAnalysis, canonical_analysis


def _prep_yxz(y, x, z, w):
    """Standardize + residualize; returns (ys_std, xs, zs, scale)."""
    from .preprocess import prepare

    xs, zs, yr, scale = prepare(x, z, y, w)
    sy = np.std(yr, axis=0, ddof=1)
    return yr / sy, xs, zs, scale


def prepared_all(y, x, z, w):
    """(ys_std, xs, zs, ca, scale): blocks + canonical pass + unit rescale."""
    ys, xs, zs, scale = _prep_yxz(y, x, z, w)
    return ys, xs, zs, canonical_analysis(xs, zs), scale


def _solve(a_mat, b_vec):
    return np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]


def ols(y, x, w=None):
    from .preprocess import residualize, standardize_columns

    xs = residualize(standardize_columns(x), w)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    yr = residualize(y_arr - y_arr.mean(), w)
    sy = yr.std(axis=0, ddof=1)
    sx = standardize_columns(x).std(axis=0, ddof=1)
    beta_std = _solve(xs, yr / sy)
    return beta_std * (sx / sy)


def tsls(y, x, z, w=None):
    ys, xs, zs, scale = _prep_yxz(y, x, z, w)
    pz_x = zs @ np.linalg.lstsq(zs, xs, rcond=None)[0]
    pz_y = zs @ np.linalg.lstsq(zs, ys, rcond=None)[0]
    return _solve(xs.T @ pz_x, xs.T @ pz_y) * scale


def kclass(y, x, z, k: float, w=None):
    """k-class: W(k) = (1-k) I + k P_Z, so k=1 is 2SLS, k=k_LIML in [0,1) LIML,
    k = N/(N-q) > 1 the Bekker many-instrument correction."""
    ys, xs, zs, scale = _prep_yxz(y, x, z, w)
    pz = zs @ np.linalg.pinv(zs)
    mk = (1.0 - k) * np.eye(len(ys)) + k * pz
    return _solve(xs.T @ mk @ xs, xs.T @ mk @ ys) * scale


def liml_k(y, x, z, w=None) -> float:
    """Smallest generalized eigenvalue of (E'E, E' M_Z E), E = [y X]."""
    ys, xs, zs = _prep_yxz(y, x, z, w)[:3]
    e = np.column_stack([ys, xs])
    pz = zs @ np.linalg.pinv(zs)
    see = e.T @ e
    sez = e.T @ (e - pz @ e)
    vals = np.linalg.eigvals(np.linalg.solve(see, sez))
    return float(np.min(vals.real))


def liml(y, x, z, w=None):
    return kclass(y, x, z, liml_k(y, x, z, w), w=w)


def fuller(y, x, z, a: float = 1.0, w=None):
    """Fuller's k: k_liml - a/(N - q_eff); convention documented in the memo."""
    ys, xs, zs = _prep_yxz(y, x, z, w)[:3]
    n_eff, q_eff = zs.shape
    k_f = liml_k(y, x, z, w) - a / (n_eff - q_eff)
    return kclass(y, x, z, k_f, w=w)


def bekker(y, x, z, w=None):
    """Bekker (1994) many-instrument correction as k-class with k = N/(N-q)."""
    ys, xs, zs = _prep_yxz(y, x, z, w)[:3]
    n_eff, q_eff = zs.shape
    return kclass(y, x, z, n_eff / (n_eff - q_eff), w=w)


def jive(y, x, z, w=None):
    """Jackknife IV (Angrist-Krueger-Imbens 1999) with (1 - h_j)-weighted sums."""
    ys, xs, zs, scale = _prep_yxz(y, x, z, w)
    g = zs @ np.linalg.solve(zs.T @ zs, np.eye(zs.shape[1]))
    h = np.einsum("ij,ij->i", g, zs)
    wm = (1.0 - h)[:, None]
    big_a = zs.T @ (wm * zs)              # q x q
    big_b = zs.T @ (wm * xs)              # q x p
    x_hat = zs @ np.linalg.solve(big_a, big_b)   # fitted leave-i-out first stage
    return _solve(x_hat.T @ xs, x_hat.T @ ys) * scale


def jive_naive(y, x, z, w=None):
    """Reference JIVE: explicit loop over (1 - h_j)-weighted sums (test-only oracle)."""
    ys, xs, zs, rescale = _prep_yxz(y, x, z, w)
    n, q = zs.shape
    zz_inv = np.linalg.inv(zs.T @ zs)
    h = np.einsum("ij,jk,ik->i", zs, zz_inv, zs)
    big_a = np.zeros((q, q))
    big_b = np.zeros((q, xs.shape[1]))
    for i in range(n):
        wi = 1.0 - h[i]
        big_a += wi * np.outer(zs[i], zs[i])
        big_b += wi * np.outer(zs[i], xs[i])
    x_hat = zs @ np.linalg.solve(big_a, big_b)
    return _solve(x_hat.T @ xs, x_hat.T @ ys) * rescale


def truncated_2sls(y, x, z, tau: float | None = None, k: int | None = None,
                   w=None, canon: CanonicalAnalysis | None = None,
                   prepared=None):
    """2SLS restricted to the top-k Z-side canonical directions.

    k = max(1, round(tau * p)) when tau given. tau = 1 recovers 2SLS exactly;
    k = 1 is just-identified IV on the leading canonical variate.
    `prepared` may pass the output of prepared_all to share a single pass.
    """
    if prepared is not None:
        ys, xs, zs, canon, scale = prepared
    else:
        ys, xs, zs, canon, scale = prepared_all(y, x, z, w)
    p_dim = xs.shape[1]
    if k is None:
        if tau is None:
            raise ValueError("truncated_2sls needs tau or k")
        k = max(1, int(round(tau * p_dim)))
    k = min(k, p_dim)
    ups_k = canon.ups[:, :k]
    pk_x = ups_k @ (ups_k.T @ xs)
    pk_y = ups_k @ (ups_k.T @ ys)
    return _solve(xs.T @ pk_x, xs.T @ pk_y) * scale


def pca_2sls(y, x, z, ell: int, w=None):
    """Meza-Singh-style PCA-2SLS: project instruments on top-ell PCs, then 2SLS."""
    ys, xs, zs, scale = _prep_yxz(y, x, z, w)
    evals, evecs = np.linalg.eigh(zs.T @ zs)
    idx = np.argsort(evals)[::-1][:ell]
    zp = zs @ evecs[:, idx]
    pz_x = zp @ np.linalg.lstsq(zp, xs, rcond=None)[0]
    pz_y = zp @ np.linalg.lstsq(zp, ys, rcond=None)[0]
    return _solve(xs.T @ pz_x, xs.T @ pz_y) * scale


def whiten_2sls(y, x, z, ridge_rel: float = 0.05, w=None):
    """Meza-Singh-style Whiten-2SLS with relative ridge on Zt'Zt."""
    ys, xs, zs, scale = _prep_yxz(y, x, z, w)
    szz = zs.T @ zs
    lam = ridge_rel * np.trace(szz) / szz.shape[0]
    evals, evecs = np.linalg.eigh(szz)
    evals = np.clip(evals, 1e-12, None)
    s_inv_sqrt = ((evals + lam) ** -0.5)[:, None] * evecs
    zw = zs @ s_inv_sqrt * np.sqrt(szz.shape[0])  # keep scale comparable
    pz_x = zw @ np.linalg.lstsq(zw, xs, rcond=None)[0]
    pz_y = zw @ np.linalg.lstsq(zw, ys, rcond=None)[0]
    return _solve(xs.T @ pz_x, xs.T @ pz_y) * scale
