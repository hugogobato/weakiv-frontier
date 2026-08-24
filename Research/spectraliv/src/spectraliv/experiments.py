"""Phase-3 experiment runners (WP-P3-S1/S2/S3).

Design notes (preregistration memo WP-P3-R0):
- ONE preprocessing + canonical pass per replication is shared by T_spec,
  select_tau, and the whole estimator battery (`fast_rep`). The mathematical
  definitions are IDENTICAL to the frozen `ivestimators` / `teststats` /
  `select_tau` code; equivalence is enforced by
  `tests/test_fast_equivalence.py` (max |diff| on beta and r^2max < 1e-8)
  before any decisive run. k-class estimators are solved through Gram
  matrices instead of n x n projectors; jive/pca/whiten are the same
  operations reordered around one eigendecomposition of Zt'Zt.
- Critical values for a cell are drawn ONCE from
  stream(master, exp, cell_id, "cv") (b_cal reps of the exact Jacobi
  ensemble) and reused across replications; this removes CV-simulation noise
  from power curves and is recorded in every manifest.
- Every result row carries (experiment, cell_id, seed); per-cell CSV plus a
  `_done` marker containing sha256; merge_results.py validates completeness
  before any gate memo is written.
"""
from __future__ import annotations

import os
import time

import numpy as np
from scipy import stats as st

from .dgps import DGPResult, make_null, make_single_spike, rho_of_kappa
from .preprocess import prepare
from .rng import cell_stream, sha256_file, stream
from .teststats import jacobi_mu_sigma
from .tw import default_tw1

MASTER_SEED = 20260823

RESULTS_ROOT_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), "Research", "weakiv_results")


def limit_threads():
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "1")


# ---------------------------------------------------------------------------
# Per-cell critical values (computed once, reused across replications)


class CellCV:
    """Exact-Jacobi critical value + secondary/naive references for a cell."""

    def __init__(self, n: int, q: int, p: int, b_cal: int, rng,
                 level: float = 0.05):
        self.n, self.q, self.p, self.b_cal, self.level = n, q, p, b_cal, level
        if p == 1:
            self.roots = None
            self.cv_exact = float(st.beta.ppf(1 - level, q / 2.0, (n - q) / 2.0))
        else:
            from .jacobiquantiles import simulate_jacobi_ensemble
            ens = simulate_jacobi_ensemble(n, q, p, b_cal, rng)
            self.roots = np.sort(ens[:, 0])
            self.cv_exact = float(np.quantile(self.roots, 1 - level,
                                              method="linear"))
        try:
            self.mu, self.sig = jacobi_mu_sigma(n, q, min(p, q))
            self.tw_ok = True
        except RuntimeError:
            self.mu = self.sig = None
            self.tw_ok = False
        if self.tw_ok:
            f_thr = default_tw1().ppf(1 - level)
            thr = self.mu + f_thr * self.sig
            self.cv_tw = float(np.exp(thr) / (1.0 + np.exp(thr)))
        else:
            self.cv_tw = None
        # naive chi-squared reference: reject if N*r2max exceeds chi2_q tail
        self.cv_chi2 = float(st.chi2.ppf(1 - level, q))
        self.f_crit95 = float(st.f.ppf(0.95, q, n - q))

    def p_exact(self, r2max: float) -> float:
        if self.roots is None:
            return float(st.beta.sf(r2max, self.q / 2.0, (self.n - self.q) / 2.0))
        ge = self.b_cal - np.searchsorted(self.roots, r2max, side="left")
        return float((ge + 0.5) / (self.b_cal + 1.0))

    def p_tw(self, r2max: float) -> float | None:
        if not self.tw_ok or not (0.0 < r2max < 1.0):
            return None
        t = (np.log(r2max / (1.0 - r2max)) - self.mu) / self.sig
        return float(default_tw1().sf(t))


# ---------------------------------------------------------------------------
# Fast shared-pass replication


def _canonical_pass(xs, zs):
    qx, rx = np.linalg.qr(xs)
    qz, rz = np.linalg.qr(zs)
    m = qx.T @ qz
    u_s, s_vals, vt = np.linalg.svd(m, full_matrices=False)
    order = np.argsort(-s_vals)
    r = s_vals[order]
    ups = qz @ vt.T[:, order]
    return r, ups, rz


def _tw_outlier_count(stat_values: np.ndarray, level: float = 0.95) -> int:
    f_thr = default_tw1().ppf(level)
    k = 0
    for flag in stat_values > f_thr:
        if flag:
            k += 1
        else:
            break
    return k


def fast_rep(y, x, z, estimators: bool = True, k_list=None,
             pca_rule="tw"):
    """One shared preprocessing/canonical pass feeding all Phase-3 outputs.

    Returns dict with keys: r2max, tau_hat, beta (dict name -> vector in
    ORIGINAL units), l_pc, k_liml, ups (Z-side variates), xs, zs, scale.
    Estimator definitions mirror ivestimators exactly (verified by
    tests/test_fast_equivalence.py).
    """
    xs, zs, yr, scale_vec = prepare(x, z, np.asarray(y, dtype=float), None)
    sy = np.std(yr, ddof=1)
    ys = yr / sy
    scale = float(scale_vec[0])
    n, p = xs.shape
    q = zs.shape[1]
    r, ups, rz = _canonical_pass(xs, zs)

    czz = rz.T @ rz
    czx = zs.T @ xs                      # q x p
    czy = zs.T @ ys                      # q,
    chol = np.linalg.cholesky(czz)
    chol_t = chol.T

    xpx = xs.T @ xs
    xpy = xs.T @ ys

    # classical first-stage Wald summary / p (preregistered KP-rk flag form;
    # see kp_rk_wald_f docstring) -- free from cached Gram factors
    gamma_fs = np.linalg.solve(chol_t, np.linalg.solve(chol, czx))
    fitted_ss = np.sum(czx * gamma_fs, axis=0)
    resid_ss = np.maximum(np.diag(xpx) - fitted_ss, 1e-300)
    out_kp_wald = float(np.sum((n - q) * fitted_ss / resid_ss) / p)

    out = {
        "r": r,
        "r2max": float(r[0] ** 2),
        "tau_hat": _tau_from_roots(n, q, r),
        "scale": scale,
        "xs": xs, "zs": zs, "ys": ys, "ups": ups,
        "chol": chol, "gram": (czz, czx),
        "kp_wald": out_kp_wald,
        "beta": {},
        "l_pc": None, "k_liml": None,
    }

    if estimators:
        sol_czx = np.linalg.solve(chol, czx)
        sol_czy = np.linalg.solve(chol, czy.reshape(-1, 1))
        xpzxs = czx.T @ np.linalg.solve(chol_t, sol_czx)     # xs'P_Z xs
        xpzys = (czx.T @ np.linalg.solve(chol_t, sol_czy)).reshape(-1)

        b_tsls = np.linalg.lstsq(xpzxs, xpzys, rcond=None)[0]
        out["beta"]["tsls"] = b_tsls

        e = np.column_stack([ys, xs])
        cez = e.T @ zs                       # 2 x q
        w_e = np.linalg.solve(chol, cez.T)   # q x 2 = L^{-1} (Z'E)
        epe = w_e.T @ w_e                    # E'P_Z E
        ee = e.T @ e
        vals = np.linalg.eigvals(np.linalg.solve(ee, ee - epe))
        k_liml = float(np.min(vals.real))
        out["k_liml"] = k_liml

        def kclass_beta(k):
            mxx = (1.0 - k) * xpx + k * xpzxs
            mxy = (1.0 - k) * xpy + k * xpzys
            return np.linalg.lstsq(mxx, mxy, rcond=None)[0]

        out["beta"]["liml"] = kclass_beta(k_liml)
        out["beta"]["fuller"] = kclass_beta(k_liml - 1.0 / (n - q))
        out["beta"]["bekker"] = kclass_beta(n / (n - q))

        ginv_zs = zs @ np.linalg.inv(czz)
        h = np.einsum("ij,ij->i", ginv_zs, zs)
        wm = (1.0 - h)
        big_a = zs.T @ (wm[:, None] * zs)
        big_b = zs.T @ (wm[:, None] * xs)
        xhat = zs @ np.linalg.solve(big_a, big_b)
        out["beta"]["jive"] = np.linalg.lstsq(xhat.T @ xs, xhat.T @ ys,
                                              rcond=None)[0]

        evals, evecs = np.linalg.eigh(czz)
        desc = np.argsort(evals)[::-1]
        evals_d = evals[desc]
        evecs_d = evecs[:, desc]
        if pca_rule == "tw":
            # data-driven PC retention: contiguous-from-top TW-outlier count
            # on the (raw-scale) instrument Gram spectrum under the white-Wishart
            # null; Johnstone-BW centering mu = (sqrt(n-1)+sqrt(q))^2 and
            # sigma = (sqrt(n-1)+sqrt(q))(1/sqrt(n-1)+1/sqrt(q))^{1/3}.
            sqn1, sqq = np.sqrt(n - 1.0), np.sqrt(float(q))
            mu_pc = (sqn1 + sqq) ** 2
            sig_pc = (sqn1 + sqq) * (1.0 / sqn1 + 1.0 / sqq) ** (1.0 / 3.0)
            t_pc = (evals_d - mu_pc) / sig_pc
            l_pc = max(1, _tw_outlier_count(t_pc))
        else:
            l_pc = int(str(pca_rule).split(":")[1])
        l_pc = min(max(l_pc, 1), q)
        zp = zs @ evecs_d[:, :l_pc]          # orthonormal columns
        inv_ev = (1.0 / evals_d[:l_pc])
        pz_x = zp @ (inv_ev[:, None] * (zp.T @ xs))
        pz_y = zp @ (inv_ev * (zp.T @ ys))
        out["beta"]["pca_l"] = np.linalg.lstsq(xs.T @ pz_x, xs.T @ pz_y,
                                               rcond=None)[0]
        out["l_pc"] = l_pc

        # whiten: zw = zs @ S^{-1/2} * sqrt(n); 2SLS against zw solved via
        # Gram matrices (identical math to ivestimators.whiten_2sls)
        lam = 0.05 * float(np.trace(czz)) / q
        ev_cl = np.clip(evals_d, 1e-12, None)
        s_inv_half = ((ev_cl + lam) ** -0.5)[:, None] * evecs_d
        zw_scale_sq = float(n)
        wzz = zw_scale_sq * (s_inv_half.T @ czz @ s_inv_half)
        wzx = np.sqrt(zw_scale_sq) * (s_inv_half.T @ czx)
        wzy = np.sqrt(zw_scale_sq) * (s_inv_half.T @ czy.reshape(-1, 1))
        # x'P_w x etc. via Gram matrices in instrument space (q x q solve)
        xpwx = wzx.T @ np.linalg.solve(wzz, wzx)
        xpwy = (wzx.T @ np.linalg.solve(wzz, wzy)).reshape(-1)
        out["beta"]["whiten"] = np.linalg.lstsq(xpwx, xpwy, rcond=None)[0]

        k_opts = [p] if k_list is None else list(k_list)
        for k in k_opts:
            kk = int(min(max(k, 1), p))
            if f"trunc_k{kk}" in out["beta"]:
                continue
            ups_k = ups[:, :kk]
            pk_x = ups_k @ (ups_k.T @ xs)
            pk_y = ups_k @ (ups_k.T @ ys)
            out["beta"][f"trunc_k{kk}"] = np.linalg.lstsq(
                xs.T @ pk_x, xs.T @ pk_y, rcond=None)[0]

    # all estimators computed on standardized blocks -> rescale to ORIGINAL
    # input units exactly as ivestimators does (*= scale_vec = sy/sx,
    # per endogenous column)
    out["beta"] = {name: np.asarray(v, dtype=float).reshape(-1) * scale_vec
                   for name, v in out["beta"].items()}

    return out


def _tau_from_roots(n: int, q: int, r: np.ndarray) -> float:
    """Mirror of select_tau.select_tau for an existing canonical pass.

    Frozen semantics: p_dim = len(r) = min(p, q); same (mu, sig) applied to
    every root; contiguous-from-top TW-outlier count; tau clipped to
    [1/p_dim, 1].
    """
    p_dim = len(r)
    mu, sig = jacobi_mu_sigma(n, q, min(p_dim, q))
    rr = np.clip(r[:p_dim], 1e-12, 1.0 - 1e-16)
    l2 = np.log(rr ** 2 / (1.0 - rr ** 2))
    t_vals = (l2 - mu) / sig
    return float(np.clip(_tw_outlier_count(t_vals) / p_dim, 1.0 / p_dim, 1.0))


# ---------------------------------------------------------------------------
# Anderson-Rubin machinery (X3)


def ar_interval_p1_from(xs1, zs, ys, chol, level: float = 0.95):
    """Cached-blocks variant of ar_interval_p1 (same math, shared pass)."""
    n, q = zs.shape
    stz_y = np.linalg.solve(chol, zs.T @ ys)
    stz_x = np.linalg.solve(chol, zs.T @ xs1)
    wy = float(stz_y @ stz_y)
    wxy = float(stz_x @ stz_x)
    ypzx = float(stz_y @ stz_x)
    ymy = float(ys @ ys)
    xmx = float(xs1 @ xs1)
    xmy = float(xs1 @ ys)
    f_crit = float(st.f.ppf(level, q, n - q))
    A = (n - q) * wxy - f_crit * q * (xmx - wxy)
    B = (n - q) * (-2.0 * ypzx) - f_crit * q * (-2.0 * xmy + 2.0 * ypzx)
    C = (n - q) * wy - f_crit * q * (ymy - wy)
    if A <= 0.0:
        return (-np.inf, np.inf)
    disc = B * B - 4.0 * A * C
    if disc <= 0.0:
        return (np.nan, np.nan)
    root = np.sqrt(disc)
    return ((-B - root) / (2.0 * A), (-B + root) / (2.0 * A))


def ar_interval_p1(y, x, z, level: float = 0.95):
    """95% AR confidence set for beta (p = 1) by exact quadratic inversion.

    AR(b) = [e'P_Z e / q] / [e'M_Z e / (n - q)] <= F_{q,n-q}(level), with
    e = ys - xs1*b. Both quadratic forms share e, so standardization cancels.
    Returns (lo, hi); (-inf, inf) if accepted everywhere; (nan, nan) if empty.
    """
    xs, zs, yr, _sc = prepare(x, z, np.asarray(y, dtype=float), None)
    ys = yr / np.std(yr, ddof=1)
    xs1 = xs[:, 0]
    n, q = zs.shape
    chol = np.linalg.cholesky(zs.T @ zs)
    stz_y = np.linalg.solve(chol, zs.T @ ys)
    stz_x = np.linalg.solve(chol, zs.T @ xs1)
    wy = float(stz_y @ stz_y)
    wxy = float(stz_x @ stz_x)
    ypzx = float(stz_y @ stz_x)
    ymy = float(ys @ ys)
    xmx = float(xs1 @ xs1)
    xmy = float(xs1 @ ys)
    f_crit = float(st.f.ppf(level, q, n - q))
    # accept iff g(b) := (n-q)*S_r(b) - f*q*S_u(b) <= 0
    A = (n - q) * wxy - f_crit * q * (xmx - wxy)
    B = (n - q) * (-2.0 * ypzx) - f_crit * q * (-2.0 * xmy + 2.0 * ypzx)
    C = (n - q) * wy - f_crit * q * (ymy - wy)
    if A <= 0.0:
        return (-np.inf, np.inf)
    disc = B * B - 4.0 * A * C
    if disc <= 0.0:
        return (np.nan, np.nan)
    root = np.sqrt(disc)
    return ((-B - root) / (2.0 * A), (-B + root) / (2.0 * A))


def ar_accepts(y, x, z, beta_true, level: float = 0.95, chol=None) -> bool:
    """AR acceptance indicator for H0: beta = beta_true (any p).

    AR = (e'P_Z e / q) / (e'M_Z e / (n-q)) ~ F(q, n-q) under H0.
    `chol` may pass the Cholesky factor of zs'zs from a shared pass to avoid
    a redundant factorization.
    """
    xs, zs, yr, _sc = prepare(x, z, np.asarray(y, dtype=float), None)
    ys = yr / np.std(yr, ddof=1)
    b_vec = np.asarray(beta_true, dtype=float).reshape(-1)
    if b_vec.size == 1:
        b_vec = np.full(xs.shape[1], b_vec[0])
    e = ys - xs @ b_vec
    n, q = zs.shape
    if chol is None:
        chol = np.linalg.cholesky(zs.T @ zs)
    w_e = np.linalg.solve(chol, zs.T @ e)            # L^{-1} Z'e
    num = max(float(w_e @ w_e), 1e-300)              # e'P_Z e
    den = max(float(e @ e) - num, 1e-300)            # e'M_Z e
    stat = (num / q) / (den / (n - q))
    return bool(stat <= st.f.ppf(level, q, n - q))


# ---------------------------------------------------------------------------
# X5 violation DGPs (additive; frozen dgps.py untouched)


def make_heavy_v(n, q, p, rng, df=5):
    """E2 violation on the FIRST STAGE: v ~ scaled t_df (unit variance)."""
    z = rng.standard_normal((n, q))
    v = rng.standard_t(df, size=(n, p)) * np.sqrt((df - 2.0) / df)
    eps = rng.standard_normal(n)
    return DGPResult(y=v[:, 0] * 0.0 + eps, x=v, z=z, beta_true=0.0,
                     theta=[0.0] * p, rho=0.0, meta={"dgp": "heavy_v"})


def make_heavy_eps(n, q, p, rng, df=5):
    """E2' on the STRUCTURAL error only: first stage Gaussian (law intact)."""
    z = rng.standard_normal((n, q))
    v = rng.standard_normal((n, p))
    eps = rng.standard_t(df, size=n) * np.sqrt((df - 2.0) / df)
    return DGPResult(y=v[:, 0] * 0.0 + eps, x=v, z=z, beta_true=0.0,
                     theta=[0.0] * p, rho=0.0, meta={"dgp": "heavy_eps"})


def make_clustered_eps(n, q, p, rng, cluster_size=25, icc=0.2):
    """A1 violation: cluster-correlated structural error (first stage clean)."""
    z = rng.standard_normal((n, q))
    v = rng.standard_normal((n, p))
    n_cl = int(np.ceil(n / cluster_size))
    eff = rng.standard_normal(n_cl) * np.sqrt(icc / (1.0 - icc))
    cl_idx = np.repeat(np.arange(n_cl), cluster_size)[:n]
    eps = eff[cl_idx] + rng.standard_normal(n) * np.sqrt(1.0 - icc)
    eps = eps / np.std(eps)
    return DGPResult(y=v[:, 0] * 0.0 + eps, x=v, z=z, beta_true=0.0,
                     theta=[0.0] * p, rho=0.0, meta={"dgp": "clustered_eps"})


# ---------------------------------------------------------------------------
# Wild-bootstrap patch (memo Section 6.4 ladder; first stage only, no Y)


def wild_boot_cv(x, z, b_boot: int, rng, level: float = 0.05,
                 hetero_resid=None) -> float:
    """Rademacher wild bootstrap of first-stage residuals -> null CV.

    Under H0 (theta = 0) pseudo-X = P_Z x + w * x_resid with w rademacher;
    recomputes r2max of (pseudo-X, z) b_boot times; returns 1-level quantile.
    Uses (X, Z) only. For heteroskedastic profiles the row-wise residuals
    carry the variance pattern, so the patch adapts the null to E1.
    """
    xs, zs, _yr, _sc = prepare(x, z, None, None)
    n = xs.shape[0]
    pz_x = zs @ np.linalg.lstsq(zs, xs, rcond=None)[0]
    resid = xs - pz_x
    stats_boot = np.empty(b_boot)
    for b in range(b_boot):
        w = rng.choice([-1.0, 1.0], size=n)[:, None]
        xb = pz_x + w * resid
        ca_r = _canonical_pass(xb, zs)[0]
        stats_boot[b] = ca_r[0] ** 2
    return float(np.quantile(stats_boot, 1.0 - level, method="linear"))


# ---------------------------------------------------------------------------
# Grid definitions (PRUNED at WP-P3-R0; see preregistration memo)


THETAS_POWER12 = [0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.32, 0.45,
                  0.62, 0.80, 0.93]
THETAS_DECISIVE8 = [0.05, 0.10, 0.18, 0.28, 0.40, 0.55, 0.72, 0.88]
ALPHAS5 = [0.1, 0.3, 0.5, 0.7, 0.9]
KAPPAS2 = [0.5, 2.0]


def q_of(n: int, alpha: float) -> int:
    return int(round(alpha * (n - 1)))


def size_grid_cells():
    cells = []
    for n in (250, 500, 1000, 2000):
        ps = [1, 2, 5] + ([25, 100] if n >= 1000 else [])
        for a in ALPHAS5:
            for p in ps:
                q = q_of(n, a)
                if not (q > p - 1 and n > q + p):
                    continue
                cells.append({"cell_id": f"n{n}_a{a}_p{p}", "n": n, "p": p,
                              "q": q, "alpha": a})
    return cells


def power_grid_cells(include_r2=True):
    cells = []
    for n in (250, 1000):
        for a in (0.1, 0.5, 0.9):
            for p in (1, 5):
                q = q_of(n, a)
                if not (q > p - 1 and n > q + p):
                    continue
                cells.append({"cell_id": f"n{n}_a{a}_p{p}", "n": n, "p": p,
                              "q": q, "alpha": a})
    if include_r2:
        # open P2-R2 question: does a BBP-type threshold emerge when p grows?
        cells.append({"cell_id": "n1000_a0.5_p25_R2", "n": 1000, "p": 25,
                      "q": q_of(1000, 0.5), "alpha": 0.5})
    return cells


def decisive_grid_cells():
    cells = []
    for a in ALPHAS5:
        for kap in KAPPAS2:
            for p, n in ((1, 1000), (5, 2000)):
                q = q_of(n, a)
                if not (q > p - 1 and n > q + p):
                    continue
                cid = f"a{a}_k{kap}_none_p{p}"
                cells.append({"cell_id": cid, "n": n, "p": p, "q": q,
                              "alpha": a, "kappa": kap})
    return cells


def robustness_cells():
    cells = []
    for p in (1, 5):
        n, a = 1000, 0.5
        q = q_of(n, a)
        for viol in ("hetero_mild", "hetero_severe", "heavy_v", "heavy_eps",
                     "clustered"):
            cells.append({"cell_id": f"{viol}_p{p}", "n": n, "p": p, "q": q,
                          "alpha": a, "violation": viol})
    return cells


# ---------------------------------------------------------------------------
# Cell runners (write schema-compliant CSVs + _done markers + manifests)


def _results_dir(exp: str, out_root: str | None = None) -> str:
    root = out_root or os.environ.get("WEAKIV_RESULTS", RESULTS_ROOT_DEFAULT)
    d = os.path.join(root, exp)
    os.makedirs(os.path.join(d, "cells"), exist_ok=True)
    return d


def _hetero_profile(name: str, n: int) -> np.ndarray:
    if name == "mild":
        t = np.arange(1, n + 1)
        return 0.6 + 0.8 * (t / n)
    if name == "severe":
        t = np.arange(1, n + 1)
        return 0.25 + 2.75 * (t / n) ** 2
    raise ValueError(name)


def make_hetero_null(n, q, p, rng, profile: str):
    """H0 DGP with heteroskedastic first-stage noise (E1; theta = 0)."""
    w = _hetero_profile(profile, n)
    z = rng.standard_normal((n, q))
    v = rng.standard_normal((n, p)) * np.sqrt(w)[:, None]
    eps = rng.standard_normal(n)
    return DGPResult(y=v[:, 0] * 0.0 + eps, x=v, z=z, beta_true=0.0,
                     theta=[0.0] * p, rho=0.0,
                     meta={"dgp": "hetero_null", "profile": profile})


def _write_done(cell_dir: str, csv_name: str, meta: dict):
    import json
    csv_path = os.path.join(cell_dir, csv_name)
    done = {"csv": csv_name, "sha256": sha256_file(csv_path), **meta}
    with open(os.path.join(cell_dir, "_done_" + csv_name + ".json"), "w") as f:
        json.dump(done, f, indent=1)


def run_size_cell(cell: dict, big_b: int = 20000, b_cal_cv: int = 4000,
                  b_cal_pval: int = 4000, out_root: str | None = None,
                  master_seed: int = MASTER_SEED) -> dict:
    """X1: empirical size of T_spec under the A1-A3 null through the full
    data-level pipeline (null DGPs; no estimators)."""
    limit_threads()
    exp = "phase3_size_grid"
    cid, n, p, q = cell["cell_id"], cell["n"], cell["p"], cell["q"]
    t0 = time.time()
    cv_rng = stream(master_seed, exp, cid, "cv")
    cv = CellCV(n, q, p, b_cal_cv, cv_rng)
    r2max = np.empty(big_b)
    for b in range(big_b):
        rng = np.random.default_rng(
            cell_stream(exp, cid, b, master_seed=master_seed).integers(1 << 31))
        dgp = make_null(n, q, p, rng)
        xs, zs, _yr, _sc = prepare(dgp.x, dgp.z, None, None)
        r2max[b] = _canonical_pass(xs, zs)[0][0] ** 2
    rej_exact = int(np.sum(r2max > cv.cv_exact))
    rej_tw = int(np.sum(r2max > cv.cv_tw)) if cv.cv_tw is not None else -1
    rej_chi2 = int(np.sum(n * r2max > cv.cv_chi2))
    rows = [
        [exp, cid, n, p, q, cell["alpha"], "exact_jacobi", "johnstone2009",
         rej_exact, big_b, master_seed],
        [exp, cid, n, p, q, cell["alpha"], "tw", "johnstone2009",
         rej_tw, big_b, master_seed],
        [exp, cid, n, p, q, cell["alpha"], "naive_chi2", "none",
         rej_chi2, big_b, master_seed],
    ]
    d = _results_dir(exp, out_root)
    cdir = os.path.join(d, "cells")
    csv_path = os.path.join(cdir, f"{cid}.csv")
    with open(csv_path, "w") as f:
        f.write("experiment,cell_id,n,p,q,alpha,cv_method,correction,"
                "rejects,B,seed\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    np.save(os.path.join(cdir, f"{cid}_r2max.npy"), r2max)
    _write_done(cdir, f"{cid}.csv", {
        "wall_s": time.time() - t0, "b_cal_cv": b_cal_cv,
        "cv_exact": cv.cv_exact, "cv_tw": cv.cv_tw,
        "mu_np": cv.mu, "sigma_np": cv.sig, "big_b": big_b,
    })
    return {"cell_id": cid, "size_exact": rej_exact / big_b,
            "size_tw": (rej_tw / big_b) if rej_tw >= 0 else None,
            "size_chi2": rej_chi2 / big_b,
            "wall_s": time.time() - t0}


def run_power_cell(cell: dict, thetas=None, reps_per_theta: int = 300,
                   b_cal_cv: int = 4000, out_root: str | None = None,
                   master_seed: int = MASTER_SEED) -> dict:
    """X2: power curves of T_spec vs baselines along a fixed 12-point theta
    grid; outlier-location check against the affine lift map g."""
    limit_threads()
    exp = "phase3_power_surface"
    thetas = THETAS_POWER12 if thetas is None else thetas
    cid, n, p, q = cell["cell_id"], cell["n"], cell["p"], cell["q"]
    a = cell["alpha"]
    t0 = time.time()
    try:
        mu, sig = jacobi_mu_sigma(n, q, min(p, q))
    except RuntimeError:
        mu = sig = None
    rows = []
    raw = {}
    for theta in thetas:
        tid = f"{cid}_th{theta}"
        cv_rng = stream(master_seed, exp, tid, "cv")
        cv = CellCV(n, q, p, b_cal_cv, cv_rng)
        r2 = np.empty(reps_per_theta)
        f10 = 0
        for b in range(reps_per_theta):
            rng = np.random.default_rng(cell_stream(
                exp, tid, b, master_seed=master_seed).integers(1 << 31))
            dgp = make_single_spike(n, q, theta, 0.0, rng, p=p, beta=0.5)
            xs, zs, _yr, _sc = prepare(dgp.x, dgp.z, None, None)
            r2[b] = _canonical_pass(xs, zs)[0][0] ** 2
            f_stat = (r2[b] / q) / ((1.0 - r2[b]) / (n - q))
            f10 += int(f_stat > 10.0)
        g_pred = a + (1.0 - a) * theta
        med = float(np.median(r2))
        q25, q75 = np.quantile(r2, [0.25, 0.75], method="linear")
        loc_err = None
        if mu is not None and 0.0 < g_pred < 1.0:
            t_med = (np.log(med / (1.0 - med)) - mu) / sig
            t_pred = (np.log(g_pred / (1.0 - g_pred)) - mu) / sig
            loc_err = abs(t_med - t_pred)
        raw[tid] = r2
        rows.append([exp, cid, n, p, q, a, theta, 0.0,
                     float(np.mean(r2 > cv.cv_exact)),
                     float(np.mean(r2 > cv.cv_tw)) if cv.cv_tw else "",
                     med, float(q25), float(q75), g_pred, reps_per_theta,
                     master_seed])
        # extra columns appended below (f10 rate, location error)
        rows[-1] = rows[-1] + [f10 / reps_per_theta,
                               "" if loc_err is None else loc_err]
    d = _results_dir(exp, out_root)
    cdir = os.path.join(d, "cells")
    csv_path = os.path.join(cdir, f"{cid}.csv")
    with open(csv_path, "w") as f:
        f.write("experiment,cell_id,n,p,q,alpha,theta,rho,power_exact,"
                "power_tw,outlier_r2_median,outlier_r2_q25,outlier_r2_q75,"
                "g_pred,B,seed,power_f10,loc_err_sigma\n")
        for r in rows:
            f.write(",".join("" if v == "" else str(v) for v in r) + "\n")
    np.savez_compressed(os.path.join(cdir, f"{cid}_raw.npz"), **raw)
    _write_done(cdir, f"{cid}.csv", {
        "wall_s": time.time() - t0, "b_cal_cv": b_cal_cv,
        "reps_per_theta": reps_per_theta, "thetas": list(thetas),
    })
    return {"cell_id": cid, "wall_s": time.time() - t0}


def kp_rk_wald_f(xs, zs) -> float:
    """First-stage Wald summary / p for the "> 10" incumbent rule.

    Preregistered form (memo Section 6): in the HOMOSKEDASTIC decisive grid
    the HC0 sandwich and the classical covariance coincide in expectation,
    so the flag uses the classical per-column Wald statistics
    W_j = (n - q) * fittedSS_j / residSS_j, summed over columns and divided
    by p (~ chi2_1 mean under H0). The HC0 variant (kp_rk_wald_hc0) is
    retained for any heteroskedastic deployment.
    """
    n, q = zs.shape
    czz = zs.T @ zs
    czx = zs.T @ xs
    chol = np.linalg.cholesky(czz)
    gamma = np.linalg.solve(chol.T, np.linalg.solve(chol, czx))
    fitted_ss = np.sum(czx * gamma, axis=0)
    total_ss = np.sum(xs * xs, axis=0)
    resid_ss = np.maximum(total_ss - fitted_ss, 1e-300)
    walds = (n - q) * fitted_ss / resid_ss
    return float(np.sum(walds) / xs.shape[1])


def kp_rk_wald_hc0(xs, zs) -> float:
    """HC0 sandwich variant of kp_rk_wald_f (column-wise, summed / p)."""
    n, q = zs.shape
    p = xs.shape[1]
    czz = zs.T @ zs
    chol = np.linalg.cholesky(czz)
    gamma = np.linalg.solve(chol.T, np.linalg.solve(chol, zs.T @ xs))
    resid = xs - zs @ gamma
    total = 0.0
    for j in range(p):
        w = resid[:, j] ** 2
        meat = (zs * w[:, None]).T @ zs
        v_mat = np.linalg.solve(chol.T, np.linalg.solve(chol, meat))
        gj = gamma[:, j]
        total += float(gj @ np.linalg.solve(v_mat, gj))
    return total / p


def run_decisive_cell(cell: dict, reps: int = 400, b_cal_cv: int = 4000,
                      beta_true: float = 0.5, delta: float = 0.1,
                      rho_hi: float = 0.894, out_root: str | None = None,
                      master_seed: int = MASTER_SEED) -> dict:
    """X3+X4 shared decisive run: coverage map + risk curves per cell.

    Per replication: one fast_rep pass -> T_spec decision, theta_hat inversion,
    envelope/F/KP pass flags, AR acceptance + length (p=1), all estimator
    betas. Aggregates written per schema; raw betas kept as npz for paired
    analysis and bootstrap intervals.
    """
    limit_threads()
    exp = "phase3_decisive_grid"
    cid = cell["cell_id"]
    n, p, q = cell["n"], cell["p"], cell["q"]
    a, kappa = cell["alpha"], cell["kappa"]
    rho = rho_of_kappa(kappa)
    t0 = time.time()
    cv_rng = stream(master_seed, exp, cid, "cv")
    cv = CellCV(n, q, p, b_cal_cv, cv_rng)

    rho_env = (rho_hi / delta - 1.0) * a
    rho_env = rho_env / (1.0 + rho_env)          # A/(1+A) form
    k_list = sorted({min(max(k, 1), p) for k in {1, 2, p}})
    est_names = ["tsls", "liml", "fuller", "bekker", "jive", "whiten",
                 "pca_l"] + [f"trunc_k{k}" for k in k_list]

    rec = {name: np.empty((reps, p)) for name in est_names}
    tau_hat_v = np.empty(reps)
    r2max_v = np.empty(reps)
    lpc_v = np.empty(reps, dtype=int)
    ar_ok = np.zeros(reps, dtype=bool)
    ar_len = np.full(reps, -1.0)
    f_pass = np.zeros(reps, dtype=bool)
    kp_pass = np.zeros(reps, dtype=bool)
    env_pass = np.zeros(reps, dtype=bool)
    theta_hat_v = np.empty(reps)

    for b in range(reps):
        rng = np.random.default_rng(
            cell_stream(exp, cid, b, master_seed=master_seed).integers(1 << 31))
        dgp = make_single_spike(n, q, cell["theta"], rho, rng, p=p,
                                beta=beta_true)
        fr = fast_rep(dgp.y, dgp.x, dgp.z, estimators=True, k_list=k_list)
        r2max_v[b] = fr["r2max"]
        tau_hat_v[b] = fr["tau_hat"]
        for name in est_names:
            vec = np.asarray(fr["beta"][name], dtype=float).reshape(-1)
            rec[name][b, :len(vec)] = vec[:p]
            if len(vec) < p:
                rec[name][b, len(vec):] = np.nan
        ar_ok[b] = ar_accepts(dgp.y, dgp.x, dgp.z, beta_true,
                              chol=fr["chol"])
        if p == 1:
            lo, hi = ar_interval_p1_from(fr["xs"][:, 0], fr["zs"], fr["ys"],
                                         fr["chol"])
            ar_len[b] = hi - lo if np.isfinite(lo) and np.isfinite(hi) else -1.0
        f_stat = (fr["r2max"] / q) / ((1.0 - fr["r2max"]) / (n - q))
        f_pass[b] = f_stat > 10.0
        kp_pass[b] = fr["kp_wald"] > 10.0
        th_hat = min(max((n * fr["r2max"] - q) / (n - q), 0.0), 1.0 - 1e-12)
        theta_hat_v[b] = th_hat
        env_pass[b] = th_hat >= rho_env
        lpc_v[b] = int(fr["l_pc"]) if fr["l_pc"] is not None else -1

    d = _results_dir(exp, out_root)
    cdir = os.path.join(d, "cells")
    base_cols = [cid, n, p, q, a, kappa, "none"]

    cov_rows = []
    for rule, mask in (("F>10", f_pass), ("KP_rk>10", kp_pass),
                       ("spectral_env", env_pass)):
        nf = int(np.sum(mask))
        cov = float(np.mean(ar_ok[mask])) if nf > 0 else ""
        cov_rows.append(base_cols + [rule, cov, nf, reps, master_seed])
    cov_path = os.path.join(cdir, f"{cid}_coverage.csv")
    with open(cov_path, "w") as f:
        f.write("cell_id,n,p,q,alpha,kappa,het,rule,ar_cov_95,n_flagged,B,"
                "seed\n")
        for r in cov_rows:
            f.write(",".join("" if v == "" else str(v) for v in r) + "\n")

    err = {name: rec[name] - beta_true for name in est_names}
    risk_rows = []
    for name in est_names:
        e_b = err[name][:, 0]                    # component 1 (headline)
        rmse = float(np.sqrt(np.nanmean(e_b**2)))
        mae = float(np.nanmean(np.abs(e_b)))
        bias = float(np.nanmean(e_b))
        sd = float(np.sqrt(max(rmse**2 - bias**2, 0.0)))
        if name.startswith("trunc_k"):
            tau_used = int(name.split("k")[1]) / p
        elif name == "pca_l":
            tau_used = float(np.mean(lpc_v)) / q
        elif name == "whiten":
            tau_used = 0.05
        else:
            tau_used = ""
        risk_rows.append(base_cols + [cell["theta"], rho, name, rmse, mae,
                                      bias, sd, tau_used, reps, master_seed])
    risk_path = os.path.join(cdir, f"{cid}_risk.csv")
    with open(risk_path, "w") as f:
        f.write("cell_id,n,p,q,alpha,kappa,theta,rho_true,het,estimator,"
                "rmse,mae,bias,sd,tau_used,B,seed\n")
        for r in risk_rows:
            f.write(",".join("" if v == "" else str(v) for v in r) + "\n")
    np.savez_compressed(os.path.join(cdir, f"{cid}_raw.npz"),
                        **{f"beta_{k}": v for k, v in rec.items()},
                        tau_hat=tau_hat_v, r2max=r2max_v, theta_hat=theta_hat_v,
                        l_pc=lpc_v, ar_ok=ar_ok, ar_len=ar_len, f_pass=f_pass,
                        kp_pass=kp_pass, env_pass=env_pass)
    _write_done(cdir, f"{cid}_coverage.csv", {"wall_s": time.time() - t0})
    _write_done(cdir, f"{cid}_risk.csv", {
        "wall_s": time.time() - t0, "rho_env": rho_env, "delta": delta,
        "rho_hi": rho_hi, "b_cal_cv": b_cal_cv, "reps": reps,
        "estimators": est_names,
    })
    return {"cell_id": cid, "cov_f10": cov_rows[0][7],
            "cov_env": cov_rows[2][7], "flag_f10": int(np.sum(f_pass)),
            "flag_env": int(np.sum(env_pass)), "wall_s": time.time() - t0}


def run_robust_cell(cell: dict, big_b: int = 4000, patch_reps: int = 250,
                    b_boot: int = 99, out_root: str | None = None,
                    master_seed: int = MASTER_SEED) -> dict:
    """X5: size drift of T_spec under violations; wild-bootstrap patch (E1).

    The reference CV is the STANDARD exact-Jacobi value (the question is how
    the shipped rule behaves under violation); the patched CV re-centers the
    null by first-stage-only wild bootstrap.
    """
    limit_threads()
    exp = "phase3_robustness"
    cid = cell["cell_id"]
    n, p, q = cell["n"], cell["p"], cell["q"]
    viol = cell["violation"]
    t0 = time.time()

    def draw(b_index: int):
        rng = np.random.default_rng(cell_stream(
            exp, cid, b_index, master_seed=master_seed).integers(1 << 31))
        if viol == "hetero_mild":
            return make_hetero_null(n, q, p, rng, "mild")
        if viol == "hetero_severe":
            return make_hetero_null(n, q, p, rng, "severe")
        if viol == "heavy_v":
            return make_heavy_v(n, q, p, rng)
        if viol == "heavy_eps":
            return make_heavy_eps(n, q, p, rng)
        if viol == "clustered":
            return make_clustered_eps(n, q, p, rng)
        raise ValueError(viol)

    cv_rng = stream(master_seed, exp, cid, "cv")
    cv = CellCV(n, q, p, 4000, cv_rng)
    r2 = np.empty(big_b)
    ar_cov = 0
    for b in range(big_b):
        dgp = draw(b)
        xs, zs, _yr, _sc = prepare(dgp.x, dgp.z, None, None)
        r2[b] = _canonical_pass(xs, zs)[0][0] ** 2
        ar_cov += int(ar_accepts(dgp.y, dgp.x, dgp.z, 0.0))
    size_std = float(np.mean(r2 > cv.cv_exact))

    rows = [[cid, n, p, q, viol, "none", size_std,
             ar_cov / big_b, big_b, master_seed]]
    if viol in ("hetero_mild", "hetero_severe"):
        rej_patch = 0
        for b in range(patch_reps):
            dgp = draw(10_000 + b)
            boot_rng = np.random.default_rng(stream(
                master_seed, exp, cid, "boot", b).integers(1 << 31))
            cv_b = wild_boot_cv(dgp.x, dgp.z, b_boot, boot_rng)
            xs, zs, _yr, _sc = prepare(dgp.x, dgp.z, None, None)
            stat = _canonical_pass(xs, zs)[0][0] ** 2
            rej_patch += int(stat > cv_b)
        rows.append([cid, n, p, q, viol, "wild_boot",
                     rej_patch / patch_reps, "", patch_reps, master_seed])

    d = _results_dir(exp, out_root)
    cdir = os.path.join(d, "cells")
    csv_path = os.path.join(cdir, f"{cid}.csv")
    with open(csv_path, "w") as f:
        f.write("cell_id,n,p,q,violation,patch,size_5pct,ar_cov_95,B,seed\n")
        for r in rows:
            f.write(",".join("" if v == "" else str(v) for v in r) + "\n")
    _write_done(cdir, f"{cid}.csv", {
        "wall_s": time.time() - t0, "b_boot": b_boot,
        "patch_reps": patch_reps, "big_b": big_b,
    })
    return {"cell_id": cid, "size_std": size_std, "wall_s": time.time() - t0}


def run_scaling(out_root: str | None = None,
                master_seed: int = MASTER_SEED) -> dict:
    """X6: workflow at target application scales.

    Individual mode: null-DGP canonical spectrum at MR-like n; summary-stats
    mode: eigendecomposition timing on synthetic spiked-LD Gram matrices up
    to q = 5000. Rows follow the scaling schema.
    """
    import resource

    limit_threads()
    exp = "phase3_scaling"
    t0 = time.time()
    machine = os.uname().nodename[:40]
    rows = []
    for (n, q, p) in ((10_000, 100, 5), (30_000, 300, 5), (100_000, 500, 5)):
        rng = np.random.default_rng(
            stream(master_seed, exp, "ind", n).integers(1 << 31))
        dgp = make_null(n, q, p, rng)
        t1 = time.time()
        xs, zs, _yr, _sc = prepare(dgp.x, dgp.z, None, None)
        _r1 = _canonical_pass(xs, zs)[0][0]
        secs = time.time() - t1
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        rows.append([n, p, q, "individual", round(secs, 3),
                     round(peak_gb, 3), machine, master_seed])
    from scipy.linalg import eigh as scipy_eigh
    for q in (500, 1000, 2000, 3500, 5000):
        rng = np.random.default_rng(
            stream(master_seed, exp, "ld", q).integers(1 << 31))
        loadings = rng.standard_normal((q, 5))
        noise = rng.standard_normal((q, q)) * (1.0 / np.sqrt(q))
        sigma = loadings @ loadings.T + noise @ noise.T \
            + 0.05 * np.eye(q) * np.sqrt(q)
        t1 = time.time()
        _vals = scipy_eigh(sigma, subset_by_index=(q - 10, q - 1),
                           eigvals_only=True)
        secs = time.time() - t1
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        rows.append(["", "", q, "summary_stats", round(secs, 3),
                     round(peak_gb, 3), machine, master_seed])

    d = _results_dir(exp, out_root)
    cdir = os.path.join(d, "cells")
    csv_path = os.path.join(cdir, "scaling_suite.csv")
    with open(csv_path, "w") as f:
        f.write("n,p,q,mode,seconds,peak_gb,machine,seed\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    _write_done(cdir, "scaling_suite.csv",
                {"wall_s": time.time() - t0})
    return {"cell_id": "scaling_suite", "rows": len(rows),
            "wall_s": time.time() - t0}


def main():
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="spectraliv Phase-3 runners")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap_run = sub.add_parser("run")
    ap_run.add_argument("--exp", required=True,
                        choices=["size_grid", "power_surface", "decisive_grid",
                                 "robustness"])
    ap_run.add_argument("--cells", default="all")
    ap_run.add_argument("--B", type=int, default=None)
    ap_run.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.exp == "size_grid":
        cells = {c["cell_id"]: c for c in size_grid_cells()}
        runner, default_b = run_size_cell, 20000
    elif args.exp == "power_surface":
        cells = {c["cell_id"]: c for c in power_grid_cells()}
        runner, default_b = run_power_cell, 300
    elif args.exp == "decisive_grid":
        cells = {}
        for th in THETAS_DECISIVE8:
            for c in decisive_grid_cells():
                cc = dict(c)
                cc["cell_id"] = c["cell_id"] + f"_th{th}"
                cc["theta"] = th
                cells[cc["cell_id"]] = cc
        runner, default_b = run_decisive_cell, 400
    else:
        cells = {c["cell_id"]: c for c in robustness_cells()}
        runner, default_b = run_robust_cell, 4000

    ids = list(cells) if args.cells == "all" else args.cells.split(",")
    for cid in ids:
        kwargs = {"out_root": args.out}
        if args.B is not None:
            kwargs["reps" if args.exp == "decisive_grid" else
                   ("reps_per_theta" if args.exp == "power_surface"
                    else "big_b")] = args.B
        res = runner(cells[cid], **kwargs) if args.exp != "decisive_grid" \
            else runner(cells[cid], **kwargs)
        print(_json.dumps(res))


if __name__ == "__main__":
    main()
