"""T_spec: spectral relevance test with exact-Jacobi and Tracy-Widom calibration.

Implements formalization_memo.md Section 5. No oracle quantities are used.
Statistic: t_tw = (logit(r_max^2) - mu)/sigma with Johnstone (2009) finite-N
constants mapped as his (m, n) := (N - q, q), uncentered baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .canoncorr import canonical_analysis
from .jacobiquantiles import (
    largest_root_pvalue,
    largest_root_quantile,
)
from .preprocess import assert_proper, prepare
from .tw import default_tw1


class TWBoundaryError(RuntimeError):
    """Raised when phi + gamma hits the pi boundary (logit map degenerates)."""


def jacobi_mu_sigma(n: int, q: int, p: int, centered: bool = False):
    """Johnstone 2009 eqs (4)-(5): mu and sigma for W(p,m,n) = logit(lambda_1).

    Mapping (toolkit F3): his m := error df = N - q (- shift), his n := q.
    """
    m = n - q - (1 if centered else 0)
    hyp = q
    mn1 = m + hyp - 1
    s2g = (min(p, hyp) - 0.5) / mn1
    s2f = (max(p, hyp) - 0.5) / mn1
    if not (0 < s2g < 1 and 0 < s2f < 1):
        raise TWBoundaryError(f"sin^2 arguments out of range: {s2g}, {s2f} (A3 violated?)")
    # sin^2(angle/2) = s2  =>  angle = 2 arcsin(sqrt(s2))   [factor 2 critical]
    gamma = 2.0 * np.arcsin(np.sqrt(s2g))
    phi = 2.0 * np.arcsin(np.sqrt(s2f))
    if phi + gamma >= np.pi - 1e-12:
        raise TWBoundaryError("phi + gamma at pi boundary; use exact Jacobi CVs")
    mu = 2.0 * np.log(np.tan((phi + gamma) / 2.0))
    sigma_cubed = (16.0 / mn1**2) / (np.sin(phi + gamma) * np.sin(phi) * np.sin(gamma))
    return mu, float(sigma_cubed ** (1.0 / 3.0))


@dataclass
class SpecTestResult:
    r2max: float
    t_tw: float | None
    mu_np: float | None
    sigma_np: float | None
    cv_exact: float
    cv_tw: float | None
    p_exact: float
    p_tw: float | None
    reject_exact: bool
    reject_tw: bool | None
    meta: dict = field(default_factory=dict)


def spec_test(x, z, w=None, level: float = 0.05, b_cal: int = 2000,
              rng=None, centered: bool = False, master_seed: int | None = None,
              canon=None) -> SpecTestResult:
    """Full T_spec pipeline. `canon` allows sharing a CanonicalAnalysis pass."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    n, p = x.shape
    q = z.shape[1]
    assert_proper(n, q, p, centered=centered)

    if canon is None:
        xs, zs, _yr, _sc = prepare(x, z, None, w)
        canon = canonical_analysis(xs, zs)
    r2max = float(canon.r[0] ** 2)

    # primary critical value: exact finite-N Jacobi
    if rng is None:
        if master_seed is None:
            rng = np.random.default_rng()
        else:
            from .rng import stream
            rng = stream(master_seed, "spec_test_cv")
    cv_exact = float(largest_root_quantile(1.0 - level, n, q, p, b_cal, rng,
                                           centered=centered)[0])
    p_exact = float(largest_root_pvalue(r2max, n, q, p, b_cal, rng,
                                        centered=centered))

    # secondary: TW approximation in logit space
    t_tw = cv_tw = p_tw = None
    mu = sig = None
    try:
        mu, sig = jacobi_mu_sigma(n, q, p, centered=centered)
        l2 = np.log(r2max / (1.0 - r2max)) if 0.0 < r2max < 1.0 else np.inf
        t_tw = float((l2 - mu) / sig)
        f_level = default_tw1().ppf(1.0 - level)
        thr = mu + f_level * sig
        cv_tw = float(np.exp(thr) / (1.0 + np.exp(thr)))
        p_tw = float(default_tw1().sf(t_tw))
    except TWBoundaryError:
        pass

    return SpecTestResult(
        r2max=r2max,
        t_tw=t_tw,
        mu_np=mu if t_tw is not None else None,
        sigma_np=sig if t_tw is not None else None,
        cv_exact=cv_exact,
        cv_tw=cv_tw,
        p_exact=p_exact,
        p_tw=p_tw,
        reject_exact=bool(r2max > cv_exact),
        reject_tw=(None if t_tw is None else bool(r2max > cv_tw)),
        meta={"n": n, "p": p, "q": q, "level": level, "b_cal": b_cal},
    )


# ---------------------------------------------------------------------------
# Naive references used by Phase-3 baselines (never by T_spec itself).

def naive_chi2_stat(r2max: float, n: int) -> float:
    """Naive chi-squared reference: N * r^2 vs chi^2_q under H0 (WRONG tail at alpha>0)."""
    return n * r2max


def naive_f_pvalue(r2max: float, n: int, q: int, centered: bool = False) -> float:
    """Exact central-F p-value of the first-stage F statistic under H0.

    F = (r^2/q) / ((1-r^2)/(N-q)) ~ F(q, N-q) exactly (toolkit F1 identity).
    This is the CORRECT null for F itself; naive here refers to using it as a
    largest-root reference without the Jacobi geometry (identical at p = 1).
    """
    from scipy import stats as st

    shift = 1 if centered else 0
    dfd = n - shift - q
    f = (r2max / q) / ((1.0 - r2max) / dfd)
    return float(st.f.sf(f, q, dfd))


def stock_yogo_interpolated_cv(q: int, target_alpha_size: float = 0.10) -> dict:
    """Placeholder for Phase-3 baseline: Stock-Yogo noncentral-F table lookup.

    Deliberately NOT implemented in Phase 2 (their tables are design-specific);
    Phase 3 X3 implements the comparison via published critical values.
    """
    raise NotImplementedError("Stock-Yogo tables enter at Phase 3 (WP-P3-S2 baselines)")
