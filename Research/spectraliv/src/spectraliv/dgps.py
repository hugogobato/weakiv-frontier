"""Data-generating processes for Phase 3 (formalization memo Sections 1, 4).

Canonical SEM parameterization (fixes Phase-1 O1 Finding 1):
    X = Z Pi sqrt(Gamma) + V,  Gamma_jj = theta_j/(1-theta_j),
    rows of Z ~ N(0, I_q), V rows iid N(0, I_p),
    eps = rho v_1 + sqrt(1-rho^2) u,   Y = beta X_1 + eps.

rho is the scale-free structural endogeneity; kappa from the plan maps to
rho = kappa/sqrt(1+kappa^2). Alignment (Pi) is random unless seeded otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DGPResult:
    y: np.ndarray
    x: np.ndarray
    z: np.ndarray
    beta_true: float | np.ndarray
    theta: list
    rho: float
    meta: dict = field(default_factory=dict)


def rho_of_kappa(kappa: float) -> float:
    if np.isinf(kappa):
        return 1.0
    return float(kappa / np.sqrt(1.0 + kappa**2))


def make_null(n: int, q: int, p: int, rng: np.random.Generator,
              beta: float = 0.0) -> DGPResult:
    """Joint irrelevance H0: X independent of Z (theta = 0), standardized."""
    z = rng.standard_normal((n, q))
    v = rng.standard_normal((n, p))
    x = v
    eps = rng.standard_normal(n)
    y = x[:, 0] * beta + eps
    return DGPResult(y=y, x=x, z=z, beta_true=beta, theta=[0.0] * p, rho=0.0,
                     meta={"dgp": "null"})


def _random_pi(q: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """m orthonormal instrument-side directions drawn uniformly (Haar)."""
    a_mat = rng.standard_normal((q, m))
    pi_mat, _ = np.linalg.qr(a_mat)
    return pi_mat


def make_single_spike(n: int, q: int, theta: float, rho: float,
                      rng: np.random.Generator, p: int = 1,
                      beta: float = 0.5,
                      hetero: str | None = None) -> DGPResult:
    """One population canonical correlation theta; remaining roots zero.

    hetero in {None, 'quadratic', 'two_group'} builds an E1 variance profile on
    the first-stage noise v (used by Phase-3 X5).
    """
    if not (0.0 < theta < 1.0):
        raise ValueError("theta in (0,1)")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho = Corr(eps, v) must be in [-1, 1]")
    gamma = theta / (1.0 - theta)
    z = rng.standard_normal((n, q))
    pi_mat = _random_pi(q, max(p, 1), rng)
    pi1 = pi_mat[:, 0]
    if hetero is None:
        v = rng.standard_normal((n, p))
    elif hetero == "quadratic":
        t_idx = np.arange(1, n + 1)
        w_prof = 0.25 + 2.75 * (t_idx / n) ** 2
        v = rng.standard_normal((n, p)) * np.sqrt(w_prof)[:, None]
    elif hetero == "two_group":
        half = n // 2
        w_prof = np.concatenate([np.ones(half), np.full(n - half, 4.0)])
        rng.shuffle(w_prof)
        v = rng.standard_normal((n, p)) * np.sqrt(w_prof)[:, None]
    else:
        raise ValueError(f"unknown hetero profile {hetero}")
    # Columns 2..p are already pure iid noise (population null roots):
    # X_j = v_j is independent of Z for j >= 2. CORRECTION RECORD (Phase 3,
    # no-silent-repairs rule): the drafted version drew ADDITIONAL extra
    # columns and stacked them onto the p existing ones (yielding 2p-1
    # columns) after an orthogonalization step that itself crashed for every
    # p > 1 ((n, p-1) @ (n, 1) matmul). Neither branch was ever exercised in
    # Phase 2 (smoke runner used the p=1 default throughout).
    x = np.sqrt(gamma) * (z @ pi1)[:, None] + v
    v1 = v[:, 0]
    u = rng.standard_normal(n)
    eps = rho * v1 + np.sqrt(max(1e-12, 1.0 - rho**2)) * u
    y = beta * x[:, 0] + eps
    return DGPResult(y=y, x=x, z=z, beta_true=beta,
                     theta=[theta] + [0.0] * (p - 1), rho=rho,
                     meta={"dgp": "single_spike", "hetero": hetero})


def make_multispike(n: int, q: int, thetas: list, rho: float,
                    rng: np.random.Generator, beta: float = 0.5) -> DGPResult:
    """R2 regime: several spikes with distinct strengths (orthogonal alignments)."""
    p = len(thetas)
    gamma = np.asarray(thetas) / (1.0 - np.asarray(thetas))
    z = rng.standard_normal((n, q))
    pi_mat = _random_pi(q, p, rng)
    signal = z @ (pi_mat * np.sqrt(gamma)[None, :])
    v = rng.standard_normal((n, p))
    x = signal + v
    u = rng.standard_normal(n)
    eps = rho * v[:, 0] + np.sqrt(max(1e-12, 1.0 - rho**2)) * u
    y = beta * x[:, 0] + eps
    return DGPResult(y=y, x=x, z=z, beta_true=beta, theta=list(thetas), rho=rho,
                     meta={"dgp": "multispike"})
