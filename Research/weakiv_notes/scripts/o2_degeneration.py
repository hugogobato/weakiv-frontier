"""WP-P1-B1 (O2 witness): p=1 degeneration of the largest-root canonical-correlation statistic.

Project: Weak-Instrument Frontier (Idea 2), Phase 1. Run date: 2026-08-23.
Master seed: 20260823 (SeedSequence spawn per cell; manifest.json logs every stream).
Conventions: no intercept. F = (r2/q)/((1-r2)/(n-q)) so under H0 F ~ F(q, n-q) exactly.
Outputs: Research/weakiv_results/phase1_o2/*.csv + manifest.json; figures in Research/weakiv_notes/figs/.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import json
import time
from multiprocessing import Pool

import numpy as np
import scipy
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(ROOT, "Research", "weakiv_results", "phase1_o2")
FIGS = os.path.join(ROOT, "Research", "weakiv_notes", "figs")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

MASTER_SEED = 20260823
ALPHAS = np.round(np.arange(0.1, 0.91, 0.1), 1)
NS = [250, 1000]
B_NULL = 2000
B_POWER = 1000
THETAS = np.round(np.linspace(0.02, 0.95, 20), 4)
KAPPA_HARM = 1.0
DELTA = 0.1
F_FOLK = 10.0


def q_of(n, alpha):
    q = int(round(alpha * (n - 1)))
    q = min(q, n - 5)
    return max(q, 2)


def solve_chol(L, U):
    W = np.linalg.solve(L, U)
    return np.linalg.solve(L.T, W)


def run_cell(args):
    n, alpha, seed = args
    q = q_of(n, alpha)
    rng = np.random.default_rng(seed)

    r2_null = np.empty(B_NULL)
    F_null = np.empty(B_NULL)
    for b in range(B_NULL):
        Z = rng.standard_normal((n, q))
        x = rng.standard_normal(n)
        L = np.linalg.cholesky(Z.T @ Z)
        u = Z.T @ x
        w = solve_chol(L, u)
        xpx = x @ x
        r2 = (u @ w) / xpx
        r2_null[b] = r2
        F_null[b] = (r2 / q) / ((1.0 - r2) / (n - q))

    T = len(THETAS)
    sq = np.sqrt(THETAS)
    sq1 = np.sqrt(1.0 - THETAS)
    r2_alt = np.empty((B_POWER, T))
    F_alt = np.empty((B_POWER, T))
    for b in range(B_POWER):
        Z = rng.standard_normal((n, q))
        pi = rng.standard_normal(q)
        pi /= np.linalg.norm(pi)
        s = Z @ pi
        V = rng.standard_normal((n, T))
        X = sq * s[:, None] + sq1 * V
        L = np.linalg.cholesky(Z.T @ Z)
        U = Z.T @ X
        W = solve_chol(L, U)
        Xh = Z @ W
        xpx = np.sum(X * X, axis=0)
        xhpx = np.sum(Xh * X, axis=0)
        r2 = xhpx / xpx
        r2_alt[b] = r2
        F_alt[b] = (r2 / q) / ((1.0 - r2) / (n - q))

    a_beta = q / 2.0
    b_beta = (n - q) / 2.0
    crit = stats.beta.ppf(0.95, a_beta, b_beta)
    power = np.mean(r2_alt > crit, axis=0)
    folk_power = np.mean(F_alt > F_FOLK, axis=0)

    def crossing(curve, level):
        idx = np.nonzero(curve >= level)[0]
        if len(idx) == 0:
            return float("nan")
        i = idx[0]
        if i == 0:
            return float(THETAS[0])
        t0, t1 = THETAS[i - 1], THETAS[i]
        p0, p1 = curve[i - 1], curve[i]
        return float(t0 + (level - p0) * (t1 - t0) / (p1 - p0)) if p1 > p0 else float(t1)

    theta_det80 = crossing(power, 0.80)
    theta_folk80 = crossing(folk_power, 0.80)

    ks = stats.kstest(r2_null, "beta", args=(a_beta, b_beta))
    rho = stats.spearmanr(F_null, r2_null).statistic
    qq_ps = np.linspace(0.01, 0.99, 199)
    qq_theory = stats.beta.ppf(qq_ps, a_beta, b_beta)
    qq_mc = np.quantile(r2_null, qq_ps)

    A = alpha * (KAPPA_HARM / DELTA - 1.0)
    theta_harm = A / (1.0 + A)

    return {
        "n": n,
        "alpha": alpha,
        "q": q,
        "seed": seed,
        "F_q95_exact": stats.f.ppf(0.95, q, n - q),
        "F_q95_chi2naive": stats.chi2.ppf(0.95, q) / q,
        "pF_gt10_null": stats.f.sf(F_FOLK, q, n - q),
        "r2_q95_beta": crit,
        "r2_q95_mc": float(np.quantile(r2_null, 0.95)),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "spearman_F_r2": float(rho),
        "theta_det80": theta_det80,
        "theta_folk80": theta_folk80,
        "theta_harm_k1_d0.1": theta_harm,
        "r2_alt_median": [float(np.median(r2_alt[:, j])) for j in range(T)],
        "power": [float(p) for p in power],
        "folk_power": [float(p) for p in folk_power],
        "qq_theory": [float(v) for v in qq_theory],
        "qq_mc": [float(v) for v in qq_mc],
    }


def main():
    ss = np.random.SeedSequence(MASTER_SEED)
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(len(NS) * len(ALPHAS))]
    cells = [(n, float(a), seeds[i]) for i, (n, a) in enumerate([(n, a) for n in NS for a in ALPHAS])]

    t0 = time.time()
    with Pool(processes=min(6, os.cpu_count() or 4)) as pool:
        results = pool.map(run_cell, cells)
    wall = time.time() - t0

    cols = [
        "n", "alpha", "q", "seed", "F_q95_exact", "F_q95_chi2naive", "pF_gt10_null",
        "r2_q95_beta", "r2_q95_mc", "ks_stat", "ks_p", "spearman_F_r2",
        "theta_det80", "theta_folk80", "theta_harm_k1_d0.1",
    ]
    with open(os.path.join(RES, "o2_thresholds.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    with open(os.path.join(RES, "o2_power_grid.json"), "w") as f:
        json.dump(
            [
                {"n": r["n"], "alpha": r["alpha"], "thetas": list(THETAS),
                 "power": r["power"], "folk_power": r["folk_power"],
                 "r2_alt_median": r["r2_alt_median"]}
                for r in results
            ],
            f,
        )

    manifest = {
        "master_seed": MASTER_SEED,
        "cell_seeds": {f"n{r['n']}_a{r['alpha']}": r["seed"] for r in results},
        "B_null": B_NULL,
        "B_power": B_POWER,
        "thetas": list(THETAS),
        "wall_seconds": wall,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "executed": "2026-08-23",
    }
    with open(os.path.join(RES, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    show = [r for r in results if r["n"] == 1000 and r["alpha"] in (0.1, 0.3, 0.5, 0.9)]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    grid = np.linspace(1e-4, 0.999, 400)
    for r in show:
        q = r["q"]
        ax[0].plot(grid, (grid / q) / ((1 - grid) / (r["n"] - q)), lw=1,
                   label=f"alpha={r['alpha']}, q={q}")
    ax[0].set_xlabel("largest squared canonical correlation $r^2$")
    ax[0].set_ylabel("first-stage $F$")
    ax[0].set_title("F is a monotone transform of $r^2$ (p=1)")
    ax[0].legend(fontsize=6)
    min_rho = min(r["spearman_F_r2"] for r in results)
    ax[1].axis("off")
    ax[1].text(0.02, 0.9, f"min Spearman rank corr(F, r^2) over all {len(results)} cells: {min_rho:.6f}", fontsize=9)
    ax[1].text(0.02, 0.75, "=> for p=1 the largest-root statistic and the first-stage F", fontsize=9)
    ax[1].text(0.02, 0.65, "carry identical information (exact algebraic monotone map).", fontsize=9)
    ax[1].text(0.02, 0.45, "Any calibration difference can only enter through the", fontsize=9)
    ax[1].text(0.02, 0.35, "critical CONSTANT, never through the statistic.", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_o2_collapse.png"))
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for n in NS:
        rows = [r for r in results if r["n"] == n]
        ax[0].plot(rows[0]["alpha"] if False else [r["alpha"] for r in rows],
                   [r["F_q95_exact"] / r["F_q95_chi2naive"] for r in rows], "o-",
                   label=f"n={n}")
    ax[0].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("alpha = q/n")
    ax[0].set_ylabel("exact 95% F quantile / naive chi2_q quantile")
    ax[0].set_title("Naive chi-squared reference degrades with alpha")
    ax[0].legend()
    r1k = sorted([r for r in results if r["n"] == 1000], key=lambda r: r["alpha"])
    a = [r["alpha"] for r in r1k]
    ax[1].plot(a, [r["theta_det80"] for r in r1k], "o-", label="theta_det (80% power, exact 5% test)")
    ax[1].plot(a, [r["theta_folk80"] for r in r1k], "s-", label="theta at which P(F>10)=80%")
    ax[1].plot(a, [r["theta_harm_k1_d0.1"] for r in r1k], "^--", label="theta_harm (kappa=1, delta=0.1)")
    ax[1].plot(a, a, ":", color="gray", label="Wachter bulk edge ~ alpha (outlier emergence)")
    ax[1].set_xlabel("alpha = q/n")
    ax[1].set_ylabel("population theta")
    ax[1].set_title("Detectability vs harm vs folklore threshold (n=1000, p=1)")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_o2_power_thresholds.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax_, r in zip(axes, [r for r in results if r["n"] == 1000 and r["alpha"] in (0.1, 0.5, 0.9)]):
        ax_.plot(r["qq_theory"], r["qq_mc"], ".", ms=2)
        ax_.plot([0, 1], [0, 1], "k:", lw=0.7)
        ax_.set_title(f"n={r['n']}, alpha={r['alpha']} (KS p={r['ks_p']:.2f})", fontsize=8)
        ax_.set_xlabel("Beta quantile")
        ax_.set_ylabel("MC quantile")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_o2_beta_qq.png"))
    plt.close(fig)

    print(f"done in {wall:.1f}s; results in {RES}")


if __name__ == "__main__":
    main()
