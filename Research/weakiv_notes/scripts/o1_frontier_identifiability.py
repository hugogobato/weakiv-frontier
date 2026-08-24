"""WP-P1-B1 (O1 witness): identifiability of the harm side of the weak-instrument frontier.

Project: Weak-Instrument Frontier (Idea 2), Phase 1. Run date: 2026-08-23.
Master seed: 20260823.
SEM (canonical form, Var(v)=1): X = Z*pi_hat*sqrt(gamma) + v with gamma = theta/(1-theta),
eps = (kappa*v + u)/sqrt(1+kappa^2); rho = Corr(eps,v) = kappa/sqrt(1+kappa^2).
Working bias approximation (verified against MC here): E[beta_2SLS] - beta ~= rho/(1 + mu2/q),
mu2 = n*theta/(1-theta); harm boundary theta_harm(A) = A/(1+A), A = alpha*(rho/delta - 1).
Outputs: Research/weakiv_results/phase1_o1/*.csv + manifest.json; figs in Research/weakiv_notes/figs/.
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
RES = os.path.join(ROOT, "Research", "weakiv_results", "phase1_o1")
FIGS = os.path.join(ROOT, "Research", "weakiv_notes", "figs")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

MASTER_SEED = 20260823
NS = [500, 2000]
ALPHAS = [0.1, 0.3, 0.5]
KAPPAS = [0.2, 0.5, 1.0, 2.0]
DELTA = 0.1
THETAS = np.round(np.linspace(0.05, 0.85, 12), 4)
B = 1000
KAPPA_HI = 2.0


def q_of(n, alpha):
    return max(int(round(alpha * (n - 1))), 2)


THETAS_CAL = np.concatenate([np.linspace(0.01, 0.98, 40), [0.985, 0.99, 0.995, 0.999]])
B_CAL = 400


def calibrate_r2_curve(n, alpha, seed):
    """Median r_max^2 as a function of population theta (first-stage-only information)."""
    q = q_of(n, alpha)
    rng = np.random.default_rng(seed)
    Tc = len(THETAS_CAL)
    sqg = np.sqrt(THETAS_CAL / (1.0 - THETAS_CAL))
    out = np.empty((B_CAL, Tc))
    for b in range(B_CAL):
        Z = rng.standard_normal((n, q))
        pi = rng.standard_normal(q)
        pi /= np.linalg.norm(pi)
        v = rng.standard_normal(n)
        X = sqg * (Z @ pi)[:, None] + v[:, None]
        L = np.linalg.cholesky(Z.T @ Z)
        W = np.linalg.solve(L, Z.T @ X)
        W = np.linalg.solve(L.T, W)
        Xh = Z @ W
        out[b] = np.sum(Xh * X, axis=0) / np.sum(X * X, axis=0)
    g = np.median(out, axis=0)
    g = np.maximum.accumulate(g)
    keep = np.concatenate([[True], np.diff(g) > 1e-12])
    return g[keep], THETAS_CAL[keep]


def mu2_of(theta, n):
    return n * theta / (1.0 - theta)


def rho_of(kappa):
    return kappa / np.sqrt(1.0 + kappa**2)


def bias_formula(theta, kappa, n, q):
    return rho_of(kappa) / (1.0 + mu2_of(theta, n) / q)


def theta_harm(alpha, kappa, delta):
    r = rho_of(kappa)
    if r <= delta:
        return 0.0
    A = alpha * (r / delta - 1.0)
    return A / (1.0 + A)


def run_cell(args):
    n, alpha, seed = args
    q = q_of(n, alpha)
    ss = np.random.SeedSequence(seed)
    main_seed, cal_seed = [int(s.generate_state(1)[0]) for s in ss.spawn(2)]
    g_cal, th_cal = calibrate_r2_curve(n, alpha, cal_seed)
    rng = np.random.default_rng(main_seed)
    T = len(THETAS)
    K = len(KAPPAS)
    gam = THETAS / (1.0 - THETAS)

    beta_hat = np.empty((B, T, K))
    r2_obs = np.empty((B, T))
    for b in range(B):
        Z = rng.standard_normal((n, q))
        pi = rng.standard_normal(q)
        pi /= np.linalg.norm(pi)
        v = rng.standard_normal(n)
        U = rng.standard_normal((n, T))
        X = np.sqrt(gam) * (Z @ pi)[:, None] + v[:, None]
        L = np.linalg.cholesky(Z.T @ Z)
        W = np.linalg.solve(L, Z.T @ X)
        W = np.linalg.solve(L.T, W)
        Xh = Z @ W
        xhpx = np.sum(Xh * X, axis=0)
        xhpxh = np.sum(Xh * Xh, axis=0)
        r2_obs[b] = xhpx / np.sum(X * X, axis=0)
        denom = xhpxh
        for k, kap in enumerate(KAPPAS):
            u = U[:, T // 2]
            eps = (kap * v + u) / np.sqrt(1.0 + kap**2)
            beta_hat[b, :, k] = (Xh.T @ eps) / denom

    bias_mc = beta_hat.mean(axis=0)
    bias_se = beta_hat.std(axis=0, ddof=1) / np.sqrt(B)

    return {
        "n": n,
        "alpha": alpha,
        "q": q,
        "seed": seed,
        "bias_mc": bias_mc,
        "bias_se": bias_se,
        "bias_form": np.array([[bias_formula(t, kap, n, q) for t in THETAS] for kap in KAPPAS]),
        "r2_median": np.median(r2_obs, axis=0),
        "r2_sd": r2_obs.std(axis=0, ddof=1),
        "r2_obs": r2_obs,
        "theta_hat_inv": np.interp(r2_obs, g_cal, th_cal),
    }


def main():
    ss = np.random.SeedSequence(MASTER_SEED)
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(len(NS) * len(ALPHAS))]
    cells = [(n, a, seeds[i]) for i, (n, a) in enumerate([(n, a) for n in NS for a in ALPHAS])]

    t0 = time.time()
    with Pool(processes=min(6, os.cpu_count() or 4)) as pool:
        results = pool.map(run_cell, cells)
    wall = time.time() - t0

    with open(os.path.join(RES, "bias_surface.csv"), "w") as f:
        f.write("n,alpha,q,kappa,rho,theta,bias_mc,bias_se,bias_formula,rel_gap\n")
        for r in results:
            for ki, kap in enumerate(KAPPAS):
                for ti, th in enumerate(THETAS):
                    bm = r["bias_mc"][ti, ki]
                    bf = r["bias_form"][ki, ti]
                    gap = (bm - bf) / abs(bf) if abs(bf) > 1e-12 else float("nan")
                    f.write(f"{r['n']},{r['alpha']},{r['q']},{kap},{rho_of(kap):.4f},{th},"
                            f"{bm:.6f},{r['bias_se'][ti, ki]:.6f},{bf:.6f},{gap:.4f}\n")

    env_rows = []
    for r in results:
        th_env = theta_harm(r["alpha"], KAPPA_HI, DELTA)
        frac_flag_plugin = np.mean(r["r2_obs"] < th_env, axis=0)
        frac_flag_inv = np.mean(r["theta_hat_inv"] < th_env, axis=0)
        flag_oracle = THETAS < th_env
        for ki, kap in enumerate(KAPPAS):
            true_harm = r["bias_mc"][:, ki] > DELTA
            tpr = float(np.mean(frac_flag_plugin[true_harm])) if true_harm.any() else float("nan")
            fpr = float(np.mean(frac_flag_plugin[~true_harm])) if (~true_harm).any() else float("nan")
            tpr_i = float(np.mean(frac_flag_inv[true_harm])) if true_harm.any() else float("nan")
            fpr_i = float(np.mean(frac_flag_inv[~true_harm])) if (~true_harm).any() else float("nan")
            tpr_oracle = float(np.mean(flag_oracle[true_harm])) if true_harm.any() else float("nan")
            fpr_oracle = float(np.mean(flag_oracle[~true_harm])) if (~true_harm).any() else float("nan")
            env_rows.append((r["n"], r["alpha"], kap, tpr, fpr, tpr_i, fpr_i,
                             tpr_oracle, fpr_oracle, float(r["r2_sd"].mean())))
    with open(os.path.join(RES, "envelope_coverage.csv"), "w") as f:
        f.write("n,alpha,kappa,tpr_plugin,fpr_plugin,tpr_inverted,fpr_inverted,"
                "tpr_oracle,fpr_oracle,mean_r2_sd\n")
        for row in env_rows:
            f.write(",".join(f"{v:.4f}" for v in row) + "\n")

    overall_tpr = np.nanmean([row[3] for row in env_rows])
    overall_tpr_i = np.nanmean([row[5] for row in env_rows])
    overall_tpr_o = np.nanmean([row[7] for row in env_rows])
    print(f"envelope TPR plugin: {overall_tpr:.4f}; inverted: {overall_tpr_i:.4f}; oracle: {overall_tpr_o:.4f}")

    manifest = {
        "master_seed": MASTER_SEED,
        "cell_seeds": {f"n{r['n']}_a{r['alpha']}": r["seed"] for r in results},
        "B": B,
        "thetas": list(THETAS),
        "kappas": KAPPAS,
        "delta": DELTA,
        "kappa_hi_envelope": KAPPA_HI,
        "wall_seconds": wall,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "executed": "2026-08-23",
    }
    with open(os.path.join(RES, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    r = [x for x in results if x["n"] == 2000 and x["alpha"] == 0.3][0]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(KAPPAS)))
    for ki, kap in enumerate(KAPPAS):
        ax[0].plot(THETAS, r["bias_mc"][:, ki], "o-", ms=3, color=colors[ki],
                   label=f"kappa={kap} (MC)")
        ax[0].plot(THETAS, r["bias_form"][ki], "--", color=colors[ki], lw=1,
                   label=f"kappa={kap} (formula)")
        ax[0].axvline(theta_harm(r["alpha"], kap, DELTA), color=colors[ki], lw=0.7, ls=":")
    ax[0].axhline(DELTA, color="k", lw=1.2, ls="-.", label=f"delta={DELTA}")
    ax[0].set_xlabel("population canonical correlation theta")
    ax[0].set_ylabel("E[beta_2SLS] - beta (beta=0)")
    ax[0].set_title(f"n={r['n']}, alpha={r['alpha']}: harm threshold shifts with kappa")
    ax[0].legend(fontsize=6)

    all_bm, all_bf, all_se = [], [], []
    for x in results:
        all_bm.append(x["bias_mc"])
        all_bf.append(x["bias_form"])
        all_se.append(x["bias_se"])
    bm = np.concatenate([a.ravel() for a in all_bm])
    bf = np.concatenate([a.ravel() for a in all_bf])
    se = np.concatenate([a.ravel() for a in all_se])
    keep = bf > 0.008
    ax[1].errorbar(bf[keep], bm[keep], yerr=2 * se[keep], fmt=".", ms=2, elinewidth=0.4)
    lims = [0, max(bf[keep].max(), bm[keep].max()) * 1.05]
    ax[1].plot(lims, lims, "k:", lw=0.8)
    ax[1].set_xlabel("leading-order formula rho/(1+mu2/q), rho=Corr(eps,v)")
    ax[1].set_ylabel("MC bias (+/-2 SE)")
    ax[1].set_title("Formula vs Monte Carlo across all cells")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_o1_bias_kappa_shift.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    r = [x for x in results if x["n"] == 2000 and x["alpha"] == 0.3][0]
    for ki, kap in enumerate(KAPPAS):
        th_h = theta_harm(r["alpha"], kap, DELTA)
        ax.axvspan(min(th_h, THETAS.min()), max(th_h, THETAS.min()), alpha=0.06, color=colors[ki])
        ax.axvline(th_h, color=colors[ki], lw=1, label=f"harm boundary kappa={kap}")
    th_env = theta_harm(r["alpha"], KAPPA_HI, DELTA)
    ax.axvline(th_env, color="crimson", lw=2, ls="--",
               label=f"identifiable envelope (kappa<={KAPPA_HI})")
    ax.set_xlabel("theta")
    ax.set_title(f"Envelope covers every harm region; n={r['n']}, alpha={r['alpha']}, delta={DELTA}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_o1_envelope.png"))
    plt.close(fig)

    print(f"done in {wall:.1f}s; results in {RES}")


if __name__ == "__main__":
    main()
