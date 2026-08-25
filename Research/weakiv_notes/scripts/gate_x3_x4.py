"""G3 adjudication statistics for X3 (coverage map) and X4 (risk curves).

Reads the 160 validated per-theta artifacts under
Research/weakiv_results/phase3_decisive_grid/cells/ (raw npz carries per-rep
AR acceptance, rule flags and all estimator betas -> paired-seed analysis).

Preregistered thresholds (weakiv_preregistration.md, verbatim):
  X3 PASS iff exists an alpha-region (>=2 adjacent alphas) with conditional
      AR coverage <= 0.90 under F>10 while spectral_env holds >= 0.93 on the
      same designs; FAIL if no divergence anywhere realistic.
  X4 PASS iff truncated family beats best baseline RMSE by >=15% in >=1
      predeclared regime cell (alpha>=0.5, theta in {0.18,0.28}, either p)
      AND never worse than best baseline by >5% in any cell; MC error via
      paired bootstrap (2000 resamples) on stored raw betas.

trunc_tauhat: k_hat = clip(round(tau_hat*p),1,p) mapped to mechanically
nearest stored k in {1,2,p} (ties -> smaller), per deviations #3.
"""
import json
import os
import sys
import numpy as np

RES = ("/home/hugo_souto/Stuff/Research/RMT/Idea2/"
       "Research/weakiv_results/phase3_decisive_grid/cells")
THETAS = [0.05, 0.10, 0.18, 0.28, 0.40, 0.55, 0.72, 0.88]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
BASE_EST = ["tsls", "liml", "fuller", "bekker", "jive", "whiten", "pca_l"]
BOOT = 2000
rng = np.random.default_rng(20260823)

cells = {}
for a in ALPHAS:
    for kap in ("0.5", "2.0"):
        for p in (1, 5):
            for t in THETAS:
                cid = f"a{a}_k{kap}_none_p{p}_th{t}"
                z = np.load(os.path.join(RES, cid + "_raw.npz"))
                ar = z["ar_ok"]
                f10, env = z["f_pass"], z["env_pass"]
                # tauhat-adaptive beta
                p_dim = p
                k_hat = np.clip(np.round(z["tau_hat"] * p_dim), 1, p_dim).astype(int)
                stored = [k for k in (1, 2, p)]
                kk = np.array([min(stored, key=lambda s: (abs(s - k), s))
                               for k in k_hat])
                bth = np.full(len(ar), np.nan)
                for k in stored:
                    m = kk == k
                    if m.any():
                        bth[m] = z[f"beta_trunc_k{k}"][m, 0]
                err = {e: z[f"beta_{e}"][:, 0] - 0.5 for e in BASE_EST}
                err["trunc_k1"] = z["beta_trunc_k1"][:, 0] - 0.5
                if p >= 2:
                    err["trunc_k2"] = z[f"beta_trunc_k2"][:, 0] - 0.5
                err["trunc_tauhat"] = bth - 0.5
                cells[cid] = {
                    "ar": ar, "f10": f10, "env": env,
                    "nf10": int(f10.sum()), "nenv": int(env.sum()),
                    "cov_f10": float(ar[f10].mean()) if f10.any() else None,
                    "cov_env": float(ar[env].mean()) if env.any() else None,
                    "err": err,
                }

# ------------------------------------------------------------------ X3 ---
print("== X3: conditional AR coverage by rule ==")
print("cell-level: F>10 cond-cov <= 0.90 AND spectral_env cond-cov >= 0.93")
by_alpha = {a: {"both": 0, "f_bad": 0, "env_ok": 0, "n_eval": 0} for a in ALPHAS}
detail = []
for cid, c in cells.items():
    a = float(cid[1:4])
    if c["nf10"] < 30:          # too few F>10 passes to speak of
        continue
    by_alpha[a]["n_eval"] += 1
    fbad = c["cov_f10"] is not None and c["cov_f10"] <= 0.90
    eok = c["cov_env"] is not None and c["nenv"] >= 30 and c["cov_env"] >= 0.93
    by_alpha[a]["f_bad"] += fbad
    by_alpha[a]["env_ok"] += eok
    by_alpha[a]["both"] += (fbad and eok)
    if fbad:
        detail.append((cid, c["nf10"], round(c["cov_f10"], 3),
                       c["nenv"], None if c["cov_env"] is None else round(c["cov_env"], 3)))
for a in ALPHAS:
    d = by_alpha[a]
    print(f"  alpha={a}: evaluable={d['n_eval']:>3}  F<=0.90: {d['f_bad']:>3}  "
          f"env>=0.93: {d['env_ok']:>3}  BOTH: {d['both']:>3}")
adj = [(a, b) for a, b in zip(ALPHAS, ALPHAS[1:])
       if by_alpha[a]["both"] > 0 and by_alpha[b]["both"] > 0]
print("adjacent alpha-region(s) with BOTH conditions:", adj or "NONE")
print("sample F-bad cells (cid, nf10, covF, nenv, covEnv):")
for d in detail[:14]:
    print("   ", d)

# ------------------------------------------------------------------ X4 ---
print("\n== X4: truncated contenders vs best baseline (RMSE, component 1) ==")
rows = []
for cid, c in cells.items():
    bb_e = min(BASE_EST, key=lambda e: np.sqrt(np.mean(c["err"][e] ** 2)))
    bb = float(np.sqrt(np.mean(c["err"][bb_e] ** 2)))
    row = {"cid": cid, "bb": bb_e, "bb_rmse": bb}
    for t in ("trunc_k1", "trunc_tauhat"):
        r = float(np.sqrt(np.mean(c["err"][t] ** 2))) / bb
        row[t] = r
    a, th = float(cid[1:4]), float(cid.rsplit("_th", 1)[1])
    row["regime"] = (a >= 0.5 and th in (0.18, 0.28))
    rows.append(row)

regime = [r for r in rows if r["regime"]]
wins = sorted([r for r in regime if min(r["trunc_k1"], r["trunc_tauhat"]) <= 0.85],
              key=lambda r: min(r["trunc_k1"], r["trunc_tauhat"]))
print(f"predeclared-regime cells: {len(regime)}; >=15% winners: {len(wins)}")
for r in wins[:8]:
    print(f"   {r['cid']}: bb={r['bb']} t_k1={r['trunc_k1']:.3f} "
          f"t_tauhat={r['trunc_tauhat']:.3f}")
if not wins:
    best = sorted(regime, key=lambda r: min(r["trunc_k1"], r["trunc_tauhat"]))[:6]
    print("   closest misses:")
    for r in best:
        print(f"   {r['cid']}: bb={r['bb']} t_k1={r['trunc_k1']:.3f} "
              f"t_tauhat={r['trunc_tauhat']:.3f}")

bad = [r for r in rows if max(r["trunc_k1"], r["trunc_tauhat"]) > 1.05]
border = [r for r in rows if abs(max(r["trunc_k1"], r["trunc_tauhat"]) - 1.05) <= 0.02]
print(f"\nnever-worse clause (>5% worse anywhere): VIOLATED in {len(bad)}/160 cells"
      f" (+{len(border)} within +-0.02 of the line)")
worst = sorted(rows, key=lambda r: -max(r["trunc_k1"], r["trunc_tauhat"]))[:8]
for r in worst:
    print(f"   {r['cid']}: t_k1={r['trunc_k1']:.3f} t_tauhat={r['trunc_tauhat']:.3f} "
          f"(vs {r['bb']})")

# paired bootstrap CI of the max-ratio statistic for the single worst offender
w0 = worst[0]
c = cells[w0["cid"]]
e_bb = c["err"][min(BASE_EST, key=lambda e: np.sqrt(np.mean(c["err"][e] ** 2)))]
cont = max(("trunc_k1", "trunc_tauhat"), key=lambda t: w0[t])
e_c = c["err"][cont]
n = len(e_bb)
idx = rng.integers(0, n, size=(BOOT, n))
ratios = np.sqrt((e_c[idx] ** 2).mean(1)) / np.sqrt((e_bb[idx] ** 2).mean(1))
lo, hi = np.quantile(ratios, [0.025, 0.975])
print(f"\npaired bootstrap ({BOOT}x) worst cell {w0['cid']} [{cont}]: "
      f"ratio 95% CI [{lo:.3f}, {hi:.3f}], point {w0[cont]:.3f}")

json.dump({"x3_by_alpha": {str(k): v for k, v in by_alpha.items()},
           "x4_worst": [{**r, } for r in worst],
           "x4_regime_wins": len(wins), "x4_violations": len(bad)},
          open(os.path.join(os.path.dirname(RES.rstrip("/cells")),
                            "analysis", "gate_x3_x4.json"), "w"), indent=1)
