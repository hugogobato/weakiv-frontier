"""Aggregate corrected AR-acceptance arrays -> X3 verdict statistics.

Consumes {cid}_arfix.npy (+ _done markers) produced by the NBE notebooks or by
recompute_ar_coverage.py locally, combined with the VALID first-stage flag
vectors stored in the existing raw.npz files. Emits
analysis/ar_coverage_fixed.csv and prints the preregistered X3 scan:

  X3 PASS iff exists an alpha-region (>=2 adjacent alphas) with conditional
  AR coverage <= 0.90 under F>10 while spectral_env holds >= 0.93 on the same
  designs (min 30 flagged reps for a conditional estimate to count).

Usage: python3 aggregate_arfix.py
"""
import csv
import json
import os
import numpy as np

ROOT = "/home/hugo_souto/Stuff/Research/RMT/Idea2"
CELLS = os.path.join(ROOT, "Research", "weakiv_results",
                     "phase3_decisive_grid", "cells")
ANA = os.path.join(ROOT, "Research", "weakiv_results", "analysis")
THETAS = [0.05, 0.10, 0.18, 0.28, 0.40, 0.55, 0.72, 0.88]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def parse_cid(cid):
    a_part, rest = cid[1:].split("_k", 1)
    k_part, rest = rest.split("_none_p", 1)
    p_part, th_part = rest.split("_th", 1)
    return float(a_part), float(k_part), int(p_part), float(th_part)


def main():
    rows = []
    for f in sorted(os.listdir(CELLS)):
        if not f.endswith("_arfix.npy"):
            continue
        cid = f[:-len("_arfix.npy")]
        arr = np.load(os.path.join(CELLS, f))
        z = np.load(os.path.join(CELLS, cid + "_raw.npz"))
        a, kap, p, th = parse_cid(cid)
        f10, env = z["f_pass"], z["env_pass"]
        covF = float(arr[f10].mean()) if f10.sum() >= 30 else ""
        covE = float(arr[env].mean()) if env.sum() >= 30 else ""
        rows.append({"cell_id": cid, "alpha": a, "kappa": kap, "p": p,
                     "theta": th, "cov_all": round(float(arr.mean()), 4),
                     "nF": int(f10.sum()), "covF_fixed": covF,
                     "nEnv": int(env.sum()), "covEnv_fixed": covE})
    out = os.path.join(ANA, "ar_coverage_fixed.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, f"({len(rows)} cells)")

    print("\nX3 scan on CORRECTED acceptance (cond cov; min 30 flagged):")
    by_alpha = {a: [0, 0] for a in ALPHAS}   # both, evaluable
    conj = []
    for r in rows:
        if r["covF_fixed"] == "" :
            continue
        by_alpha[r["alpha"]][1] += 1
        ok = (r["covF_fixed"] <= 0.90 and r["covEnv_fixed"] != ""
              and r["covEnv_fixed"] >= 0.93)
        by_alpha[r["alpha"]][0] += ok
        if ok:
            conj.append(r["cell_id"])
    for a in ALPHAS:
        b, e = by_alpha[a]
        print(f"  alpha={a}: BOTH={b} / evaluable={e}")
    adj = [(a, b) for a, b in zip(ALPHAS, ALPHAS[1:])
           if by_alpha[a][0] > 0 and by_alpha[b][0] > 0]
    print("adjacent alpha-region(s):", adj or "NONE")
    print("conjunction cells:", conj or "NONE")
    json.dump({"by_alpha": {str(a): by_alpha[a] for a in ALPHAS},
               "conjunction_cells": conj},
              open(os.path.join(ANA, "x3_corrected.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
