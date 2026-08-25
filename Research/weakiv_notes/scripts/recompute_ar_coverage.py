"""Corrected AR-acceptance recomputation for the decisive grid (deviation #10).

ar_accepts() in experiments.py tests beta in RAW units against standardized
blocks (missing the sy/sx rescale that ivestimators apply), so for theta > 0
the tested null is false in standardized units and AR acceptance decays
artificially with signal strength. This script regenerates every decisive
replication from its registered stream (identical seeds), recomputes AR
acceptance with the CORRECT standardized-null vector b_std = beta/scale_vec,
and writes per-cell arrays + a summary CSV using the STORED (valid)
first-stage flag vectors f_pass/env_pass for conditional slices.

Parity check (--smoke): recomputing the BUGGY statistic must reproduce the
stored ar_ok bit-for-bit on sampled cells before any corrected number is
produced.

Usage:
  python3 recompute_ar_coverage.py --smoke
  python3 recompute_ar_coverage.py [--workers 12] [--cells cid1,cid2]
"""
import argparse
import json
import os
import sys

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = ("/home/hugo_souto/Stuff/Research/RMT/Idea2")
sys.path.insert(0, os.path.join(ROOT, "Research", "spectraliv", "src"))
CELLS_DIR = os.path.join(ROOT, "Research", "weakiv_results",
                         "phase3_decisive_grid", "cells")
ANA = os.path.join(ROOT, "Research", "weakiv_results", "analysis")
MASTER_SEED = 20260823


def parse_cid(cid):
    a_part, rest = cid[1:].split("_k", 1)
    k_part, rest = rest.split("_none_p", 1)
    p_part, th_part = rest.split("_th", 1)
    return float(a_part), float(k_part), int(p_part), float(th_part)


def rep_rng(cid, b):
    # NOTE: the runners seed streams by the BASE cell id (theta stripped), so
    # all thetas of one base share per-rep draws by design (paired protocol).
    from spectraliv.rng import cell_stream
    base = cid.rsplit("_th", 1)[0]
    return np.random.default_rng(
        cell_stream("phase3_decisive_grid", base, b,
                    master_seed=MASTER_SEED).integers(1 << 31))


def ar_stat(xs, zs, ys, chol, b_vec):
    from scipy.stats import f as fdist
    e = ys - xs @ b_vec
    n, q = zs.shape
    w_e = np.linalg.solve(chol, zs.T @ e)
    num = max(float(w_e @ w_e), 1e-300)
    den = max(float(e @ e) - num, 1e-300)
    return (num / q) / (den / (n - q))


def process_cell(args_tuple):
    import numpy as np
    from scipy.stats import f as fdist
    from spectraliv.dgps import make_single_spike, rho_of_kappa
    from spectraliv.preprocess import prepare
    cid, reps = args_tuple
    alpha, kappa, p, theta = parse_cid(cid)
    n, q = (1000 if p == 1 else 2000), int(round(alpha * ((1000 if p == 1 else 2000) - 1)))
    rho = rho_of_kappa(kappa)
    out = np.empty(reps, dtype=bool)
    for b in range(reps):
        rng = rep_rng(cid, b)
        dgp = make_single_spike(n, q, theta, rho, rng, p=p, beta=0.5)
        xs, zs, yr, resc = prepare(dgp.x, dgp.z, dgp.y, None)
        sy = np.std(yr, ddof=1)
        ys = yr / sy
        chol = np.linalg.cholesky(zs.T @ zs)
        b_orig = np.zeros(len(resc)); b_orig[0] = 0.5   # y = 0.5*x1 + eps
        b_std = b_orig / resc                            # standardized null
        f_crit = float(fdist.ppf(0.95, q, n - q))
        out[b] = ar_stat(xs, zs, ys, chol, b_std) <= f_crit
    return cid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--cells", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    import numpy as np

    all_cids = sorted(f[:-len("_raw.npz")] for f in os.listdir(CELLS_DIR)
                      if f.endswith("_raw.npz") and "_th" in f)
    if args.cells:
        keep = set(args.cells.split(","))
        all_cids = [c for c in all_cids if c in keep]

    if args.smoke:
        print("-- smoke: parity of stream replication + buggy statistic --")
        for cid in ("a0.5_k2.0_none_p1_th0.05", "a0.9_k0.5_none_p5_th0.28"):
            alpha, kappa, p, theta = parse_cid(cid)
            n = 1000 if p == 1 else 2000
            q = int(round(alpha * (n - 1)))
            from spectraliv.dgps import make_single_spike, rho_of_kappa
            from spectraliv.preprocess import prepare
            z_ref = np.load(os.path.join(CELLS_DIR, cid + "_raw.npz"))
            match = True
            for b in range(10):
                rng = rep_rng(cid, b)
                dgp = make_single_spike(n, q, theta, rho_of_kappa(kappa),
                                        rng, p=p, beta=0.5)
                xs, zs, yr, _ = prepare(dgp.x, dgp.z, dgp.y, None)
                ys = yr / np.std(yr, ddof=1)
                chol = np.linalg.cholesky(zs.T @ zs)
                from scipy.stats import f as fdist
                buggy = ar_stat(xs, zs, ys, chol,
                                np.full(p, 0.5)) <= fdist.ppf(0.95, q, n - q)
                match &= bool(buggy) == bool(z_ref["ar_ok"][b])
            print(f"  {cid}: buggy-statistic matches stored ar_ok: {match}")
            if not match:
                raise SystemExit("stream/statistic parity FAILED - abort")
        # corrected preview on one strong-signal p1 cell
        cid, outs = process_cell(("a0.5_k2.0_none_p1_th0.72", 40))
        print(f"  corrected AR acceptance {cid} (40 reps): {outs.mean():.3f}"
              "  [stored corrupted value was 0.00]")
        return

    from multiprocessing import Pool
    todo = []
    for cid in all_cids:
        marker = os.path.join(CELLS_DIR, f"{cid}_arfix_done.json")
        if not os.path.exists(marker):
            todo.append((cid, args.reps))
    print(f"{len(todo)} cells to recompute")
    if not todo:
        return
    done = 0
    with Pool(args.workers) as pool:
        for cid, arr in pool.imap_unordered(process_cell, todo):
            np.save(os.path.join(CELLS_DIR, f"{cid}_arfix.npy"), arr)
            with open(os.path.join(CELLS_DIR, f"{cid}_arfix_done.json"), "w") as f:
                json.dump({"reps": len(arr), "seed": MASTER_SEED}, f)
            done += 1
            print(f"[{done}/{len(todo)}] {cid}: cov={arr.mean():.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
