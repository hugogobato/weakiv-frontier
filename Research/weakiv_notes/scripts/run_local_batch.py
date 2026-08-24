"""Optional LOCAL execution of Phase-3 cells (opportunistic).

Reads colab_plan_v2.csv, runs every cell flagged local_eligible=YES that has
no _done markers yet, with a worker pool capped at 4 processes, BLAS threads
= 1, nice 10 (machine-etiquette rules). Results write directly to
Research/weakiv_results/. Safe to re-run: completed cells are skipped.

Usage: python3 run_local_batch.py [--max-workers 4]
"""
import argparse
import csv
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.nice(10)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
PLAN = os.path.join(ROOT, "Research", "weakiv_notes", "colab_plan_v2.csv")
OUT_ROOT = os.path.join(ROOT, "Research", "weakiv_results")


def done_markers(exp, cid):
    d = os.path.join(OUT_ROOT, exp, "cells")
    if exp == "phase3_decisive_grid":
        names = [f"_done_{cid}_coverage.csv.json", f"_done_{cid}_risk.csv.json"]
    else:
        names = [f"_done_{cid}.csv.json"]
    return all(os.path.exists(os.path.join(d, nm)) for nm in names)


def run_job(item):
    from spectraliv.experiments import (
        run_decisive_cell,
        run_power_cell,
        run_robust_cell,
        run_size_cell,
    )
    exp, cid = item["experiment"], item["cell_id"]
    t0 = __import__("time").time()
    cell = dict(item["cell"])
    if exp == "phase3_size_grid":
        run_size_cell(cell, big_b=int(item["row"]["B"]), b_cal_cv=4000,
                      out_root=OUT_ROOT)
    elif exp == "phase3_power_surface":
        run_power_cell(cell, reps_per_theta=300, b_cal_cv=4000,
                       out_root=OUT_ROOT)
    elif exp == "phase3_decisive_grid":
        run_decisive_cell(cell, reps=400, b_cal_cv=4000, out_root=OUT_ROOT)
    elif exp == "phase3_robustness":
        run_robust_cell(cell, big_b=4000, patch_reps=250, b_boot=99,
                        out_root=OUT_ROOT)
    return {"cell_id": cid, "wall_s": round(__import__("time").time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(ROOT, "Research", "spectraliv", "src"))

    with open(PLAN) as f:
        rows = list(csv.DictReader(f))
    jobs = []
    for r in rows:
        if r["local_eligible"] != "YES" or r["cell_id"].endswith("_patch"):
            continue
        exp, cid = r["experiment"], r["cell_id"]
        if done_markers(exp, cid):
            continue
        cell = {"n": int(r["n"]), "p": int(r["p"]), "q": int(r["q"]),
                "alpha": float(r["alpha"]),
                "cell_id": cid.split("_th")[0]
                if exp == "phase3_decisive_grid" else cid}
        if exp == "phase3_decisive_grid":
            cell["kappa"] = float(r["kappa"])
            cell["theta"] = float(r["theta"])
        elif exp == "phase3_robustness":
            cell["violation"] = cid.rsplit("_p", 1)[0]
        jobs.append({"experiment": exp, "cell_id": cid, "cell": cell,
                     "row": r})

    print(f"{len(jobs)} local-eligible pending cells")
    if not jobs:
        return
    from multiprocessing import Pool

    with Pool(args.max_workers) as pool:
        for res in pool.imap_unordered(run_job, jobs):
            print(json.dumps(res))


if __name__ == "__main__":
    main()
