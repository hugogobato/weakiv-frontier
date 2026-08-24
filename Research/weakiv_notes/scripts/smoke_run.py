"""WP-P2-I2: Smoke run and cost model (one-seed small grid).

Runs B=100 seeds x 6-cell mini-grid through the FULL pipeline (T_spec +
estimator battery) in isolated child processes, records wall time and peak RSS
per cell, fits a transparent cost model, and emits colab_plan.csv assigning
every planned Phase-3 grid cell to LOCAL or NOTEBOOK-i.

Outputs:
    Research/weakiv_notes/smoke_run/smoke_cells.csv
    Research/weakiv_notes/smoke_run/manifest.json
    Research/weakiv_notes/colab_plan.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import resource
import time
from multiprocessing import Process, Queue

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SMOKE = os.path.join(ROOT, "Research", "weakiv_notes", "smoke_run")
os.makedirs(SMOKE, exist_ok=True)

CELLS = [
    ("c1_n250_p1_q25", 250, 1, 25),
    ("c2_n250_p2_q125", 250, 2, 125),
    ("c3_n500_p5_q250", 500, 5, 250),
    ("c4_n1000_p1_q700", 1000, 1, 700),
    ("c5_n1000_p25_q500", 1000, 25, 500),
    ("c6_n2000_p50_q1000", 2000, 50, 1000),
]
BIG_B = 100
THETA = 0.35
RHO = 0.5
MASTER_SEED = 20260823


def run_cell(cell_id, n, p, q, out_q):
    import spectraliv
    from spectraliv.dgps import make_single_spike
    from spectraliv.ivestimators import (
        bekker, fuller, jive, liml, pca_2sls, prepared_all, tsls,
        truncated_2sls,
    )
    from spectraliv.select_tau import select_tau
    from spectraliv.teststats import spec_test

    rng_master = np.random.default_rng(MASTER_SEED + sum(ord(ch) for ch in cell_id))
    t0 = time.time()
    est_names = ["tsls", "liml", "fuller", "bekker", "jive",
                 "trunc_tauhat", "trunc_tau1", "pca_l"]
    acc = {k: [] for k in est_names}
    rej = 0
    tau_sum = 0.0
    for b in range(BIG_B):
        dgp = make_single_spike(n, q, THETA, RHO,
                                rng=np.random.default_rng(rng_master.integers(1 << 31)),
                                beta=0.5)
        ys, xs, zs, ca, _rescale = prepared_all(dgp.y, dgp.x, dgp.z, None)
        res = spec_test(dgp.x, dgp.z, canon=ca, b_cal=300,
                        rng=np.random.default_rng(rng_master.integers(1 << 31)))
        rej += res.reject_exact
        tau = select_tau(dgp.x, dgp.z, canon=ca)
        tau_sum += tau
        acc["tsls"].append(float(tsls(dgp.y, dgp.x, dgp.z)[0]))
        acc["liml"].append(float(liml(dgp.y, dgp.x, dgp.z)[0]))
        acc["fuller"].append(float(fuller(dgp.y, dgp.x, dgp.z)[0]))
        acc["bekker"].append(float(bekker(dgp.y, dgp.x, dgp.z)[0]))
        acc["jive"].append(float(jive(dgp.y, dgp.x, dgp.z)[0]))
        acc["trunc_tauhat"].append(float(truncated_2sls(dgp.y, dgp.x, dgp.z, tau=tau)[0]))
        acc["trunc_tau1"].append(float(truncated_2sls(dgp.y, dgp.x, dgp.z, tau=1.0)[0]))
        acc["pca_l"].append(float(pca_2sls(dgp.y, dgp.x, dgp.z, ell=p)[0]))
    wall = time.time() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out_q.put({
        "cell_id": cell_id, "n": n, "p": p, "q": q, "B": BIG_B,
        "wall_s": wall, "rss_mb": rss_mb, "reps_per_s": BIG_B / wall,
        "size_est": rej / BIG_B, "tau_mean": tau_sum / BIG_B,
        **{f"mean_{k}": float(np.nanmean(v)) for k, v in acc.items()},
        "spectraliv_version": spectraliv.__version__,
    })


def fit_cost_model(rows):
    """t_per_rep ~ a0 + a1*f1 + a2*f2 + a3*f3 with f1=n(p+q), f2=n q^2, f3=n p^2.

    Non-negative least squares: compute time cannot decrease with workload;
    plain OLS went negative on extrapolation (corrected after first smoke).
    """
    from scipy.optimize import nnls

    X = np.array([[1.0, r["n"] * (r["p"] + r["q"]),
                   r["n"] * r["q"] ** 2, r["n"] * r["p"] ** 2] for r in rows])
    y = np.array([r["wall_s"] / r["B"] for r in rows])
    coef, _res = nnls(X, y)
    pred = X @ coef
    relerr = np.abs(pred - y) / y
    return coef, float(relerr.max()), float(np.median(pred / y))


def predict_time(coef, n, p, q, big_b):
    feats = np.array([1.0, n * (p + q), n * q ** 2, n * p ** 2])
    return float(coef @ feats) * big_b


def _is_num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _load_rows():
    import csv

    rows = []
    with open(os.path.join(SMOKE, "smoke_cells.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if _is_num(v) else v) for k, v in r.items()})
    return rows


def main():
    import sys

    if "--plan-only" in sys.argv:
        rows = _load_rows()
        coef, max_rel, med_ratio = fit_cost_model(rows)
        ram_pred_ratio = float(np.median([
            r["rss_mb"] / ((r["n"] * (r["p"] + r["q"]) * 8) / 1024.0 ** 2 + 0.2)
            for r in rows]))
    else:
        out_q = Queue()
        procs = []
        for cell_id, n, p, q in CELLS:
            pr = Process(target=run_cell, args=(cell_id, n, p, q, out_q))
            pr.start()
            procs.append(pr)
        rows = [out_q.get() for _ in CELLS]
        for pr in procs:
            pr.join()

        order = {cid: i for i, (cid, *_r) in enumerate(CELLS)}
        rows.sort(key=lambda r: order[r["cell_id"]])

        cols = list(rows[0].keys())
        with open(os.path.join(SMOKE, "smoke_cells.csv"), "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r[c]) for c in cols) + "\n")
        coef, max_rel, med_ratio = fit_cost_model(rows)
        ram_pred_ratio = float(np.median([
            r["rss_mb"] / ((r["n"] * (r["p"] + r["q"]) * 8) / 1024.0 ** 2 + 0.2)
            for r in rows]))

    # ---- enumerate Phase-3 grids and assign placements ----
    alphas5 = [0.1, 0.3, 0.5, 0.7, 0.9]
    plan = []

    def q_of(n_, a_):
        return int(round(a_ * (n_ - 1)))

    # size grid (X1): null-only reps are ~55 percent of pipeline cost; use 0.6 factor
    for n_ in (250, 500, 1000, 2000):
        ps = [1, 2, 5] + ([25, 100] if n_ >= 1000 else [])
        for a_ in alphas5:
            q_ = q_of(n_, a_)
            for p_ in ps:
                if not (q_ > p_ - 1 and n_ > q_ + p_):
                    continue
                t_h = predict_time(coef, n_, p_, q_, 20000) * 0.6 / 3600.0
                gb = (n_ * (p_ + q_) * 8) / 1024.0 ** 3 * ram_pred_ratio + 0.2
                plan.append(["size_grid", f"n{n_}_a{a_}_p{p_}", n_, p_, q_, a_,
                             "", "", "", 20000, t_h, gb])

    # power surface (X2): sweeps of 20 thetas x 400 reps
    for n_ in (250, 1000):
        for a_ in (0.1, 0.5, 0.9):
            q_ = q_of(n_, a_)
            for p_ in (1, 5):
                if not (q_ > p_ - 1 and n_ > q_ + p_):
                    continue
                t_h = predict_time(coef, n_, p_, q_, 20 * 400) / 3600.0
                gb = (n_ * (p_ + q_) * 8) / 1024.0 ** 3 * ram_pred_ratio + 0.2
                plan.append(["power_surface", f"n{n_}_a{a_}_p{p_}", n_, p_, q_, a_,
                             "", "", "", 8000, t_h, gb])

    # decisive grid (X3/X4/X5): full factorial, 400 reps per cell
    kappas = [0.2, 0.5, 1.0, 2.0]
    heteros = ["none", "mild", "severe"]
    for a_ in alphas5:
        for kap in kappas:
            for hi, het in enumerate(heteros):
                for p_ in (1, 5):
                    n_ = 1000 if p_ == 1 else 2000
                    q_ = q_of(n_, a_)
                    if not (q_ > p_ - 1 and n_ > q_ + p_):
                        continue
                    t_h = predict_time(coef, n_, p_, q_, 20 * 400) / 3600.0
                    gb = (n_ * (p_ + q_) * 8) / 1024.0 ** 3 * ram_pred_ratio + 0.2
                    plan.append(["decisive_grid",
                                 f"a{a_}_k{kap}_{het}_p{p_}", n_, p_, q_, a_,
                                 kap, "", het, 8000, t_h, gb])

    # placement
    LOCAL_T_H, LOCAL_RAM, NB_CAP_H, NB_MAX = 2.0, 4.0, 6.0, 40
    nb_counter = 0
    nb_load = []
    with open(os.path.join(ROOT, "Research", "weakiv_notes", "colab_plan.csv"), "w") as f:
        f.write("experiment,cell_id,n,p,q,alpha,kappa,theta,hetero,B,"
                "pred_wall_h_local,pred_ram_gb,placement,notebook_id\n")
        # local first
        rest = []
        for row in plan:
            if row[10] < LOCAL_T_H and row[11] < LOCAL_RAM:
                placement, nid = "LOCAL", ""
            else:
                rest.append(row)
                continue
            f.write(",".join(str(x) for x in row) + f",{placement},{nid}\n")
        # bin-pack the rest into notebooks <= NB_CAP_H hours
        rest.sort(key=lambda r: -r[10])
        for row in rest:
            placed = False
            for i, load in enumerate(nb_load):
                if load + row[10] <= NB_CAP_H:
                    nb_load[i] += row[10]
                    placement, nid = "NOTEBOOK", f"NB{i:02d}"
                    placed = True
                    break
            if not placed:
                nb_load.append(row[10])
                placement, nid = "NOTEBOOK", f"NB{len(nb_load)-1:02d}"
            f.write(",".join(str(x) for x in row) + f",{placement},{nid}\n")

    manifest = {
        "master_seed": MASTER_SEED,
        "big_b": BIG_B,
        "theta": THETA,
        "rho": RHO,
        "cost_coef": list(map(float, coef)),
        "cost_features": ["1", "n(p+q)", "n*q^2", "n*p^2"],
        "cost_fit": "nnls (non-negative; OLS extrapolated negative after first smoke)",
        "cost_max_rel_err": max_rel,
        "cost_median_pred_over_meas": med_ratio,
        "ram_overhead_factor": ram_pred_ratio,
        "grid_total_serial_hours": float(sum(r[10] for r in plan)),
        "notebooks_needed": len(nb_load),
        "notebook_cap": 40,
        "notebook_cap_exceeded": bool(len(nb_load) > 40),
        "planning_flag": ("GRID_PRUNING_REQUIRED_AT_WP_P3_R0: enumerated Phase-3 "
                          "factorial exceeds the 40-notebook budget; prune theta "
                          "points / hetero levels or accept multi-day local runs"),
        "numpy": np.__version__,
        "script_sha256": __import__("hashlib").sha256(
            open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "executed": "2026-08-23",
    }
    with open(os.path.join(SMOKE, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in
                      ("cost_max_rel_err", "cost_median_pred_over_meas",
                       "ram_overhead_factor", "notebooks_needed")}, indent=1))
    print("cells:")
    for r in rows:
        print(f"  {r['cell_id']}: {r['wall_s']:.1f}s  {r['rss_mb']:.0f} MB  "
              f"size={r['size_est']:.2f} tau={r['tau_mean']:.2f}")


if __name__ == "__main__":
    main()
