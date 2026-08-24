"""WP-P3-R0 placement generator: colab_plan_v2.csv (pruned grids).

Uses MEASURED per-rep anchors from bench_phase3.py (2026-08-24) instead of the
Phase-2 NNLS model (which priced the redundant multi-pass battery). Emits the
pruned decisive/size/power/robustness grids with placements and notebook
bucket assignment under the plan's policy (LOCAL if < 2 h AND < 4 GB;
notebooks packed <= 6 h target). Given current machine load (loadavg ~45 on
2026-08-24), ALL cells are routed NOTEBOOK-first; LOCAL eligibility is still
recorded so run_local_batch.py can pick cells up opportunistically.
"""
import csv
import json
import os

ROOT = "/home/hugo_souto/Stuff/Research/RMT/Idea2"
OUT = os.path.join(ROOT, "Research", "weakiv_notes", "colab_plan_v2.csv")

# measured full-battery seconds/rep, (n, q) -> s  [bench_phase3 + interpolation]
FULL_ANCHORS = {
    (1000, 100): 0.42, (1000, 300): 0.53, (1000, 500): 0.741,
    (1000, 700): 1.07, (1000, 900): 1.50,
    (2000, 200): 0.779, (2000, 600): 1.15, (2000, 1000): 1.936,
    (2000, 1400): 4.30, (2000, 1800): 8.386,
}


def q_of(n, alpha):
    return int(round(alpha * (n - 1)))


def full_rate(n, q):
    ks = sorted(FULL_ANCHORS, key=lambda k: (k[0], abs(k[1] - q)))
    same_n = [k for k in FULL_ANCHORS if k[0] == n]
    lo = max([k for k in same_n if k[1] <= q],
             key=lambda k: k[1], default=(n, min(k[1] for k in same_n)))
    hi = min([k for k in same_n if k[1] >= q],
             key=lambda k: k[1], default=(n, max(k[1] for k in same_n)))
    if lo == hi:
        return FULL_ANCHORS[lo]
    w = (q - lo[1]) / (hi[1] - lo[1])
    return FULL_ANCHORS[lo] * (1 - w) + FULL_ANCHORS[hi] * w


def stat_rate(n, q):
    """Null-only statistic pass: fit 0.03 + c*n*q^2 calibrated on
    (1000,500)=0.105 and (1000,900)=0.230."""
    c = (0.230 - 0.105) / (1000 * (900**2 - 500**2))
    return 0.03 + c * n * q ** 2


def main():
    sys_path = os.path.join(ROOT, "Research", "spectraliv", "src")
    import sys
    sys.path.insert(0, sys_path)
    from spectraliv.experiments import (
        ALPHAS5, KAPPAS2, THETAS_DECISIVE8, THETAS_POWER12,
        decisive_grid_cells, robustness_cells,
    )

    rows = []
    # --- X1 size grid ---
    for n in (250, 500, 1000, 2000):
        ps = [1, 2, 5] + ([25, 100] if n >= 1000 else [])
        for a in ALPHAS5:
            for p in ps:
                q = q_of(n, a)
                if not (q > p - 1 and n > q + p):
                    continue
                big_b = 20000 if n <= 1000 else 8000
                hrs = stat_rate(n, q) * big_b / 3600.0
                ram = 0.2 + n * (p + q) * 8 * 70 / 1024 ** 3
                rows.append(["phase3_size_grid", f"n{n}_a{a}_p{p}", n, p, q, a,
                             "", "", "", big_b, hrs, ram])

    # --- X2 power surface ---
    for n in (250, 1000):
        for a in (0.1, 0.5, 0.9):
            for p in (1, 5):
                q = q_of(n, a)
                if not (q > p - 1 and n > q + p):
                    continue
                secs = stat_rate(n, q) * len(THETAS_POWER12) * 300
                hrs = secs / 3600.0
                ram = 0.2 + n * (p + q) * 8 * 70 / 1024 ** 3
                rows.append(["phase3_power_surface", f"n{n}_a{a}_p{p}", n, p,
                             q, a, "", "", "", 3600, hrs, ram])
    rows.append(["phase3_power_surface", "n1000_a0.5_p25_R2", 1000, 25, 500,
                 0.5, "", "", "", 3600, stat_rate(1000, 500) * 12 * 300 / 3600,
                 0.45])

    # --- X3/X4 decisive grid ---
    for a in ALPHAS5:
        for kap in KAPPAS2:
            for p, n in ((1, 1000), (5, 2000)):
                q = q_of(n, a)
                if not (q > p - 1 and n > q + p):
                    continue
                for th in THETAS_DECISIVE8:
                    hrs = full_rate(n, q) * 400 / 3600.0
                    ram = 0.2 + n * (p + q) * 8 * 70 / 1024 ** 3
                    rows.append(["phase3_decisive_grid",
                                 f"a{a}_k{kap}_none_p{p}_th{th}", n, p, q, a,
                                 kap, th, "none", 400, hrs, ram])

    # --- X5 robustness ---
    for cell in robustness_cells():
        base = stat_rate(cell["n"], cell["q"]) * 1.6
        hrs = base * 4000 / 3600.0
        rows.append(["phase3_robustness", cell["cell_id"], cell["n"],
                     cell["p"], cell["q"], cell["alpha"], "", "",
                     cell["violation"], 4000, hrs, 0.4])
        if cell["violation"] in ("hetero_mild", "hetero_severe"):
            hrs_patch = base * 99 * 250 / 3600.0
            rows.append(["phase3_robustness", cell["cell_id"] + "_patch",
                         cell["n"], cell["p"], cell["q"], cell["alpha"], "",
                         "", cell["violation"] + "+wild_boot", 250,
                         hrs_patch, 0.4])

    # --- X6 scaling ---
    rows.append(["phase3_scaling", "scaling_suite", "", "", "", "", "", "", "",
                 "", 0.5, 3.0])

    total_h = sum(r[10] for r in rows)
    nb_cap = 6.0
    heavy = sorted([r for r in rows if r[10] >= 2.0 or r[11] >= 4.0],
                   key=lambda r: -r[10])
    light = [r for r in rows if r not in heavy]
    nb_load = []
    for r in heavy + light[::-1]:
        placed = False
        for i, load in enumerate(nb_load):
            if load + r[10] <= nb_cap:
                nb_load[i] += r[10]
                r.append("NOTEBOOK")
                r.append(f"NB{i:02d}")
                placed = True
                break
        if not placed:
            nb_load.append(r[10])
            r.append("NOTEBOOK")
            r.append(f"NB{len(nb_load)-1:02d}")
        r.append("YES" if (r[10] < 2.0 and r[11] < 4.0) else "NO")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("experiment,cell_id,n,p,q,alpha,kappa,theta,hetero,B,"
                   "pred_wall_h,pred_ram_gb,placement,notebook_id,"
                   "local_eligible".split(","))
        w.writerows(rows)
    print(json.dumps({"cells": len(rows), "serial_hours": round(total_h, 1),
                      "notebooks": len(nb_load),
                      "max_nb_h": round(max(nb_load), 2)}, indent=1))


if __name__ == "__main__":
    main()
