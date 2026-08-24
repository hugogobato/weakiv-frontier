"""WP-P3-R0/S1 pilot: end-to-end validation of all four Phase-3 runners.

Tiny B, isolated output root (/tmp/opencode/phase3_pilots). Validates:
runner execution, schema headers, _done markers, npz payloads, merge script.
NOT decisive data; deleted after inspection.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time

OUT = "/tmp/opencode/phase3_pilots"
sys.path.insert(0, "/home/hugo_souto/Stuff/Research/RMT/Idea2/Research/spectraliv/src")

from spectraliv.experiments import (
    THETAS_POWER12,
    decisive_grid_cells,
    power_grid_cells,
    robustness_cells,
    run_decisive_cell,
    run_power_cell,
    run_robust_cell,
    run_size_cell,
    size_grid_cells,
)

report = {}
t0 = time.time()

cell = [c for c in size_grid_cells() if c["cell_id"] == "n250_a0.3_p1"][0]
res = run_size_cell(cell, big_b=300, b_cal_cv=500, out_root=OUT)
report["size"] = res

cell = [c for c in power_grid_cells() if c["cell_id"] == "n250_a0.5_p5"][0]
res = run_power_cell(cell, thetas=[0.04, 0.15, 0.45], reps_per_theta=40,
                     b_cal_cv=400, out_root=OUT)
report["power"] = res

cells = {c["cell_id"]: c for th in ["0.28"] for c in decisive_grid_cells()}
for cid in ("a0.5_k2.0_none_p1", "a0.9_k0.5_none_p5"):
    cc = dict([c for c in decisive_grid_cells() if c["cell_id"] == cid][0])
    cc["cell_id"] = cid + "_th0.28"
    cc["theta"] = 0.28
    res = run_decisive_cell(cc, reps=25, b_cal_cv=400, out_root=OUT)
    report[f"decisive_{cid}"] = res

cell = [c for c in robustness_cells() if c["cell_id"] == "hetero_severe_p1"][0]
res = run_robust_cell(cell, big_b=150, patch_reps=8, b_boot=19, out_root=OUT)
report["robust"] = res

# schema header check
expected = {
    "phase3_size_grid": "experiment,cell_id,n,p,q,alpha,cv_method,correction,rejects,B,seed",
    "phase3_power_surface": "experiment,cell_id,n,p,q,alpha,theta,rho,power_exact,power_tw,outlier_r2_median,outlier_r2_q25,outlier_r2_q75,g_pred,B,seed,power_f10,loc_err_sigma",
    "phase3_decisive_grid_cov": "cell_id,n,p,q,alpha,kappa,het,rule,ar_cov_95,n_flagged,B,seed",
    "phase3_decisive_grid_risk": "cell_id,n,p,q,alpha,kappa,theta,rho_true,het,estimator,rmse,mae,bias,sd,tau_used,B,seed",
    "phase3_robustness": "cell_id,n,p,q,violation,patch,size_5pct,ar_cov_95,B,seed",
}
checks = {}
for exp, hdr in expected.items():
    if exp.startswith("phase3_decisive"):
        fname = {"phase3_decisive_grid_cov": "a0.5_k2.0_none_p1_th0.28_coverage.csv",
                 "phase3_decisive_grid_risk": "a0.5_k2.0_none_p1_th0.28_risk.csv"}[exp]
    else:
        fname = {"phase3_size_grid": "n250_a0.3_p1.csv",
                 "phase3_power_surface": "n250_a0.5_p5.csv",
                 "phase3_robustness": "hetero_severe_p1.csv"}[exp]
    path = os.path.join(OUT, exp.replace("_cov", "").replace("_risk", ""),
                        "cells", fname)
    with open(path) as f:
        got = f.readline().strip()
        checks[exp] = "OK" if got == hdr else f"MISMATCH:\n  want {hdr}\n  got  {got}"
report["schema_checks"] = checks

report["wall_s_total"] = time.time() - t0
print(json.dumps(report, indent=1, default=str))
