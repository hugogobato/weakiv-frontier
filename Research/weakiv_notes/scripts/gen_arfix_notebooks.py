"""Generate NBE* notebooks recomputing DECISIVE-grid AR acceptance correctly.

Reason (deviation #11): ar_accepts() tests beta in raw units against
standardized blocks; for theta > 0 the tested null is false in standardized
units, corrupting every decisive-grid AR acceptance value. This rerun
regenerates each registered replication from its stream (seeded by BASE cell
id - thetas share draws by the paired protocol) and recomputes acceptance
with the correct standardized-null vector b_std = beta / (sy/sx).

First-stage flags f_pass/env_pass stored in the existing raw.npz remain VALID
(they never touch y), so corrected conditional coverages combine stored flags
with corrected acceptance arrays ({cid}_arfix.npy).

Usage: python3 gen_arfix_notebooks.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from gen_notebooks import (  # noqa: E402
    DOWNLOAD_CELL,
    REPO_REF,
    REPO_URL,
    SETUP_CELL,
)

OUTDIR = os.path.join(ROOT, "Research", "weakiv_notes", "notebooks")
CAP_H = 2.5

RUNNER = '''import json, os, shutil, time, traceback
import numpy as np
from scipy.stats import f as fdist
from spectraliv.dgps import make_single_spike, rho_of_kappa
from spectraliv.preprocess import prepare
from spectraliv.rng import cell_stream

OUT_ROOT = "/content/results"
MASTER_SEED = 20260823
CELLS = {cells}

CELLDIR = os.path.join(OUT_ROOT, "phase3_decisive_grid", "cells")
os.makedirs(CELLDIR, exist_ok=True)


def base_of(cid):
    return cid.rsplit("_th", 1)[0]


def parse_cid(cid):
    a_part, rest = cid[1:].split("_k", 1)
    k_part, rest = rest.split("_none_p", 1)
    p_part, th_part = rest.split("_th", 1)
    return float(a_part), float(k_part), int(p_part), float(th_part)


def process(cid, reps):
    a, kap, p, theta = parse_cid(cid)
    n = 1000 if p == 1 else 2000
    q = int(round(a * (n - 1)))
    rho = rho_of_kappa(kap)
    out = np.empty(reps, dtype=bool)
    for b in range(reps):
        rng = np.random.default_rng(
            cell_stream("phase3_decisive_grid", base_of(cid), b,
                        master_seed=MASTER_SEED).integers(1 << 31))
        dgp = make_single_spike(n, q, theta, rho, rng, p=p, beta=0.5)
        xs, zs, yr, resc = prepare(dgp.x, dgp.z, dgp.y, None)
        sy = np.std(yr, ddof=1)
        ys = yr / sy
        chol = np.linalg.cholesky(zs.T @ zs)
        e = ys - xs @ (np.full(p, 0.5) / resc)
        w_e = np.linalg.solve(chol, zs.T @ e)
        num = max(float(w_e @ w_e), 1e-300)
        den = max(float(e @ e) - num, 1e-300)
        stat = (num / q) / (den / (n - q))
        out[b] = stat <= fdist.ppf(0.95, q, n - q)
    return out


def ckpt(tag):
    try:
        zp = "/content/results_NBE_ckpt_%s" % tag
        shutil.make_archive(zp, "zip", OUT_ROOT)
        from google.colab import files
        files.download(zp + ".zip")
        print("[ckpt] downloaded:", zp + ".zip")
    except Exception as e:
        print("(checkpoint download skipped):", e)


summary = []
for item in CELLS:
    cid = item["cell_id"]
    mk = os.path.join(CELLDIR, cid + "_arfix_done.json")
    if os.path.exists(mk):
        print("[skip]", cid)
        continue
    t0 = time.time()
    try:
        arr = process(cid, item["reps"])
        np.save(os.path.join(CELLDIR, cid + "_arfix.npy"), arr)
        with open(mk, "w") as f:
            json.dump({{"reps": len(arr), "seed": MASTER_SEED}}, f)
        print("[done] %s in %.1fs cov=%.4f" % (cid, time.time() - t0,
                                               arr.mean()))
        summary.append([cid, round(time.time() - t0, 1)])
        ckpt(cid.replace(".", ""))
    except Exception:
        print("[FAIL]", cid)
        traceback.print_exc()

with open(os.path.join(OUT_ROOT, "manifest_ARFIX.json"), "w") as f:
    json.dump({{"notebook_id": NB_ID, "git_sha": GIT_SHA,
                "cells": [c["cell_id"] for c in CELLS],
                "timings_s": summary}}, f, indent=1)
print(json.dumps(summary, indent=1))
'''


def est_h(q, reps=400):
    # crude single-core model calibrated on the local smoke (dominated by the
    # Cholesky of Z'Z and the QR inside prepare): ~0.9 s/rep at q=1799
    return reps * max(0.08, 0.9 * (q / 1799.0) ** 2) / 3600.0


def main():
    cells_dir = os.path.join(ROOT, "Research", "weakiv_results",
                             "phase3_decisive_grid", "cells")
    cids = sorted(f[:-len("_raw.npz")] for f in os.listdir(cells_dir)
                  if f.endswith("_raw.npz") and "_th" in f)
    items = []
    for cid in cids:
        a = float(cid[1:4])
        p = int(cid.rsplit("_p", 1)[1].split("_")[0])
        n = 1000 if p == 1 else 2000
        q = int(round(a * (n - 1)))
        items.append({"cell_id": cid, "reps": 400, "q": q})
    items.sort(key=lambda x: -est_h(x["q"]))
    buckets, cur, curh = [], [], 0.0
    for it in items:
        h = est_h(it["q"])
        if cur and curh + h > CAP_H:
            buckets.append(cur)
            cur, curh = [], 0.0
        cur.append(it)
        curh += h
    if cur:
        buckets.append(cur)
    index = []
    for k, b in enumerate(buckets, 1):
        nb_id = "NBE%d" % k
        header = (
            "# Weak-Instrument Frontier | Corrected AR-acceptance recompute "
            "(%s)\n\nDeviation #11 remediation: regenerates the registered "
            "replications (identical streams) and recomputes AR acceptance "
            "with the standardized-null vector. Writes `%s` per cell plus "
            "`_done` markers; checkpoint zip after every cell.\n" % (nb_id, "{cid}_arfix.npy"))
        nb = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"provenance": []},
                         "kernelspec": {"name": "python3",
                                        "display_name": "Python 3"}},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": header},
                {"cell_type": "code", "metadata": {}, "execution_count": None,
                 "outputs": [],
                 "source": SETUP_CELL.format(repo_url=REPO_URL,
                                             repo_ref=REPO_REF, nb_id=nb_id)},
                {"cell_type": "code", "metadata": {}, "execution_count": None,
                 "outputs": [],
                 "source": RUNNER.format(cells=json.dumps(b, indent=1))},
                {"cell_type": "code", "metadata": {}, "execution_count": None,
                 "outputs": [], "source": DOWNLOAD_CELL.format(nb_id=nb_id)},
            ],
        }
        path = os.path.join(OUTDIR, "%s.ipynb" % nb_id)
        open(path, "w").write(json.dumps(nb, indent=1))
        index.append({"notebook": "%s.ipynb" % nb_id, "cells": len(b),
                      "pred_h": round(sum(est_h(i["q"]) for i in b), 2)})
        print("wrote", path, "(%d cells, ~%.1f h)" % (len(b), sum(est_h(i["q"]) for i in b)))
    with open(os.path.join(OUTDIR, "_index_arfix_retry.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
