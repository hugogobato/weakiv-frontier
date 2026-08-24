"""Generate self-contained Colab notebooks from colab_plan_v2.csv.

Each notebook clones this repo (runtime sha recorded in its manifest),
installs spectraliv, runs its assigned cells with per-cell checkpoint/resume
(_done markers), then zips results and triggers the mandatory
download-fallback block.

Robustness "_patch" rows are cost accounting only: their compute runs inside
the base cell's run_robust_cell call; the generator folds their predicted
hours into the base cell and never dispatches them separately.
"""
import csv
import json
import os
from collections import defaultdict

ROOT = "/home/hugo_souto/Stuff/Research/RMT/Idea2"
PLAN = os.path.join(ROOT, "Research", "weakiv_notes", "colab_plan_v2.csv")
OUTDIR = os.path.join(ROOT, "Research", "weakiv_notes", "notebooks")

REPO_URL = "https://github.com/hugogobato/weakiv-frontier.git"
REPO_REF = "main"

SETUP_CELL = '''import json, os, subprocess, sys, time

REPO_URL = "{repo_url}"
REPO_REF = "{repo_ref}"
NB_ID = "{nb_id}"

if not os.path.isdir("weakiv-frontier"):
    subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_REF,
                    REPO_URL], check=True)
os.chdir("weakiv-frontier")
GIT_SHA = subprocess.check_output(
    ["git", "rev-parse", "HEAD"]).decode().strip()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                "./Research/spectraliv"], check=True)
sys.path.insert(0, "Research/spectraliv/src")
print("repo sha:", GIT_SHA)
'''

RUNNER_CELL = '''import traceback
from spectraliv import __version__ as pkg_version
from spectraliv.experiments import (
    run_size_cell, run_power_cell, run_decisive_cell, run_robust_cell,
    run_scaling,
)

OUT_ROOT = "/content/results"
MASTER_SEED = 20260823

B_PARAMS = {{
    "phase3_size_grid": 20000,
    "phase3_power_surface": 300,
    "phase3_decisive_grid": 400,
    "phase3_robustness": 4000,
}}
PATCH_PARAMS = {{"patch_reps": 250, "b_boot": 99}}

CELLS = {cells}

def done_markers(exp, cid):
    d = os.path.join(OUT_ROOT, exp, "cells")
    if exp == "phase3_decisive_grid":
        names = ["_done_%s_coverage.csv.json" % cid,
                 "_done_%s_risk.csv.json" % cid]
    else:
        names = ["_done_%s.csv.json" % cid]
    return all(os.path.exists(os.path.join(d, nm)) for nm in names)

summary = []
for item in CELLS:
    exp, cid = item["experiment"], item["cell_id"]
    if done_markers(exp, cid):
        print("[skip] %s (done markers present)" % cid)
        continue
    t0 = time.time()
    try:
        cell = dict(item["cell"])
        if exp == "phase3_size_grid":
            run_size_cell(cell, big_b=B_PARAMS[exp], b_cal_cv=4000,
                          out_root=OUT_ROOT)
        elif exp == "phase3_power_surface":
            run_power_cell(cell, reps_per_theta=B_PARAMS[exp], b_cal_cv=4000,
                           out_root=OUT_ROOT)
        elif exp == "phase3_decisive_grid":
            run_decisive_cell(cell, reps=B_PARAMS[exp], b_cal_cv=4000,
                              out_root=OUT_ROOT)
        elif exp == "phase3_robustness":
            run_robust_cell(cell, big_b=B_PARAMS[exp],
                            patch_reps=PATCH_PARAMS["patch_reps"],
                            b_boot=PATCH_PARAMS["b_boot"],
                            out_root=OUT_ROOT)
        elif exp == "phase3_scaling":
            run_scaling(out_root=OUT_ROOT)
        print("[done] %s in %.1fs" % (cid, time.time() - t0))
        summary.append([cid, round(time.time() - t0, 1)])
    except Exception:
        print("[FAIL] %s" % cid)
        traceback.print_exc()

manifest = {{
    "notebook_id": NB_ID, "git_sha": GIT_SHA,
    "package_version": pkg_version, "master_seed": MASTER_SEED,
    "b_params": B_PARAMS, "patch_params": PATCH_PARAMS,
    "cells": [c["cell_id"] for c in CELLS],
    "timings_s": summary,
}}
with open(os.path.join(OUT_ROOT, "manifest_%s.json" % NB_ID), "w") as f:
    json.dump(manifest, f, indent=1)
print(json.dumps(manifest, indent=1))
'''

DOWNLOAD_CELL = '''import shutil, os
os.chdir("/content")
shutil.make_archive("results_{nb_id}", "zip", "/content/results")
try:
    from google.colab import files
    files.download("results_{nb_id}.zip")
    print("Downloaded:", "results_{nb_id}.zip")
except Exception as e:
    print("(Not on Colab / download skipped):", e)
'''


def load_and_fold():
    with open(PLAN) as f:
        rows = list(csv.DictReader(f))
    folded = {}
    for r in rows:
        if r["placement"] != "NOTEBOOK":
            continue
        cid = r["cell_id"]
        key = (r["experiment"], cid[:-6] if cid.endswith("_patch") else cid)
        entry = folded.setdefault(key, {"rows": [], "extra_h": 0.0})
        if cid.endswith("_patch"):
            entry["extra_h"] += float(r["pred_wall_h"])
        else:
            entry["rows"].append(r)
    items = []
    for (exp, cid), v in sorted(folded.items()):
        if not v["rows"]:
            continue
        r = v["rows"][0]
        items.append({"row": r, "pred_h": float(r["pred_wall_h"]) + v["extra_h"]})
    return items


def build_cells_cfg(items):
    cells_cfg = []
    for it in items:
        r = it["row"]
        exp, cid = r["experiment"], r["cell_id"]
        if exp == "phase3_scaling":
            cells_cfg.append({"experiment": exp, "cell_id": cid, "cell": {}})
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
        cells_cfg.append({"experiment": exp, "cell_id": cid, "cell": cell})
    return cells_cfg


def make_nb(nb_id, items):
    header = (
        "# Weak-Instrument Frontier | Phase-3 decisive grid\n\n"
        "**Notebook:** `%s` | **Preregistration:** "
        "`Research/weakiv_preregistration.md` v1.0 (2026-08-24)\n\n"
        "**Slice:** %d cells, predicted **%.1f h** (cap 6 h) | "
        "master_seed `20260823`\n\n"
        "Per-cell checkpoints (`_done` markers carrying sha256) make this "
        "notebook resumable; re-running skips completed cells."
        % (nb_id, len(items), sum(i["pred_h"] for i in items))
    )
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"colab": {"provenance": []},
                     "kernelspec": {"name": "python3",
                                    "display_name": "Python 3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": header},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [],
             "source": SETUP_CELL.format(repo_url=REPO_URL, repo_ref=REPO_REF,
                                         nb_id=nb_id)},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [],
             "source": RUNNER_CELL.format(cells=json.dumps(
                 build_cells_cfg(items), indent=1))},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": DOWNLOAD_CELL.format(nb_id=nb_id)},
        ],
    }
    return json.dumps(nb, indent=1)


def main():
    items = load_and_fold()
    buckets = defaultdict(list)
    for it in items:
        buckets[it["row"]["notebook_id"]].append(it)
    os.makedirs(OUTDIR, exist_ok=True)
    index = []
    for nb_id, bucket in sorted(buckets.items()):
        with open(os.path.join(OUTDIR, "%s.ipynb" % nb_id), "w") as f:
            f.write(make_nb(nb_id, bucket))
        index.append({
            "notebook": "%s.ipynb" % nb_id,
            "cells": len(bucket),
            "predicted_h": round(sum(i["pred_h"] for i in bucket), 2),
            "experiments": sorted({i["row"]["experiment"] for i in bucket}),
        })
    with open(os.path.join(OUTDIR, "_index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
