"""Generate retry notebooks for Phase-3 cells that did not finish on Colab.

Reuses the exact SETUP/RUNNER/DOWNLOAD cell templates from gen_notebooks.py
(same repo pin, same B params, same done-marker resume logic) and dispatches
only the cells that are still missing from the downloaded results zips:

    NB00R <- NB00: n2000_a0.9_p1            (twin n2000_a0.7_p1  ~6.0 h)
    NB01R <- NB01: n2000_a0.9_p2            (twin n2000_a0.7_p2  ~6.3 h)
    NB02R <- NB02: n2000_a0.9_p5            (twin n2000_a0.7_p5  ~5.2 h)
    NB03R <- NB03: n2000_a0.9_p25           (twin n2000_a0.7_p25 ~6.2 h)
    NB04R <- NB04: n2000_a0.9_p100          (twin n2000_a0.7_p100~6.7 h)
    NB12R <- NB12: n2000_a0.5_p2/p25/p5     (twin n2000_a0.5_p1  ~3.7 h ea)

Size-grid wall time is independent of `alpha` (the null DGP depends only on
n, q, p), so the observed twin runtimes bound the retry runtimes.

One addition over the original template: after EVERY completed cell the
runner archives /content/results and triggers a browser download of the
partial zip, so a session that dies mid-notebook keeps all finished cells.

Usage: python3 gen_retry_notebooks.py
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
    build_cells_cfg,
    load_and_fold,
)

OUTDIR = os.path.join(ROOT, "Research", "weakiv_notes", "notebooks")

RETRIES = {
    "NB00R": ["n2000_a0.9_p1"],
    "NB01R": ["n2000_a0.9_p2"],
    "NB02R": ["n2000_a0.9_p5"],
    "NB03R": ["n2000_a0.9_p25"],
    "NB04R": ["n2000_a0.9_p100"],
    "NB12R": ["n2000_a0.5_p2", "n2000_a0.5_p25", "n2000_a0.5_p5"],
}

EST_H = {
    "n2000_a0.9_p1": 6.0, "n2000_a0.9_p2": 6.3, "n2000_a0.9_p5": 5.2,
    "n2000_a0.9_p25": 6.2, "n2000_a0.9_p100": 6.7,
    "n2000_a0.5_p2": 3.7, "n2000_a0.5_p25": 3.7, "n2000_a0.5_p5": 3.7,
}

SOURCE_NB = {
    "NB00R": "NB00", "NB01R": "NB01", "NB02R": "NB02",
    "NB03R": "NB03", "NB04R": "NB04", "NB12R": "NB12",
}

RUNNER_RETRY = '''import shutil, traceback, time
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


def checkpoint_download(tag):
    """Zip everything so far and push it to the browser. A dead session then
    only loses the cell that was in flight, never finished ones."""
    try:
        zpath = "/content/results_%s_ckpt_%s" % (NB_ID, tag)
        shutil.make_archive(zpath, "zip", OUT_ROOT)
        from google.colab import files
        files.download(zpath + ".zip")
        print("[ckpt] downloaded:", zpath + ".zip")
    except Exception as e:
        print("(checkpoint download skipped):", e)


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
                            b_boot=PATCH_PARAMS["b_boot"], out_root=OUT_ROOT)
        elif exp == "phase3_scaling":
            run_scaling(out_root=OUT_ROOT)
        print("[done] %s in %.1fs" % (cid, time.time() - t0))
        summary.append([cid, round(time.time() - t0, 1)])
        checkpoint_download(cid.replace(".", ""))
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


def make_retry_nb(nb_id, items):
    src = SOURCE_NB[nb_id]
    lines = [
        "# Weak-Instrument Frontier | Phase-3 RETRY (%s leftovers)" % src,
        "",
        "**Notebook:** `%s` | **Retries unfinished cells of:** `%s.ipynb` | "
        "repo pinned to `%s` @ `main` (sha used by all completed runs)"
        % (nb_id, src, REPO_URL),
        "",
        "| cell | experiment | expected wall |",
        "|---|---|---|",
    ]
    for it in items:
        r = it["row"]
        lines.append("| `%s` | %s | ~%.1f h |"
                     % (r["cell_id"], r["experiment"], EST_H[r["cell_id"]]))
    lines += [
        "",
        "Expected wall times come from the already-completed size-grid twins "
        "(size-grid compute does not depend on `alpha`, only `n, q, p`). "
        "Keep each notebook on its own runtime; total session budget should "
        "stay under Colab's limit.",
        "",
        "After every finished cell the runner zips `/content/results` and "
        "triggers a browser download (`*_ckpt_*` files), so a dropped "
        "session only loses the cell in flight.",
    ]
    header = "\n".join(lines)
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
             "source": RUNNER_RETRY.format(cells=json.dumps(
                 build_cells_cfg(items), indent=1))},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": DOWNLOAD_CELL.format(nb_id=nb_id)},
        ],
    }
    return json.dumps(nb, indent=1)


def main():
    by_cid = {it["row"]["cell_id"]: it for it in load_and_fold()}
    index = []
    for nb_id, cids in RETRIES.items():
        missing = [c for c in cids if c not in by_cid]
        if missing:
            raise SystemExit("unknown cell ids for %s: %s" % (nb_id, missing))
        items = [by_cid[c] for c in cids]
        path = os.path.join(OUTDIR, "%s.ipynb" % nb_id)
        with open(path, "w") as f:
            f.write(make_retry_nb(nb_id, items))
        index.append({
            "notebook": "%s.ipynb" % nb_id,
            "retry_of": SOURCE_NB[nb_id],
            "cells": [it["row"]["cell_id"] for it in items],
            "experiments": sorted({it["row"]["experiment"] for it in items}),
            "pred_h": round(sum(float(it["row"]["pred_wall_h"])
                                for it in items), 2),
            "expected_h": round(sum(EST_H[c] for c in cids), 2),
        })
        print("wrote", path)
    with open(os.path.join(OUTDIR, "_index_retry.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
