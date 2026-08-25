"""Generate retry notebooks for the 121 lost decisive-grid (base,theta) states.

Root cause (deviations #6/#7): run_decisive_cell writes {base}_coverage.csv /
{base}_risk.csv / {base}_raw.npz in OVERWRITE mode while the plan enumerates
8 thetas per base cell; earlier thetas of the same base inside one Colab
session were clobbered before download. Recovery captured 39/160 states; this
regenerates the missing 121 with IDENTICAL seeds/streams (results are the
registered runs, not a redesign).

Fix applied at RUNNER level (package code stays pinned @ b4e10ed): after each
run_decisive_cell call the runner copies the outputs to their per-theta names
({base}_th{t}_*) including sha256 done markers, matching merge_results.py's
registry. Per-cell checkpoint downloads kept from the NBxxR generation.

Usage: python3 gen_decisive_retry.py
"""
import csv
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
from gen_retry_notebooks import RUNNER_RETRY  # noqa: E402

PLAN = os.path.join(ROOT, "Research", "weakiv_notes", "colab_plan_v2.csv")
OUTDIR = os.path.join(ROOT, "Research", "weakiv_notes", "notebooks")
RESULTS = os.path.join(ROOT, "Research", "weakiv_results")

CAP_H = 6.0
SAFETY = 1.15          # decisive preds tracked actuals within ~10%


def missing_cells():
    dc = os.path.join(RESULTS, "phase3_decisive_grid", "cells")
    have = {f[:-len("_coverage.csv")] for f in os.listdir(dc)
            if f.endswith("_coverage.csv")}
    rows = []
    with open(PLAN) as f:
        for r in csv.DictReader(f):
            if r["experiment"] != "phase3_decisive_grid":
                continue
            cid = r["cell_id"]
            if cid in have:
                continue
            base, th = cid.rsplit("_th", 1)
            rows.append({"row": r, "base": base, "theta": float(th),
                         "pred_h": float(r["pred_wall_h"])})
    return sorted(rows, key=lambda x: -x["pred_h"])


RENAME_BLOCK = '''
        if exp == "phase3_decisive_grid":
            base = cell["cell_id"]; suf = "_th%s" % cell["theta"]
            cdir = os.path.join(OUT_ROOT, exp, "cells")
            for s in ("_coverage.csv", "_risk.csv", "_raw.npz"):
                src = os.path.join(cdir, base + s)
                if os.path.exists(src):
                    shutil.copyfile(src, os.path.join(cdir, base + suf + s))
                    if s != "_raw.npz":
                        dm = os.path.join(cdir, "_done_" + base + s + ".json")
                        dn = os.path.join(cdir, "_done_" + base + suf + s
                                          + ".json")
                        if os.path.exists(dm):
                            shutil.copyfile(dm, dn)
'''

RUNNER_DECISIVE = RUNNER_RETRY.replace(
    '        print("[done] %s in %.1fs" % (cid, time.time() - t0))\n'
    '        summary.append([cid, round(time.time() - t0, 1)])',
    '        print("[done] %s in %.1fs" % (cid, time.time() - t0))\n'
    '        summary.append([cid, round(time.time() - t0, 1)])\n'
    + RENAME_BLOCK.rstrip("\n"))


def make_nb(nb_id, items):
    est = sum(i["pred_h"] * SAFETY for i in items)
    lines = [
        "# Weak-Instrument Frontier | Phase-3 decisive-grid RERUN (%s)" % nb_id,
        "",
        "**Purpose:** regenerate the %d decisive (base,theta) states lost to "
        "the session-overwrite issue (deviations #6/#7). Identical seeds and "
        "streams as the registered runs; repo pinned `%s` @ `%s`."
        % (len(items), REPO_URL, REPO_REF),
        "",
        "| slice | value |",
        "|---|---|",
        "| cells | %d |" % len(items),
        "| predicted wall | %.1f h (+%d%% safety => ~%.1f h) |"
        % (sum(i["pred_h"] for i in items), int((SAFETY - 1) * 100), est),
        "| master seed | 20260823 |",
        "",
        "Per-theta output capture (`_th*` filenames + sha256 markers) is "
        "built into the runner; every finished cell also triggers a "
        "checkpoint zip download.",
    ]
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"colab": {"provenance": []},
                     "kernelspec": {"name": "python3",
                                    "display_name": "Python 3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [],
             "source": SETUP_CELL.format(repo_url=REPO_URL, repo_ref=REPO_REF,
                                         nb_id=nb_id)},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [],
             "source": RUNNER_DECISIVE.format(cells=json.dumps(
                 [{"experiment": "phase3_decisive_grid", "cell_id": i["row"]["cell_id"],
                   "cell": {"n": int(i["row"]["n"]), "p": int(i["row"]["p"]),
                            "q": int(i["row"]["q"]),
                            "alpha": float(i["row"]["alpha"]),
                            "kappa": float(i["row"]["kappa"]),
                            "theta": i["theta"],
                            "cell_id": i["base"]}}
                  for i in items], indent=1))},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": DOWNLOAD_CELL.format(nb_id=nb_id)},
        ],
    }
    return json.dumps(nb, indent=1)


def main():
    miss = missing_cells()
    print(f"{len(miss)} missing decisive states, "
          f"{sum(m['pred_h'] for m in miss):.2f} predicted h")
    buckets, cur, curh = [], [], 0.0
    for m in miss:
        if cur and curh + m["pred_h"] > CAP_H:
            buckets.append(cur); cur, curh = [], 0.0
        cur.append(m); curh += m["pred_h"]
    if cur:
        buckets.append(cur)
    index = []
    for k, b in enumerate(buckets, 1):
        nb_id = "NBD%d" % k
        path = os.path.join(OUTDIR, "%s.ipynb" % nb_id)
        open(path, "w").write(make_nb(nb_id, b))
        index.append({"notebook": "%s.ipynb" % nb_id, "cells": len(b),
                      "cells_list": [i["row"]["cell_id"] for i in b],
                      "pred_h": round(sum(i["pred_h"] for i in b), 2)})
        print("wrote", path, f"({len(b)} cells, {sum(i['pred_h'] for i in b):.2f} pred h)")
    with open(os.path.join(OUTDIR, "_index_decisive_retry.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
