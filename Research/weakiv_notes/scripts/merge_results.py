"""Phase-3 merge & validation script (plan Section 7, compute policy).

Validates that every registered cell of an experiment is present with a
passing sha256 `_done` marker, then concatenates per-cell CSVs into
`Research/weakiv_results/<exp>/<exp>.csv`. No gate memo may be written from
partial merges (plan rule).

Usage:
    python3 merge_results.py --exp phase3_size_grid
    python3 merge_results.py --exp all
"""
import argparse
import csv
import hashlib
import json
import os
import sys

ROOT = os.environ.get(
    "WEAKIV_ROOT",
    "/home/hugo_souto/Stuff/Research/RMT/Idea2")
RESULTS = (os.environ["WEAKIV_RESULTS"] if "WEAKIV_RESULTS" in os.environ
           else os.path.join(ROOT, "Research", "weakiv_results"))

REGISTRY = {
    "phase3_size_grid": {
        "module": "spectraliv.experiments:size_grid_cells",
        "files": ["{cid}.csv"],
    },
    "phase3_power_surface": {
        "module": "spectraliv.experiments:power_grid_cells",
        "files": ["{cid}.csv"],
    },
    "phase3_decisive_grid": {
        "module": "spectraliv.experiments:decisive_theta_cells",
        "files": ["{cid}_coverage.csv", "{cid}_risk.csv"],
    },
    "phase3_robustness": {
        "module": "spectraliv.experiments:robustness_cells",
        "files": ["{cid}.csv"],
    },
}


def decisive_theta_cells():
    from spectraliv.experiments import THETAS_DECISIVE8, decisive_grid_cells

    out = []
    for th in THETAS_DECISIVE8:
        for c in decisive_grid_cells():
            cc = dict(c)
            cc["cell_id"] = c["cell_id"] + f"_th{th}"
            cc["theta"] = th
            out.append(cc)
    return out


def cells_for(exp):
    if exp == "phase3_decisive_grid":
        return decisive_theta_cells()
    from spectraliv.experiments import (
        power_grid_cells,
        robustness_cells,
        size_grid_cells,
    )
    return {
        "phase3_size_grid": size_grid_cells,
        "phase3_power_surface": power_grid_cells,
        "phase3_robustness": robustness_cells,
    }[exp]()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_exp(exp):
    reg = REGISTRY[exp]
    exp_dir = os.path.join(RESULTS, exp)
    cell_dir = os.path.join(exp_dir, "cells")
    if not os.path.isdir(cell_dir):
        return {"experiment": exp, "status": "MISSING", "ok": 0,
                "missing": "whole directory"}
    cells = cells_for(exp)
    rows, problems, ok_cells = [], [], 0
    for cell in cells:
        cid = cell["cell_id"]
        cell_ok = True
        for tmpl in reg["files"]:
            fname = tmpl.format(cid=cid)
            fpath = os.path.join(cell_dir, fname)
            done_path = os.path.join(cell_dir, "_done_" + fname + ".json")
            if not (os.path.exists(fpath) and os.path.exists(done_path)):
                problems.append(f"{cid}: missing {fname} or marker")
                cell_ok = False
                continue
            with open(done_path) as f:
                done = json.load(f)
            if done.get("sha256") != sha256_file(fpath):
                problems.append(f"{cid}: checksum mismatch {fname}")
                cell_ok = False
        if cell_ok:
            ok_cells += 1
            for tmpl in reg["files"]:
                with open(os.path.join(cell_dir, tmpl.format(cid=cid))) as f:
                    rdr = csv.reader(f)
                    header = next(rdr)
                    for r in rdr:
                        rows.append(r)
    if problems:
        return {"experiment": exp, "status": "INCOMPLETE",
                "ok_cells": ok_cells, "total_cells": len(cells),
                "problems": problems[:20], "n_problems": len(problems)}
    out_path = os.path.join(exp_dir, exp + ".csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return {"experiment": exp, "status": "MERGED", "cells": len(cells),
            "rows": len(rows), "out": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    args = ap.parse_args()
    exps = list(REGISTRY) if args.exp == "all" else [args.exp]
    for exp in exps:
        res = merge_exp(exp)
        print(json.dumps(res, indent=1))
        if res["status"] in ("MISSING", "INCOMPLETE"):
            sys.exit(1)


if __name__ == "__main__":
    main()
