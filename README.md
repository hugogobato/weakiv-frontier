# weakiv-frontier

Phase-3 execution repository for the **Weak-Instrument Frontier** project
(Idea 2: canonical correlations, Jacobi ensembles, and high-dimensional
instrumental variables).

The Colab notebooks in `Research/weakiv_notes/notebooks/` clone THIS repo,
install the pinned `spectraliv` package from it, and run their assigned grid
cells with per-cell checkpoint/resume. All stochastic steps draw from master
seed `20260823` via the stream conventions frozen in
`Research/seeds.yaml`; metrics, tolerances, grids, and decision rules are
preregistered in `Research/weakiv_preregistration.md` (v1.0, 2026-08-24)
BEFORE any decisive run.

## Layout

| Path | Content |
|---|---|
| `Research/spectraliv/` | The tested Python package (`src/spectraliv/`, `tests/`) |
| `Research/seeds.yaml` | Master seed manifest + per-experiment seed policy |
| `Research/schemas.md` | Fixed result-file column dictionaries |
| `Research/weakiv_preregistration.md` | WP-P3-R0 preregistration memo (append-only after first decisive run) |
| `Research/weakiv_notes/colab_plan_v2.csv` | Pruned Phase-3 grids: costs, placements, notebook buckets |
| `Research/weakiv_notes/notebooks/` | Self-contained Colab notebooks (`NB00` ... `NB16`) |
| `Research/weakiv_notes/scripts/` | Generators, benchmark, pilot, merge validator, optional local runner |

## How to run (Colab)

1. Upload any `NBxx.ipynb` to Colab and "Run all". It clones this repo at
   branch `main` (the resolved commit sha is recorded into its manifest),
   installs the package, and runs its slice.
2. Each cell writes schema-compliant CSVs plus `_done` markers (sha256) so a
   restarted notebook skips finished work.
3. At the end the results directory is zipped and downloaded automatically
   (mandatory download-fallback block).
4. Unzip downloads into `Research/weakiv_results/<experiment>/...` locally,
   then validate + merge:

```
python3 Research/weakiv_notes/scripts/merge_results.py --exp all
```

The gate memo G3 is written only from complete, checksum-validated merges.

## Recommended execution order

1. Any `phase3_size_grid` notebooks (correctness layer X1 gates everything).
2. `power_surface` notebook (X2).
3. Decisive-grid notebooks (X3/X4), then robustness (X5), then scaling (X6).

Predicted cost per notebook is in its header cell and in `_index.json`
(total ~98 serial hours across 17 notebooks; cap 6 h each).

## Reproduce any single number

Every result row carries `(experiment, cell_id, seed)`; streams are
`stream(20260823, experiment, cell_id[, rep_index])` with critical values
from `stream(20260823, experiment, cell_id, "cv")`. See `seeds.yaml`.
