# Result schemas (fixed column dictionaries; enforced by merge scripts)

Every parquet/csv produced in Phase 3+ must match one of these schemas exactly.
Each row additionally carries the seed provenance required by `Research/seeds.yaml`.
Amended 2026-08-24 at WP-P3-R0 freeze (before any results exist; see
weakiv_preregistration.md Section 9, item 3): power_surface gains power_f10 and
loc_err_sigma; risk_curves estimator names realized as trunc_k{k} with the
tau=1 anchor trunc_kp (analysis maps to trunc_tauhat/trunc_tau1 post hoc);
robustness violation set realized as hetero_mild/hetero_severe/heavy_v/
heavy_eps/clustered (heavy-tail split first-stage vs structural).

## size_grid (X1)
| column | type | notes |
|---|---|---|
| experiment | str | "size_grid" |
| cell_id | str | e.g. n1000_a0.5_p5 |
| n, p, q | int | design dimensions |
| alpha | float | q/N target |
| cv_method | str | "exact_jacobi" or "tw" |
| correction | str | "johnstone2009" / "none" (ablation) |
| rejects | int | rejections at 5 pct level |
| B | int | replications (default 2e4) |
| seed | int | per-cell master stream id |

## power_surface (X2)
experiment, cell_id, n, p, q, alpha, theta, rho, power_exact, power_tw,
outlier_r2_median, outlier_r2_q25, outlier_r2_q75, g_pred, B, seed,
power_f10, loc_err_sigma

## coverage_map (X3)
experiment=decisive_grid subset; columns: cell_id, n, p, q, alpha, kappa, het,
rule ("F>10", "KP_rk>10", "spectral_env"), ar_cov_95, n_flagged, B, seed

## risk_curves (X4)
cell_id, n, p, q, alpha, kappa, theta, rho_true, het, estimator
("tsls","liml","fuller","bekker","jive","trunc_k1","trunc_k2","trunc_kp",
"pca_l","whiten"), rmse, mae, bias, sd, tau_used, B, seed

## robustness (X5)
cell_id, n, p, q, violation ("hetero_mild","hetero_severe","heavy_v",
"heavy_eps","clustered"), patch ("none","wild_boot"), size_5pct, ar_cov_95,
B, seed

## scaling (X6)
n, p, q, mode ("individual","summary_stats"), seconds, peak_gb, machine, seed
