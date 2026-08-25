# WP-P3-R0: Preregistration memo (Phase 3)

**Project:** Weak-Instrument Frontier (Idea 2) | **Version:** 1.0 | **Stamped:** 2026-08-24
**Status:** FROZEN before any decisive run (plan rule WP-P3-R0; G2 condition discharged herein).
**Amendment policy:** this file is append-only after the first decisive run. Any deviation is recorded in Section 9 (Deviations log) with timestamp and reason; original text below is never edited.

---

## 0. Resolution of the G2 planning flag (`GRID_PRUNING_REQUIRED_AT_WP_P3_R0`)

The Phase-2 cost model priced the enumerated decisive factorial at ~6,100 serial hours against a 40-notebook (~400 h, target ≤ 6 h each) budget. Three changes close the gap by a factor of ~70:

1. **Fast shared-pass evaluation (implementation, not definition change).** One preprocessing + canonical pass per replication now feeds T_spec, tau selection, and the entire estimator battery (`spectraliv.experiments.fast_rep`). Estimator definitions are bit-compatible with the frozen `ivestimators`/`select_tau` code paths; equality is enforced by `tests/test_fast_equivalence.py` (max |Δbeta| < 1e-8 on tsls/liml/fuller/bekker/jive/pca/whiten/trunc_k and |Δtau| < 1e-12 at (n,q,p) ∈ {(300,40,1),(350,90,2),(400,150,5)}), which PASSED on 2026-08-24 before this memo was frozen. Measured speedup vs the Phase-2 smoke cost model: 4.9x at (1000,1,500), 5.7x at (1000,1,900), 10.4x at (2000,5,1000), 6.3x at (2000,5,1800).
2. **Grid pruning (below).** kappa {0.2,0.5,1,2} -> {0.5,2.0}; heteroskedasticity moved OUT of the decisive grid into X5 entirely; theta sweep 20 -> 8 (decisive) / 12 fixed points (power); size-grid B trimmed only at n = 2000 (20,000 -> 8,000; binomial se still 0.0024 vs the ±0.01 threshold).
3. **Measured per-rep costs** (benchmark script `weakiv_notes/scripts/bench_phase3.py`, BLAS threads = 1): full battery 0.74-1.50 s/rep at n=1000 p=1 (q=500-900); 0.78-8.39 s/rep at n=2000 p=5 (q=200-1800); statistic-only pass 0.10-0.23 s/rep at n=1000.

**Revised totals:** decisive grid ≈ 37 serial h; size grid ≈ 43 h; power surface ≈ 1.5 h; robustness ≈ 6 h; scaling ≈ 1 h. Total ≈ 89 serial hours -> ~14 self-contained notebooks at ≤ 6 h each (~70 notebook-hours), or a bounded local batch. The 40-notebook cap and the ≤ 240 notebook-hour margin target are both met with ≥ 3x headroom. RAM never exceeds 1 GB per worker (measured peak 287 MB in Phase 2 at larger n·q than most cells).

---

## 1. Frozen grids

### X1 size grid (`phase3_size_grid`) — claim: calibrated size
- Cells: n ∈ {250, 500, 1000, 2000} x alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9} x p ∈ {1, 2, 5}, plus p ∈ {25, 100} at n ∈ {1000, 2000}; A3 filter applied (55 cells total).
- B = 20,000 null replications/cell (data-level DGPs through the FULL pipeline), except n = 2000: B = 8,000.
- Critical values: exact Jacobi per cell drawn ONCE from stream(master, exp, cid, "cv") with b_cal = 4,000 (p = 1: closed-form Beta ppf). Secondary columns reported from the same reps: TW CV (cv_method = tw), naive chi-squared reference N*r^2max > chi2_{q,.95} (cv_method = naive_chi2, correction = none).
- Raw r^2max arrays saved per cell for QQ plots and tail checks.

### X2 power surface (`phase3_power_surface`) — claims: power at predicted outliers; location match
- Cells: (n, alpha) ∈ ({250, 1000} x {0.1, 0.5, 0.9}) x p ∈ {1, 5} (12 cells), plus ONE R2 emergence cell (n=1000, alpha=0.5, q=500, p=25) carrying the open P2-R2 falsifier (does a BBP-type detection threshold emerge as p grows?).
- theta grid FIXED COMMON, 12 points: {0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.32, 0.45, 0.62, 0.80, 0.93}. No adaptivity (no researcher dof).
- rho = 0 declared (first-stage-only experiment; Y plays no role in T_spec power or outlier location).
- B = 300 reps per theta point, per-cell streams stream(master, exp, cid_th{t}, b).
- Recorded per (cell, theta): power_exact, power_tw, P(F>10) empirical detector rate, outlier quantiles (median/q25/q75 of r^2max), affine g_pred = alpha + (1-alpha)*theta, and loc_err_sigma = |t_median - t_gpred| where t(x) = (logit(x) - mu_np)/sigma_np.
- Thresholds (preregistered): outlier location within 2 sigma_np of prediction for theta >= 0.07 (away from the TW edge where skewness is known to bias inversions; documented Phase-1 Finding 3).

### X3 coverage map + X4 risk curves (`phase3_decisive_grid`) — headline experiments
- Factors: alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9} x kappa ∈ {0.5, 2.0} (rho ∈ {0.447, 0.894}) x p ∈ {1, 5} x theta ∈ {0.05, 0.10, 0.18, 0.28, 0.40, 0.55, 0.72, 0.88} = 160 cells; n = 1000 (p = 1) / 2000 (p = 5); q = round(alpha*(n-1)); het = none (violations live in X5 only).
- B = 400 replications/cell, seeds cell_stream(exp, cid, b); beta_true = 0.5; alignment redrawn per replicate (Haar).
- Per replication, ONE fast pass records: r^2max; tau_hat; betas of tsls, liml, fuller, bekker, jive, pca_l (TW-rule retention), whiten (ridge_rel = 0.05), trunc_k1, trunc_k2 (p >= 2 only), trunc_kp (= tau 1 == 2SLS identity anchor); AR acceptance of the true beta at 95 pct (+ interval length at p = 1 via exact quadratic inversion, verified against brute force in tests); F>10, KP-rk>10, spectral-env pass flags; theta_hat = clip((N r^2max - q)/(N - q), 0, 1).
- Decision rules frozen NOW:
  - F>10: pass iff first-stage F > 10 (incumbent practice).
  - KP_rk>10: pass iff HC0-robust first-stage Wald/p statistic > 10.
  - spectral_env: pass iff theta_hat >= rho_env(alpha; delta = 0.1, rho_hi = 0.894), rho_env = A/(1+A), A = alpha(rho_hi/delta - 1). delta and rho_hi are DECLARED policy constants (memo Section 7); the envelope's TPR >= 95 pct requirement is re-tested here on the wider grid.
- Primary metric hierarchy (unchanged from plan): (1) size calibration [X1], (2) power/location at matched alternatives [X2], (3) RMSE [X4]; coverage comparisons [X3] read as conditional-on-pass realized AR coverage with paired seeds across rules.
- Thresholds (preregistered, verbatim from plan matrix):
  - X3 PASS iff exists an alpha-region (>= 2 adjacent alphas) with conditional AR coverage <= 0.90 under F>10 while spectral_env holds >= 0.93 on the same designs; FAIL if no divergence anywhere realistic.
  - X4 PASS iff truncated family beats the best baseline RMSE by >= 15 pct in >= 1 predeclared regime — predeclared HERE as: alpha >= 0.5, theta ∈ {0.18, 0.28}, either p — AND is never worse than best baseline by > 5 pct in any cell (paired-seed comparison; MC error allowance via paired bootstrap on the stored raw betas, 2,000 resamples). trunc_tauhat and trunc_k1 are the declared contenders; trunc_kp anchors the tau=1==2SLS identity.

### X5 robustness ladder (`phase3_robustness`)
- Violations at (n=1000, alpha=0.5, q=500), p ∈ {1, 5}: hetero_mild (w_i = 0.6+0.8 i/n), hetero_severe (w_i = 0.25+2.75(i/n)^2) on first-stage noise; heavy_v (V ~ scaled t5); heavy_eps (structural error scaled t5; first stage Gaussian); clustered_eps (cluster random effects, cluster size 25, ICC 0.2). All NULL designs (theta = 0): the question is size drift of the SHIPPED rule.
- B = 4,000 unpatched per cell: size_5pct vs the standard exact-Jacobi CV + AR coverage (beta_true = 0).
- Patch (E1 ladder, memo 6.4): wild bootstrap, rademacher signs on first-stage residuals, B_boot = 99, evaluated on 250 fresh reps for hetero_mild/severe only. Uses (X, Z) only (anti-leakage preserved).
- Predictions registered NOW (falsifiable): heavy_eps/clustered leave T_spec size calibrated (first stage untouched) but damage AR coverage (scope boundary, not our failure); hetero_* drift T_spec size upward, patched drift <= +0.02 counts as success; unpatchable drift > 0.05 triggers the plan's fail rule.

### X6 scaling (`phase3_scaling`)
- Individual mode: wall/RSS at (n, q, p) = (1e4, 100, 5), (3e4, 300, 5), (1e5, 500, 5), plus one full-battery rep at (2e5, 1000, 5) if RAM permits; summary-stats mode: synthetic spiked-LD Gram matrices at q ∈ {500, 1000, 2000, 3500, 5000}: eigendecomposition + canonical-spectrum timing.
- Threshold: fits the notebook envelope at MR-like scale (q <= 5e3).

## 2. Seed manifest (frozen)
- master_seed = 20260823 everywhere; conventions exactly as `Research/seeds.yaml`: per-replication cell_stream(exp, cid, b); critical-value stream stream(master, exp, cid[, "_th{t}"], "cv") with b_cal = 4,000; spec_test internal cv-stream reserved for compatibility runs only (not used by runners).
- seeds.yaml is amended by this memo (pre-prune values quoted in Section 9); no other seed edits.

## 3. Comparison protocol (equal information/budgets)
- Equal data: every rule/statistic within a replication sees the SAME draw; all estimators consume the same prepared blocks.
- Tuning budgets: our tau_hat and the pca TW-retention use internal data-driven rules with NO free constants; whiten ridge_rel = 0.05 fixed ex ante (their paper's default); NO cross-method validation splits are needed because no method tunes on held-out data. Any late addition of a tuned variant would be logged as a deviation.
- Failed runs: preserved with reasons in the deviations log; never silently rerun with different seeds (resume uses identical streams; checkpoint markers carry sha256).

## 4. Analysis & merge rules
- `merge_results.py` validates every registered cell (presence + checksum) before concatenation; gate memo G3 is written only from merged tables.
- Paired-seed analysis for X3/X4 (same draws across rules/estimators); uncertainty from paired bootstrap over replications, never rank averages alone.
- Figures generated by numbered scripts only; no hand-edited figures.

## 5. Compute placement (per plan policy: LOCAL if predicted < 2 h AND < 4 GB; else NOTEBOOK)
- LOCAL: size-grid n <= 1000 cells, power surface, robustness unpatched rows, scaling, decisive p = 1 cells. Local Pool capped at 4 workers (machine currently loaded by unrelated jobs; etiquette rule), BLAS threads = 1, nice 10.
- NOTEBOOK: all n = 2000 cells (size p >= 5, decisive p = 5) + robustness patch rows; packed ≤ 6 h/notebook round-robin → est. 12-16 notebooks, each self-contained (embedded package tarball + config + seed manifest + per-cell checkpoints + mandatory download-fallback block).
- Notebook budget after pruning: ~70 notebook-hours << 240 margin target. FLAG RESOLVED.

## 6. Implementation notes binding the runners
- fast_rep equivalence test MUST be green (it is; see §0) and stays in CI.
- CellCV reuses one ensemble draw per cell for both cv_exact and p-values (empirical sf with the +0.5/(B+1) continuity form, matching jacobiquantiles.largest_root_pvalue).
- pca_l retention: contiguous-from-top TW-outlier count on the raw-scale instrument Gram spectrum with Johnstone-BW centering mu = (sqrt(n-1)+sqrt(q))^2, sigma = (sqrt(n-1)+sqrt(q))(1/sqrt(n-1)+1/sqrt(q))^{1/3} at level 95 pct, floor l = 1.
- KP-rk proxy: the exact Kleibergen-Paap rk Wald statistic needs a qp x qp sandwich (infeasible at q ~ 1800). In the HOMOSKEDASTIC decisive grid the HC0 and classical covariance forms coincide in expectation, so the "> 10" flag uses the classical per-column first-stage Wald statistics W_j = (n-q)*fittedSS_j/residSS_j, summed over columns and divided by p (~ 1 under H0), computed from cached Gram factors at zero extra cost. The HC0 variant (`kp_rk_wald_hc0`) is implemented and reserved for heteroskedastic deployments.
- AR interval (p = 1): exact inversion of the F-form AR inequality (quadratic in beta_0); empty sets recorded as length -1; (-inf, inf) impossible at these df but handled.

## 7. Declared constants (policy, not estimates)
delta = 0.1; rho_hi = 0.894 (kappa = 2 correspondence); beta_true = 0.5; level = 0.05 throughout; b_cal = 4,000; B_boot = 99; hetero profiles as in §X5; theta grids as in §X2/X3.

## 8. Gate consequences (unchanged from plan)
Judgment at G3 strictly by the thresholds above. Give-up rules (KILL testing layer / KILL frontier claim / KILL estimation layer / INCREMENTAL-ONLY / PIVOT-to-R2) are quoted verbatim from plan Section 7 and are NOT restated or modified here.

## 9. Deviations log (append-only; empty at freeze)

| # | Timestamp | Item | Deviation | Reason |
|---|---|---|---|---|
| 1 | 2026-08-24 (at freeze) | dgps.make_single_spike p > 1 crash | Fixed frozen module: drafted version orthogonalized extra noise columns against realized v[:,0] and crashed for EVERY p > 1 ((n,p-1) @ (n,1) matmul mismatch); replaced with plain iid noise columns per the documented contract ("remaining columns pure noise"). Regression covered by test_dgps (p=2 path) and new equivalence tests. | Never exercised in Phase 2: the smoke runner called make_single_spike without p (default 1) for ALL cells, so labeled-p cells actually ran p = 1 data. Phase-2 cost-model labels therefore carried p from config while compute was p = 1; impact minor (cost coefficient of n*p^2 measured ~= 0). Recorded per no-silent-repairs rule. |
| 2 | 2026-08-24 (at freeze) | seeds.yaml amendment | phase3 entries updated to pruned design: power_surface {B: 300/theta, thetas: 12 fixed points, R2 cell added}; decisive_grid {B: 400, kappas: [0.5, 2.0], hetero: [none], thetas: 8 points}; size_grid {B: 20000 (8000 at n=2000)}. Pre-prune values were {B_default: 400, thetas: "20-point sweep", kappas: [0.2, 0.5, 1.0, 2.0], hetero: [none, mild, severe]} (B_size 20000 everywhere). | The pruning act itself, mandated by gate review G2 condition. |
| 3 | 2026-08-24 (at freeze) | schemas.md amendment | power_surface gains power_f10, loc_err_sigma; risk_curves estimator names realized as trunc_k{k} with trunc_kp = tau-1 anchor (analysis maps to trunc_tauhat/trunc_tau1 post hoc from raw npz); robustness violation set realized as hetero_mild/hetero_severe/heavy_v/heavy_eps/clustered (heavy-tail split into first-stage vs structural variants, both preregistered in §1-X5). | Schema completion at registration time, before any results exist. |
| 4 | 2026-08-24 (pre-run) | Toolchain binding | All decisive compute executes from GitHub repo `hugogobato/weakiv-frontier` @ commit `6e941d14487b...` (main); notebooks record the resolved sha into their manifests at runtime. Equivalence tests green at this sha. | Reproducibility: pins code to the frozen memo before first decisive run. |
| 5 | 2026-08-24 (pre-run) | Execution routing | ALL cells routed NOTEBOOK-first (17 notebooks, colab_plan_v2.csv) because the workstation was under unrelated load (loadavg 45-55 on 8 visible cores) making even <2 h local cells impractical; `run_local_batch.py` remains available for opportunistic local execution of the same registered cells with identical seeds. Grids, seeds, thresholds unchanged. Placement is an operational choice, not a scientific one. | Machine-etiquette rule; notebook budget has >= 3x headroom. |
| 6 | 2026-08-25 (post-run, analysis time) | run_decisive_cell risk CSV column labels | Executed writer emits base_cols + [theta, rho_true, ...] while the header declares [..., theta, rho_true, het]; net effect: columns 7-9 are label-shifted (field "rho_true" holds theta, field "het" holds rho_true). Cosmetic only (all values present, het == none everywhere in the decisive grid). Analysis parses positionally; package code stays pinned @ b4e10ed. | Discovered during merge validation of downloaded zips; no statistical quantity affected. |
| 7 | 2026-08-25 (post-run) | Decisive-grid session-overwrite data loss | run_decisive_cell writes {base}_coverage.csv / _risk.csv / _raw.npz in OVERWRITE mode keyed by base cell id, so within one notebook session each later theta clobbered the previous theta's artifacts of the same base before download. Cross-notebook recovery (each notebook's LAST theta per base survives in its zip) restored 39/160 registered states; the remaining 121 are re-dispatched as NBD1-NBD5 with IDENTICAL seeds/streams (cell_stream(exp, cid_th{t}, b)) plus a runner-level copy step persisting per-theta filenames ({base}_th{t}_* including sha256 done markers) so the merge registry is satisfied and the loss mode cannot recur. Package code unchanged (still b4e10ed). | Implementation defect discovered at analysis time; reruns are the registered runs, not redesigns. |
| 8 | 2026-08-25 (post-run, analysis time) | Size-grid B at n = 2000 | All executed size cells used B = 20,000 (runner B_PARAMS constant), including n = 2000 where this memo planned B = 8,000. Conservative direction only: binomial se 0.0015 vs planned 0.0024; no threshold affected; raw r^2max arrays retained for all 80 cells. | Notebook template constant was not specialized per n; detected post hoc from _done markers (big_b field). |
| 9 | 2026-08-25 (operational) | Retry executions | Size-grid cells that exceeded Colab session limits were re-run as NB00R-NB04R and NB12R (identical streams; per-cell checkpoint-zip downloads added to retry runners so a dead session loses only the in-flight cell). All 80 registered size cells now validated by sha256 markers via merge_results.py. | Operational completion of the same registered cells. |

*Nothing below this line may exist at freeze time.*
