# Research Plan: The Weak-Instrument Frontier (Idea 2)

**Project:** Canonical Correlations, Jacobi Ensembles, and High-Dimensional Instrumental Variables
**Date:** 2026-08-23
**Source idea:** Random_Matrix_Research_Ideas.md, Idea 2
**Plan version:** 1.0 (initial planning; simulation and application evidence not yet produced)
**Classification:** PROMISING BUT UNPROVEN (see Section 1)

---

## 1. Executive verdict

1. **Current classification:** `promising but unproven`. Prior-art and validity gates are provisionally clear; simulation, application, and inference evidence has not been produced.
2. **Confidence and evidence level:** Moderate. Load-bearing sources verified at E2-E3 (Section 14); the primary-vocabulary novelty scan returned zero hits on arXiv (executed 2026-08-23); secondary-vocabulary sweeps are not yet done (scheduled as WP-P1-A1).
3. **One-sentence proposed contribution:** Replace the folkloric "F > 10" weak-instrument rule with an RMT-derived weak-instrument frontier (the minimal population canonical correlation that is simultaneously detectable and harmful) plus a Tracy-Widom/Jacobi-calibrated joint relevance test valid when the number of instruments grows with sample size (q/n to alpha), together with exact large-system risk formulas for spectrally truncated 2SLS.
4. **Contribution vs engine vs application vs decoration:**
   - *Real contribution:* items C1 (calibrated relevance test), C2 (weak-instrument frontier), C3 (truncated-2SLS risk theory), framed by C6 ("retiring F > 10").
   - *Engine:* Jacobi-ensemble quantile computation, Tracy-Widom evaluation, deterministic-equivalent solvers.
   - *Application:* Mendelian randomization (many SNP instruments) and Angrist-Krueger-style dummy instruments.
   - *Decoration to cut if pressed:* universality proofs for non-Gaussian errors (keep as empirical robustness), minimax lower bounds for composite alternatives (defer or drop).
5. **Strongest reason it could become a strong field paper:** a verifiably empty interface (zero hits for "weak instruments" AND "random matrix"/"Tracy-Widom" on arXiv, checked 2026-08-23), a complete off-the-shelf mathematical toolkit (Johnstone 2008/2009, BBP-type spikes, Wachter limits), and a memorable, testable practical headline: the classical relevance rule is systematically miscalibrated when q/n = alpha is not negligible.
6. **Strongest reason it could fail or become incremental:** (a) in the single-endogenous-regressor case (p = 1, the most common applied setting), the Jacobi ensemble degenerates and the classical F statistic may already be near-optimal, confining the contribution to multivariate-exposure settings such as multivariable MR; (b) the closest competitor (Meza and Singh, arXiv:2512.22697) may absorb the estimation layer, leaving testing only; (c) heteroskedasticity may break exact calibration in ways that collapse to existing robust practice.
7. **Next unresolved gate:** Gate G0/G1 (fatal-flaw certificate plus completed multi-vocabulary prior-art sweep), executed inside Phase 1.
8. **Single cheapest decisive next action:** run WP-P1-B1 (one afternoon of simulations checking the p = 1 degeneration worry and the identifiability of the frontier quantity) alongside WP-P1-A1 (bounded prior-art sweep). Both together require less than a day of compute.

---

## 2. Idea reconstruction and claim decomposition

**Scientific problem.** Instrument relevance (first-stage strength) determines whether IV inference is trustworthy, but classical diagnostics assume the number of instruments q is fixed or negligible relative to n. Modern applications (Mendelian randomization with hundreds of SNPs, dummy-instrument strategies with q/n around 0.1-0.3) live outside that regime, where the spectral theory of canonical correlations (Jacobi ensembles, Tracy-Widom edges, BBP-type spikes, Wachter bulk) governs the finite-sample behavior of every relevance statistic and of spectrally regularized 2SLS.

**Unit of analysis and observable data.** Independent observations i = 1, ..., n of (Y_i, X_i, Z_i): outcome scalar, p-dimensional endogenous regressor block, q-dimensional instrument vector. Regimes: R1 (p in {1, 2, 3} fixed, q/n to alpha in (0, 1)) and R2 (p growing, p/n to c_p > 0, q/n to alpha).

**Intervention/decision problem.** Decide whether the instrument set is strong enough for reliable IV inference (relevance decision), choose how much spectral information to retain (truncation level), and report valid confidence sets for beta.

**Estimands/targets.**
- Structural parameter beta in Y = beta' X + epsilon.
- Relevance parameter theta: the vector of squared population canonical correlations between standardized X and Z; the scalar headline is theta_max.
- The frontier function rho-star(alpha, p, delta, kappa): minimal theta_max such that (a) departure from irrelevance is detectable at level 5 with power 80 percent, and (b) the worst-case asymptotic 2SLS bias exceeds delta, where kappa indexes structural endogeneity strength (correlation of epsilon with the first-stage error).

**Proposed method/mechanism.** Test statistic: largest sample canonical correlation of residualized, standardized (X, Z), calibrated by exact finite-sample Jacobi quantiles (Gaussian case) and Tracy-Widom approximation with finite-n corrections (Johnstone 2009). Estimator: 2SLS computed after projecting the first-stage fitted values onto the top-tau fraction of sample canonical directions, with tau chosen by a rule derived from the estimated spectrum. Mechanism: under irrelevance the entire canonical spectrum follows a known random-matrix law, so both detection thresholds and optimal retention levels are computable in closed form or cheap simulation.

**Assumptions (baseline).** A1: iid sampling, homoskedastic Gaussian first-stage errors. A2: standardized designs (population covariance of (X, Z) block-diagonal under the null). A3: q < n with n - q bounded away from 0 as needed by the Jacobi parameter regime. Extensions: E1 heteroskedasticity, E2 non-Gaussian errors, E3 factor-structured instruments (link to Meza-Singh).

**Intended evidence.** Null-calibration grids, power surfaces against predicted spike locations, realized-size-distortion maps for decisions conditioned on F > 10 versus the spectral rule, risk curves for truncated versus classical estimators, then MR and AK applications.

**Audience/venue tier.** Econometrics theory and applied econometrics audiences (target tier: Econometrica / Quantitative Economics / Journal of Econometrics / Econometric Theory; statistics alternative: Annals of Statistics / JRSS-B). Fit assessment deferred to Gate G6.

### Claim decomposition

| Claim | Type | Feasibility | Novelty | Importance | Evidence status | Keep/cut/pivot |
|---|---|---|---|---|---|---|
| C1: TW/Jacobi-calibrated joint relevance test, uniformly valid in the q/n to alpha regime | Contribution | High (exact null ensemble known) | High (zero-hit interface) | High (weak IV is pervasive) | UNTESTED | Keep (headline) |
| C2: weak-instrument frontier rho-star(alpha, p, delta, kappa), modern replacement for F > 10 | Contribution | Medium (depends on identifiability of kappa, see O2) | High | Very high (practical rule) | UNTESTED | Keep (headline) |
| C3: deterministic-equivalent risk of spectrally truncated 2SLS, optimal truncation | Contribution/engine | Medium-high | Medium (overlaps Meza-Singh, who assume factor structure; ours is exact limits under iid designs) | High | UNTESTED | Keep, positioned against Meza-Singh |
| C4: applications to MR and AK-style dummy instruments | Application | Medium (data access) | Medium | High | UNTESTED | Keep after G3 |
| C5: universality of calibration under heteroskedastic/non-Gaussian errors | Enabling/decoration | Medium (heavy theory) | Low-medium | Medium (needed for credibility) | UNTESTED | Empirical robustness first; proof only at Phase 5 if earned |
| C6: demonstration that F > 10 is anticonservative or ultraconservative as a function of alpha | Framing/contribution | High | High | Very high | UNTESTED | Keep (memorable headline) |

**Load-bearing contribution:** C1 + C2 (testing and frontier). **Load-bearing assumption:** the Gaussian/homoskedastic Jacobi null is close enough to reality, after robust corrections, that calibration carries to applications. **Most dangerous prior-art collision:** Meza-Singh (arXiv:2512.22697) for C3; any econometrics paper using extreme eigenvalues of the first-stage design that the secondary sweep might reveal. **Strongest simple baseline:** first-stage F with Stock-Yogo critical values (practice) and Anderson-Rubin intervals (valid inference). **Cheapest decisive experiment:** WP-P1-B1 sanity simulations. **Hardest credible referee objection:** see Section 5.7. **Attractive component to cut if it dilutes the paper:** the minimax detection-boundary program (lower bounds for composite alternatives); keep the empirical power maps instead.

---

## 3. Fatal-flaw certificate (Gate G0)

Status: PROVISIONAL PASS with two open items (O1, O2). Full certificate to be produced by Phase 1.

Checks performed at planning time (reasoning-level):

1. **Definitions and mathematical typing.** Coherent. Canonical correlations between X (n x p) and Z (n x q) are well-defined for p + q <= n; the Jacobi parameter regime requires attention when q/n approaches 1 (encode in assumption A3).
2. **Consistency and non-vacuity.** Under irrelevance (population canonical correlations all zero), the squared sample canonical correlations of jointly Gaussian (X, Z) follow the Jacobi ensemble (Johnstone 2008, verified). Non-vacuous: the null is a real hypothesis (Pi = 0).
3. **Identification under the actual observation scheme.** Open item O1: the harm side of the frontier (2SLS bias exceeding delta) depends on kappa (strength of structural endogeneity), which is not identified from data alone. Repair path (to be validated in WP-P1-B1): report rho-star as a family of curves indexed by kappa, or as a conservative bound computed from an identifiable upper envelope; if neither yields a reportable, honest object, the frontier claim pivots from "identifiable rule" to "phase diagram" (still valuable, lower applied punch).
4. **Existence/well-definedness of targets.** beta, theta, and the detectability half of rho-star are well-defined population objects. The harm half inherits O1.
5. **Information leakage.** The test uses only (X, Z); the truncation rule for estimation must be tuned on first-stage information or validation splits only, never on the structural outcome Y (else tuning leaks the moment condition). Encoded in WP-P2-F1.
6. **Method-target match.** The relevance test targets joint irrelevance, matching the decision problem; the truncation estimator targets beta, matching the estimation problem. No mismatch detected.
7. **Smallest counterexamples / degenerate cases.** Open item O2: p = 1 degeneracy. With one endogenous regressor there is a single population canonical correlation and the Jacobi ensemble collapses; the largest-root statistic is a monotone transform of the classical F. The contribution then rests on (a) exact finite-n calibration replacing folklore thresholds, (b) the harm frontier, and (c) truncation theory. Must quantify honestly in WP-P1-B1 whether the p = 1 case leaves any practical gain; if not, the paper leads with R2 (growing p), which changes the audience.
8. **Vacuity via known equivalences.** For fixed q, largest-root statistics coincide with Roy's test and the problem reduces to classical MANOVA; nothing new. The new content lives strictly in the q/n = alpha regime; verified that the classical weak-IV literature works with trace/F statistics there, discarding spectral shape (Bekker 1994; Chao et al. 2012, E2).

No material defect demonstrated at planning time. Gate G0 decision (to finalize after WP-P1-B1): expected CONDITIONAL GO, conditions being resolution of O1 and O2.

---

## 4. Verified prior art and nearest-neighbor map

### Search log

Executed 2026-08-23:

| Query | Database | Result |
|---|---|---|
| all:"weak instruments" AND (all:"random matrix" OR all:"tracy-widom") | arXiv API | 0 hits |
| id 2512.22697 (Meza-Singh abstract inspection) | arXiv API | verified, see S1 |
| bibliographic queries for Johnstone 2008/2009, Bekker 1994, Andrews-Moreira-Stock | Crossref | verified, see Section 14 |

Planned in WP-P1-A1 (quota: stop after two consecutive empty sweeps across three databases): "canonical correlation" AND "instrumental variables" (arXiv, Crossref, RePEc/IDEAS); "many instruments" AND ("extreme eigenvalue" OR "largest root" OR "Roy") ; "first stage" AND "Tracy-Widom"; "Jacobi ensemble" AND econometrics venues; "concentration parameter" AND "many instruments"; SSRN and NBER working-paper search; citations forward from Bekker 1994, Chao et al. 2012, and Meza-Singh; backward search from Johnstone 2009 citing papers in econometrics journals.

### Nearest-neighbor table

| Source | Same problem? | Same target? | Same method? | Same evidence? | Remaining gap | Direct-hit risk |
|---|---|---|---|---|---|---|
| Stock-Yogo (2005), fixed-q weak-instrument rules | Yes | Rule for relevance | F statistic, fixed-q limits | Size-distortion tables | q/n to alpha regime; spectral info unused | Low (incumbent to beat) |
| Andrews-Moreira-Stock (2006 Econometrica; 2007 JoE) | Yes (weak-IV inference) | Valid tests/CI under weakness | Similarity/CLR theory, fixed q | Admissibility and power calc | Many-instrument regime; spectral relevance statistic | Medium (CLR remains the gold standard to condition on) |
| Bekker (1994); Chao-Swanson-Hausman-Newey-Woutersen (2012) | Yes (many instruments) | Consistency/bias of k-class and JIVE | Trace-level limits | Asymptotics | Discard eigenvalue configuration; no testing layer, no frontier | Medium |
| Johnstone (2008 Ann. Statist.; 2009 AoAS) | No (pure MRT) | Largest-root distributions | Jacobi/TW theory | Theorems | No IV/causal connection made | Low (tool supplier) |
| Wachter (1980) | No | Limiting spectral measure of discriminant ratios | MANOVA LSD | Theorem | Unused for IV | Low |
| Benaych-Georges-Nadakuditi (2012) | No | Spikes of rectangular information-plus-noise | Outlier formulas | Theorem | Not applied to canonical correlations under relevance | Low |
| Meza-Singh (arXiv:2512.22697), verified | Adjacent (IV with many noisy covariates/instruments) | beta estimation rates under factor repetition | Spectrally (canonical-correlation) regularized 2SLS | Upper/lower bounds, guidance on regularization | No relevance testing, no frontier, no weak-IV-robust inference; assumes factor structure rather than deriving exact iid-design limits | Highest for C3; must be cited and raced in simulations |
| Onatski (2009, 2010 REStat) | Adjacent (factor testing in macro) | Number of factors | Eigenvalue-ratio tests | Asymptotics | Different estimand | Low (template) |
| Belloni-Chen-Chernozhukov-Hansen (2012 Econometrica) | Adjacent (many instruments) | Optimal sparse instruments | Lasso selection | Rates | Sparsity mechanism, not spectral; complementary | Low-medium |
| Donald-Newey (2001 Econometrica) | Adjacent (number of instruments choice) | Estimator bias tradeoff | Higher-order MSE | Asymptotics | No spectral/null-distribution layer | Low |

### Source-backed novelty statement

Verified as of 2026-08-23: no located work provides (i) a relevance test whose null calibration exploits the exact Jacobi/TW law in the q/n to alpha regime, (ii) a weak-instrument frontier connecting detectability to IV harm, or (iii) weak-IV-robust inference conditioned on a spectral relevance statistic. The estimation layer (iii-a: spectrally regularized 2SLS) overlaps Meza-Singh, who prove rates under factor structure; exact deterministic-equivalent limits under iid designs and the link between optimal truncation and the detection threshold remain open. Positioning must therefore lead with C1 + C2 + C6, race C3 against Meza-Singh's estimator, and treat C3 alone as incremental.

**Strongest incumbent:** Andrews-Moreira-Stock CLR (valid inference) and the Stock-Yogo F > 10 rule plus Kleibergen-Paap rk F (daily practice). **Strongest simple baseline for every decisive experiment:** first-stage F with Stock-Yogo critical values and AR intervals.

**Gate G1 verdict (provisional):** CONDITIONAL GO. Conditions: complete WP-P1-A1 sweep; inspect Meza-Singh beyond the abstract (exact estimand, whether their regularization guidance touches detection thresholds).

---

## 5. Impact thesis and skeptical-referee test

1. **Why the problem matters.** Weak instruments invalidate the most widely used identification strategy in economics and epidemiology; practitioners rely on a rule of thumb calibrated in 2005 for a regime (small q) that many modern datasets violate.
2. **What changes if it works.** Relevance decisions become calibrated functions of the measured spectrum rather than folklore; MR pipelines gain a defensible screen at GWAS scale; the choice of how much first-stage spectral information to retain becomes principled.
3. **Who would use or cite it.** Applied microeconomists using dummy-instrument designs; MR geneticists (hundreds of thousands of users of two-sample MR pipelines); econometricians working on many-instrument limits; statisticians in high-dimensional MANOVA.
4. **Why a simpler incumbent is insufficient.** The F statistic is a trace functional: it is blind to the eigenvalue configuration that governs both detectability (edge fluctuations) and 2SLS bias (alignment of signal directions). Stock-Yogo critical values are fixed-q objects whose meaning degrades as q/n grows; AR/CLR are valid but answer a different question (they do not tell you whether your instruments are informative before you commit to a design).
5. **Why the contribution is not merely a combination.** The frontier concept (detectable is not the same as harmless, and vice versa) is new in this space and produces qualitative claims (regions where visible spikes contribute no bias; regions where undetectable confounding-like weakness still biases) that no incumbent tool articulates.
6. **Most damaging plausible referee paragraph.** "With one endogenous regressor, the largest canonical correlation is a monotone transformation of the first-stage F, so the entire Jacobi apparatus collapses to recalibrating a statistic everyone already computes; the interesting multivariate regime (p growing) is rare in practice; and under the heteroskedasticity that prevails in applications the exact null breaks, leaving the authors with bootstrap patches indistinguishable from existing robust practice."
7. **Evidence needed to answer it.** (a) Quantify the p = 1 case honestly: show either that exact finite-n calibration plus the frontier changes real decisions where F > 10 does not (realized-coverage maps at q/n = alpha), or concede p = 1 and lead with multivariable MR where p = 50 to 500 is routine. (b) Demonstrate a heteroskedastic-robust variant with controlled size in the grid (empirically first). (c) Show at least one crossover region at realistic alpha where F-conditioned inference has realized coverage below 0.90 while the spectral rule maintains it.

### Impact dimensions (planning-time scores; UNTESTED entries marked)

| Dimension | Score | Notes |
|---|---|---|
| Problem importance | 3 | Weak IV affects a broad applied literature |
| Novelty after prior art | 2 | Zero-hit primary vocabulary; estimation layer adjacent to Meza-Singh |
| Mechanism or insight | 2 | Sharp, falsifiable mechanism (spectral thresholds), UNTESTED |
| Empirical advantage | UNTESTED | Pending Gate G3 |
| Applied value | UNTESTED | Pending Gate G4 |
| Generality | 2 | Applies to any many-instrument design |
| Credibility | UNTESTED | Pending G3/G4 |
| Paper coherence | 2 | One story: retire F > 10, replace with frontier |

---

## 6. Dependency graph and gate map

```text
G0 validity + G1 prior art            (Phase 1, ACTIVE)
    -> G2 enabling formalization      (Phase 2, DORMANT UNTIL GATE G1)
        -> prototype smoke test       (Phase 2)
            -> G3 simulation engine   (Phase 3, DORMANT UNTIL GATE G2)
                -> G4 applied value   (Phase 4, DORMANT UNTIL GATE G3)
                    -> G5 theory investment   (Phase 5, DORMANT UNTIL GATE G4)
                        -> G6 submission case (Phase 5, DORMANT UNTIL GATE G5)
```

Read-only data-feasibility checks (WP-P4-F0) may run during Phase 3 without violating the gate ordering. No application analysis and no substantial theorem-proving may begin before Gates G3 and G4 respectively, except enabling propositions tagged in Phase 2.

---

## 7. Phase-by-phase execution program

Five phases. Every phase carries explicit give-up rules. Phases 2-5 remain dormant behind their gates; work packages specify exact outputs, verification, and compute.

---

### PHASE 1: Verification preflight (Gates G0 and G1)

**Purpose and scientific question.** Is the idea responsible to invest in? Specifically: does the closest literature absorb it, and do the two open defects (O1 frontier identifiability, O2 p = 1 degeneration) have witnesses?

**Prerequisites.** None.

#### WP-P1-A1: Multi-vocabulary prior-art sweep

- Status: ACTIVE
- Gate served: G1
- Objective: close the novelty question across neighboring vocabularies.
- Why this changes a decision: a direct hit kills the project now instead of after months.
- Inputs: search log in Section 4; reference manager.
- Preconditions: none.
- Actions:
  1. Run the ten query families listed in Section 4 on arXiv, Crossref, RePEc/IDEAS, and SSRN; inclusion criterion: any work performing relevance testing, weak-IV frontiers, or spectral calibration in the q/n = alpha regime.
  2. Forward-citation sweep from Bekker (1994), Chao et al. (2012), Meza-Singh; backward sweep from Johnstone (2009) filtered to econometrics journals.
  3. Read Meza-Singh beyond the abstract: extract exact estimand, assumptions, whether their regularization guidance involves detection thresholds.
  4. Write the evidence register.
- Outputs and exact paths: `Research/WeakIV_Evidence_Register.md` (nearest-neighbor table v2, >= 12 sources, each with verification level E0-E4).
- Verification (mechanical): every claimed query recorded with database, date, hit count. (Scientific): each keep/kill decision cites E2+ evidence.
- Scientific pass rule: no E3 direct hit on the testing/frontier layer.
- Scientific fail rule: any E3 source providing TW/Jacobi-calibrated relevance testing or the frontier in the many-instrument regime.
- Gate consequence: FAIL triggers KILL or PIVOT memo (`WeakIV_Research_Diagnostic.md`), no further phases.
- Dependencies: none. Can run in parallel with WP-P1-B1, WP-P1-C1.
- Compute, RAM, expected runtime: browsing + reading, < 4 hours human/agent time, no compute.
- Likely trap: searching only the project's own vocabulary; missing econometrics-journal papers phrased as "identification strength" or "concentration parameter".
- Recovery action: add vocabulary families iteratively until two consecutive empty sweeps.

#### WP-P1-A2: Classical-toolkit content inspection

- Status: ACTIVE
- Gate served: G0 (assumption consistency), enables Phase 2.
- Objective: extract the exact formulas the implementation will need, from primary sources.
- Why this changes a decision: prevents implementing the wrong ensemble or wrong centering constants.
- Inputs: Johnstone (2008, DOI 10.1214/08-AOS605), Johnstone (2009, DOI 10.1214/08-AOAS220), Wachter (1980, DOI 10.1214/aos/1176345134), Benaych-Georges-Nadakuditi (2012), BBP (2005).
- Preconditions: library access.
- Actions:
  1. Extract from Johnstone (2008): the Jacobi ensemble definition with parameters (n, p, q), the centering mu and scaling sigma-squared for the largest root, rate statement.
  2. Extract from Johnstone (2009): the approximate-null recipe (improved mu_np, sigma_np) usable at finite n.
  3. Extract the rectangular/spike outlier formula applicable to sample canonical correlations under a single population spike (Benaych-Georges-Nadakuditi 2012, JMVA).
  4. Record each as: exact statement, assumptions line-by-line, what must be adapted.
- Outputs and exact paths: `Research/weakiv_notes/toolkit_formulas.md` with equation-level notes and source page references.
- Verification (mechanical): each formula accompanied by source quote/location. (Scientific): a second reader reproduces the parameter mapping for one worked example.
- Scientific pass rule: parameter mapping consistent across two independent derivations.
- Scientific fail rule: irreconcilable ambiguity in the finite-n corrections.
- Gate consequence: fail downgrades Phase 2 to pure finite-n Monte Carlo quantiles (slower but safe).
- Dependencies: none; parallel with WP-P1-A1, WP-P1-B1.
- Compute: reading only.
- Likely trap: mixing complex-real convention (beta = 2) with real data (beta = 1) ensembles.
- Recovery action: tag every formula with its beta value.

#### WP-P1-B1: Fatal-flaw numeric witnesses (O1 and O2)

- Status: ACTIVE
- Gate served: G0
- Objective: settle the two open defects with small, decisive simulations.
- Why this changes a decision: O2 determines the paper's lead regime; O1 determines whether the frontier is an identifiable rule or a phase diagram.
- Inputs: toolkit formulas (may start concurrently using standard Jacobi simulation via triangular/beta-distribution construction).
- Preconditions: Python environment with numpy/scipy.
- Actions:
  1. Script `weakiv_notes/scripts/o2_degeneration.py`: for n in {250, 1000}, alpha in {0.1, ..., 0.9}, p = 1: simulate exact null canonical correlations; verify largest root is a monotone transform of F (compute rank correlation = 1); quantify where exact Jacobi quantiles differ from naive chi-squared/F approximations used by Stock-Yogo interpolation; plot.
  2. Script `weakiv_notes/scripts/o1_frontier_identifiability.py`: build a small SEM (Y = beta X + epsilon, X = Z Pi + V, correlated epsilon-V with strength kappa); show numerically whether the bias-exceeds-delta event, as a function of theta_max, is separable across kappa; test whether an identifiable upper envelope (using residualized outcomes only) covers the truth across kappa in {0.2, 0.5, 1, 2}.
  3. One-page memo interpreting both.
- Outputs and exact paths: `Research/weakiv_notes/scripts/o2_degeneration.py`, `o1_frontier_identifiability.py`, `Research/weakiv_notes/o1_o2_memo.md`, figures under `Research/weakiv_notes/figs/`.
- Verification (mechanical): scripts run from clean interpreter, seeds logged, figures produced. (Scientific): memo states clearly which quantities are identifiable from (X, Z, Y) and which are not, with witness plots.
- Scientific pass rule: monotone-transform check confirms/refutes collapse cleanly; envelope coverage >= 95 percent across kappa grid (or the failure is documented and the phase-diagram pivot adopted).
- Scientific fail rule: frontier harm-side depends on unidentifiable kappa AND no conservative identifiable envelope exists.
- Gate consequence: contributes to G0 verdict; fail on O1 forces PIVOT of C2 to phase-diagram framing (documented as such, not silently repaired).
- Dependencies: none hard; benefits from WP-P1-A2. Parallel with WP-P1-A1, WP-P1-C1.
- Compute, RAM, runtime: < 30 min wall, < 1 GB, laptop-scale.
- Likely trap: concluding from one kappa value.
- Recovery action: always sweep kappa.

#### WP-P1-C1: Application data feasibility (read-only)

- Status: ACTIVE
- Gate served: early audit for Phase 4 (does not begin the application).
- Objective: confirm data access for MR (summary statistics for 2-3 exposure-outcome pairs, LD reference panel) and AK (quarter-of-birth Census extracts, public replications).
- Preconditions: none.
- Actions: locate and record provider, release, unit, access conditions, schema links for: IEU OpenGWAS or GWAS Catalog entries for candidate traits; 1000 Genomes/HRC LD panel; AK replication datasets (Angrist-Krueger 1991 extracts, Bound-Jaeger-Baker 1995 materials). Record sizes only; no outcome analysis.
- Outputs and exact paths: `Research/weakiv_notes/data_feasibility.md`.
- Verification: every dataset row has provider + access status + license note.
- Pass rule: at least one MR pair and one AK extract accessible.
- Fail rule: no accessible MR summary statistics (pivot to semi-synthetic genotypes + published summaries only).
- Gate consequence: shapes Phase 4 design; cannot block Phases 2-3.
- Dependencies: none; parallel with all Phase 1 WPs.
- Compute: trivial. Likely trap: downloading full UK Biobank-scale individual data (unnecessary; summary stats suffice). Recovery: stick to summary statistics.

**Phase 1 deliverables.** Evidence register, toolkit formulas, O1/O2 memo, data feasibility note, completed G0 certificate, G1 verdict.

**Gate G0/G1 evidence required.** Certificate with witnesses for O1/O2 resolutions; nearest-neighbor table at E2+ for all keep/kill decisions.

**Phase 1 give-up rules (decide at gate review):**

- **KILL** if WP-P1-A1 finds an E3 direct hit absorbing C1 or C2 (calibrated relevance testing or frontier in the many-instrument regime).
- **KILL** if O1 fails both repair paths (frontier harm-side unidentifiable and no honest envelope), AND the phase-diagram reframing is judged below the user's impact bar.
- **PIVOT** if O2 shows the p = 1 case retains no practical gain: lead regime becomes R2 (growing p, multivariable MR); rerun G1 scoping under the new audience.
- **CONDITIONAL GO** otherwise, with named bounded tasks carried into Phase 2 (expected: none).

**Estimated compute/data.** Negligible (< 1 hour machine time total, < 1 GB RAM).

**Risks and recovery.** Sweep fatigue (mitigate with stop quota); over-reading Meza-Singh (extract only what bears on testing/frontier gaps).

---

### PHASE 2: Minimum enabling formalization and prototype (Gate G2)

Status: DORMANT UNTIL GATE G1. **Theory cap:** no asymptotic theorem-proving in this phase. Only definitions, the algorithm specification, and enabling propositions without which the code would be meaningless (exact null ensemble, outlier-formula conjecture to be checked numerically before any proof attempt). Everything substantive waits for Phase 5.

**Purpose and scientific question.** Can the test and the truncated estimator be implemented and tested honestly, without oracle knowledge?

**Prerequisites.** Phase 1 outputs: toolkit formulas, O1/O2 resolutions.

#### WP-P2-F1: Formalization memo

- Status: DORMANT UNTIL GATE G1
- Gate served: G2
- Objective: pin down regimes, hypotheses, estimands, assumptions, algorithms.
- Actions:
  1. Declare regimes R1/R2 (Section 2) and the null H0: Pi = 0 (joint irrelevance) versus H1(theta): maximal population canonical correlation = theta.
  2. Assumptions ledger A1-A3, extensions E1-E3, each with a violation-detection hook.
  3. Specify test T_spec: standardize X, Z; residualize both against included exogenous covariates (none in baseline); statistic t_n = (r_max^2 - mu_np)/sigma_np with Johnstone (2009) finite-n corrections; critical value from exact finite-n Jacobi Monte Carlo quantiles (primary) and TW1 approximation (secondary); report both.
  4. Specify estimator beta-hat(tau): 2SLS on first-stage fitted values projected onto top-tau sample canonical directions; tau-selection rule tau-hat from the estimated spectrum under the fitted spiked model (no Y involvement; anti-leakage invariant stated and unit-tested).
  5. Enabling propositions, tagged: P1 (direct, Johnstone 2008): under A1-A3 with theta = 0 the squared sample canonical correlations are exactly Jacobi-distributed; P2 (adaptation, Benaych-Georges-Nadakuditi 2012): outlier location of r_max under a single spike theta; P3 (working hypothesis, to be verified numerically in Phase 3, proved only in Phase 5 if earned): asymptotic 2SLS bias is monotone in theta along worst-case alignment.
  6. State the frontier definition formally, incorporating the O1 resolution (family over kappa or identifiable envelope).
- Outputs and exact paths: `Research/weakiv_notes/formalization_memo.md`.
- Verification: memo reviewed against assumptions ledger; every algorithm step implementable without oracle quantities (checklist).
- Pass rule: unambiguous implementable spec; anti-leakage property stated as testable invariant.
- Fail rule: any algorithm step requires true theta, true Pi direction, or kappa.
- Gate consequence: fail forces redesign or narrower regime before any coding.
- Dependencies: WP-P1-A2. Parallel with WP-P2-I1 start.
- Compute: none. Likely trap: silent reliance on population standardization. Recovery: make standardization an explicit, tested preprocessing module.

#### WP-P2-I1: Reference implementation `specraliv`

- Status: DORMANT UNTIL GATE G1 (coding starts once memo skeleton exists; tests frozen after G2)
- Gate served: G2
- Objective: transparent, tested Python package.
- Actions:
  1. Modules: `rng.py` (seed streams), `canoncorr.py` (SVD/QR-based sample canonical correlations), `jacobiquantiles.py` (exact null quantiles via beta-ensemble simulation; vectorized over replications), `tw.py` (TW1 cdf/quantiles via precomputed tables or Bornemann quadrature), `teststats.py` (t_n, finite-n corrections, p-values), `ivestimators.py` (2SLS, LIML, Fuller-k, JIVE, Bekker, truncated-2SLS, Meza-Singh-style CCR), `dgps.py` (all Phase 3 DGPs).
  2. Unit tests: canoncorr matches manual formula on 3x2 examples; p = 1 Jacobi quantile equals Beta-distribution identity; truncated-2SLS(tau = 1) == 2SLS and tau -> 0 approaches OLS-within-canonical-subspace; anti-leakage invariant (tau-hat unchanged when Y is permuted); JIVE reproduces published small-example values.
  3. Package layout, pyproject.toml, pinned dependencies.
- Outputs and exact paths: `Research/specraliv/` (package), `Research/specraliv/tests/`, CI-runnable pytest suite.
- Verification (mechanical): `pytest` green from clean env. (Scientific): statistical identities above hold to 1e-8.
- Pass/fail rules: any failing identity blocks Phase 3.
- Gate consequence: green suite is G2 evidence.
- Dependencies: WP-P2-F1. Parallel with WP-P2-I2 design.
- Compute: development only. Likely trap: slow naive Jacobi simulation blocking later scale. Recovery: vectorize via QR of Gaussian pairs (O(npq) per replicate), profile in I2.

#### WP-P2-I2: Smoke run and cost model

- Status: DORMANT UNTIL GATE G2 (runs immediately after tests pass)
- Objective: one-seed, small-grid end-to-end run producing the compute-cost model that decides local versus Colab placement of every Phase 3 cell.
- Actions: run 100 seeds x 6-cell mini-grid; record wall time and peak RSS per cell (resource.getrusage); fit cost model time(n, p, q, B); emit `colab_plan.csv` assigning each planned grid cell to LOCAL or NOTEBOOK-i with predicted hours and RAM.
- Outputs and exact paths: `Research/weakiv_notes/smoke_run/`, `Research/weakiv_notes/colab_plan.csv`.
- Verification: smoke artifacts exist; cost model predicts measured times within 2x.
- Pass rule: all unit-tested estimators execute end-to-end on the mini-grid.
- Fail rule: any estimator crashes or dominates runtime budget by 10x (profile and fix before Phase 3).
- Dependencies: WP-P2-I1.
- Compute: < 20 min, < 2 GB. Likely trap: forgetting BLAS thread oversubscription. Recovery: set threads = 1 per worker in all scripts.

**Phase 2 deliverables.** Formalization memo, tested package, smoke run, colab plan.

**Gate G2 evidence.** Memo + green test suite + smoke artifacts + anti-leakage test.

**Phase 2 give-up rules:**

- **KILL (layer-level)** if the test cannot be defined without oracle knowledge (true theta or kappa) in any predeclared variant, or if the exact-Jacobi quantile implementation disagrees with direct Monte Carlo of canonical correlations beyond MC error (misformulation witness).
- **PIVOT** if calibration is implementable only for p = 1 (where O2 showed little gain): pivot the testing layer to linear-spectral-statistic tests (sum-of-logs functionals), which survive growing p; rerun G2.
- **INCREMENTAL-ONLY** if both testing and frontier layers fail here but truncated-2SLS limits remain: the surviving project is an estimation note adjacent to Meza-Singh, below the stated bar; terminate unless the user accepts an incremental paper.

**Compute.** Development machine only; < 4 GB throughout.

---

### PHASE 3: Simulation-first falsification (Gate G3)

Status: DORMANT UNTIL GATE G2. This is the decisive phase: every claim must face an experiment able to refute it, with thresholds preregistered before the decisive runs.

**Purpose and scientific question.** Does the spectral test beat fair strong baselines for the claimed reason, and does the predicted frontier describe simulated 2SLS harm?

**Prerequisites.** G2 evidence; preregistration memo signed before decisive runs.

#### Claims-to-experiments matrix

| Claim | Mechanism | DGP | Metric | Baseline | Ablation | Threshold (preregister in WP-P3-R0) | Falsifier | Output |
|---|---|---|---|---|---|---|---|---|
| X1 calibrated size | exact/asymptotic Jacobi null | A1-A3 null, grid (n, p, alpha) | empirical rejection at 5 pct | chi-squared/F naive CV; KP-rk F | TW vs exact-Jacobi CV; no finite-n correction | size within +/-0.01 of nominal everywhere | sustained miscalibration in easiest case | `results/size_grid.parquet` |
| X2 power at predicted outliers | BBP-type spike emergence | single-spike theta sweep | power curve vs theta; outlier location match | Stock-Yogo F rule; AR test | spike-direction randomization | outlier location within 2 sigma_np of prediction | systematic location mismatch after one ansatz revision | `results/power_surface.parquet` |
| X3 F > 10 miscalibration map | trace statistic blind to edge | kappa, alpha grid | realized 95 pct AR-interval coverage among designs passing each rule | F > 10; KP-rk > 10 | rule variants | exists alpha-region with coverage <= 0.90 under F > 10 and >= 0.93 under spectral rule | no divergence anywhere realistic | `results/coverage_map.parquet` |
| X4 truncated-2SLS risk | spectral retention trades bias/variance | (alpha, theta, kappa, heterosk) grid | RMSE(beta-hat), MAE | 2SLS, LIML, Fuller, JIVE, Bekker, CCR (Meza-Singh-style) | tau-rule variants; oracle-tau | >= 15 pct RMSE improvement over best baseline in >= 1 predeclared regime; no regime worse than best baseline by > 5 pct | worse everywhere or wins nowhere | `results/risk_curves.parquet` |
| X5 robustness ladder | universality (empirical) | heteroskedastic (variance f(Z)), clustered, heavy-tailed epsilon | size, coverage | robust F variants | bootstrap-CV variant | size drift <= +0.02 with bootstrap patch | drift > 0.05 unpatchable | `results/robustness.parquet` |
| X6 scaling | cost model | n up to 2e5 (MR-like q up to 5e3 via summary-statistics mode) | seconds, peak GB | none | none | fits notebook envelope | infeasible at target scales | `results/scaling.parquet` |

#### Work packages

##### WP-P3-R0: Preregistration memo (before any decisive run)

- Objective: lock metrics, tolerances, seed lists (main grid: 400 replications per cell for power/risk; 2x10^4 for size cells; seed stream IDs fixed), comparison protocol (equal data, equal tuning budgets: both rules may use a common validation split for any data-driven constant), primary metric hierarchy (size calibration first; then power at matched alternatives; then RMSE), failure rules (as in the table above).
- Outputs: `Research/weakiv_preregistration.md` (version-stamped; edits after first decisive run forbidden, append-only deviations log).
- Verification: memo committed before timestamped result files exist.
- Gate consequence: defines G3 judgment.

##### WP-P3-S1: Correctness simulations (X1, X2)

- Objective: verify null calibration and spike predictions before any comparison.
- Grid: n in {250, 500, 1000, 2000}; alpha in {0.1, 0.3, 0.5, 0.7, 0.9} (respecting n - q > p + 1); p in {1, 2, 5} for R1 and {25, 100} at n >= 1000 for R2; B_size = 2x10^4 (up to 10^5 at small configs for TW tails).
- Outputs: `Research/weakiv_results/size_grid.parquet`, `power_surface.parquet`, QQ plots vs TW1, `figs/size_heatmap.png`, `figs/power_vs_prediction.png`.
- Verification: all cells present with seeds; schema check.
- Pass rule: X1 threshold met; X2 location match.
- Fail rule: any sustained miscalibration in the easiest (Gaussian, homoskedastic) case after finite-n corrections.
- Compute: pilot-measured; est. 2-8 h local across cells; cells predicted > 2 h or > 4 GB go to notebooks (see compute policy).

##### WP-P3-S2: Decisive comparisons (X3, X4) and robustness (X5)

- Objective: the headline experiments.
- Grid: alpha in {0.1, ..., 0.9}; kappa in {0.2, 0.5, 1, 2}; theta in 20-point sweep; heterosk level in {0, mild, severe}; 400 reps/cell.
- Baselines tuned fairly (validation split for any constant); failed runs preserved with reasons.
- Outputs: `risk_curves.parquet`, `coverage_map.parquet`, `robustness.parquet`; figures `figs/risk_curves.png`, `figs/coverage_heatmap.png`.
- Verification + pass/fail: per matrix thresholds.
- Compute: largest phase; est. 20-60 h serial-equivalent; sharded per policy below.

##### WP-P3-S3: Scaling study (X6)

- Objective: demonstrate the workflow at target application scale (summary-statistics mode: spectral computations on LD matrices of dimension q up to 5x10^3).
- Outputs: `scaling.parquet`, `figs/runtime_scaling.png`.

##### WP-P3-M1: Gate memo G3

- Objective: standalone memo per contract: preregistered expectations vs observed, deviations log, strongest baseline's best case, our failure region, ablation attribution, practical effect sizes, single G3 decision.
- Outputs: `Research/weakiv_gate_memo_G3.md`.

**Simulation-gate give-up rules:**

- **KILL testing layer** if size calibration (X1) fails in the easiest case after the predeclared correction ladder, or if power (X2) never exceeds the Stock-Yogo/AR baselines by >= 5 percentage points anywhere in the predeclared favorable region.
- **KILL frontier claim** if simulated bias surfaces contradict the predicted frontier beyond tolerance after exactly one permitted ansatz revision (mismatch is falsification, not noise; second mismatch ends the claim).
- **KILL estimation layer** if truncated-2SLS is worse than the best of {2SLS, LIML, Fuller, JIVE, CCR} in every predeclared regime (X4 falsifier).
- **INCREMENTAL-ONLY (terminal)** if all layers merely match incumbents with no regime of practical dominance (e.g., F > 10 proves adequate across alpha in {0.1, ..., 0.9}): the "retire F > 10" headline dies and the residue is below the stated bar; preserve evidence and stop.
- **PIVOT** if wins exist only under R2 (growing p): narrow the paper to multivariable MR and rerun G3 judgment on the narrowed grid.

**Compute and Colab policy (applies to all Phase 3 WPs).**

- Pilot first (WP-P2-I2 cost model). Route each grid cell by measured cost: LOCAL if predicted < 2 h wall AND peak RAM < 4 GB; otherwise NOTEBOOK.
- Notebook envelope: Google Colab notebooks are created for any experiment predicted to exceed 2 hours or 4 GB RAM. Up to 40 independent, self-contained notebooks are available; each runs up to 10 h with roughly 12 GB RAM (GPU unnecessary here: pure numpy/scipy CPU linear algebra).
- Sharding scheme: assign grid cells round-robin to notebooks so each finishes comfortably inside 10 h (target <= 6 h for margin). Each notebook is self-contained: installs pinned deps, embeds the DGP/config dict, seed manifest, and cell list; writes one `results_<cellid>.csv` (plus `_done` marker with sha256); ends with the mandatory download fallback:

```python
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
```

- Merge script `merge_results.py` validates completeness against the seed manifest (every cell x seed present, checksums match) before any gate memo is written.
- Local parallelism etiquette: multiprocessing Pool capped at 10 workers (physical cores), BLAS threads = 1 per worker, check free RAM before launch, leave headroom for concurrent experiments on this machine; never nested parallelism.

---

### PHASE 4: Applications (Gate G4)

Status: DORMANT UNTIL GATE G3. Read-only feasibility (WP-P1-C1) already ran; no application analysis before G3 passes.

**Purpose and scientific question.** Do the spectral decisions change real conclusions in scientifically consequential settings?

#### WP-P4-A1: Mendelian randomization (semi-synthetic + summary statistics)

- Objective: test whether spectral relevance decisions reclassify exposures relative to F > 10 / KP-rk at GWAS scale, and whether those reclassifications correspond to honest differences in downstream validity.
- Design: 2-3 predeclared exposure-outcome pairs (from WP-P1-C1); harmonization, LD clumping frozen before comparative results; q in the hundreds-to-thousands; leave-one-chromosome-out (LOCO) so conclusions do not hinge on one LD structure.
- Controls: negative (phenotype permutation, non-targeting constructs where applicable: statistics must come out null-calibrated); positive (planted genetic effects at known theta: recovery must track predicted power).
- Predeclared primary novel finding (one sentence, frozen in memo before unblinding real outcomes): at GWAS-scale q, a nontrivial fraction (>= 10 percent) of exposure-outcome pairs passing F > 10 exhibit realized AR-interval coverage <= 0.90 in LOCO splits, while the spectral-frontier rule excludes them.
- Outputs: `Research/weakiv_apps/mr/mr_analysis_memo.md`, `mr_loco_results.parquet`, `figs/mr_reclassification.png`.
- Pass rule: positive controls recover; negative controls calibrated; primary finding holds in >= 2 of 3 pairs.
- Give-up rules: **KILL** if positive controls fail (diagnostic cannot separate planted weak instruments from noise at declared sizes: identification-adjacent failure). **INCREMENTAL-ONLY** if no reclassification occurs in any pair and no covariance-level finding emerges (method adds nothing in practice). **PIVOT** to fully-synthetic + published-summaries replication if individual-level or LD access fails.
- Compute: LD-matrix spectral computations are modest; any genome-scale pass predicted > 2 h goes to a notebook per policy (est. 1-3 notebooks).

#### WP-P4-A2: Angrist-Krueger-style dummy instruments

- Objective: benchmark on the canonical many-dummy design (quarter-of-birth x state interactions, q in {180, 510}); reproduce the established qualitative picture (AK 1991; Bound-Jaeger-Baker 1995 critique), then overlay spectral decisions and AR-conditioned intervals.
- Predeclared finding: the spectral rule either (a) flags the 510-instrument specification as weaker than F suggests (consistent with BJP's critique, now with a quantitative distance-to-frontier), or (b) certifies it, contradicting the folklore, in which case the contradiction is investigated as a bug before being claimed as discovery.
- Outputs: `Research/weakiv_apps/ak/ak_memo.md`, `ak_results.parquet`.
- Pass rule: trusted-benchmark reproduction achieved; a defensible new quantitative statement about distance-to-frontier.
- Give-up rules: **KILL/INCREMENTAL-ONLY** if results identical to classical analysis with no interpretable distance-to-frontier statement. Data-access failure pivots to published-extract replication only.

**Gate G4 evidence.** Per contract: benchmark reproduction, identification diagnostics, sensitivity/placebo survival, changed understanding versus incumbents, honest uncertainty.

**Phase 4 give-up rules (aggregate):** **KILL** if both applications yield nothing beyond incumbents (no reclassification, no new quantitative insight, showcase-only residue) or if any unresolved contradiction with established evidence remains after investigation.

---

### PHASE 5: Evidence-earned theory and paper consolidation (Gates G5 and G6)

Status: DORMANT UNTIL GATE G4. Substantial mathematics starts only here, only for claims that survived.

**Purpose.** Prove the smallest set of theorems that secure the surviving load-bearing claims; consolidate the smallest coherent paper.

#### Theory target table (ordered by decision value; invoke `math-theory-dev-plan` at this point)

| Target | Why earned | Statement sketch | Tag | Source result | Adaptation gap | Numerical falsifier | Stop rule |
|---|---|---|---|---|---|---|---|
| T1 exact null => uniform size control | explains X1 | Under A1-A3, T_spec's null distribution is the Jacobi largest root with Johnstone (2009) corrections; size converges uniformly over the parameter grid | direct | Johnstone 2008 Thm (largest-root TW), 2009 corrections | residualization + standardization steps | X1 grid | 1 week; if blocked, ship exact-Jacobi CVs (already implemented, no theorem needed) |
| T2 spike outlier location for canonical correlations | explains X2 | Under single spike theta, r_max^2 concentrates at the BGN rectangular outlier formula past the BBP-type threshold | adaptation | Benaych-Georges-Nadakuditi 2012; BBP 2005 | map their additive-spiked model to canonical-correlation geometry | X2 location match | 2 weeks; else state as verified conjecture with numerics |
| T3 frontier characterization (detectability half) | secures C2 | Minimal detectable theta as function of (alpha, p, n); power limits of LSS tests | adaptation | Onatski-Moreira-Hallin 2013 power limits; BBP | two-point Le Cam argument in the Jacobi geometry | X2 power surface | 2 weeks; lower bound may be dropped without killing the paper |
| T4 deterministic-equivalent risk of truncated 2SLS + optimal tau | secures C3 | Exact limiting MSE as explicit function of (tau, alpha, theta, kappa); tau-optimum coincides with (or deviates from) the detection threshold | adaptation | Wachter 1980 LSD; Dobriban-Wager-style resolvent calculus | double-Wishart geometry with endogenous second stage | X4 risk curves | 3-4 weeks; highest value if estimation layer survived |
| T5 heteroskedastic robustness | protects credibility | LSS CLT under E1 giving size control of patched statistic | conjecture | Bao-Pan-Zhou-type CLTs; El Karoui concentration | heteroskedastic variance profile | X5 | attempt last; bootstrap fallback already shipped empirically |

Rules: try to prove the high-value result before downgrading it; tag honestly as direct/adaptation/conjecture; every target names its numerical falsifier; any theorem that merely decorates surviving evidence is cut.

**Gate G5 give-up rules:** if T1-T4 targets resist after their bounded windows and simulations carry the story unaided, take GO-WRITE (no ornamental theorems). **KILL** if theory would be compensating for failed evidence (that state was supposed to die at G3).

#### WP-P5-P1: Paper consolidation and referee stress test

- Smallest coherent story options: Option A (testing + frontier flagship, C1+C2+C6, with C3 as supporting section) if X1-X3 passed strongly; Option B (estimation-limits flagship, C3, racing Meza-Singh) if testing died but X4 passed; cut decorative modules accordingly.
- Skeptical-referee pass answering the Section 5.6 paragraph with produced evidence; reproducibility audit; venue-fit comparison based on recent comparable papers (fit reasoning, not acceptance prediction).
- Outputs: `Research/weakiv_paper/outline_option_A_or_B.md`, referee-pass memo.
- Gate G6 give-up: if the contribution statement still answers "why not F > 10 + AR?" with promised rather than produced evidence, downgrade to INCREMENTAL-ONLY and stop.

---

## 8. Simulation study (summary reference)

Full specification lives in Phase 3 (claims-to-experiments matrix, DGP ladder order: noiseless known-truth case, correct stochastic case, null case, baseline-favorable case, mechanism-favorable case, crossover grid, nuisance misspecification, realistic measurement error, application violations, runtime scaling). Non-negotiables: preregistration before decisive runs; equal information and tuning budgets; uncertainty across replications (never average ranks alone); failed runs preserved with reasons; one primary metric per claim.

## 9. Applied study (summary reference)

Full protocols in Phase 4. Separation maintained: read-only feasibility (Phase 1) precedes; confirmatory application follows G3 only; preprocessing frozen before comparative results; contradictions with established evidence treated as bugs until investigated.

## 10. Deferred theory program

Table in Phase 5 is the complete theory program; it is deliberately narrower than the original idea's speculative wish list (minimax composite-alternative lower bounds and non-Gaussian universality proofs are cut or demoted unless evidence earns them). Each retained target maps to a verified source (Section 14) with an adaptation gap and a numerical falsifier.

## 11. Risk register

| Risk | Probability | Damage | Earliest detector | Prevention | Recovery | Terminal? | Owner package |
|---|---|---|---|---|---|---|---|
| Direct prior-art hit in secondary vocabulary | Low-med | Kills novelty | WP-P1-A1 | Broad query families, forward citations | Pivot scope memo | Yes if E3 on C1/C2 | WP-P1-A1 |
| p = 1 degeneration confines gains | Medium | Narrows paper | WP-P1-B1 | Honest early quantification | Lead with R2/multivariable MR | No (pivot) | WP-P1-B1 |
| Frontier harm-side unidentifiable (kappa) | Medium | Weakens C2 headline | WP-P1-B1 | Envelope repair path | Phase-diagram framing | Only if envelope also fails and bar uncompromised | WP-P1-B1 |
| Meza-Singh absorbs C3 | Medium | Loss of estimation layer | WP-P1-A1 step 3 | Race their estimator in X4 | Testing-led Option A | No | WP-P3-S2 |
| Weak-baseline illusion (wins only vs untuned F) | Medium | False engine | WP-P3-R0 protocol | Tuned baselines, AR gold standard | None if persists | Terminal at G3 | WP-P3-S2 |
| Heteroskedasticity breaks calibration | High | Practical adoption blocked | WP-P3-S2 X5 | Bootstrap patch designed upfront | Restrict claims to homoskedastic theory + empirical robustness | No unless unpatchable | WP-P3-S2 |
| Size miscalibration in easiest case | Low-med | Kills testing layer | WP-P3-S1 X1 | Exact-Jacobi CVs (finite-n, no asymptotics) | Drop testing layer | Terminal for C1 | WP-P3-S1 |
| MR data access fails | Medium | Loses flagship application | WP-P1-C1 | Multiple providers | Fully-synthetic pivot | No (pivot) | WP-P1-C1 |
| Computational infeasibility at genome scale | Low | Blocks X6/apps | WP-P3-S3 | Summary-statistics mode; notebook sharding | Reduce q via clumping | No | WP-P3-S3 |
| Decorative theory creep | Medium | Wasted months | Phase 5 stop rules | Bounded windows per target | Cut to conjectures | No | WP-P5 |
| Diffuse paper story | Medium | Desk-reject tier | WP-P5-P1 | Option A/B discipline | Cut modules | No | WP-P5-P1 |

## 12. Reproducibility and artifact map

- Repository layout: `Research/specraliv/` (package + tests), `Research/weakiv_notes/` (memos, formulas, scripts, figs), `Research/weakiv_results/` (parquet/csv per experiment, one directory per grid, `manifest.json` with seeds and sha256 per artifact), `Research/weakiv_apps/`, `Research/weakiv_paper/`.
- Environment: `pyproject.toml` with pinned versions; `requirements_freeze.txt` generated after G2; notebooks install from the freeze file.
- Seeds: master seed file `seeds.yaml` (stream IDs per experiment, per cell); every result row carries `(experiment, cell_id, seed)`.
- Raw vs processed boundary: application raw extracts stored read-only under `weakiv_apps/data_raw/` (never edited); processed harmonized panels under `data_processed/` with generation scripts.
- Result schemas: fixed column dictionaries in `schemas.md` (e.g., size grid: `n, p, q, alpha, cv_method, correction, rejects, B, seed_range`); merge script enforces schemas.
- Checkpointing: every notebook/local job writes per-cell results plus `_done` markers; resume logic skips completed cells; checksum validation before merging.
- Figure/table provenance: every figure generated by a numbered script in `scripts_figures/`; no hand-edited figures.
- Colab notebook template requirements (mandatory for any notebook): self-contained (deps, config dict, seeds, cell list embedded), <= 10 h predicted runtime, <= 12 GB RAM, per-cell checkpoints, output download fallback block (snippet in Phase 3), and a header cell recording notebook index, grid slice, and parent commit/hash of the package.
- Reproduce-any-claim command pattern: `python -m specraliv.experiments run --config configs/<experiment>.yml --cells <ids>`; every memo references the configs that generate its numbers.

## 13. Immediate actions (stop at Gate G0/G1)

Only these four, all in Phase 1:

1. WP-P1-A1: run the ten planned query families; produce `Research/WeakIV_Evidence_Register.md`. (No theory, no coding beyond throwaway queries.)
2. WP-P1-A2: extract and record the toolkit formulas in `Research/weakiv_notes/toolkit_formulas.md`.
3. WP-P1-B1: run `o2_degeneration.py` and `o1_frontier_identifiability.py`; write `Research/weakiv_notes/o1_o2_memo.md`.
4. WP-P1-C1: write `Research/weakiv_notes/data_feasibility.md`.

Then hold a gate review: fill the G0 certificate, make the G1 call (GO / CONDITIONAL GO / PIVOT / KILL). Nothing in Phases 2-5 starts before that review.

## 14. References (verification level at 2026-08-23)

1. Baik, J., Ben Arous, G., Peche, S. (2005). Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. Annals of Probability 33(5), 164-182. Level: E2. URL: https://doi.org/10.1214/009117905000000234 <!-- DOI to re-verify at WP-P1-A2; flag -->
2. Bekker, P. A. (1994). Alternative Approximations to the Distributions of Instrumental Variable Estimators. Econometrica 62(3), 657-681. Level: E3 (Crossref record inspected; title corrected from the idea dossier). URL: https://doi.org/10.2307/2951662
3. Belloni, A., Chen, D., Chernozhukov, V., Hansen, C. (2012). Sparse Models and Methods for Optimal Instruments with an Application to Eminent Domain. Econometrica 80(5), 2369-2429. Level: E1 (metadata from memory; verify at WP-P1-A1). <!-- URL/DOI to add; flag -->
4. Benaych-Georges, F., Nadakuditi, R. R. (2012). The singular values and vectors of low rank perturbations of large rectangular random matrices. Journal of Multivariate Analysis 111, 120-135. Level: E2. URL: https://doi.org/10.1016/j.jmva.2012.04.002 <!-- DOI to re-verify at WP-P1-A2; flag -->
5. Chao, J. C., Swanson, N. R., Hausman, J. A., Newey, W. K., Woutersen, T. (2012). Asymptotic distribution of JIVE in a heteroskedastic linear regression model with many instruments. Level: E1 (venue/volume unconfirmed). <!-- URL/DOI to add; flag -->
6. Donald, S. G., Newey, W. K. (2001). Choosing the Number of Instruments. Econometrica 69(5), 1161-1191. Level: E2. URL: https://doi.org/10.1111/1468-0262.00238
7. Andrews, D. W. K., Moreira, M. J., Stock, J. H. (2006). Optimal Two-Sided Invariant Similar Tests for Instrumental Variables Regression. Econometrica 74(3), 715-752. Level: E2. URL: https://doi.org/10.1111/j.1468-0262.2006.00680.x (Note: the idea dossier attributed "performance of conditional Wald tests" to Econometrica 2006; that paper is Andrews-Moreira-Stock (2007), Journal of Econometrics 139(1), 116-132, https://doi.org/10.1016/j.jeconom.2006.06.007, E3.)
8. Johnstone, I. M. (2008). Multivariate analysis and Jacobi ensembles: Largest eigenvalue, Tracy-Widom limits and rates of convergence. Annals of Statistics 36(6), 2638-2716. Level: E3 (record inspected; theorem-level extraction at WP-P1-A2). URL: https://doi.org/10.1214/08-AOS605
9. Johnstone, I. M. (2009). Approximate null distribution of the largest root in multivariate analysis. Annals of Applied Statistics 3(4), 1616-1633. Level: E2 (note: Ann. Appl. Statist., not Biometrika as stated in the idea dossier). URL: https://doi.org/10.1214/08-AOAS220
10. Meza, I., Singh, R. (2025). Canonical correlation regression with noisy data. arXiv:2512.22697. Level: E3 (abstract inspected; full-text read at WP-P1-A1 step 3). URL: https://arxiv.org/abs/2512.22697
11. Onatski, A. (2009). Testing Hypotheses About the Number of Factors in Large Factor Models. Econometrica 77(5), 1447-1479. Level: E1. <!-- URL/DOI to add; flag --> Onatski, A. (2010). Determining the Number of Factors from Empirical Distribution of Eigenvalues. Review of Economics and Statistics 92(4), 1004-1016. Level: E1. <!-- URL/DOI to add; flag -->
12. Onatski, A., Moreira, M. J., Hallin, M. (2013). Asymptotic Power of Sphericity Tests for High-Dimensional Data. Annals of Statistics 41(3), 1204-1229. Level: E2. URL: https://doi.org/10.1214/13-AOS1100 <!-- DOI to re-verify; flag -->
13. Stock, J. H., Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression. In: Identification and Inference for Econometric Models: Essays in Honor of Thomas Rothenberg (Andrews and Stock, eds.), 80-108. Cambridge University Press. Level: E2. <!-- Book chapter, no standalone DOI; NBER WP t0192 as locator; flag -->
14. Wachter, K. W. (1980). The limiting empirical measure of multiple discriminant ratios. Annals of Statistics 8(4), 937-957. Level: E2. URL: https://doi.org/10.1214/aos/1176345134

Sources marked with flag comments carry no verified URL/DOI yet and must be resolved at WP-P1-A1/A2 before any gate decision relies on them (rubric rule: UNVERIFIED evidence cannot carry a go decision).

---

*End of plan. Current state: Phase 1 COMPLETE (G0/G1 GO, see Research/weakiv_notes/gate_review_G0_G1.md); Phase 2 COMPLETE (G2 GO with WP-P3-R0 grid-pruning condition, see Research/weakiv_notes/gate_review_G2.md); Phase 3 dormant until the preregistration memo resolves the notebook-budget flag.*
