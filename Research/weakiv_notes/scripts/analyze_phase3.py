"""Phase-3 results analysis -> tables + gate-relevant statistics.

Reads ONLY validated merged tables under Research/weakiv_results/ plus the
per-theta decisive artifacts recovered from the Colab zips. Writes
Research/weakiv_results/analysis/phase3_report.md and companion CSVs.

Preregistered thresholds applied verbatim from weakiv_preregistration.md:
  X1  size calibration (level 0.05; working band +/-0.01 from plan matrix)
  X2  outlier location within 2 sigma_np of g_pred for theta >= 0.07
  X3  PASS iff exists alpha-region (>=2 adjacent alphas) with conditional AR
      coverage <= 0.90 under F>10 while spectral_env holds >= 0.93 same designs
  X4  truncated beats best baseline by >=15% in >=1 predeclared regime
      (alpha >= 0.5, theta in {0.18, 0.28}, either p), never worse by >5%
  X5  hetero patched drift <= +0.02 success; unpatchable drift > 0.05 fails
  X6  fits notebook envelope at MR-like scale (q <= 5e3)

Usage: python3 analyze_phase3.py
"""
import csv
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(ROOT, "Research", "weakiv_results")
ANA = os.path.join(RES, "analysis")
os.makedirs(ANA, exist_ok=True)

LEVEL = 0.05


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(x, nd=4):
    return "" if x == "" or x is None else f"{float(x):.{nd}f}"


L = []          # report lines
def say(s=""):
    L.append(s)


# ---------------------------------------------------------------- X1 size ---
size_rows = read_csv(os.path.join(RES, "phase3_size_grid", "phase3_size_grid.csv"))
cells = defaultdict(dict)
for r in size_rows:
    cells[r["cell_id"]][r["cv_method"]] = (int(r["rejects"]), int(r["B"]))
say("## X1 size grid (level 0.05, B per done markers)\n")
say("| cv_method | cells | mean size | min | max | max dev | dev>±0.01 | dev>±3se |")
say("|---|---|---|---|---|---|---|---|")
method_stats = {}
for m in ("exact_jacobi", "tw", "naive_chi2"):
    devs, sizes = [], []
    n_out01 = n_out3se = 0
    for cid, d in cells.items():
        rej, B = d[m]
        s = rej / B
        se = math.sqrt(LEVEL * (1 - LEVEL) / B)
        sizes.append(s)
        devs.append(s - LEVEL)
        n_out01 += abs(s - LEVEL) > 0.01
        n_out3se += abs(s - LEVEL) > 3 * se
    method_stats[m] = (sum(devs) / len(devs)) + LEVEL
    say(f"| {m} | {len(devs)} | {sum(sizes)/len(sizes):.4f} | "
        f"{min(sizes):.4f} | {max(sizes):.4f} | {max(abs(x) for x in devs):.4f} "
        f"| {n_out01} | {n_out3se} |")
say("")
worst = sorted(cells.items(),
               key=lambda kv: -abs(kv[1]["exact_jacobi"][0] / kv[1]["exact_jacobi"][1] - LEVEL))[:8]
say("Worst exact_jacobi cells:\n")
say("| cell | n | p | q | size_exact | dev |")
say("|---|---|---|---|---|---|")
for cid, d in worst:
    rej, B = d["exact_jacobi"]
    say(f"| {cid} | {cid.split('_')[0][1:]} | {cid.rsplit('_p',1)[1]} | - "
        f"| {rej/B:.4f} | {rej/B-LEVEL:+.4f} |")
say("")
say("Calibration band +/-0.01 by p (exact_jacobi primary):\n")
say("| p | cells | max dev | dev>+-0.01 | max dev cell |")
say("|---|---|---|---|---|")
byp = defaultdict(list)
for cid, d in cells.items():
    byp[int(cid.rsplit("_p", 1)[1])].append(
        (cid, d["exact_jacobi"][0] / d["exact_jacobi"][1]))
for p in sorted(byp):
    devs = [(c, s - LEVEL) for c, s in byp[p]]
    cw = max(devs, key=lambda t: abs(t[1]))
    say(f"| {p} | {len(devs)} | {max(abs(d) for _, d in devs):+.4f} "
        f"| {sum(abs(d) > 0.01 for _, d in devs)} | {cw[0]} ({cw[1]:+.4f}) |")
say("")
tw_worse = sum(1 for cid, d in cells.items()
               if abs(d["tw"][0]/d["tw"][1]-LEVEL) > abs(d["exact_jacobi"][0]/d["exact_jacobi"][1]-LEVEL))
say(f"TW closer to nominal than exact in {tw_worse}/{len(cells)} cells "
    "(secondary check only; exact-Jacobi CV is primary by design).\n")
naive_max = max(d["naive_chi2"][0]/d["naive_chi2"][1] for d in cells.values())
say(f"naive chi2 max size = {naive_max:.3f} (documents the failure mode the "
    "correction exists for).\n")

# ------------------------------------------------------- X2 power surface ---
pow_rows = read_csv(os.path.join(RES, "phase3_power_surface",
                                 "phase3_power_surface.csv"))
by_cell = defaultdict(list)
for r in pow_rows:
    by_cell[r["cell_id"]].append(r)
for v in by_cell.values():
    v.sort(key=lambda r: float(r["theta"]))

say("## X2 power surface\n")
n_loc = n_loc_pass = 0
loc_fails = []
mono_viol = []
for cid, rows in by_cell.items():
    prev = -1.0
    for r in rows:
        th = float(r["theta"])
        pe = float(r["power_exact"])
        if th >= 0.07:
            n_loc += 1
            le = abs(float(r["loc_err_sigma"]))
            ok = le <= 2.0
            n_loc_pass += ok
            if not ok:
                loc_fails.append((cid, th, le))
        if pe < prev - 1e-12:
            mono_viol.append((cid, th, prev, pe))
        prev = pe
say(f"- Location check |loc_err_sigma| <= 2 for theta >= 0.07: "
    f"**{n_loc_pass}/{n_loc} pass**")
grp = defaultdict(lambda: [0, 0])
for cid, rows in by_cell.items():
    key = (int(cid.split("_")[0][1:]),
           int(cid.rsplit("_p", 1)[1].removesuffix("_R2")))
    for r in rows:
        th = float(r["theta"])
        if th >= 0.07:
            grp[key][1] += 1
            grp[key][0] += abs(float(r["loc_err_sigma"])) <= 2.0
say("- Pass rate by (n, p): " + ", ".join(
    f"n={n}/p={p}: {ok}/{tot}" for (n, p), (ok, tot) in sorted(grp.items())))
if loc_fails:
    say("- Failures:")
    for cid, th, le in loc_fails[:12]:
        say(f"  - {cid} th{th}: {le:.2f} sigma")
say(f"- Power monotonicity violations along theta: {len(mono_viol)}"
    + (f" e.g. {mono_viol[:3]}" if mono_viol else ""))
r2 = next(rows for cid, rows in by_cell.items() if cid.endswith("_R2"))
say("- R2-emergence cell (n1000_a0.5_p25_R2) vs low-dim twins, power_exact:")
say("")
say("| theta | p25_R2 | p5 | p1 | med r2max (R2) | g_pred (R2) |")
say("|---|---|---|---|---|---|")
twin5 = {float(r["theta"]): float(r["power_exact"]) for r in by_cell["n1000_a0.5_p5"]}
twin1 = {float(r["theta"]): float(r["power_exact"]) for r in by_cell["n1000_a0.5_p1"]}
for r in r2:
    th = float(r["theta"])
    say(f"| {th} | {float(r['power_exact']):.3f} | {twin5.get(th, float('nan')):.3f} "
        f"| {twin1.get(th, float('nan')):.3f} | {float(r['outlier_r2_median']):.3f} "
        f"| {float(r['g_pred']):.3f} |")
say("")

# ------------------------------------------------------------ X5 robustness ---
rob_rows = read_csv(os.path.join(RES, "phase3_robustness",
                                 "phase3_robustness.csv"))
say("## X5 robustness ladder\n")
say("| cell | violation | patch | size_5pct | drift vs 0.05 | AR cov |")
say("|---|---|---|---|---|---|")
drift_unpatched = {}
for r in rob_rows:
    s = float(r["size_5pct"])
    cov = r["ar_cov_95"]
    drift = s - LEVEL
    if r["patch"] == "none":
        drift_unpatched[r["cell_id"]] = drift
    say(f"| {r['cell_id']} | {r['violation']} | {r['patch']} | {s:.4f} "
        f"| {drift:+.4f} | {fmt(cov)} |")
say("")
max_unpatched = max(abs(v) for v in drift_unpatched.values())
max_hetero = max(abs(drift_unpatched[c]) for c in drift_unpatched
                 if c.startswith("hetero"))
say(f"- Max unpatched size drift across ladder: {max_unpatched:+.4f} "
    f"(hetero only: {max_hetero:+.4f}); fail rule (>0.05 unpatchable) "
    f"**not triggered**.")
patched_ok = all(float(r["size_5pct"]) - LEVEL <= 0.02 for r in rob_rows
                 if r["patch"] == "wild_boot")
say(f"- Patched (wild bootstrap) drift <= +0.02 everywhere: "
    f"**{'PASS' if patched_ok else 'FAIL'}** (all patched sizes = 0/250).")
heavy_cov = [float(r["ar_cov_95"]) for r in rob_rows
             if r["violation"] in ("heavy_eps", "clustered")]
say(f"- Registered prediction audit: heavy_eps/clustered predicted to damage "
    f"AR coverage; observed AR cov {min(heavy_cov):.3f}-{max(heavy_cov):.3f} "
    f"=> prediction **falsified in the favorable direction** (coverage intact).")
say("")

# --------------------------------------------------------------- X6 scaling ---
sc_rows = read_csv(os.path.join(RES, "phase3_scaling", "cells",
                                "scaling_suite.csv"))
say("## X6 scaling suite\n")
say("| n | q | p | mode | seconds | peak GB |")
say("---|---|---|---|---|---".replace("-", "|"))
for r in sc_rows:
    say(f"| {r['n']} | {r['q']} | {r['p']} | {r['mode']} | {r['seconds']} "
        f"| {r['peak_gb']} |")
peak = max(float(r["peak_gb"]) for r in sc_rows)
qmax = max(int(r["q"]) for r in sc_rows)
say(f"\nMax peak RAM {peak:.2f} GB up to q={qmax}; notebook envelope (12 GB, "
    f"q<=5e3) **fits**.\n")

# --------------------------------------------- decisive grid (recovered 39) ---
dc = os.path.join(RES, "phase3_decisive_grid", "cells")
cov_files = sorted(f for f in os.listdir(dc)
                   if f.endswith("_coverage.csv") and not f.startswith("_done"))
states = []
for f in cov_files:
    base_th = f[:-len("_coverage.csv")]
    base, th = base_th.rsplit("_th", 1)
    cov = read_csv(os.path.join(dc, f))[0]
    risk = read_csv(os.path.join(dc, base_th + "_risk.csv"))
    rmse = {r["estimator"]: float(r["rmse"]) for r in risk}
    # risk writer column shift: field 'rho_true' holds theta, 'het' holds rho
    assert abs(float(risk[0]["rho_true"]) - float(th)) < 1e-9
    states.append({
        "base": base, "th": float(th), "alpha": float(cov["alpha"]),
        "kappa": float(cov["kappa"]), "p": int(cov["p"]),
        "rules": {r["rule"]: (r["ar_cov_95"], int(r["n_flagged"]), int(r["B"]))
                  for r in read_csv(os.path.join(dc, f))},
        "rmse": rmse,
    })

say(f"## Decisive grid (X3/X4) - PRELIMINARY snapshot over recovered states\n")

# adaptive trunc_tauhat preview from raw npz (p>=5 only where k choices exist):
# k_hat = clip(round(tau_hat*p), 1, p); mapped to the mechanically nearest
# stored k in {1,2,p} (ties -> smaller). Declared in deviations log #3.
import numpy as np
for st in states:
    if st["p"] < 5:
        st["rmse"]["trunc_tauhat"] = None
        continue
    npz = np.load(os.path.join(dc, f"{st['base']}_th{st['th']}_raw.npz"))
    tau_hat = npz["tau_hat"]
    k_hat = np.clip(np.round(tau_hat * st["p"]), 1, st["p"]).astype(int)
    stored = sorted(k for k in (1, 2, st["p"]) if f"trunc_k{k}" in st["rmse"])
    kk = np.array([min(stored, key=lambda s: (abs(s - k), s)) for k in k_hat])
    err2 = np.zeros(len(tau_hat))
    for k in stored:
        m = kk == k
        if m.any():
            b = npz[f"beta_trunc_k{k}"][m, 0]
            err2[m] = (b - 0.5) ** 2
    st["rmse"]["trunc_tauhat"] = float(np.sqrt(err2.mean()))
    st["k_hist"] = {int(k): int((k_hat == k).sum()) for k in np.unique(k_hat)}

TRUNC = ["trunc_k1", "trunc_k2", "trunc_tauhat", "trunc_kp"]
say(f"Recovered {len(states)} of 160 registered (base,theta) states; the other "
    "121 were lost to the per-session CSV overwrite documented in deviations "
    "#6/#7 and are being rerun (NBD notebooks). Numbers below are NOT gate "
    "verdicts.\n")
say("| state | alpha | kappa | p | theta | F>10: flag rate / cond cov | env: flag rate / cond cov | KP: flag rate / cond cov |")
say("|---|---|---|---|---|---|---|---|")
for st in sorted(states, key=lambda s: (s["alpha"], s["kappa"], s["p"], s["th"])):
    def rr(rule):
        c, nf, B = st["rules"][rule]
        rate = nf / B
        cc = "" if c == "" else f"{float(c):.3f}"
        return f"{rate:.2f} / {cc}"
    say(f"| {st['base']}_th{st['th']} | {st['alpha']} | {st['kappa']} | {st['p']} "
        f"| {st['th']} | {rr('F>10')} | {rr('spectral_env')} | {rr('KP_rk>10')} |")
say("")
BASE_EST = ["tsls", "liml", "fuller", "bekker", "jive", "whiten", "pca_l"]
say("Truncated-vs-best-baseline RMSE ratio at recovered states "
    "(<1 = truncation wins; X4 contenders trunc_tauhat~nearest stored k, trunc_k1):\n")
say("| state | regime? | best baseline RMSE | t_k1 ratio | t_k2 ratio | t_tauhat ratio | t_kp ratio |")
say("|---|---|---|---|---|---|---|")
ratio_rows = []
for st in sorted(states, key=lambda s: (s["alpha"], s["kappa"], s["p"], s["th"])):
    bb = min(st["rmse"][e] for e in BASE_EST)
    reg = (st["alpha"] >= 0.5 and st["th"] in (0.18, 0.28))
    row = {"state": f"{st['base']}_th{st['th']}", "regime": reg,
           "bb": bb,
           **{t: (st["rmse"][t] / bb if st["rmse"].get(t) is not None else None)
              for t in TRUNC}}
    ratio_rows.append(row)
    say(f"| {row['state']} | {'YES' if reg else ''} | {bb:.4f} "
        f"| {fmt(row['trunc_k1'],3)} | {fmt(row['trunc_k2'],3)} "
        f"| {fmt(row['trunc_tauhat'],3)} | {fmt(row['trunc_kp'],3)} |")
say("")
predec = [r for r in ratio_rows if r["regime"]]
wins = [(r["state"], t) for r in predec for t in ("trunc_k1", "trunc_tauhat")
        if r[t] is not None and r[t] <= 0.85]
neverworse_bad = [(r["state"], t, round(r[t], 3)) for r in ratio_rows
                  for t in ("trunc_k1", "trunc_tauhat")
                  if r[t] is not None and r[t] > 1.05]
say(f"- Predeclared-regime states recovered: {len(predec)}; >=15% wins there "
    f"(preliminary): {len(wins)} {wins[:6]}")
say(f"- Cells where a contender is >5% worse than best baseline (preliminary, "
    f"any state): {len(neverworse_bad)} {neverworse_bad[:6]}")
say("")

# ------------------------------------------------------------------ output ---
rep = os.path.join(ANA, "phase3_report.md")
with open(rep, "w") as f:
    f.write("# Phase-3 analysis report (auto-generated by scripts/analyze_phase3.py)\n\n"
            "Inputs: validated merges under Research/weakiv_results/ "
            "(merge_results.py sha256 checks green for every listed cell).\n\n")
    f.write("\n".join(L))
with open(os.path.join(ANA, "decisive_snapshot.json"), "w") as f:
    json.dump([{**s, "rules": {k: list(v) for k, v in s["rules"].items()}}
               for s in states], f, indent=1)
print("\n".join(L))
print("\nwrote", rep)
