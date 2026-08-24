"""WP-P3-R0 benchmark: fast shared-pass path vs the Phase-2 cost model.

Measures per-replication wall time of (a) the full fast_rep battery and
(b) the null-only statistic pass, at representative decisive/size-grid
configs; compares against the Phase-2 NNLS cost model to produce the
compute table for the preregistration memo.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import time

import numpy as np

from spectraliv.dgps import make_single_spike
from spectraliv.experiments import _canonical_pass, fast_rep
from spectraliv.preprocess import prepare

COST_COEF_P2 = [0.046402864733551284, 4.227491628427171e-06,
                5.766583220930669e-09, 0.0]


def p2_predict(n, p, q, B=1):
    feats = np.array([1.0, n * (p + q), n * q ** 2, n * p ** 2])
    return float(np.array(COST_COEF_P2) @ feats) * B


CONFIGS = [
    ("decisive_p1_mid", 1000, 1, 500),
    ("decisive_p1_hi", 1000, 1, 900),
    ("decisive_p5_lo", 2000, 5, 200),
    ("decisive_p5_mid", 2000, 5, 1000),
    ("decisive_p5_hi", 2000, 5, 1800),
    ("size_n1000_p25_a0.5", 1000, 25, 500),
]

for name, n, p, q in CONFIGS:
    rng = np.random.default_rng(42)
    dgp = make_single_spike(n, q, 0.30, 0.5, rng, p=p, beta=0.5)
    reps = 8 if n <= 1000 else 5
    t0 = time.time()
    for _ in range(reps):
        fast_rep(dgp.y, dgp.x, dgp.z, estimators=True,
                 k_list=sorted({1, 2, p}))
    full = (time.time() - t0) / reps
    t0 = time.time()
    for _ in range(max(reps, 6)):
        xs, zs, _y, _s = prepare(dgp.x, dgp.z, None, None)
        _canonical_pass(xs, zs)
    stat_only = (time.time() - t0) / max(reps, 6)
    print(f"{name:24s} n={n} p={p:3d} q={q:5d}  "
          f"full={full:7.3f}s/rep  stat_only={stat_only:7.4f}s/rep  "
          f"p2_model_full_battery={p2_predict(n, p, q):7.3f}s")
