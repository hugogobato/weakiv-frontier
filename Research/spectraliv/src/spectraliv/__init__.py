"""spectraliv: spectral calibration for weak-instrument testing and truncated IV estimation.

Modules
-------
rng              deterministic seed streams
preprocess       standardization + residualization (explicit, tested)
canoncorr        sample canonical correlations via QR/SVD
jacobiquantiles  exact null (Jacobi ensemble) quantiles; Beta closed form at p = 1
tw               Tracy-Widom beta=1 distribution via Hastings-McLeod Painleve II
teststats        T_spec statistic, Johnstone-2009 constants, p-values
ivestimators     2SLS/LIML/Fuller/Bekker/JIVE/truncated-2SLS/PCA-2SLS/Whiten-2SLS
select_tau       first-stage-only tau selection (anti-leakage invariant)
dgps             data-generating processes for Phase 3 grids
"""

from .canoncorr import canonical_analysis
from .teststats import spec_test, jacobi_mu_sigma
from .ivestimators import (
    ols,
    tsls,
    kclass,
    liml,
    fuller,
    bekker,
    jive,
    truncated_2sls,
    pca_2sls,
    whiten_2sls,
)
from .select_tau import select_tau

__version__ = "0.1.0"

__all__ = [
    "canonical_analysis",
    "spec_test",
    "jacobi_mu_sigma",
    "ols",
    "tsls",
    "kclass",
    "liml",
    "fuller",
    "bekker",
    "jive",
    "truncated_2sls",
    "pca_2sls",
    "whiten_2sls",
    "select_tau",
    "__version__",
]
