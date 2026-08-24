# spectraliv

Reference implementation for the Weak-Instrument Frontier project (Idea 2).
Phase 2 artifact (WP-P2-I1); see `../weakiv_notes/formalization_memo.md` for
the specification this package implements.

## Modules

| module | content |
|---|---|
| `rng` | deterministic seed streams (master seed 20260823, `Research/seeds.yaml`) |
| `preprocess` | explicit standardization + residualization, A3 properness assertion |
| `canoncorr` | QR/SVD canonical correlations + variates (single shared pass) |
| `jacobiquantiles` | exact Jacobi-ensemble null quantiles; Beta closed form at p=1 |
| `tw` | Tracy-Widom beta=1 via Bornemann Fredholm determinants (stable; validated to 13 digits against F2(-2)=0.413224142505123) |
| `teststats` | T_spec test: statistic, Johnstone-2009 constants, dual p-values |
| `ivestimators` | 2SLS, LIML, Fuller, Bekker, JIVE, truncated-2SLS, PCA-2SLS, Whiten-2SLS |
| `select_tau` | first-stage-only tau selection (anti-leakage invariant) |
| `dgps` | Phase-3 DGP ladder (null / single-spike / multi-spike / heteroskedastic) |

## Quick start

```python
from spectraliv import spec_test, select_tau, tsls, truncated_2sls
res = spec_test(X, Z)              # exact-Jacobi CVs (primary), TW (secondary)
tau = select_tau(X, Z)             # uses (X,Z) only; anti-leakage tested
beta = truncated_2sls(y, X, Z, tau=tau)
```

## Tests

```
python -m pytest tests -q            # fast suite
python -m pytest tests -m slow       # statistical validations (+ ~10 min)
```

Key identity tests enforce the plan's Phase-2 contract: Beta closed form,
exact-ensemble-vs-direct-MC agreement (KILL check), tau=1 == 2SLS,
anti-leakage under Y permutation, JIVE dual implementation, TW1 moments and
Bornemann reference value, P2 affine lift map with mandatory numeric check.
