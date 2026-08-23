# Surrogate Inverse-Variance Weighting for Intervention Harvesting

Reproducibility code for the CIKM 2026 short paper:

> **Surrogate Inverse-Variance Weighting for Intervention Harvesting on Heavy-Tailed Click Logs**
> Alessandro Magnani (Coupang), Min Xie (Coupang). CIKM 2026.

## Method summary

Intervention harvesting estimates position bias (examination propensities
`p_k`) from click logs by comparing the click-through rates of the *same*
(query, document) pair when it appears at two different ranks. The classic
estimator of Agarwal et al. (2019) forms a ratio of summed CTRs across all such
pairs with **uniform** weights (`omega_i = 1`). On heavy-tailed logs, where a
pair's impression counts `N_k` and `N_kp` at the two ranks are wildly imbalanced,
uniform weighting lets a few noisy, low-count cells dominate the variance.

This package studies **count-only** weighting schemes — weights that depend only
on the impression counts, not on clicks, so they remain **ratio-unbiased**:

| Scheme | Weight `omega_i` | Notes |
| --- | --- | --- |
| `original` (uniform) | `1` | Agarwal et al. 2019 baseline |
| `min` | `min(N_k, N_kp)` | proposed |
| `harmonic` | `N_k · N_kp / (N_k + N_kp)` | proposed; **surrogate IVW optimum** |

The paper shows that harmonic weighting is the **surrogate inverse-variance
weighting (IVW)** optimum: it minimises a relevance-agnostic, count-only
surrogate of the delta-method estimator variance, and a Cauchy–Schwarz argument
guarantees it reduces this surrogate relative to uniform weighting. `min` is a
simpler robust alternative. (A biased `clipped` scheme is also included for
comparison.)

## Installation

```bash
pip install -e .                       # core library (numpy, pandas, tqdm)
pip install -e .[experiments,dev]      # + matplotlib, scikit-learn, torch, pytest
```

Only `AllPairsEstimator` needs `torch`; it is imported lazily, so the core
estimators and the test suite run without a PyTorch installation.

## Quickstart

```python
import pandas as pd
from ivw_harvesting import AdjacentChainEstimator

# Aggregated click log: each (query, doc) shown at two ranks with imbalanced counts.
df = pd.DataFrame({
    "query_id":    [1, 1, 2, 2],
    "doc_id":      ["a", "a", "b", "b"],
    "position":    [1, 2, 1, 2],
    "impressions": [1000, 100, 500, 2000],
    "clicks":      [300, 20, 150, 500],
})

estimator = AdjacentChainEstimator(weighting="harmonic")  # surrogate IVW optimum
propensities = estimator(
    df, query_col="query_id", doc_col="doc_id",
    imps_col="impressions", clicks_col="clicks",
)
print(propensities)   # columns: position, examination (position 1 == 1.0)
```

## Estimators and weighting schemes

Estimators (all in `ivw_harvesting.estimators`, re-exported at top level):

| Estimator | Description |
| --- | --- |
| `AdjacentChainEstimator` | chains CTR ratios between adjacent ranks |
| `AllPairsEstimator` | fits a PBM over all rank pairs (requires `torch`) |
| `PivotEstimator` | CTR ratios against a fixed pivot rank |

Weighting schemes (pass as the `weighting=` argument, or supply a callable
`weight_fn(N_k, N_kp) -> (omega_k, omega_kp)`):

`original`, `min`, `harmonic`, `variance_reduced` (alias of `min`),
`clipped_2`, `clipped_5`, `clipped_10`.

Weight functions and variance diagnostics live in `ivw_harvesting.weighting`
(`get_weight_fn`, `weight_original`, `weight_min`, `weight_harmonic`,
`weight_clipped`, `weight_adaptive`, `variance_term`, `variance_decomposition`).

## Input formats

1. **Binary**: one row per impression with a `click` column (0/1).
2. **Aggregated**: rows carry `impressions` and `clicks` counts (pass the column
   names via `imps_col` / `clicks_col`). Ensure `clicks <= impressions`.

## Reproducing the experiments

See [`experiments/README.md`](experiments/README.md). In short:

```bash
# Synthetic (no external data):
python experiments/synthetic_imbalance_sweep.py

# Yahoo semi-synthetic (needs Yahoo LTR Set 1):
python experiments/yahoo_semisynthetic.py --data-dir /path/to/Yahoo_ltr

# Downstream IPS-LTR nDCG:
python experiments/downstream_ltr.py --data-dir /path/to/Yahoo_ltr --pilot
```

Outputs are written under `./results/` (gitignored).

## Data

The experiments use publicly available datasets that are **not redistributed** here:

* **Yahoo Learning to Rank Challenge, Set 1** — <https://webscope.sandbox.yahoo.com>
  (requires a Yahoo Webscope account). Used for semi-synthetic relevance grades.
* **KDD Cup 2012 Track 2** click logs — from the KDD Cup 2012 archive.

## Tests

```bash
python -m pytest -q
```

The test suite covers the weighting functions, intervention-set construction,
and an end-to-end PBM recovery check for the estimators. It runs without `torch`
thanks to a stub in `tests/conftest.py`.

## Supplement

Proofs and additional derivations referenced by the paper are provided in
[`supplement/`](supplement/) (the supplement PDF is added separately).

## Citation

```bibtex
@inproceedings{magnani2026surrogate,
  title     = {Surrogate Inverse-Variance Weighting for Intervention Harvesting on Heavy-Tailed Click Logs},
  author    = {Magnani, Alessandro and Xie, Min},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Rome, Italy}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License and attribution

Released under the [MIT License](LICENSE) (© 2026 Alessandro Magnani, Min Xie).

The estimators and intervention-set construction are **derived from the
[ultr-bias-toolkit](https://github.com/philipphager/ultr-bias-toolkit) by Philipp
Hager** (open source), adapted and extended here with the count-only surrogate-IVW
(`min` / `harmonic`) weighting schemes. The underlying intervention-harvesting
methodology and the uniform-weighting baseline are due to Agarwal et al.,
"Estimating Position Bias without Intrusive Interventions" (WSDM 2019). See
[`NOTICE`](NOTICE) for full attribution.
