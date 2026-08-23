"""ivw_harvesting: surrogate inverse-variance weighting for intervention harvesting.

Count-only weighting schemes (min, harmonic) for intervention-harvesting
position-bias estimation, as described in "Surrogate Inverse-Variance Weighting
for Intervention Harvesting on Heavy-Tailed Click Logs" (Magnani & Xie, CIKM 2026).
"""

from ivw_harvesting.estimators import (
    AdjacentChainEstimator,
    AllPairsEstimator,
    PivotEstimator,
)
from ivw_harvesting.intervention_sets import (
    build_intervention_sets,
    normalize_bias,
)
from ivw_harvesting.weighting import (
    get_weight_fn,
    variance_decomposition,
    variance_term,
    weight_adaptive,
    weight_clipped,
    weight_harmonic,
    weight_min,
    weight_original,
)

__version__ = "1.0.0"

__all__ = [
    "AdjacentChainEstimator",
    "AllPairsEstimator",
    "PivotEstimator",
    "build_intervention_sets",
    "normalize_bias",
    "get_weight_fn",
    "variance_decomposition",
    "variance_term",
    "weight_adaptive",
    "weight_clipped",
    "weight_harmonic",
    "weight_min",
    "weight_original",
    "__version__",
]
