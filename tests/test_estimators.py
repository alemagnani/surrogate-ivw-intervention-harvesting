"""End-to-end tests for the intervention-harvesting estimators.

Simulates Position-Based Model (PBM) clicks with known per-position propensities
p_k and checks that:
  * AdjacentChainEstimator recovers the (decreasing) bias ordering,
  * the harmonic and min weighting schemes run and give finite propensities,
  * the clicks<=impressions validation raises.
"""

import numpy as np
import pandas as pd
import pytest

from ivw_harvesting import (
    AdjacentChainEstimator,
    PivotEstimator,
)


NUM_POSITIONS = 5
ETA = 1.0


def _true_propensities(num_positions: int, eta: float) -> np.ndarray:
    p = np.array([1.0 / k**eta for k in range(1, num_positions + 1)])
    return p / p[0]


def _simulate_pbm(seed: int = 0, num_docs: int = 400) -> pd.DataFrame:
    """Simulate aggregated PBM clicks for adjacent (k, k+1) pairs.

    Each doc is shown at two adjacent positions with plentiful impressions,
    so the adjacent-chain estimator has enough signal to recover p_k.
    """
    rng = np.random.default_rng(seed)
    true_bias = _true_propensities(NUM_POSITIONS, ETA)
    rows = []
    doc_id = 0
    for pos in range(1, NUM_POSITIONS):
        p_k = true_bias[pos - 1]
        p_kp = true_bias[pos]
        for _ in range(num_docs):
            rel = rng.uniform(0.2, 0.6)
            n_k = int(rng.integers(500, 2000))
            n_kp = int(rng.integers(500, 2000))
            c_k = int(rng.binomial(n_k, rel * p_k))
            c_kp = int(rng.binomial(n_kp, rel * p_kp))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                         "impressions": n_k, "clicks": c_k})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                         "impressions": n_kp, "clicks": c_kp})
            doc_id += 1
    return pd.DataFrame(rows)


def test_adjacent_chain_recovers_bias_ordering():
    """AdjacentChain should recover a strictly decreasing examination curve."""
    df = _simulate_pbm(seed=1)
    est = AdjacentChainEstimator(weighting="original")
    result = est(df, query_col="query_id", doc_col="doc_id",
                 imps_col="impressions", clicks_col="clicks")

    exam = result.sort_values("position")["examination"].values
    assert len(exam) == NUM_POSITIONS
    assert np.all(np.isfinite(exam))
    # Position 1 is the reference (examination == 1).
    assert np.isclose(exam[0], 1.0)
    # Examination must be monotonically decreasing (bias ordering).
    assert np.all(np.diff(exam) < 0), f"Not decreasing: {exam}"

    # Estimated curve should be reasonably close to the true 1/k propensities.
    true_bias = _true_propensities(NUM_POSITIONS, ETA)
    assert np.max(np.abs(exam - true_bias)) < 0.1


@pytest.mark.parametrize("scheme", ["harmonic", "min"])
def test_harmonic_and_min_run_and_are_finite(scheme):
    """Harmonic and min weighting run end-to-end and give finite propensities."""
    df = _simulate_pbm(seed=2)
    est = AdjacentChainEstimator(weighting=scheme)
    result = est(df, query_col="query_id", doc_col="doc_id",
                 imps_col="impressions", clicks_col="clicks")

    exam = result.sort_values("position")["examination"].values
    assert len(exam) == NUM_POSITIONS
    assert np.all(np.isfinite(exam))
    assert np.all(exam > 0)
    # Should also recover the decreasing ordering.
    assert np.all(np.diff(exam) < 0), f"Not decreasing for {scheme}: {exam}"


def test_pivot_estimator_runs_with_harmonic():
    """PivotEstimator runs with harmonic weighting and gives finite output."""
    df = _simulate_pbm(seed=3)
    est = PivotEstimator(pivot_rank=1, weighting="harmonic")
    result = est(df, query_col="query_id", doc_col="doc_id",
                 imps_col="impressions", clicks_col="clicks")
    exam = result.sort_values("position")["examination"].values
    assert np.all(np.isfinite(exam))
    assert np.isclose(exam[0], 1.0)


def test_clicks_exceed_impressions_raises():
    """The clicks<=impressions validation should raise a ValueError."""
    bad_df = pd.DataFrame({
        "query_id": [0, 0],
        "doc_id": [0, 0],
        "position": [1, 2],
        "impressions": [10, 10],
        "clicks": [11, 3],  # 11 > 10
    })
    est = AdjacentChainEstimator(weighting="harmonic")
    with pytest.raises(ValueError, match="Clicks must be <= impressions"):
        est(bad_df, query_col="query_id", doc_col="doc_id",
            imps_col="impressions", clicks_col="clicks")
