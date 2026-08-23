"""Tests for the weighting module and its weighting schemes."""

import numpy as np
import pandas as pd
import pytest

from ivw_harvesting.intervention_sets import build_intervention_sets
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


# ---------------------------------------------------------------------------
# Weight function unit tests
# ---------------------------------------------------------------------------

def test_weight_original():
    N_k = np.array([100.0, 200.0, 50.0])
    N_kp = np.array([50.0, 100.0, 300.0])
    ok, okp = weight_original(N_k, N_kp)
    assert np.all(ok == 1.0)
    assert np.all(okp == 1.0)


def test_weight_min():
    N_k = np.array([100.0, 50.0])
    N_kp = np.array([50.0, 200.0])
    ok, okp = weight_min(N_k, N_kp)
    np.testing.assert_array_equal(ok, [50.0, 50.0])
    np.testing.assert_array_equal(okp, [50.0, 50.0])


def test_weight_harmonic():
    N_k = np.array([100.0, 100.0])
    N_kp = np.array([100.0, 900.0])
    ok, okp = weight_harmonic(N_k, N_kp)
    # harmonic(100, 100) = 50; harmonic(100, 900) = 90
    np.testing.assert_allclose(ok, [50.0, 90.0])
    np.testing.assert_allclose(okp, [50.0, 90.0])


def test_weight_harmonic_equals_min_when_equal():
    """When N_k == N_kp, harmonic == N/2 and min == N. Different, not equal."""
    N = np.array([100.0, 200.0])
    oh, _ = weight_harmonic(N, N)
    om, _ = weight_min(N, N)
    # harmonic(N,N) = N/2, min(N,N) = N
    np.testing.assert_allclose(oh, N / 2)
    np.testing.assert_allclose(om, N)


def test_weight_clipped_symmetric():
    """When N_k == N_kp, clipping changes nothing."""
    N = np.array([100.0, 200.0])
    ok, okp = weight_clipped(N, N, tau=5.0)
    np.testing.assert_array_equal(ok, N)
    np.testing.assert_array_equal(okp, N)


def test_weight_clipped_asymmetric():
    """With tau=2 and ratio=10, majority gets clipped."""
    N_k = np.array([100.0])
    N_kp = np.array([10.0])
    ok, okp = weight_clipped(N_k, N_kp, tau=2.0)
    # omega_k = min(100, 2*10) = 20
    np.testing.assert_allclose(ok, [20.0])
    # omega_kp = min(10, 2*100) = 10 (no clip)
    np.testing.assert_allclose(okp, [10.0])


def test_weight_adaptive_reduces_to_harmonic_when_equal():
    """When alpha == beta, adaptive == harmonic."""
    N_k = np.array([100.0, 200.0])
    N_kp = np.array([50.0, 300.0])
    alpha = beta = 1.0
    oa, _ = weight_adaptive(N_k, N_kp, alpha, beta)
    oh, _ = weight_harmonic(N_k, N_kp)
    np.testing.assert_allclose(oa, oh)


def test_get_weight_fn_all_schemes():
    for scheme in ["original", "variance_reduced", "min", "harmonic",
                   "clipped_2", "clipped_5", "clipped_10"]:
        fn = get_weight_fn(scheme)
        N = np.array([100.0, 200.0])
        ok, okp = fn(N, N)
        assert ok.shape == N.shape
        assert okp.shape == N.shape


def test_get_weight_fn_unknown_raises():
    with pytest.raises(ValueError, match="Unknown weighting scheme"):
        get_weight_fn("nonexistent")


# ---------------------------------------------------------------------------
# Ratio unbiasedness: harmonic and min preserve E[A]/E[B] = p_k/p_kp
# For a single pair (q,d), A = omega * CTR_k, B = omega * CTR_kp
# E[A]/E[B] = E[omega * C_k / N_k] / E[omega * C_kp / N_kp]
#           = omega * p_k / (omega * p_kp) = p_k / p_kp (since omega is count-based)
# ---------------------------------------------------------------------------

def test_harmonic_ratio_unbiased_analytically():
    """Verify that for a single doc, omega cancels in expectation."""
    p_k, p_kp = 0.5, 0.2
    rel = 0.3
    N_k, N_kp = np.array([1000.0]), np.array([100.0])

    omega_h, _ = weight_harmonic(N_k, N_kp)

    expected_A = omega_h[0] * rel * p_k
    expected_B = omega_h[0] * rel * p_kp

    ratio = expected_A / expected_B
    assert abs(ratio - p_k / p_kp) < 1e-9


# ---------------------------------------------------------------------------
# Integration: build_intervention_sets with the schemes
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "doc_id": ["a", "a", "b", "b"],
        "position": [1, 2, 1, 2],
        "impressions": [1000, 100, 500, 2000],
        "clicks": [300, 20, 150, 500],
    })


def test_harmonic_weighting_build(sample_df):
    result = build_intervention_sets(
        sample_df, "query_id", "doc_id", "impressions", "clicks",
        weighting="harmonic",
    )
    pos_1_2 = result[(result.position_0 == 1) & (result.position_1 == 2)]
    assert not pos_1_2.empty

    # doc a: N_k=1000, N_kp=100, omega=1000*100/1100~90.9, factor_0=90.9/1000
    # weighted_c_0(a) = 300 * 90.9/1000 ~ 27.27
    # doc b: N_k=500, N_kp=2000, omega=500*2000/2500=400, factor_0=400/500=0.8
    # weighted_c_0(b) = 150 * 0.8 = 120
    # total c_0 = 27.27 + 120 = 147.27
    expected_c0 = 300 * (100 / 1100) + 150 * (2000 / 2500)
    assert np.isclose(pos_1_2.c_0.iloc[0], expected_c0, rtol=1e-4)


def test_clipped_weighting_build(sample_df):
    result = build_intervention_sets(
        sample_df, "query_id", "doc_id", "impressions", "clicks",
        weighting="clipped_5",
    )
    pos_1_2 = result[(result.position_0 == 1) & (result.position_1 == 2)]
    assert not pos_1_2.empty

    # doc a: N_k=1000, N_kp=100, tau=5: omega_k=min(1000,500)=500, factor=500/1000=0.5
    # doc b: N_k=500, N_kp=2000, tau=5: omega_k=min(500,10000)=500, factor=500/500=1.0
    expected_c0 = 300 * 0.5 + 150 * 1.0
    assert np.isclose(pos_1_2.c_0.iloc[0], expected_c0, rtol=1e-4)


def test_callable_weight_fn(sample_df):
    """Test passing a custom callable weight function."""
    from ivw_harvesting.weighting import weight_harmonic

    result = build_intervention_sets(
        sample_df, "query_id", "doc_id", "impressions", "clicks",
        weighting=weight_harmonic,
    )
    result_str = build_intervention_sets(
        sample_df, "query_id", "doc_id", "impressions", "clicks",
        weighting="harmonic",
    )
    pd.testing.assert_frame_equal(
        result.sort_values(["position_0", "position_1"]).reset_index(drop=True),
        result_str.sort_values(["position_0", "position_1"]).reset_index(drop=True),
    )


def test_all_schemes_produce_valid_output(sample_df):
    """All schemes should return non-empty DataFrames with expected columns."""
    for scheme in ["original", "min", "harmonic", "clipped_2", "clipped_5", "clipped_10"]:
        result = build_intervention_sets(
            sample_df, "query_id", "doc_id", "impressions", "clicks",
            weighting=scheme,
        )
        assert len(result) > 0
        for col in ["position_0", "position_1", "c_0", "c_1", "not_c_0", "not_c_1"]:
            assert col in result.columns, f"Missing {col} for scheme {scheme}"


# ---------------------------------------------------------------------------
# Variance diagnostics
# ---------------------------------------------------------------------------

def test_variance_term_basic():
    omega = np.array([1.0, 1.0, 1.0])
    N_r = np.array([10.0, 10.0, 10.0])
    # V = sum(1/10) / 9 = 3/10 / 9 = 1/30
    expected = (3 * (1.0 / 10.0)) / 9.0
    assert np.isclose(variance_term(omega, N_r), expected)


def test_variance_decomposition_cauchy_schwarz():
    """Cauchy-Schwarz: D_k + D_kp <= 0 always (harmonic reduces sum of V_k + V_kp)."""
    rng = np.random.default_rng(0)
    N_k = rng.integers(10, 500, size=100).astype(float)
    N_kp = rng.integers(10, 500, size=100).astype(float)
    alpha, beta = 3.0, 5.0

    decomp = variance_decomposition(N_k, N_kp, alpha, beta)
    # By Cauchy-Schwarz, D_k + D_kp <= 0 and term1 = beta*(D_k+D_kp) <= 0
    assert decomp["D_k"] + decomp["D_kp"] <= 1e-10, (
        f"D_k + D_kp should be <= 0, got {decomp['D_k'] + decomp['D_kp']}"
    )
    assert decomp["term1"] <= 1e-10, f"Expected term1 <= 0, got {decomp['term1']}"


def test_variance_decomposition_total_reduction_systematic():
    """Total variance reduction is negative under systematic imbalance (N_k >> N_kp)."""
    rng = np.random.default_rng(42)
    # Strongly systematic: N_k >> N_kp for all pairs
    N_k = rng.integers(200, 500, size=100).astype(float)
    N_kp = rng.integers(10, 30, size=100).astype(float)  # always small
    # alpha < beta: position k has higher propensity (pos k is earlier, lower k index)
    alpha, beta = 0.5, 2.0

    decomp = variance_decomposition(N_k, N_kp, alpha, beta)
    assert decomp["total"] <= 1e-10, (
        f"Expected total variance reduction <= 0, got {decomp['total']}"
    )
