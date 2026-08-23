import numpy as np
import pandas as pd
import pytest

from ivw_harvesting.intervention_sets import (
    build_intervention_sets,
    build_intervention_sets_aggregated,
    build_intervention_sets_binary,
    build_intervention_sets_variance_reduced,
)


def test_binary_and_aggregated_produces_same_results():
    """Test that both input formats produce identical results."""
    binary_df = pd.DataFrame({
        "query_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "doc_id": ["a", "b", "a", "c", "d", "e", "d", "f"],
        "position": [1, 2, 3, 4, 1, 2, 3, 4],
        "click": [1, 0, 0, 0, 1, 1, 0, 0]
    })

    aggregated_df = pd.DataFrame({
        "query_id": [1, 1, 1, 2, 2, 2],
        "doc_id": ["a", "b", "c", "d", "e", "f"],
        "position": [1, 2, 4, 1, 2, 4],  # Missing position 3 for test
        "impressions": [2, 1, 1, 2, 1, 1],
        "clicks": [1, 0, 0, 1, 1, 0]
    })

    extra_row = pd.DataFrame({
        "query_id": [1, 2],
        "doc_id": ["a", "d"],
        "position": [3, 3],
        "impressions": [1, 1],
        "clicks": [0, 0]
    })

    aggregated_df = pd.concat([aggregated_df, extra_row], ignore_index=True)

    binary_result = build_intervention_sets(binary_df, "query_id", "doc_id")
    aggregated_result = build_intervention_sets(
        aggregated_df, "query_id", "doc_id",
        imps_col="impressions", clicks_col="clicks"
    )

    binary_result = binary_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    aggregated_result = aggregated_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)

    assert binary_result.shape == aggregated_result.shape
    assert all(binary_result.position_0 == aggregated_result.position_0)
    assert all(binary_result.position_1 == aggregated_result.position_1)
    assert set(binary_result.columns) == set(aggregated_result.columns)


def test_repeated_rows_in_aggregated_format():
    """Test that repeated query-document-position rows get properly aggregated."""
    repeated_df = pd.DataFrame({
        "query_id": [1, 1, 1, 1, 1, 2, 2, 2],
        "doc_id": ["a", "a", "b", "b", "a", "c", "c", "d"],
        "position": [1, 1, 2, 2, 3, 1, 1, 2],
        "impressions": [100, 200, 150, 50, 300, 400, 100, 250],
        "clicks": [30, 50, 40, 10, 60, 120, 30, 50]
    })

    preaggregated_df = pd.DataFrame({
        "query_id": [1, 1, 1, 2, 2],
        "doc_id": ["a", "b", "a", "c", "d"],
        "position": [1, 2, 3, 1, 2],
        "impressions": [300, 200, 300, 500, 250],
        "clicks": [80, 50, 60, 150, 50]
    })

    repeated_result = build_intervention_sets(
        repeated_df, "query_id", "doc_id",
        imps_col="impressions", clicks_col="clicks"
    )
    preaggregated_result = build_intervention_sets(
        preaggregated_df, "query_id", "doc_id",
        imps_col="impressions", clicks_col="clicks"
    )

    repeated_result = repeated_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    preaggregated_result = preaggregated_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)

    assert repeated_result.shape == preaggregated_result.shape
    for col in ["c_0", "c_1", "not_c_0", "not_c_1"]:
        assert np.allclose(repeated_result[col], preaggregated_result[col], rtol=0.01)


def test_aggregated_input_validation():
    """Test validation that clicks <= impressions."""
    invalid_df = pd.DataFrame({
        "query_id": [1],
        "doc_id": ["a"],
        "position": [1],
        "impressions": [5],
        "clicks": [6]  # More clicks than impressions
    })

    with pytest.raises(ValueError):
        build_intervention_sets(
            invalid_df, "query_id", "doc_id",
            imps_col="impressions", clicks_col="clicks"
        )


def test_aggregation_in_aggregated_format():
    """Test that repeated rows get properly aggregated."""
    df = pd.DataFrame({
        "query_id": [1, 1, 1, 1],
        "doc_id": ["a", "a", "b", "b"],
        "position": [1, 1, 2, 2],
        "impressions": [3, 2, 5, 3],
        "clicks": [2, 1, 3, 1]
    })

    result = build_intervention_sets_aggregated(
        df, "query_id", "doc_id", "impressions", "clicks"
    )

    assert len(result) == 2
    row = result[result.position_0 == 1].iloc[0]
    assert np.isclose(row.c_0, 0.6, rtol=0.01)
    assert np.isclose(row.c_1, 0.6, rtol=0.01)


def test_large_aggregated_dataset():
    """Test with a larger aggregated dataset."""
    df = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "doc_id": ["a", "b", "c", "d"],
        "position": [1, 2, 1, 2],
        "impressions": [1000, 1000, 2000, 2000],
        "clicks": [300, 200, 500, 250]
    })

    result = build_intervention_sets_aggregated(
        df, "query_id", "doc_id", "impressions", "clicks"
    )

    assert len(result) == 2
    assert "c_0" in result.columns
    assert "c_1" in result.columns
    assert "not_c_0" in result.columns
    assert "not_c_1" in result.columns


def test_variance_reduced_weighting_basic():
    """Test the basic functionality of variance-reduced (min) weighting."""
    df = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "doc_id": ["a", "a", "b", "b"],
        "position": [1, 2, 1, 2],
        "impressions": [1000, 100, 500, 2000],
        "clicks": [300, 20, 150, 500]
    })

    vr_result = build_intervention_sets(
        df, "query_id", "doc_id", "impressions", "clicks",
        weighting="variance_reduced"
    )
    vr_result = vr_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)

    # doc a: clicks=300, weight=min(1000,100)/1000=0.1 -> 30
    # doc b: clicks=150, weight=min(500,2000)/500=1.0 -> 150 ; total=180
    pos_1_2 = vr_result[(vr_result.position_0 == 1) & (vr_result.position_1 == 2)]
    assert not pos_1_2.empty
    assert np.isclose(pos_1_2.c_0.iloc[0], 180, rtol=0.01)


def test_variance_reduced_weighting_repeated_rows():
    """Test variance-reduced weighting with repeated rows that need aggregation."""
    repeated_df = pd.DataFrame({
        "query_id": [1, 1, 1, 1, 2, 2],
        "doc_id": ["a", "a", "b", "b", "c", "c"],
        "position": [1, 1, 2, 2, 1, 2],
        "impressions": [800, 200, 300, 100, 500, 600],
        "clicks": [240, 60, 60, 20, 150, 120]
    })

    preaggregated_df = pd.DataFrame({
        "query_id": [1, 1, 2],
        "doc_id": ["a", "b", "c"],
        "position": [1, 2, 1],
        "impressions": [1000, 400, 500],
        "clicks": [300, 80, 150]
    })
    preaggregated_extra = pd.DataFrame({
        "query_id": [2],
        "doc_id": ["c"],
        "position": [2],
        "impressions": [600],
        "clicks": [120]
    })
    preaggregated_df = pd.concat([preaggregated_df, preaggregated_extra], ignore_index=True)

    repeated_result = build_intervention_sets(
        repeated_df, "query_id", "doc_id",
        imps_col="impressions", clicks_col="clicks", weighting="variance_reduced"
    )
    preaggregated_result = build_intervention_sets(
        preaggregated_df, "query_id", "doc_id",
        imps_col="impressions", clicks_col="clicks", weighting="variance_reduced"
    )

    repeated_result = repeated_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)
    preaggregated_result = preaggregated_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(repeated_result, preaggregated_result, check_dtype=False)


def test_variance_reduced_weighting_extreme_imbalance():
    """Test variance-reduced weighting with extreme impression imbalances."""
    df = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "doc_id": ["a", "a", "b", "b"],
        "position": [1, 2, 1, 2],
        "impressions": [10000, 10, 20, 5000],
        "clicks": [3000, 2, 5, 1000]
    })

    vr_result = build_intervention_sets(
        df, "query_id", "doc_id", "impressions", "clicks",
        weighting="variance_reduced"
    )
    vr_result = vr_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)

    # doc a: 3000 * min(10000,10)/10000=0.001 -> 3 ; doc b: 5 * 1.0 -> 5 ; total=8
    pos_1_2 = vr_result[(vr_result.position_0 == 1) & (vr_result.position_1 == 2)]
    assert not pos_1_2.empty
    assert np.isclose(pos_1_2.c_0.iloc[0], 8, rtol=0.01)


def test_aggregated_format_conversion():
    """Test that build_intervention_sets_aggregated correctly handles click rates."""
    df = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "doc_id": ["a", "a", "b", "b"],
        "position": [1, 2, 1, 2],
        "impressions": [500, 100, 200, 800],
        "clicks": [150, 20, 60, 160]
    })

    result_df = build_intervention_sets_aggregated(df, "query_id", "doc_id", "impressions", "clicks")

    pos_1_2 = result_df[(result_df.position_0 == 1) & (result_df.position_1 == 2)]
    assert not pos_1_2.empty
    assert np.isclose(pos_1_2.c_0.iloc[0], 0.6, rtol=0.01)

    pos_2_1 = result_df[(result_df.position_0 == 2) & (result_df.position_1 == 1)]
    assert not pos_2_1.empty
    assert np.isclose(pos_2_1.c_0.iloc[0], 0.4, rtol=0.01)


def test_variance_reduced_weighting_raw_counts():
    """Test that variance-reduced weighting properly applies weights to click counts."""
    df = pd.DataFrame({
        "query_id": [1, 1, 2, 2],
        "doc_id": ["a", "a", "b", "b"],
        "position": [1, 2, 1, 2],
        "impressions": [500, 100, 200, 800],
        "clicks": [150, 20, 60, 160]
    })

    vr_result = build_intervention_sets(
        df, "query_id", "doc_id", "impressions", "clicks",
        weighting="variance_reduced"
    )
    vr_result = vr_result.sort_values(["position_0", "position_1"]).reset_index(drop=True)

    # doc a: 150 * min(500,100)/500=0.2 -> 30 ; doc b: 60 * 1.0 -> 60 ; total=90
    pos_1_2 = vr_result[(vr_result.position_0 == 1) & (vr_result.position_1 == 2)]
    assert not pos_1_2.empty
    assert np.isclose(pos_1_2.c_0.iloc[0], 90, rtol=0.01)

    # doc a pos2: 20 * min(500,100)/100=1.0 -> 20 ; doc b pos2: 160 * 0.25 -> 40 ; total=60
    pos_2_1 = vr_result[(vr_result.position_0 == 2) & (vr_result.position_1 == 1)]
    assert not pos_2_1.empty
    assert np.isclose(pos_2_1.c_0.iloc[0], 60, rtol=0.01)
