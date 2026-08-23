from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ivw_harvesting.weighting import get_weight_fn


def build_intervention_sets(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
    imps_col: Optional[str] = None,
    clicks_col: Optional[str] = None,
    weighting: Union[str, Callable] = "original",
) -> pd.DataFrame:
    """Build intervention sets for position bias estimation.

    Args:
        df: DataFrame with click data.
        query_col: Column containing query identifiers.
        doc_col: Column containing document identifiers.
        imps_col: Column with impression counts (aggregated format).
        clicks_col: Column with click counts (aggregated format).
        weighting: Weighting scheme. Either a string name (see
            ``get_weight_fn`` for supported values) or a callable
            ``weight_fn(N_k, N_kp) -> (omega_k, omega_kp)``.

    Returns:
        DataFrame with columns position_0, position_1, c_0, c_1,
        not_c_0, not_c_1.
    """
    df = df.copy()

    if imps_col is not None and clicks_col is not None:
        # Aggregated format
        if weighting == "original":
            return build_intervention_sets_aggregated(df, query_col, doc_col, imps_col, clicks_col)
        weight_fn = get_weight_fn(weighting) if isinstance(weighting, str) else weighting
        return build_intervention_sets_weighted(df, query_col, doc_col, imps_col, clicks_col, weight_fn)
    else:
        # Binary format: convert to aggregated first
        if weighting == "original":
            return build_intervention_sets_binary(df, query_col, doc_col)
        df_agg = _binary_to_aggregated(df, query_col, doc_col)
        weight_fn = get_weight_fn(weighting) if isinstance(weighting, str) else weighting
        return build_intervention_sets_weighted(
            df_agg, query_col, doc_col,
            imps_col="impressions", clicks_col="clicks",
            weight_fn=weight_fn,
        )


def build_intervention_sets_binary(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
) -> pd.DataFrame:
    """Original implementation for binary click data (0/1)."""
    df["no_click"] = 1 - df["click"]
    df = (
        df.groupby([query_col, doc_col, "position"])
        .agg(
            clicks=("click", "sum"),
            no_clicks=("no_click", "sum"),
            impressions=(doc_col, "count"),
        )
        .reset_index()
    )

    df["c"] = df["clicks"] / df["impressions"]
    df["not_c"] = df["no_clicks"] / df["impressions"]
    df = df.merge(df, on=[query_col, doc_col], suffixes=["_0", "_1"])

    df = (
        df.groupby(["position_0", "position_1"])
        .agg(
            c_0=("c_0", "sum"),
            c_1=("c_1", "sum"),
            not_c_0=("not_c_0", "sum"),
            not_c_1=("not_c_1", "sum"),
        )
        .reset_index()
    )

    return df


def build_intervention_sets_aggregated(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
    imps_col: str,
    clicks_col: str,
) -> pd.DataFrame:
    """Implementation for pre-aggregated data with impression and click counts."""
    if (df[clicks_col] > df[imps_col]).any():
        raise ValueError(
            f"Found rows where {clicks_col} > {imps_col}. Clicks must be <= impressions."
        )

    df = _get_aggregated_data(df, query_col, doc_col, imps_col, clicks_col)

    df["c"] = df["c"] / df["impressions"]
    df["not_c"] = df["not_c"] / df["impressions"]

    df = df.merge(df, on=[query_col, doc_col], suffixes=["_0", "_1"])

    df = (
        df.groupby(["position_0", "position_1"])
        .agg(
            c_0=("c_0", "sum"),
            c_1=("c_1", "sum"),
            not_c_0=("not_c_0", "sum"),
            not_c_1=("not_c_1", "sum"),
        )
        .reset_index()
    )

    return df


def build_intervention_sets_weighted(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
    imps_col: str,
    clicks_col: str,
    weight_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Generic weighted intervention sets using a caller-supplied weight function.

    Args:
        weight_fn: ``weight_fn(N_k, N_kp) -> (omega_k, omega_kp)`` where
            omega_k and omega_kp are per-pair weights.  They may differ
            (e.g. clipped weights) which breaks ratio-unbiasedness.

    The contribution of pair i to position k's sum is::

        omega_k^i * C_k^i / N_k^i = omega_k^i * CTR_k^i
    """
    if (df[clicks_col] > df[imps_col]).any():
        raise ValueError(
            f"Found rows where {clicks_col} > {imps_col}. Clicks must be <= impressions."
        )

    df = _get_aggregated_data(df, query_col, doc_col, imps_col, clicks_col)

    merged = df.merge(df, on=[query_col, doc_col], suffixes=["_0", "_1"])

    N_k = merged["impressions_0"].values.astype(float)
    N_kp = merged["impressions_1"].values.astype(float)
    omega_k, omega_kp = weight_fn(N_k, N_kp)

    # factor_0 = omega_k / N_k  ->  clicks_0 * factor_0 = omega_k * CTR_0
    merged["weighted_c_0"] = merged["c_0"] * omega_k / N_k
    merged["weighted_c_1"] = merged["c_1"] * omega_kp / N_kp
    merged["weighted_not_c_0"] = merged["not_c_0"] * omega_k / N_k
    merged["weighted_not_c_1"] = merged["not_c_1"] * omega_kp / N_kp

    result = (
        merged.groupby(["position_0", "position_1"])
        .agg(
            c_0=("weighted_c_0", "sum"),
            c_1=("weighted_c_1", "sum"),
            not_c_0=("weighted_not_c_0", "sum"),
            not_c_1=("weighted_not_c_1", "sum"),
        )
        .reset_index()
    )

    return result


def build_intervention_sets_variance_reduced(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
    imps_col: str,
    clicks_col: str,
) -> pd.DataFrame:
    """Min-weighting (variance-reduced).  Kept for backward compatibility."""
    from ivw_harvesting.weighting import weight_min

    return build_intervention_sets_weighted(
        df, query_col, doc_col, imps_col, clicks_col, weight_fn=weight_min
    )


def _binary_to_aggregated(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
) -> pd.DataFrame:
    """Convert binary click data to aggregated format."""
    df = df.copy()
    df["no_click"] = 1 - df["click"]
    df_agg = (
        df.groupby([query_col, doc_col, "position"])
        .agg(
            clicks=("click", "sum"),
            no_clicks=("no_click", "sum"),
            impressions=(doc_col, "count"),
        )
        .reset_index()
    )
    return df_agg


def _get_aggregated_data(
    df: pd.DataFrame,
    query_col: str,
    doc_col: str,
    imps_col: str,
    clicks_col: str,
) -> pd.DataFrame:
    """Aggregate data by (query, document, position) and return raw counts."""
    df = df.copy()

    df["no_clicks"] = df[imps_col] - df[clicks_col]

    df = (
        df.groupby([query_col, doc_col, "position"])
        .agg(
            clicks=(clicks_col, "sum"),
            no_clicks=("no_clicks", "sum"),
            impressions=(imps_col, "sum"),
        )
        .reset_index()
    )

    df["c"] = df["clicks"]
    df["not_c"] = df["no_clicks"]

    return df


def normalize_bias(examination: pd.Series) -> pd.Series:
    examination /= examination.values[0]
    return examination.fillna(0)
