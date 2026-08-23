"""Intervention-harvesting position-bias estimators.

Three estimators share a common interface; each accepts a ``weighting`` scheme
(string name or callable) that selects the count-only weighting applied when
building the intervention sets:

* :class:`AdjacentChainEstimator` - chains CTR ratios between adjacent ranks.
* :class:`AllPairsEstimator`      - fits a PBM over all rank pairs (needs torch).
* :class:`PivotEstimator`         - CTR ratios against a fixed pivot rank.

``torch`` is imported lazily inside :class:`AllPairsEstimator` only, so the rest
of the package (and the test suite) works without a PyTorch installation.
"""

import logging
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ivw_harvesting.assertions import assert_columns_in_df
from ivw_harvesting.intervention_sets import build_intervention_sets, normalize_bias

logger = logging.getLogger(__name__)


class AdjacentChainEstimator:
    def __init__(
        self,
        weighting: Union[
            str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        ] = "original",
    ):
        """Initialize the Adjacent Chain estimator.

        Args:
            weighting: Weighting scheme.  Either a string name
                ("original", "variance_reduced", "min", "harmonic",
                "clipped_2", "clipped_5", "clipped_10") or a callable
                ``weight_fn(N_k, N_kp) -> (omega_k, omega_kp)``.
        """
        self.weighting = weighting

    def __call__(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        doc_col: str = "doc_id",
        imps_col: Optional[str] = None,
        clicks_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Estimate position bias using the adjacent chain method.

        Args:
            df: DataFrame with click data.
            query_col: Name of the column containing query identifiers.
            doc_col: Name of the column containing document identifiers.
            imps_col: Optional column with impression counts (for aggregated data).
            clicks_col: Optional column with click counts (for aggregated data).

        Returns:
            DataFrame with position and examination probability.
        """
        logger.info("Position bias between adjacent/neighboring ranks")
        logger.info(f"Using weighting scheme: {self.weighting}")

        if imps_col is not None and clicks_col is not None:
            assert_columns_in_df(df, ["position", query_col, doc_col, imps_col, clicks_col])
            df = build_intervention_sets(
                df, query_col, doc_col, imps_col, clicks_col,
                weighting=self.weighting,
            )
        else:
            assert_columns_in_df(df, ["position", query_col, doc_col, "click"])
            df = build_intervention_sets(
                df, query_col, doc_col,
                weighting=self.weighting,
            )

        # Filter interventions between adjacent pairs, prepend exam=1.0 for position 1:
        pos_1_df = df[(df.position_0 == 1) & (df.position_1 == 1)]
        adjacent_pair_df = df[df.position_1 == df.position_0 + 1]
        adjacent_pair_df = adjacent_pair_df.sort_values(["position_0", "position_1"])
        df = pd.concat([pos_1_df, adjacent_pair_df])

        # Compute click ratio between neighboring ranks:
        df["examination"] = (df["c_1"] / df["c_0"]).fillna(0)
        df["examination"] = df.examination.cumprod()

        df = df.rename(columns={"position_1": "position"})
        return df[["position", "examination"]].reset_index(drop=True)


class PivotEstimator:
    def __init__(
        self,
        pivot_rank: int = 1,
        weighting: Union[
            str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        ] = "original",
    ):
        """Initialize the Pivot estimator.

        Args:
            pivot_rank: Position to use as pivot (default=1).
            weighting: Weighting scheme.  Either a string name
                ("original", "variance_reduced", "min", "harmonic",
                "clipped_2", "clipped_5", "clipped_10") or a callable
                ``weight_fn(N_k, N_kp) -> (omega_k, omega_kp)``.
        """
        self.pivot_rank = pivot_rank
        self.weighting = weighting

    def __call__(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        doc_col: str = "doc_id",
        imps_col: Optional[str] = None,
        clicks_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Estimate position bias using the pivot method.

        Args:
            df: DataFrame with click data.
            query_col: Name of the column containing query identifiers.
            doc_col: Name of the column containing document identifiers.
            imps_col: Optional column with impression counts (for aggregated data).
            clicks_col: Optional column with click counts (for aggregated data).

        Returns:
            DataFrame with position and examination probability.
        """
        logger.info(f"Position bias between rank k and pivot rank: {self.pivot_rank}")
        logger.info(f"Using weighting scheme: {self.weighting}")

        if imps_col is not None and clicks_col is not None:
            assert_columns_in_df(df, ["position", query_col, doc_col, imps_col, clicks_col])
            df = build_intervention_sets(
                df, query_col, doc_col, imps_col, clicks_col,
                weighting=self.weighting,
            )
        else:
            assert_columns_in_df(df, ["position", query_col, doc_col, "click"])
            df = build_intervention_sets(
                df, query_col, doc_col,
                weighting=self.weighting,
            )

        # Filter interventions with pivot rank in first positions:
        df = df[df.position_0 == self.pivot_rank]

        # Compute CTR ratio between position k and the pivot rank:
        df["examination"] = (df["c_1"] / df["c_0"]).fillna(0)
        df.examination = normalize_bias(df.examination)

        df = df.rename(columns={"position_1": "position"})
        return df[["position", "examination"]]


class AllPairsEstimator:
    def __init__(
        self,
        epochs: int = 5_000,
        lr: float = 0.01,
        batch_size: int = 512,
        weighting: Union[
            str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        ] = "original",
    ):
        """Initialize the All Pairs estimator.

        Args:
            epochs: Number of training epochs (default=5000).
            lr: Learning rate (default=0.01).
            batch_size: Batch size for training (default=512).
            weighting: Weighting scheme.  Either a string name
                ("original", "variance_reduced", "min", "harmonic",
                "clipped_2", "clipped_5", "clipped_10") or a callable
                ``weight_fn(N_k, N_kp) -> (omega_k, omega_kp)``.
        """
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weighting = weighting

    def __call__(
        self,
        df: pd.DataFrame,
        query_col: str = "query_id",
        doc_col: str = "doc_id",
        imps_col: Optional[str] = None,
        clicks_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Estimate position bias using the all pairs method (neural PBM fit).

        Args:
            df: DataFrame with click data.
            query_col: Name of the column containing query identifiers.
            doc_col: Name of the column containing document identifiers.
            imps_col: Optional column with impression counts (for aggregated data).
            clicks_col: Optional column with click counts (for aggregated data).

        Returns:
            DataFrame with position and examination probability.
        """
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset

        logger.info("Position bias estimation using global all pairs estimator")
        logger.info(f"Using weighting scheme: {self.weighting}")

        n_positions = df.position.nunique()
        max_position = df.position.max()

        if imps_col is not None and clicks_col is not None:
            assert_columns_in_df(df, ["position", query_col, doc_col, imps_col, clicks_col])
            df = build_intervention_sets(
                df, query_col, doc_col, imps_col, clicks_col,
                weighting=self.weighting,
            )
        else:
            assert_columns_in_df(df, ["position", query_col, doc_col, "click"])
            df = build_intervention_sets(
                df, query_col, doc_col,
                weighting=self.weighting,
            )

        df = df[df.position_0 != df.position_1].copy()

        class _AllPairsDataset(Dataset):
            def __init__(self, data):
                self.k = torch.tensor(data["position_0"].values)
                self.k_prime = torch.tensor(data["position_1"].values)
                self.c_0 = torch.tensor(data["c_0"].values, dtype=torch.float)
                self.not_c_0 = torch.tensor(data["not_c_0"].values, dtype=torch.float)

            def __len__(self):
                return len(self.k)

            def __getitem__(self, idx):
                return self.k[idx], self.k_prime[idx], self.c_0[idx], self.not_c_0[idx]

        class _AllPairsPBM(nn.Module):
            def __init__(self, n_pos):
                super().__init__()
                k = n_pos + 1
                self.examination = nn.Sequential(nn.Embedding(k, 1), nn.Sigmoid())
                self.relevance = nn.Sequential(nn.Embedding(k * k, 1), nn.Sigmoid())

            def forward(self, k, k_prime):
                examination = self.examination(k)
                relevance = self.relevance(k * k_prime)
                return (examination * relevance).squeeze(dim=-1)

            def get_position_bias(self, k):
                positions = torch.arange(k) + 1
                examination = self.examination(positions).squeeze()
                return positions, examination.detach()

        dataset = _AllPairsDataset(df)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = _AllPairsPBM(n_positions)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for _ in tqdm(range(self.epochs), desc="Maximizing weighted cross entropy..."):
            for batch in loader:
                k, k_prime, c_0, not_c_0 = batch
                optimizer.zero_grad()
                y_predict = model(k, k_prime)
                loss = -(
                    c_0 * torch.log(y_predict) + not_c_0 * torch.log(1 - y_predict)
                ).sum()
                loss.backward()
                optimizer.step()

        position, examination = model.get_position_bias(max_position)
        df = pd.DataFrame({"position": position, "examination": examination})
        df.examination = normalize_bias(df.examination)
        return df
