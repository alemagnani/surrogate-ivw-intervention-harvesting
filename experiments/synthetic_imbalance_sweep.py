"""Synthetic imbalance sweep (no external data required).

PBM with p_k = (1/k)^eta, eta=1.0, M=10 positions.  Sweeps a traffic split and a
per-doc impression-imbalance ratio; for each configuration runs many independent
trials and reports MSE, variance, and bias^2 for each weighting scheme applied to
the AdjacentChain estimator.

Reproduces the synthetic imbalance-sweep figures/table (uniform vs min vs
harmonic vs clipped) from the paper.

Usage:
    python experiments/synthetic_imbalance_sweep.py                 # full run
    python experiments/synthetic_imbalance_sweep.py --trials 200    # faster
    python experiments/synthetic_imbalance_sweep.py --out-dir results/imbalance_sweep

Outputs (default ./results/imbalance_sweep/):
    exp1_results.csv
    exp1_variance_vs_imbalance.png
    exp1_variance_vs_split.png
    exp1_variance_reduction.png
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from ivw_harvesting import AdjacentChainEstimator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_POSITIONS = 10
ETA = 1.0
NUM_DOCS = 50            # docs shown at both positions in each adjacent pair

TRAFFIC_SPLITS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]  # fraction anchor docs
IMBALANCE_RATIOS = [1, 2, 5, 10, 20, 50, 100]          # N_k / N_kp for anchor docs

SCHEMES = ["original", "min", "harmonic", "clipped_5"]
SCHEME_LABELS = {
    "original": "Original (uniform)",
    "min": "Min",
    "harmonic": "Harmonic",
    "clipped_5": r"Clipped $\tau$=5",
}
SCHEME_COLORS = {
    "original": "black",
    "min": "blue",
    "harmonic": "red",
    "clipped_5": "green",
}


def true_propensities(num_positions: int, eta: float) -> np.ndarray:
    p = np.array([1.0 / k**eta for k in range(1, num_positions + 1)])
    return p / p[0]


def generate_dataset(
    true_bias: np.ndarray,
    num_docs: int,
    base_impressions: int,
    imbalance_ratio: float,
    traffic_split: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate aggregated click data for adjacent pairs (anchor-item model).

    ``traffic_split`` fraction of docs are "anchor" items with many impressions
    at position k and few at k' (varying per-doc ratios).  The rest are "regular"
    items with balanced small impressions.  This produces systematic imbalance
    with varying per-doc ratios -- the regime where harmonic outperforms uniform.
    """
    rows = []
    doc_id = 0
    num_positions = len(true_bias)
    n_anchor = max(1, int(num_docs * traffic_split))
    n_regular = num_docs - n_anchor

    for pos in range(1, num_positions):
        p_k = true_bias[pos - 1]
        p_kp = true_bias[pos]

        for _ in range(n_anchor):
            rel = rng.uniform(0.1, 0.5)
            n_k = int(rng.uniform(base_impressions * imbalance_ratio * 0.5,
                                  base_impressions * imbalance_ratio * 2.0))
            n_kp = int(rng.uniform(1, base_impressions + 1))
            n_k = max(n_k, 1)
            c_k = int(rng.binomial(n_k, rel * p_k))
            c_kp = int(rng.binomial(n_kp, rel * p_kp))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                         "impressions": n_k, "clicks": c_k})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                         "impressions": n_kp, "clicks": c_kp})
            doc_id += 1

        for _ in range(n_regular):
            rel = rng.uniform(0.1, 0.5)
            n_k = int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5))
            n_kp = int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5))
            n_k = max(n_k, 1)
            n_kp = max(n_kp, 1)
            c_k = int(rng.binomial(n_k, rel * p_k))
            c_kp = int(rng.binomial(n_kp, rel * p_kp))
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos,
                         "impressions": n_k, "clicks": c_k})
            rows.append({"query_id": 0, "doc_id": doc_id, "position": pos + 1,
                         "impressions": n_kp, "clicks": c_kp})
            doc_id += 1

    return pd.DataFrame(rows)


def run_trial(true_bias, num_docs, base_impressions, imbalance_ratio, traffic_split, rng):
    df = generate_dataset(true_bias, num_docs, base_impressions,
                          imbalance_ratio, traffic_split, rng)
    estimates = {}
    for scheme in SCHEMES:
        est = AdjacentChainEstimator(weighting=scheme)
        try:
            result = est(df, query_col="query_id", doc_col="doc_id",
                         imps_col="impressions", clicks_col="clicks")
            vals = result.set_index("position")["examination"].values
            if len(vals) != len(true_bias):
                vals = np.full(len(true_bias), np.nan)
        except Exception:
            vals = np.full(len(true_bias), np.nan)
        estimates[scheme] = vals
    return estimates


def run_sweep(traffic_splits, imbalance_ratios, num_trials, base_impressions, seed):
    true_bias = true_propensities(NUM_POSITIONS, ETA)
    records = []
    total = len(traffic_splits) * len(imbalance_ratios)
    pbar = tqdm(total=total, desc="Imbalance sweep")

    for split in traffic_splits:
        for ratio in imbalance_ratios:
            rng = np.random.default_rng(seed)
            trial_results = {s: [] for s in SCHEMES}
            for _ in range(num_trials):
                est = run_trial(true_bias, NUM_DOCS, base_impressions, ratio, split, rng)
                for s in SCHEMES:
                    trial_results[s].append(est[s])
            for s in SCHEMES:
                arr = np.array(trial_results[s])
                mean_est = np.nanmean(arr, axis=0)
                var_est = np.nanvar(arr, axis=0)
                bias2 = (mean_est - true_bias) ** 2
                mse = var_est + bias2
                records.append({
                    "traffic_split": split,
                    "imbalance_ratio": ratio,
                    "scheme": s,
                    "mse": float(np.nanmean(mse)),
                    "variance": float(np.nanmean(var_est)),
                    "bias2": float(np.nanmean(bias2)),
                })
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)


def plot_sweep_results(results: pd.DataFrame, results_dir: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(results_dir, exist_ok=True)

    # Plot 1: variance / MSE vs imbalance ratio at split = 0.80.
    split_val = 0.80
    sub = results[results["traffic_split"] == split_val]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for s in SCHEMES:
        d = sub[sub["scheme"] == s].sort_values("imbalance_ratio")
        axes[0].plot(d["imbalance_ratio"], d["variance"], marker="o",
                     color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])
        axes[1].plot(d["imbalance_ratio"], d["mse"], marker="o",
                     color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])
    for ax, metric in zip(axes, ["Variance", "MSE"]):
        ax.set_xlabel("Imbalance ratio (N_k / N_{k'})")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs. Imbalance (traffic split={split_val})")
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp1_variance_vs_imbalance.png"), dpi=150)
    plt.close()

    # Plot 2: variance / MSE vs traffic split at imbalance ratio = 10.
    ratio_val = 10
    sub2 = results[results["imbalance_ratio"] == ratio_val]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for s in SCHEMES:
        d = sub2[sub2["scheme"] == s].sort_values("traffic_split")
        axes[0].plot(d["traffic_split"] * 100, d["variance"], marker="o",
                     color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])
        axes[1].plot(d["traffic_split"] * 100, d["mse"], marker="o",
                     color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])
    for ax, metric in zip(axes, ["Variance", "MSE"]):
        ax.set_xlabel("Traffic split (% anchor docs)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs. Traffic Split (ratio={ratio_val})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp1_variance_vs_split.png"), dpi=150)
    plt.close()

    # Plot 3: variance reduction relative to original at split = 0.80.
    fig, ax = plt.subplots(figsize=(10, 6))
    orig = results[results["scheme"] == "original"][
        ["traffic_split", "imbalance_ratio", "variance"]
    ].rename(columns={"variance": "var_orig"})
    for s in SCHEMES:
        if s == "original":
            continue
        d = results[results["scheme"] == s].merge(orig, on=["traffic_split", "imbalance_ratio"])
        d["var_reduction"] = (1 - d["variance"] / d["var_orig"]) * 100
        d_fixed = d[d["traffic_split"] == split_val].sort_values("imbalance_ratio")
        ax.plot(d_fixed["imbalance_ratio"], d_fixed["var_reduction"], marker="o",
                color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Imbalance ratio (N_k / N_{k'})")
    ax.set_ylabel("Variance reduction (%)")
    ax.set_title(f"Variance reduction vs. original (traffic split={split_val})")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp1_variance_reduction.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {results_dir}/")


def print_latex_table(results: pd.DataFrame) -> None:
    sub = results[results["traffic_split"] == 0.80]
    print("\n--- LaTeX Table: Variance (traffic split=80/20) ---")
    print(r"\begin{tabular}{l" + "r" * len(IMBALANCE_RATIOS) + "}")
    print(r"\hline")
    print(f"{'Scheme':<20} & " + " & ".join(f"ratio={r}" for r in IMBALANCE_RATIOS) + r" \\")
    print(r"\hline")
    for s in SCHEMES:
        row = sub[sub["scheme"] == s].sort_values("imbalance_ratio")
        vals = " & ".join(f"{v:.4f}" for v in row["variance"].values)
        print(f"{SCHEME_LABELS[s]:<20} & {vals} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=5000, help="trials per configuration")
    ap.add_argument("--base-impressions", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/imbalance_sweep")
    ap.add_argument("--no-plots", action="store_true", help="skip matplotlib plots")
    args = ap.parse_args()

    print("Running synthetic imbalance sweep")
    print(f"  Positions: {NUM_POSITIONS}, eta={ETA}, trials={args.trials}")
    print(f"  Traffic splits: {TRAFFIC_SPLITS}")
    print(f"  Imbalance ratios: {IMBALANCE_RATIOS}")

    results = run_sweep(TRAFFIC_SPLITS, IMBALANCE_RATIOS, args.trials,
                        args.base_impressions, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "exp1_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    if not args.no_plots:
        plot_sweep_results(results, args.out_dir)
    print_latex_table(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
