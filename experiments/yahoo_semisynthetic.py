"""Yahoo LTR semi-synthetic experiment.

Uses real query-document relevance labels from the Yahoo Learning to Rank
Challenge dataset (Set 1) as ground-truth relevance, then simulates clicks under
the Position-Based Model with the anchor-item impression-imbalance model.

"Semi-synthetic" because:
  * Relevance rel(q,d) comes from real human-judged grades (0-4),
  * Clicks are simulated: P(C=1|q,d,k) = p_k * rel(q,d),
  * Impression imbalance follows the same anchor-item model as the synthetic sweep.

The Yahoo dataset is NOT redistributed here. Obtain it from
https://webscope.sandbox.yahoo.com (Learning to Rank Challenge, Set 1) and point
the script at the directory containing set1.{train,valid,test}.txt.

Usage:
    python experiments/yahoo_semisynthetic.py --data-dir /path/to/Yahoo_ltr
    YAHOO_LTR_DIR=/path/to/Yahoo_ltr python experiments/yahoo_semisynthetic.py

Outputs (default ./results/yahoo/):
    exp3_results.csv
    exp3_variance_vs_imbalance.png
    exp3_variance_reduction.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from ivw_harvesting import AdjacentChainEstimator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_POSITIONS = 10
ETA = 1.0
NUM_TRIALS = 200
QUERIES_PER_TRIAL = 150
BASE_IMPRESSIONS = 20

TRAFFIC_SPLITS = [0.80]
IMBALANCE_RATIOS = [1, 2, 5, 10, 20, 50, 100]

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


def resolve_yahoo_file(data_dir: str, filename: str) -> str:
    """Return the path to a Yahoo split file, raising a clear error if missing."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {filename} in data directory {data_dir!r}.\n"
            "Provide the Yahoo LTR Set 1 directory via --data-dir or the "
            "YAHOO_LTR_DIR environment variable. The Yahoo dataset is not "
            "redistributed; obtain it from https://webscope.sandbox.yahoo.com."
        )
    return path


def load_yahoo(path: str) -> pd.DataFrame:
    """Load a Yahoo SVMLight file; return DataFrame with query_id, doc_id, rel."""
    from sklearn.datasets import load_svmlight_file

    print(f"Loading {path} ...")
    _, y, qids = load_svmlight_file(path, query_id=True)
    y = y.astype(int)
    df = pd.DataFrame({"query_id": qids, "rel_grade": y})
    df["doc_id"] = df.groupby("query_id").cumcount()
    # Convert grade to continuous relevance: (2^grade - 1) / 15.
    df["rel"] = (2.0 ** df["rel_grade"] - 1.0) / 15.0
    return df[["query_id", "doc_id", "rel"]]


def true_propensities(num_positions: int, eta: float) -> np.ndarray:
    p = np.array([1.0 / k**eta for k in range(1, num_positions + 1)])
    return p / p[0]


def generate_dataset(rel_lookup, query_ids, true_bias, base_impressions,
                     imbalance_ratio, traffic_split, rng):
    """Generate click data using Yahoo relevance grades + anchor-item impressions."""
    rows = []
    global_doc_id = 0
    num_positions = len(true_bias)

    for pos in range(1, num_positions):
        p_k = true_bias[pos - 1]
        p_kp = true_bias[pos]
        for qid in query_ids:
            rels = rel_lookup[qid]
            n_docs = len(rels)
            if n_docs == 0:
                continue
            n_anchor = max(1, int(n_docs * traffic_split))
            n_regular = n_docs - n_anchor
            perm = rng.permutation(n_docs)

            for idx in range(n_anchor):
                rel = float(rels[perm[idx]])
                n_k = max(1, int(rng.uniform(base_impressions * imbalance_ratio * 0.5,
                                             base_impressions * imbalance_ratio * 2.0)))
                n_kp = max(1, int(rng.uniform(1, base_impressions + 1)))
                c_k = int(rng.binomial(n_k, rel * p_k))
                c_kp = int(rng.binomial(n_kp, rel * p_kp))
                rows.append({"query_id": qid, "doc_id": global_doc_id,
                             "position": pos, "impressions": n_k, "clicks": c_k})
                rows.append({"query_id": qid, "doc_id": global_doc_id,
                             "position": pos + 1, "impressions": n_kp, "clicks": c_kp})
                global_doc_id += 1

            for idx in range(n_anchor, n_anchor + n_regular):
                rel = float(rels[perm[idx]])
                n_k = max(1, int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5)))
                n_kp = max(1, int(rng.uniform(base_impressions * 0.5, base_impressions * 1.5)))
                c_k = int(rng.binomial(n_k, rel * p_k))
                c_kp = int(rng.binomial(n_kp, rel * p_kp))
                rows.append({"query_id": qid, "doc_id": global_doc_id,
                             "position": pos, "impressions": n_k, "clicks": c_k})
                rows.append({"query_id": qid, "doc_id": global_doc_id,
                             "position": pos + 1, "impressions": n_kp, "clicks": c_kp})
                global_doc_id += 1

    return pd.DataFrame(rows)


def run_sweep(rel_lookup, all_query_ids, num_trials, queries_per_trial, seed=0):
    true_bias = true_propensities(NUM_POSITIONS, ETA)
    records = []
    rng_main = np.random.default_rng(seed)
    total = len(TRAFFIC_SPLITS) * len(IMBALANCE_RATIOS)
    pbar = tqdm(total=total, desc="Yahoo sweep")

    for split in TRAFFIC_SPLITS:
        for ratio in IMBALANCE_RATIOS:
            rng = np.random.default_rng(rng_main.integers(0, 2**31))
            trial_results = {s: [] for s in SCHEMES}
            for _ in range(num_trials):
                q_idx = rng.choice(len(all_query_ids), size=queries_per_trial, replace=False)
                trial_qids = [all_query_ids[i] for i in q_idx]
                df = generate_dataset(rel_lookup, trial_qids, true_bias,
                                      BASE_IMPRESSIONS, ratio, split, rng)
                for scheme in SCHEMES:
                    est = AdjacentChainEstimator(weighting=scheme)
                    try:
                        r = est(df, query_col="query_id", doc_col="doc_id",
                                imps_col="impressions", clicks_col="clicks")
                        vals = r.set_index("position")["examination"].values
                        if len(vals) != len(true_bias):
                            vals = np.full(len(true_bias), np.nan)
                    except Exception:
                        vals = np.full(len(true_bias), np.nan)
                    trial_results[scheme].append(vals)
            for s in SCHEMES:
                arr = np.array(trial_results[s])
                mean_est = np.nanmean(arr, axis=0)
                var_est = np.nanvar(arr, axis=0)
                bias2 = (mean_est - true_bias) ** 2
                records.append({
                    "traffic_split": split,
                    "imbalance_ratio": ratio,
                    "scheme": s,
                    "mse": float(np.nanmean(var_est + bias2)),
                    "variance": float(np.nanmean(var_est)),
                    "bias2": float(np.nanmean(bias2)),
                })
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)


def plot_results(results: pd.DataFrame, results_dir: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(results_dir, exist_ok=True)
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
        ax.set_xlabel("Imbalance ratio")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs. Imbalance (Yahoo, split={split_val})")
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Yahoo semi-synthetic: weighting scheme comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp3_variance_vs_imbalance.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    orig = results[results["scheme"] == "original"][
        ["traffic_split", "imbalance_ratio", "variance"]
    ].rename(columns={"variance": "var_orig"})
    for s in ["min", "harmonic", "clipped_5"]:
        d = results[results["scheme"] == s].merge(orig, on=["traffic_split", "imbalance_ratio"])
        d["reduction"] = (1 - d["variance"] / d["var_orig"]) * 100
        d_fixed = d[d["traffic_split"] == split_val].sort_values("imbalance_ratio")
        ax.plot(d_fixed["imbalance_ratio"], d_fixed["reduction"], marker="o",
                color=SCHEME_COLORS[s], label=SCHEME_LABELS[s])
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Imbalance ratio")
    ax.set_ylabel("Variance reduction vs. original (%)")
    ax.set_title(f"Yahoo semi-synthetic: variance reduction (split={split_val})")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp3_variance_reduction.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {results_dir}/")


def print_summary(results: pd.DataFrame) -> None:
    sub = results[results["traffic_split"] == 0.80]
    print("\n--- Variance at split=80/20 (Yahoo semi-synthetic) ---")
    print(f"{'Scheme':<22} " + " ".join(f"r={r:>3}" for r in IMBALANCE_RATIOS))
    for s in SCHEMES:
        row = sub[sub["scheme"] == s].sort_values("imbalance_ratio")
        vals = " ".join(f"{v:.5f}" for v in row["variance"].values)
        print(f"{SCHEME_LABELS[s]:<22} {vals}")

    print("\n--- Harmonic variance reduction vs original ---")
    for ratio in IMBALANCE_RATIOS:
        o = sub[(sub["scheme"] == "original") & (sub["imbalance_ratio"] == ratio)]["variance"].values[0]
        h = sub[(sub["scheme"] == "harmonic") & (sub["imbalance_ratio"] == ratio)]["variance"].values[0]
        print(f"  ratio={ratio:>4}: original={o:.5f}  harmonic={h:.5f}  "
              f"reduction={(1 - h / o) * 100:+.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("YAHOO_LTR_DIR"),
                    help="directory containing set1.{train,valid,test}.txt "
                         "(or set YAHOO_LTR_DIR)")
    ap.add_argument("--split-file", default="set1.test.txt",
                    help="which Yahoo file to use for relevance grades")
    ap.add_argument("--trials", type=int, default=NUM_TRIALS)
    ap.add_argument("--queries-per-trial", type=int, default=QUERIES_PER_TRIAL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/yahoo")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if not args.data_dir:
        print("ERROR: no data directory. Pass --data-dir /path/to/Yahoo_ltr or "
              "set YAHOO_LTR_DIR.", file=sys.stderr)
        sys.exit(1)

    path = resolve_yahoo_file(args.data_dir, args.split_file)
    yahoo_df = load_yahoo(path)
    print(f"Loaded {len(yahoo_df):,} (query, doc) pairs from "
          f"{yahoo_df['query_id'].nunique():,} queries.")

    rel_lookup = {qid: grp["rel"].values for qid, grp in yahoo_df.groupby("query_id")}
    rel_lookup = {qid: rels for qid, rels in rel_lookup.items() if len(rels) >= 2}
    all_query_ids = sorted(rel_lookup.keys())
    print(f"Queries with >=2 docs: {len(all_query_ids):,}")

    results = run_sweep(rel_lookup, all_query_ids, args.trials, args.queries_per_trial, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "exp3_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    if not args.no_plots:
        plot_results(results, args.out_dir)
    print_summary(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
