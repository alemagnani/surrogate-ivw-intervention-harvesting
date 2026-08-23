"""Downstream IPS-LTR evaluation: does better propensity *estimation* translate
into better *ranking* (nDCG)?

Both uniform and harmonic are *consistent* propensity estimators, so any downstream
ranking gain can only appear when propensities are estimated from LIMITED data (few
impressions), where harmonic's lower variance yields a more accurate p_hat_k. This
script isolates that regime on the Yahoo LTR Set 1 relevance grades.

Design (semi-synthetic)
-----------------------
Two decoupled logs, as in a real pipeline (estimate propensities once, reuse):

 1. HARVESTING log -> estimate per-position propensities p_hat_k under each
    weighting scheme, via AdjacentChainEstimator. We sweep BASE_IMPRESSIONS to
    move from the noisy (low-data) regime to the well-powered regime.

 2. EVALUATION set -> a logging policy ranks each query's docs by a noisy version
    of true relevance and assigns positions 1..M. Clicks are simulated under PBM.
    We recover IPS-debiased relevance rel_hat = clickrate / p_hat_pos, rank docs
    by rel_hat, and score nDCG@K against the true grades. Ranking by IPS-debiased
    relevance is the Bayes-optimal IPS-LTR ranker, a faithful upper bound on what
    a learned IPS-LTR model targets.

Anchors compared per data budget:
    naive    : p_hat_k = 1 (no position correction)      -> lower bound
    uniform  : Agarwal et al. weighting (the baseline)
    harmonic : our weight
    oracle   : true p_k                                  -> upper bound

The Yahoo dataset is NOT redistributed. Obtain it from
https://webscope.sandbox.yahoo.com and point the script at its directory.

Usage:
    python experiments/downstream_ltr.py --data-dir /path/to/Yahoo_ltr --pilot
    YAHOO_LTR_DIR=/path/to/Yahoo_ltr python experiments/downstream_ltr.py

Outputs (default ./results/downstream_ltr/):
    downstream_ndcg.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from ivw_harvesting import AdjacentChainEstimator

# Reuse the Yahoo loader + harvesting-log generator from the sibling experiment.
from yahoo_semisynthetic import (
    generate_dataset,
    load_yahoo,
    resolve_yahoo_file,
    true_propensities,
)

NUM_POSITIONS = 10
ETA = 1.0
TRAFFIC_SPLIT = 0.80
IMBALANCE_RATIO = 20  # heavy-tailed regime where weighting matters

# Data-budget sweep: base impressions in the HARVESTING log. Low => noisy p_hat.
IMPRESSION_BUDGETS = [3, 5, 10, 20, 50]

# Evaluation set.
EVAL_IMPRESSIONS = 10      # sessions per (q,d) in the eval log
LOGGING_NOISE = 0.5        # std of gaussian noise added to rel for logging policy
NDCG_K = 10
MIN_DOCS = 10              # queries need at least this many docs to rank


def ndcg_at_k(ranked_rel_grades: np.ndarray, k: int) -> float:
    """nDCG@k given the true relevance grades in the *predicted* rank order."""
    gains = (2.0 ** ranked_rel_grades - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains[:k] * discounts[:k]))
    ideal = np.sort(ranked_rel_grades)[::-1]
    ideal_gains = (2.0 ** ideal - 1.0)
    idcg = float(np.sum(ideal_gains[:k] * discounts[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def build_eval_log(rel_lookup, grade_lookup, query_ids, true_bias, rng):
    """One evaluation ranking per query under a noisy-relevance logging policy."""
    rows = []
    for qid in query_ids:
        rels = rel_lookup[qid]
        grades = grade_lookup[qid]
        n_docs = len(rels)
        if n_docs < MIN_DOCS:
            continue
        noisy = rels + rng.normal(0, LOGGING_NOISE, size=n_docs)
        order = np.argsort(noisy)[::-1][:NUM_POSITIONS]
        for pos_idx, d in enumerate(order):
            p_k = true_bias[pos_idx]
            rel = float(rels[d])
            n = EVAL_IMPRESSIONS
            clicks = int(rng.binomial(n, p_k * rel))
            rows.append({
                "query_id": qid,
                "position": pos_idx + 1,
                "clickrate": clicks / n,
                "grade": float(grades[d]),
            })
    return pd.DataFrame(rows)


def score_ndcg(eval_df, p_hat, k):
    """Rank each query by IPS-debiased relevance clickrate / p_hat_pos; mean nDCG@k."""
    pos = eval_df["position"].values - 1
    rel_hat = eval_df["clickrate"].values / np.maximum(p_hat[pos], 1e-8)
    tmp = eval_df.assign(rel_hat=rel_hat)
    scores = []
    for _, g in tmp.groupby("query_id"):
        gg = g.sort_values("rel_hat", ascending=False)
        scores.append(ndcg_at_k(gg["grade"].values, k))
    return float(np.mean(scores)) if scores else np.nan


def estimate_propensity(harvest_df, scheme, num_positions):
    est = AdjacentChainEstimator(weighting=scheme)
    try:
        r = est(harvest_df, query_col="query_id", doc_col="doc_id",
                imps_col="impressions", clicks_col="clicks")
        vals = r.set_index("position")["examination"].values
        if len(vals) != num_positions or np.any(~np.isfinite(vals)):
            return None
        return vals
    except Exception:
        return None


def run(data_path, cache_path, out_dir, pilot):
    num_trials = 10 if pilot else 200
    queries_per_trial = 60 if pilot else 150
    budgets = [5, 20] if pilot else IMPRESSION_BUDGETS

    print(f"Loading Yahoo relevance from {data_path} ...")
    if cache_path and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        df = load_yahoo(data_path)
        # load_yahoo returns rel; recover integer grade for nDCG gains.
        df["grade"] = np.round(np.log2(df["rel"] * 15.0 + 1.0)).astype(int)
        if cache_path:
            try:
                df.to_parquet(cache_path)
            except Exception as e:
                print(f"  (skipping cache write: {e})")
    print(f"  {df.query_id.nunique()} queries, {len(df)} docs")

    rel_lookup = {q: g["rel"].values for q, g in df.groupby("query_id")}
    grade_lookup = {q: g["grade"].values for q, g in df.groupby("query_id")}
    all_qids = list(rel_lookup.keys())

    true_bias = true_propensities(NUM_POSITIONS, ETA)
    naive_p = np.ones(NUM_POSITIONS)

    rng_main = np.random.default_rng(0)
    records = []

    for budget in budgets:
        per_scheme = {s: [] for s in ["naive", "uniform", "harmonic", "oracle"]}
        for _ in range(num_trials):
            rng = np.random.default_rng(rng_main.integers(0, 2**31))
            q_idx = rng.choice(len(all_qids), size=queries_per_trial, replace=False)
            trial_qids = [all_qids[i] for i in q_idx]

            harvest = generate_dataset(
                rel_lookup, trial_qids, true_bias,
                base_impressions=budget, imbalance_ratio=IMBALANCE_RATIO,
                traffic_split=TRAFFIC_SPLIT, rng=rng,
            )
            p_uniform = estimate_propensity(harvest, "original", NUM_POSITIONS)
            p_harmonic = estimate_propensity(harvest, "harmonic", NUM_POSITIONS)
            if p_uniform is None or p_harmonic is None:
                continue

            eval_df = build_eval_log(rel_lookup, grade_lookup, trial_qids, true_bias, rng)
            if eval_df.empty:
                continue
            per_scheme["naive"].append(score_ndcg(eval_df, naive_p, NDCG_K))
            per_scheme["uniform"].append(score_ndcg(eval_df, p_uniform, NDCG_K))
            per_scheme["harmonic"].append(score_ndcg(eval_df, p_harmonic, NDCG_K))
            per_scheme["oracle"].append(score_ndcg(eval_df, true_bias, NDCG_K))

        row = {"budget": budget, "n_trials": len(per_scheme["uniform"])}
        for s in ["naive", "uniform", "harmonic", "oracle"]:
            arr = np.array(per_scheme[s])
            row[f"ndcg_{s}"] = float(np.mean(arr)) if len(arr) else np.nan
            row[f"se_{s}"] = float(np.std(arr) / np.sqrt(len(arr))) if len(arr) else np.nan
        u = np.array(per_scheme["uniform"])
        h = np.array(per_scheme["harmonic"])
        d = h - u
        row["delta_h_minus_u"] = float(np.mean(d)) if len(d) else np.nan
        row["delta_se"] = float(np.std(d) / np.sqrt(len(d))) if len(d) else np.nan
        records.append(row)
        print(
            f"budget={budget:3d} | naive={row['ndcg_naive']:.4f} "
            f"uniform={row['ndcg_uniform']:.4f} harmonic={row['ndcg_harmonic']:.4f} "
            f"oracle={row['ndcg_oracle']:.4f} | "
            f"delta(h-u)={row['delta_h_minus_u']:+.4f} +/- {row['delta_se']:.4f}"
        )

    out = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "downstream_ndcg.csv")
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("YAHOO_LTR_DIR"),
                    help="directory containing the Yahoo split file "
                         "(or set YAHOO_LTR_DIR)")
    ap.add_argument("--split-file", default="set1.valid.txt",
                    help="Yahoo file to use for relevance grades (smaller = faster)")
    ap.add_argument("--cache", default=None,
                    help="optional parquet cache path for the parsed relevance table")
    ap.add_argument("--out-dir", default="results/downstream_ltr")
    ap.add_argument("--pilot", action="store_true", help="fast smoke test")
    args = ap.parse_args()

    if not args.data_dir:
        print("ERROR: no data directory. Pass --data-dir /path/to/Yahoo_ltr or "
              "set YAHOO_LTR_DIR.", file=sys.stderr)
        sys.exit(1)

    path = resolve_yahoo_file(args.data_dir, args.split_file)
    run(path, args.cache, args.out_dir, args.pilot)


if __name__ == "__main__":
    main()
