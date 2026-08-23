# Experiments

Each script installs the package (`pip install -e .[experiments]`) and writes its
outputs under a local `./results/` directory (gitignored). Run them from the
repository root.

| Script | Reproduces | External data |
| --- | --- | --- |
| `synthetic_imbalance_sweep.py` | Synthetic imbalance-sweep figures + variance table (uniform vs min vs harmonic vs clipped) | None |
| `yahoo_semisynthetic.py` | Yahoo semi-synthetic variance / variance-reduction figures | Yahoo LTR Set 1 |
| `downstream_ltr.py` | Downstream IPS-LTR nDCG vs data-budget table (naive / uniform / harmonic / oracle) | Yahoo LTR Set 1 |

## 1. Synthetic imbalance sweep (no data needed)

```bash
python experiments/synthetic_imbalance_sweep.py                 # full run
python experiments/synthetic_imbalance_sweep.py --trials 200    # faster
python experiments/synthetic_imbalance_sweep.py --no-plots      # CSV + LaTeX table only
```

Writes `results/imbalance_sweep/exp1_results.csv` and three PNG figures, and
prints a LaTeX variance table for the 80/20 traffic split.

## 2. Yahoo semi-synthetic

Requires the Yahoo Learning to Rank Challenge dataset (Set 1). It is **not**
redistributed here; obtain it from <https://webscope.sandbox.yahoo.com>. Point
the script at the directory that contains `set1.train.txt`, `set1.valid.txt`,
`set1.test.txt` via `--data-dir` or the `YAHOO_LTR_DIR` environment variable.

```bash
python experiments/yahoo_semisynthetic.py --data-dir /path/to/Yahoo_ltr
# or
export YAHOO_LTR_DIR=/path/to/Yahoo_ltr
python experiments/yahoo_semisynthetic.py --trials 50
```

Writes `results/yahoo/exp3_results.csv` and two PNG figures.

## 3. Downstream IPS-LTR

Also requires the Yahoo dataset. Uses `set1.valid.txt` by default (smaller/faster).

```bash
python experiments/downstream_ltr.py --data-dir /path/to/Yahoo_ltr --pilot   # smoke test
python experiments/downstream_ltr.py --data-dir /path/to/Yahoo_ltr           # full run
```

Writes `results/downstream_ltr/downstream_ndcg.csv`.

## Data

* **Yahoo Learning to Rank Challenge (Set 1)** — <https://webscope.sandbox.yahoo.com>
  (requires a Yahoo Webscope account). SVMLight format; only relevance grades and
  query ids are used.
* **KDD Cup 2012 Track 2** click logs — obtainable from the KDD Cup 2012 archive.
  Neither dataset is redistributed in this repository.
