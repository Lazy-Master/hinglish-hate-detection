"""
evaluate.py  (M2 — Baseline & stats)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN THIS MODULE (Day-6 dry run):
  • What it does : Computes Macro-F1 (+ per-class P/R/F1), MCC, balanced accuracy, a bootstrap
                   95% CI (1,000 resamples), and McNemar's test between B0 and A0.
  • Inputs       : prediction files under results/ (true + predicted per model/seed).
  • Outputs      : augmented results/results.csv with ci_lo, ci_hi, mcnemar_p, etc.
  • The one gotcha: A ~1.3K test set cannot resolve a 0.02 F1 gap — that's why the CI + McNemar
                   exist. If the CI spans zero, REPORT "no statistically detectable difference".
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score
from statsmodels.stats.contingency_tables import mcnemar


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro"))


def bootstrap_ci(y_true, y_pred, metric=macro_f1, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    out = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        out.append(metric(y_true[s], y_pred[s]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return metric(y_true, y_pred), float(lo), float(hi)


def mcnemar_paired(y_true, yp_a, yp_b):
    b = int(np.sum((yp_a == y_true) & (yp_b != y_true)))
    c = int(np.sum((yp_a != y_true) & (yp_b == y_true)))
    r = mcnemar([[0, b], [c, 0]], exact=False, correction=True)
    return b, c, float(r.statistic), float(r.pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/")
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()
    # PLACEHOLDER: load predictions, run metrics + CI + McNemar, write back to results.csv
    raise NotImplementedError("Compute all metrics and append CI/McNemar columns here.")


if __name__ == "__main__":
    main()
