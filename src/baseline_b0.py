"""
baseline_b0.py  (M2 — Baseline & stats)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN THIS MODULE (Day-6 dry run):
  • What it does : Trains the B0 baseline — char n-gram TF-IDF + word n-gram TF-IDF, hstacked,
                   classified by a balanced LinearSVC. Also emits majority/random floors.
  • Inputs       : data/processed/{train,val,test}.csv (text, label). Post text only.
  • Outputs      : rows appended to results/results.csv (model, seed, macro_f1, mcc, ...).
  • The one gotcha: This is the HONEST reference (~0.62–0.70 Macro-F1), not a strawman.
                   Sublinear_tf + class_weight='balanced' matter under imbalance.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, matthews_corrcoef

SEEDS = [42, 43, 44]


def build_pipeline():
    vec = FeatureUnion([
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ("w", TfidfVectorizer(analyzer="word",    ngram_range=(1, 2), sublinear_tf=True)),
    ])
    return Pipeline([("vec", vec), ("clf", LinearSVC(class_weight="balanced"))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()
    df = pd.read_csv("data/processed/train.csv")
    # PLACEHOLDER: run 3 seeds, compute macro_f1 + mcc, append to results/results.csv
    raise NotImplementedError("Train B0 across seeds and write results rows here.")


if __name__ == "__main__":
    main()
