"""
make_tables.py  (M4 — Repo / demo / slides)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN THIS MODULE (Day-6 dry run):
  • What it does : Reads results/results.csv and generates the report's Table 1 (main comparison
                   w/ CIs), Table 2 (ablation), and Figure 1 (Macro-F1 CI bar chart). All scripted.
  • Inputs       : results/results.csv (the single source of truth for every number).
  • Outputs      : figures/table1.png, figures/table2.png, figures/figure1_ci.png (+ optional .tex).
  • The one gotcha: NO number in the report may be hand-typed — it must come from this script.
                   The CI bar chart is the slide everyone remembers; build it even if rushed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--out", default="figures/")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    # PLACEHOLDER: build Table 1, Table 2, and Figure 1 (error bars = 95% CI).
    raise NotImplementedError("Render tables + CI bar chart from results.csv here.")


if __name__ == "__main__":
    main()
