"""
advanced_a0.py  (M3 — Models)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN THIS MODULE (Day-6 dry run):
  • What it does : Fine-tunes A0 — MuRIL (google/muril-base-cased) on single-post text.
                   Also runs the XLM-R control.
  • Inputs       : data/processed/{train,val,test}.csv (text, label). Needs a GPU.
  • Outputs      : best_model/ (best seed by val Macro-F1) + rows in results/results.csv +
                   logs/a0_seed*.log.
  • The one gotcha: MuRIL is pretrained on Indian languages INCLUDING transliterated text — our
                   exact input. The open dataset has no parent posts, so this is text-only (context
                   is future work). Early-stop on val Macro-F1.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse

# Hyperparameters (plan §2.2)
MID = "google/muril-base-cased"
LR = 2e-5
EPOCHS = 4
BATCH = 16
GRAD_ACCUM = 2
MAX_LEN = 256
WARMUP = 0.1
SEEDS = [42, 43, 44]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MID)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    args = ap.parse_args()
    # PLACEHOLDER: build Dataset (tokenise text), Trainer with weighted CE,
    # warmup 10%, dropout 0.1, early stop on val macro_f1, 3 seeds, save best_model/.
    raise NotImplementedError("Implement the MuRIL fine-tune + XLM-R control here.")


if __name__ == "__main__":
    main()
