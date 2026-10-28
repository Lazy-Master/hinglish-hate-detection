"""
predict.py  (M4 — inference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN THIS MODULE (Day-6 dry run):
  • What it does : Loads best_model/ and exposes a single predict(target_post, parent_post) -> probs.
  • Inputs       : a target post (and optional parent). Uses the SAME de-identification as training.
  • Outputs      : P(HATE) and P(NON-HATE), latency in ms, and top-3 attributed tokens.
  • The one gotcha: de-identify BEFORE tokenising, and build the parent [SEP] target input exactly
                   as in advanced_a0.py — mismatch here silently breaks the demo.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse


def predict(target: str, parent: str = ""):
    # PLACEHOLDER: load best_model/, deidentify, build input, return probs + latency.
    raise NotImplementedError("Wire up model loading + inference here.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--parent", default="")
    args = ap.parse_args()
    print(predict(args.target, args.parent))


if __name__ == "__main__":
    main()
