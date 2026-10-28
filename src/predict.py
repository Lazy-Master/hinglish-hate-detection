"""
predict.py  (M4 — inference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN THIS MODULE (Day-6 dry run):
  • What it does : Loads best_model/ and exposes predict(text) -> probs.
  • Inputs       : a single post. Uses the SAME de-identification as training.
  • Outputs      : P(HATE) and P(NON-HATE), latency in ms, and top-3 attributed tokens.
  • The one gotcha: de-identify BEFORE tokenising, and tokenise exactly as in advanced_a0.py —
                   mismatch here silently breaks the demo.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse


def predict(text: str):
    # PLACEHOLDER: load best_model/, deidentify, tokenise, return probs + latency.
    raise NotImplementedError("Wire up model loading + inference here.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    print(predict(args.text))


if __name__ == "__main__":
    main()
