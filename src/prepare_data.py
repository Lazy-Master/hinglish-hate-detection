"""
prepare_data.py  (M1 — Data)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN THIS MODULE (Day-6 dry run):
  • What it does : Loads the dataset, de-identifies PII, and writes a stratified,
                   conversation-grouped split to data/processed/processed.csv.
  • Inputs       : --plan A (HASOC, password-gated) or --plan B (nikitadesai/hasoc,
                   ungated CC-BY-NC-4.0). Raw gated data is NEVER committed.
  • Outputs      : data/processed/processed.csv + data/processed/DATASET.md (counts + sha256).
  • The one gotcha: Split by CONVERSATION (GroupShuffleSplit), never by row — otherwise
                   train/test leakage inflates scores. Emojis are KEPT.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import re
import json
import glob
import hashlib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def deidentify(text: str) -> str:
    """@handles -> <USER>, URLs -> <URL>, collapse whitespace. Emojis KEPT."""
    text = re.sub(r"https?://\S+", " <URL> ", text)
    text = re.sub(r"@\w+", " <USER> ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_plan_b():
    """Placeholder: load_dataset('nikitadesai/hasoc'); carve 80/10/10 stratified split."""
    raise NotImplementedError("Wire up Hugging Face load_dataset('nikitadesai/hasoc') here.")


def flatten_plan_a(root: str) -> pd.DataFrame:
    """Placeholder: flatten HASOC ICHCL tree JSON (Story/Tweet_id/data.json + labels.json)."""
    raise NotImplementedError("Implement the walk() flattener from the plan §13.")


def write_split(df: pd.DataFrame, out_csv: str):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(df, groups=df["conv"]))
    # carve val out of tr the same way; record exact counts in DATASET.md
    df.iloc[tr].to_csv(out_csv, index=False)
    sha = hashlib.sha256(open(out_csv, "rb").read()).hexdigest()
    print(f"wrote {out_csv}  sha256={sha}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["A", "B"], default="B")
    ap.add_argument("--out", default="data/processed/processed.csv")
    args = ap.parse_args()
    df = load_plan_b() if args.plan == "B" else flatten_plan_a("Data")
    df["text_clean"] = df["text"].map(deidentify)
    write_split(df, args.out)


if __name__ == "__main__":
    main()
