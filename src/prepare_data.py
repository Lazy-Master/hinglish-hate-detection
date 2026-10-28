"""
prepare_data.py  (M1 — Data)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN THIS MODULE (Day-6 dry run):
  • What it does : Loads the open dataset, de-identifies PII, and writes the processed
                   train / val / test splits to data/processed/.
  • Inputs       : pankajbiswas6/prism-hinglish-hate-speech (MIT, ungated, on Hugging Face).
                   --lang hinglish keeps only the romanised code-mixed rows (~4.8K);
                   --lang all keeps the full 29.5K English + Hindi + Hinglish corpus.
  • Outputs      : data/processed/{train,val,test}.csv + data/processed/DATASET.md (counts + sha256).
  • The one gotcha: The dataset ALREADY ships a split — use it, do NOT re-split or shuffle.
                   Emojis are KEPT; only @handles and URLs are masked.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_ID = "pankajbiswas6/prism-hinglish-hate-speech"
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def deidentify(text: str) -> str:
    """@handles -> <USER>, URLs -> <URL>, collapse whitespace. Emojis KEPT."""
    text = re.sub(r"https?://\S+", " <URL> ", str(text))
    text = re.sub(r"@\w+", " <USER> ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hinglish",
                    choices=["hinglish", "english", "hindi", "all"],
                    help="filter by the dataset's `lang` column ('all' = no filter)")
    ap.add_argument("--out", default="data/processed", help="output directory")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_ID)
    rows = []

    for hf_split, name in SPLIT_MAP.items():
        if hf_split not in ds:
            continue
        df = ds[hf_split].to_pandas()
        if args.lang != "all":
            df = df[df["lang"] == args.lang].reset_index(drop=True)
        df["text"] = df["text"].map(deidentify)
        df = df[["text", "label"]]

        path = outdir / f"{name}.csv"
        df.to_csv(path, index=False)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()

        n_hate = int((df["label"] == 1).sum())
        rows.append((name, len(df), n_hate, len(df) - n_hate, sha))
        print(f"{name:>5}: {len(df):>6} rows  (hate={n_hate})  -> {path}")

    total = sum(r[1] for r in rows)
    hate_total = sum(r[2] for r in rows)
    md = [
        "# Dataset provenance (data/processed/)",
        "",
        f"- **Dataset:** `{DATASET_ID}` (Hugging Face)",
        "- **Licence:** MIT — fully open, ungated, attribution required.",
        f"- **Filter:** `lang == \"{args.lang}\"`" + (" (no filter)" if args.lang == "all" else ""),
        f"- **Total rows:** {total} (hate={hate_total}, non-hate={total - hate_total})",
        "",
        "| Split | Rows | Hate | Non-hate | sha256 |",
        "|---|---|---|---|---|",
    ]
    for name, n, h, nh, sha in rows:
        md.append(f"| {name} | {n} | {h} | {nh} | `{sha[:16]}…` |")
    md += [
        "",
        "> Provenance: the card states this is derived from a Kaggle `combined_hate_speech_dataset`",
        "> compilation — cite it as a *derived open dataset* and note this in Limitations.",
        "> No parent posts are available, so conversational context is future work.",
        "",
    ]
    (outdir / "DATASET.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {outdir / 'DATASET.md'}")


if __name__ == "__main__":
    main()
