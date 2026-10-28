# REPRODUCE.md

Exact, ordered commands to reproduce every number in the report. Run from the repo root.
Each step lists its **expected runtime** and **hardware** so a reviewer can sanity-check.

> Prerequisites: Python 3.10+, CUDA GPU only needed for Steps 3–5 (Days 1–3). Everything after Day 3 is CPU-only.

## Environment

```bash
pip install -r requirements.txt
```

## Step 1 — Data preparation (M1) · CPU · ~5 min
```bash
python src/prepare_data.py --plan B --out data/processed/processed.csv
```
Produces `data/processed/processed.csv` (de-identified, conversation-grouped split).
Records exact train/val/test counts and the split's sha256 in `data/processed/DATASET.md`.

## Step 2 — Baselines + floors (M2) · CPU · ~5 min
```bash
python src/baseline_b0.py --seeds 42 43 44 --out results/results.csv
```
Writes majority-floor, random-floor, and B0 rows to `results/results.csv`.

## Step 3 — A0 MuRIL + context (M3) · GPU · ~1.5 h (3 seeds)
```bash
python src/advanced_a0.py --model google/muril-base-cased --seeds 42 43 44 \
       --epochs 4 --batch 16 --lr 2e-5 --max-len 256
```
Saves `best_model/` (best seed by val Macro-F1) and appends A0 rows to `results/results.csv`.
Training logs: `logs/a0_seed42.log` … `logs/a0_seed44.log`.

## Step 4 — Ablation + control (M3) · GPU · ~1 h
```bash
python src/advanced_a0.py --model google/muril-base-cased --seeds 42 --no-context \
       --out results/results.csv          # A0-nocontext
python src/advanced_a0.py --model xlm-roberta-base --seeds 42 \
       --out results/results.csv           # A0-xlmr control
```

## Step 5 — Metrics, bootstrap CI, McNemar (M2) · CPU · ~15 min
```bash
python src/evaluate.py --preds results/ --out results/results.csv
```
Adds Macro-F1 [95% CI], MCC, balanced accuracy, and the McNemar p-value between B0 and A0.

## Step 6 — Tables + figures (M4) · CPU · ~2 min
```bash
python src/make_tables.py --csv results/results.csv --out figures/
```
Generates Table 1 (main comparison), Table 2 (ablation), and Figure 1 (CI bar chart).

## Step 7 — Inference / demo (M4) · CPU · < 1 min to load
```bash
python app/app.py          # Gradio demo at http://localhost:7860
```

## Expected totals
- **GPU:** ~2.5 h (Steps 3–4). **CPU:** ~30 min (rest).
- If a GPU is unavailable, A0 at **1 seed still passes** the Day-3 gate (G3); report single-seed and say so.

## Integrity notes
- Raw gated data is **never** committed (see `.gitignore`).
- `results/results.csv` is committed with timestamps; every report figure traces to a row here.
- If any number lands far outside the bands in the plan §5.1, suspect a data/leakage bug before celebrating.
