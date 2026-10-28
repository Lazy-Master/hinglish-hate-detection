# Code-Mixed Hinglish Hate Speech Detection

<p align="center">
  <img src="https://img.shields.io/badge/visibility-private-red?style=flat-square&logo=github&logoColor=white" alt="private"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.4-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Model-MuRIL%20base%20cased-FF6F00?style=flat-square&logo=huggingface&logoColor=white" alt="MuRIL"/>
  <img src="https://img.shields.io/badge/Dataset-CC--BY--NC--4.0-lightgrey?style=flat-square" alt="Dataset licence"/>
  <img src="https://img.shields.io/badge/Metric-Macro--F1-0969DA?style=flat-square" alt="Macro-F1"/>
</p>

**Stack:** Python · scikit-learn · PyTorch · Hugging Face Transformers · Gradio · pandas / NumPy · Matplotlib

---

## 1. Problem statement

Social media in India is overwhelmingly **code-mixed**: a single post mixes Hindi and English, usually with Hindi *romanised* (written in Latin script, e.g. *"bhai ye kis level ki ghatiya soch hai"*). Monolingual hate-speech systems break on this distribution because:

- romanisation shatters tokenisation and vocabulary coverage,
- the *same* reply can be hateful or harmless depending on its **parent post** (conversational context),
- available code-mixed datasets are small and noisy (HASOC ICHCL inter-annotator κ ≈ 0.47).

We frame the task as **Identification of Conversational Hate in Code-Mixed Language** (HASOC 2021 ICHCL Subtask 2): binary HOF (hate/offensive) vs NONE, optionally using the parent post as context.

## 2. The two models

| | Model | Backbone | Input | Why |
|---|---|---|---|---|
| 🔵 **B0 (baseline)** | Char n-gram TF-IDF + Linear SVM | `TfidfVectorizer` + `LinearSVC` | post text only | Classical, GPU-free reference that survives romanisation/noise. Expect Macro-F1 ≈ 0.62–0.70 — **not a strawman**. |
| 🟢 **A0 (advanced)** | Context-aware MuRIL transformer | `google/muril-base-cased` (~178M) | `parent_post [SEP] target_post` | MuRIL is pretrained on Indian languages **including transliterated text** — our exact input distribution. |

**Fairness contract** *(paste verbatim into the report §Methodology):*
> *"Both models are trained and evaluated on the identical stratified split, using the same de-identification pipeline, the same decision-threshold policy (0.5, tuned only on the validation split), the same three random seeds, and the same held-out test set. The only differences between B0 and A0 are the representation learner and the use of conversational context, both of which are properties of the models under comparison rather than of the experimental setup."*

## 3. Dataset

| | Plan A (preferred) | Plan B (fallback, used if no HASOC password in 24h) |
|---|---|---|
| Source | **HASOC 2021 ICHCL** (FIRE shared task) | `nikitadesai/hasoc` on Hugging Face |
| Size | 5,740 train / 1,348 official test | train split only → carve **stratified 80/10/10** |
| Access | Password-gated (register via Google Form) | `gated: false`, **CC-BY-NC-4.0** |
| Labels | HOF / NONE | HOF / NOT + SHOF / CHOF |

> ⚠️ Plan B's HF repo ships only `traindata-basic.csv` (the declared `testdata_basic.csv` is absent) and is **Devanagari-heavy** — filter by script and report the distribution honestly.
> Third fallback: `pankajbiswas6/prism-hinglish-hate-speech` (filter `language == 'hinglish'`), described only as a *mirrored compilation of Bohra et al. (2018)* — never cite as a research dataset.

**De-identification (required):** `@handles → <USER>`, URLs `→ <URL>`, names stripped; slurs masked (`b****d`) in all examples. Emojis are **kept**. Raw gated data is **never** committed (see `.gitignore`).

## 4. Evaluation (4 criteria, not 6)

1. **Effectiveness (primary):** Macro-F1 + per-class P/R/F1 (HASOC's official metric).
2. **Imbalance robustness:** MCC + balanced accuracy.
3. **Statistical reliability:** Bootstrap 95% CI (1,000 resamples) + McNemar's test.
4. **Cost:** train minutes · param count · ms/inference.

Plus **Figure 1**: a Macro-F1 bar chart with 95% CI error bars (majority-floor / B0 / A0-nocontext / A0).

## 5. Results (fill from `results/results.csv`)

| Model | Macro-F1 | MCC | Train time | ms/infer |
|---|---|---|---|---|
| Majority floor | _TBD_ | _TBD_ | 0 | – |
| **B0** TF-IDF + LinearSVC | _TBD_ | _TBD_ | < 2 min (CPU) | _TBD_ |
| A0 − context (ablation) | _TBD_ | _TBD_ | ~25 min (T4) | _TBD_ |
| **A0** MuRIL + context | _TBD_ | _TBD_ | ~25 min (T4) | _TBD_ |
| A0 with XLM-R (control) | _TBD_ | _TBD_ | ~30 min (T4) | _TBD_ |

> Report as one sentence, e.g. *"A0 outperformed B0 by ΔMacro-F1 = +0.06 [95% CI 0.03–0.09]; McNemar χ² = 27.4, p < 0.001."* If not significant, **say so** — a null result is a finding, not a failure.

## 6. Repository layout

```
hinglish-hate-detection/
├── README.md               ← you are here
├── REPRODUCE.md            ← exact commands, in order, with expected runtime
├── requirements.txt        ← pinned versions
├── CONTRIBUTING.md         ← contribution matrix: who did what (MVP evidence)
├── .gitignore
├── data/
│   ├── raw/                ← gated/licenced raw data (gitignored)
│   └── processed/          ← de-identified CSV + DATASET.md (sha256)
├── src/
│   ├── prepare_data.py     ← M1: loader + PII stripper + stratified split
│   ├── baseline_b0.py      ← M2: TF-IDF char 2-5gram + LinearSVC
│   ├── advanced_a0.py      ← M3: MuRIL + context, 3 seeds
│   ├── evaluate.py         ← M2: metrics, bootstrap CI, McNemar
│   ├── make_tables.py      ← M4: results.csv → tables + figures
│   └── predict.py          ← M4: export model for inference
├── results/results.csv     ← COMMITTED. Every report number traces here.
├── figures/                ← committed PNGs for report & slides
├── app/app.py              ← M4: Gradio demo
├── notebooks/01_eda.ipynb
├── report/main.pdf
├── slides/presentation.pdf
└── demo/demo_video.mp4
```

> **Every number in the report must appear in `results/results.csv`.** Every table/figure is generated by script, never hand-typed.

## 7. How to run

```bash
pip install -r requirements.txt
# 1) Data (M1) — produces data/processed/processed.csv
python src/prepare_data.py --plan B --out data/processed/processed.csv
# 2) Baseline (M2, CPU ~2 min)
python src/baseline_b0.py --seeds 42 43 44 --out results/results.csv
# 3) Advanced (M3, GPU ~25 min/seed)
python src/advanced_a0.py --model google/muril-base-cased --seeds 42 43 44
# 4) Metrics + CI + McNemar (M2)
python src/evaluate.py --preds results/ --out results/results.csv
# 5) Tables + figures (M4)
python src/make_tables.py --csv results/results.csv --out figures/
# 6) Demo (M4)
python app/app.py
```

See `REPRODUCE.md` for the authoritative, ordered command list with expected runtimes.

## 8. Team & access

- **Visibility:** Private repository (course activity).
- **Owner:** `@Lazy-Master`
- **Collaborators (write):** `@SohamVaidya12` — **Soham Nilesh Vaidya** (`sohamvaidya333@gmail.com`), M.Tech AI&DS. Invite sent with **write** access. ✅
- **Workstreams:** M1 Data · M2 Baseline & stats · M3 Models · M4 Repo/demo/slides. See `CONTRIBUTING.md`.

## 9. AI-use statement

> Code in this repository was developed with AI assistance; **all experimental results were produced by executing the committed scripts on the cited public datasets**, and every figure in the report is generated by `src/make_tables.py` from `results/results.csv`. Training logs are committed under `logs/`. This statement plus a traceable `results.csv` is our defence against "were these numbers real?".

## 10. Licence

- **Code:** _[to be decided — recommend MIT or Apache-2.0 for the submission]_.
- **Data:** Plan B (`nikitadesai/hasoc`) is **CC-BY-NC-4.0** — academic use permitted, attribution required. Plan A (HASOC 2021) is research-use under task terms.
- Cite the **dataset and its overview paper** (Satapara et al., FIRE 2021, CEUR Vol-3159 T1-2), not just the download URL.

## 11. References (shortlist — full list in the plan, §11)

- Satapara et al. (2021). *Overview of HASOC Subtrack at FIRE 2021: Conversational Hate Speech…* CEUR Vol-3159 T1-2.
- Khanuja et al. (2021). *MuRIL: Multilingual Representations for Indian Languages.* arXiv:2103.10730.
- Kakati & Dandotiya (2024). *…code-mixed Indian languages… DConvBLSTM-MuRIL ensemble.* SNAM 14(1):108. — peer-reviewed precedent for MuRIL on code-mixed hate.
- Singh et al. (2025). *Rethinking Hate Speech Detection… Can LLMs Replace Traditional Models?* arXiv:2506.12744 — IndoHateMix; Future Work.
- Sheth et al. (2026). *Beyond Monolingual Assumptions: A Survey on Code-Switched NLP…* ACL 2026.

---

*Repository scaffold generated from the team's 7-day execution plan (prepared 28 Oct 2026). Fill placeholders as the 7-day schedule progresses; protect the Minimum Viable Submission (§3.2 of the plan) above all bonuses.*
