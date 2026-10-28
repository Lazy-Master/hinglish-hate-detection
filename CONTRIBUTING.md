# CONTRIBUTING.md — Contribution Matrix

This is **evidence** for the "every member must explain the work" requirement.
Fill the matrix on Day 1 and keep it honest. Each member should be able to present *any* module at the Day-6 cross-module dry run.

## Team & workstreams

| Member | GitHub handle | Primary workstream | Backup |
|---|---|---|---|
| _TBD_ | `@Lazy-Master` (owner) | _TBD (M1–M4)_ | – |
| Soham Nilesh Vaidya | `@SohamVaidya12` | _TBD_ | _TBD_ |
| _TBD_ | _add handle_ | _TBD_ | _TBD_ |
| _TBD_ | _add handle_ | _TBD_ | _TBD_ |

## Module ownership (M1–M4)

| Module | Owner | File(s) |
|---|---|---|
| M1 — Data: loader, PII stripper, stratified split | _TBD_ | `src/prepare_data.py` |
| M2 — Baseline + statistics (B0, bootstrap, McNemar) | _TBD_ | `src/baseline_b0.py`, `src/evaluate.py` |
| M3 — Models (A0, ablation, control) | _TBD_ | `src/advanced_a0.py` |
| M4 — Repo, demo, slides, report assembly | _TBD_ | `app/app.py`, `src/make_tables.py`, `src/predict.py` |

## Commit rules (scores 3 marks on the GitHub graph)
- ≥ 2 commits per member per day; commits spread across all 7 days.
- Readable messages: `feat(baseline): TF-IDF char 2-5gram + LinearSVC, macroF1 0.67` — never `update`.
- All members must appear in **Insights → Contributors**.

## How to explain this module (put at the top of every source file)
Each `src/*.py` carries a 4-bullet header: what it does · inputs · outputs · the one gotcha.
