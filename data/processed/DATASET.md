# Dataset provenance (data/processed/)

This file records exactly which data was used and its integrity hash, so every result is traceable.

| Field | Value |
|---|---|
| Plan used | _TBD (A or B)_ |
| Source | _TBD_ |
| Licence | Plan B: CC-BY-NC-4.0 (attribution required). Plan A: HASOC 2021 task terms. |
| Raw rows | _TBD_ |
| Train / Val / Test | _TBD / TBD / TBD_ (stratified 80/10/10, split by conversation) |
| `processed.csv` sha256 | _TBD_ |
| Inter-annotator κ | HASOC ICHCL ≈ 0.47 (label noise ceiling — report as a limitation) |
| Script distribution | _TBD (Devanagari vs romanised)_ |
| De-identification | `@handle → <USER>`, `URL → <URL>`, names stripped, emojis kept, slurs masked |

> Fill the counts and sha256 after `python src/prepare_data.py` runs on Day 1.
