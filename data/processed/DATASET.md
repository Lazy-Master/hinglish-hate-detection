# Dataset provenance (data/processed/)

This file records exactly which data was used and its integrity hash, so every result is traceable.
It is **regenerated** by `src/prepare_data.py`; the values below are the expected shape until you run it.

| Field | Value |
|---|---|
| Dataset | `pankajbiswas6/prism-hinglish-hate-speech` (Hugging Face) |
| Licence | **MIT** — fully open, ungated, attribution required |
| Access | `load_dataset("pankajbiswas6/prism-hinglish-hate-speech")` (no password) |
| Filter | `lang == "hinglish"` (romanised code-mixed) |
| Columns | `text`, `label` (0 = non-hate, 1 = hate) |
| Upstream splits | train 17,704 / validation 2,950 / test 8,852 (used as-is, not re-split) |
| Hinglish subset | ~4,783 rows across splits |
| Label balance (full) | non-hate 53.5% / hate 46.5% |
| `train.csv` sha256 | _TBD — printed by `prepare_data.py`_ |
| De-identification | `@handle → <USER>`, `URL → <URL>`, emojis kept |

> **Provenance:** the card states this is derived from a Kaggle `combined_hate_speech_dataset`
> compilation — cite it as a *derived open dataset*, and note this in Limitations.
> **No parent posts:** conversational context is unavailable here → future work.
