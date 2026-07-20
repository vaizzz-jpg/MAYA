# MAYA — Coding Standards (Canonical)

Aligned with Phase 0/1 standards in the SRS and architecture docs.

## Python

- Python 3.11+
- PEP 8
- Type hints on public functions
- Docstrings on modules and public APIs
- Logging (never `print()` for operational output)
- `pathlib.Path` over `os.path` when cleaner
- Dataclasses / Enums where they clarify domain concepts
- No God modules; reusable helpers instead of copy-paste

## Phase 2 specific

- Stream images one-at-a-time (or small batches) — never load the full dataset into RAM
- Skip + log corrupt/unreadable files; never crash the pipeline
- Configuration only via `dataset_config.py` / environment — no hardcoded paths in algorithms
- Prefer Pillow for decode/metadata; OpenCV only when CV ops truly require it
- Prefer `collections.Counter` and pandas for summaries; matplotlib for plots
