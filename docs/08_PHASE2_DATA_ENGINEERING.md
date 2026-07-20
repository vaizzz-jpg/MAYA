# MAYA — Phase 2 Evidence Data Engineering

## Purpose

Produce a clean, balanced, validated, memory-efficient image corpus and a reusable PyTorch `Dataset` / `DataLoader` stack—**without** training models or touching the web app.

## Architecture

```
dataset/raw/          # immutable source (Kaggle dump as-is)
        │
        ▼
ai/datasets/*         # inspect → integrity → stats → sample → preprocess → validate
        │
        ▼
dataset/processed/{train,validation,test}/{REAL,FAKE}
        │
        ▼
MayaImageDataset + create_dataloader()
```

## Modules (SRP)

| File | Responsibility |
|------|----------------|
| `dataset_config.py` | Paths, sizes, seed, formats, batch defaults |
| `inventory.py` | Structure discovery + inventory counts |
| `integrity.py` | Corrupt / zero-byte / duplicate detection |
| `statistics.py` | Distributions + size/aspect stats |
| `visualization.py` | Plots + sample grids |
| `sampling.py` | Seeded balanced split selection |
| `preprocessing.py` | RGB resize-to-224 writers (no normalize) |
| `validation.py` | Folder/count balance checks |
| `reports.py` | Markdown/JSON/CSV writers |
| `dataset.py` | `torch.utils.data.Dataset` |
| `dataloader.py` | Configurable DataLoader factory |
| `pipeline.py` | Orchestrates steps 1–6 |

## Target split (seeded)

| Split | REAL | FAKE |
|-------|------|------|
| train | 1800 | 1800 |
| validation | 300 | 300 |
| test | 500 | 500 |
| **Total** | | **5200** |

## Run

```bash
# Place Kaggle extract under dataset/raw/ (or set MAYA_RAW_DATASET_DIR)
python scripts/run_dataset_pipeline.py
python -m pytest tests/test_dataset.py -q
```

## Outputs (`dataset/reports/`)

- `dataset_report.md`
- `integrity_report.json`
- `validation_report.md`
- `dataset_summary.csv`
- `class_distribution.png`
- `sample_grid.png`
- `dataset_metadata.json` (also under `dataset/versions/<id>/`)

## Phase 2.5

See [`09_PHASE25_ENGINEERING_REVIEW.md`](09_PHASE25_ENGINEERING_REVIEW.md), [`DATASET.md`](DATASET.md), and [`DATASET_VERSIONING.md`](DATASET_VERSIONING.md).
