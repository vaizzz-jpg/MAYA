# MAYA Dataset Versioning

## Why version

Training experiments, papers, and forensic reproducibility require a **named, checksummed** corpus identity. Versioning separates:

- **raw/** — immutable source dump  
- **processed/** — active training-ready working set  
- **versions/<id>/** — sealed metadata + file manifest for that working set  

## Layout

```
dataset/versions/
├── CURRENT          # text file: active version id (e.g. v1)
├── future/          # reserved placeholder for upcoming seals
└── v1/
    ├── README.md
    ├── dataset_metadata.json
    └── manifest.json
```

## What is (and is not) copied

To protect disk on an **8 GB RAM / limited-SSD** workstation, version folders **do not duplicate** the ~5200 JPEG files. Pixels remain in `dataset/processed/`. The version seal records:

- counts, seed, resolution, formats  
- streaming `path|size` corpus checksum  
- relative path manifest  

Rebuilding from the same seed + raw source reproduces the same assignment.

## Creating / sealing a version

```bash
# Full pipeline (rebuilds processed/, then seals)
python scripts/run_dataset_pipeline.py

# Seal an existing validated processed corpus
python scripts/seal_dataset_version.py
```

Override version id:

```powershell
$env:MAYA_DATASET_VERSION = "v2"
```

## Integrating FaceForensics++ / Celeb-DF

1. Extract under a folder with `REAL`/`FAKE` (or alias) directory names  
2. Point `MAYA_RAW_DATASET_DIR` at that extract  
3. Set `MAYA_DATASET_VERSION=v2` (or `celebdf-v1`)  
4. Adjust split budgets in env / `DatasetConfig` if needed  
5. Run `python scripts/run_dataset_pipeline.py`  

Core modules (`inventory` → `sampling` → `preprocessing` → `MayaImageDataset`) stay unchanged; only configuration and the input tree differ.

## Metadata location

| Path | Role |
|------|------|
| `dataset/versions/<id>/dataset_metadata.json` | Canonical seal |
| `dataset/reports/dataset_metadata.json` | Convenience copy beside QA reports |
