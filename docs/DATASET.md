# MAYA Dataset Documentation

## Project dataset name

**MAYA-Kaggle-Balanced** (`dataset_version`: `v1`)

## Dataset source

- Primary catalogue: Kaggle-derived image URLs in `archive/FINAL_DATASET.csv` (immutable)
- Materialized on disk via `ai/datasets/csv_ingest.py` into `dataset/raw/{REAL,FAKE}`
- Local class folders are discovered automatically (aliases: real/fake/authentic/deepfake/…)

## Original dataset size

| Metric | Value |
|--------|-------|
| Inventoried images (raw) | **5490** |
| REAL (readable) | **2790** |
| FAKE (readable) | **2700** |
| Formats | `.jpg` |
| Geometry (sampled) | width 127–1080 · height 128–2079 |

## Trimmed (processed) dataset

| Split | REAL | FAKE | Total |
|-------|------|------|-------|
| train | 1800 | 1800 | 3600 |
| validation | 300 | 300 | 600 |
| test | 500 | 500 | 1000 |
| **Overall** | **2600** | **2600** | **5200** |

## Image resolution

- Processed: **224 × 224** RGB JPEG
- Aspect ratio preserved via short-side resize + center crop
- **No** mean/std normalization is baked into files (applied later in PyTorch transforms)

## Folder structure

```
dataset/
├── raw/                    # immutable source images
│   ├── REAL/
│   └── FAKE/
├── processed/              # active working corpus (PyTorch reads this)
│   ├── train/{REAL,FAKE}/
│   ├── validation/{REAL,FAKE}/
│   └── test/{REAL,FAKE}/
├── versions/
│   ├── CURRENT             # pointer to active version id
│   ├── future/             # reserved for upcoming corpora
│   └── v1/
│       ├── dataset_metadata.json
│       ├── manifest.json
│       └── README.md
└── reports/                # QA artefacts + metadata copy
```

## Sampling strategy

- Seeded random sampling (`random.Random(seed)`)
- Disjoint train / validation / test per class
- Balanced REAL/FAKE within each split
- **Random seed:** `42` (override with `MAYA_RANDOM_SEED`)

## Hardware constraints

- Windows 11 · **8 GB RAM** · no dedicated GPU required
- Incremental I/O · lazy `Dataset` loading · `num_workers=0` by default
- Version seal avoids duplicating JPEG bytes (metadata + manifest only)

## Reason for trimming

Full raw inventory (~5.5k) exceeds a comfortable training footprint on 8 GB RAM when combined with model weights and DataLoader buffering. The 5200-image balanced corpus keeps class parity and leaves headroom for Phase 4 training.

## Known limitations

- Content is JPG-centric in the current raw dump
- Integrity is Pillow verify/load (not cryptographic bit-for-bit of remote sources)
- Default corpus checksum seals `path|size` (set content hashing if stronger seals are needed)
- Raw catalog mixed resolution remains until processed

## Future expansion

Registered corpora (same pipeline, new `MAYA_RAW_DATASET_DIR` / version id):

| Key | Display name |
|-----|----------------|
| `faceforensics++` | FaceForensics++ |
| `celeb-df` | Celeb-DF |

No architectural change is required: place labeled `REAL`/`FAKE` (or alias) folders under raw, bump `MAYA_DATASET_VERSION` (e.g. `v2`), re-run the pipeline.

## Related docs

- [`DATASET_VERSIONING.md`](DATASET_VERSIONING.md)
- [`08_PHASE2_DATA_ENGINEERING.md`](08_PHASE2_DATA_ENGINEERING.md)
- [`09_PHASE25_ENGINEERING_REVIEW.md`](09_PHASE25_ENGINEERING_REVIEW.md)
