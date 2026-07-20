# MAYA — Media Authenticity Analyzer

AI-powered **digital evidence investigation platform** for authenticity assessment, explainable analysis, and professional investigation workflows.

> Deepfake detection is one component—not the whole product.

## Current status

| Phase | Status |
|-------|--------|
| Phase 0 — Design | Complete (`docs/`) |
| Phase 1 — Foundation | Complete |
| Phase 2 — Evidence data engineering | Complete |
| Phase 2.5 — Dataset pipeline review & optimization | Complete |
| Phase 3.1 — AI model architecture | Complete |
| Phase 3.2 — AI training engine | Complete |
| Phase 3.3 — AI validation & reporting | Complete |
| Phase 3.4 — Investigation inference | Complete |
| Phase 3.5 — AI performance benchmarks | Complete |
| Phase 3+ — App auth / APIs | Pending |

## Hardware requirements

- Windows 11
- **8 GB RAM**
- No dedicated GPU required for dataset engineering
- Prefer `num_workers=0` DataLoader defaults

## Installation

```bash
# From repository root
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

copy .env.example .env
```

## Dataset workflow

```bash
# 1) Place Kaggle REAL/FAKE images in dataset/raw/
#    or materialize from catalogue CSV, or:
#    $env:MAYA_RAW_DATASET_DIR = "D:\path\to\extract"

# 2) Build balanced 224×224 corpus + reports
python scripts/run_dataset_pipeline.py

# 3) (Optional) Seal / re-seal version metadata for an existing processed set
python scripts/seal_dataset_version.py

# 4) Verify Dataset / DataLoader / versioning
python -m pytest tests/test_dataset.py -q
```

Details:

- [`docs/DATASET.md`](docs/DATASET.md)
- [`docs/DATASET_VERSIONING.md`](docs/DATASET_VERSIONING.md)
- [`docs/08_PHASE2_DATA_ENGINEERING.md`](docs/08_PHASE2_DATA_ENGINEERING.md)
- [`docs/09_PHASE25_ENGINEERING_REVIEW.md`](docs/09_PHASE25_ENGINEERING_REVIEW.md)
- [`docs/PHASE3_SPRINT1.md`](docs/PHASE3_SPRINT1.md)
- [`docs/PHASE3_SPRINT2.md`](docs/PHASE3_SPRINT2.md)
- [`docs/PHASE3_SPRINT3.md`](docs/PHASE3_SPRINT3.md)
- [`docs/PHASE3_SPRINT4.md`](docs/PHASE3_SPRINT4.md)
- [`docs/PHASE3_SPRINT5.md`](docs/PHASE3_SPRINT5.md)

## Training (Phase 3 Sprint 2)

```bash
python scripts/train.py --profile debug
python scripts/train.py --profile development
python scripts/train.py --profile production

python -m pytest tests/test_training.py -q
```

Logs: `logs/training.log` · Artefacts: `artifacts/phase3/sprint2/`

## Evaluation (Phase 3 Sprint 3)

```bash
python scripts/evaluate.py --threshold 0.5
python -m pytest tests/test_evaluation.py -q
```

Artefacts: `artifacts/phase3/sprint3/`

## Inference (Phase 3 Sprint 4)

```bash
python scripts/predict.py path\to\image.jpg
python scripts/predict.py --folder path\to\images
python -m pytest tests/test_inference.py -q
```

Artefacts: `artifacts/phase3/sprint4/`

## Benchmarks (Phase 3 Sprint 5)

```bash
python scripts/benchmark.py
python scripts/benchmark.py --runs 20
python -m pytest tests/test_benchmark.py -q
```

Artefacts: `artifacts/phase3/benchmark/`

## Processing pipeline

```
raw/ → inventory → integrity → statistics/plots
     → seeded sample → preprocess (224 RGB)
     → validate → seal versions/v1 + dataset_metadata.json
```

PyTorch access:

```python
from ai.datasets import create_dataloader, build_split_dataset
from ai.datasets.dataset_config import SplitName

loader = create_dataloader(SplitName.TRAIN, batch_size=16, transform_name="train")
```

## Folder structure

```
MAYA/
├── ai/datasets/       # Config, utils, inventory → DataLoader (no Flask)
├── backend/           # Flask application (Phase 1+)
├── frontend/          # Templates & static assets
├── dataset/
│   ├── raw/           # Immutable source
│   ├── processed/     # Active train/val/test
│   ├── versions/      # Sealed metadata (v1, future, …)
│   └── reports/       # QA artefacts
├── docs/
├── tests/
└── scripts/
```

## Dataset versioning

Active version pointer: `dataset/versions/CURRENT`  
Sealed metadata: `dataset/versions/v1/dataset_metadata.json`  

Future corpora (**FaceForensics++**, **Celeb-DF**) plug in via config / `MAYA_RAW_DATASET_DIR` without changing pipeline architecture.

## Quick start (Phase 1 app shell)

```bash
python backend/run.py
```

- Home: http://127.0.0.1:5000/
- Health: http://127.0.0.1:5000/health

## Documentation

Start here: [`docs/00_PHASE0_INDEX.md`](docs/00_PHASE0_INDEX.md)

## Architecture

**Presentation → Application → Business Logic → AI Analysis → Storage**

Dataset code lives under `ai/` and must not import Flask.

## License / use

Intended for academic and authorized investigative training contexts. Not a consumer public scanner.
