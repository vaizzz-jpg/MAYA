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
| Phase 4.1 — Explainability (Grad-CAM foundation) | Complete |
| Phase 4.2 — Multi-Explainer Framework | Complete |
| Phase 4.3 — Explanation Analytics & Trust | Complete |
| Phase 4.4 — Explainability Validation & Benchmark | Complete |
| Phase 4.5 — Advanced Explainability & Trust Layer | Complete |
| Phase 3 Product — Auth / Cases / Evidence / APIs | Complete |
| Phase 5 — Product / reports / hardening | Pending |

Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md)

## Hardware requirements

- Windows 11
- **8 GB RAM**
- CPU-first (no dedicated GPU required)
- Prefer `num_workers=0` DataLoader defaults

## Installation

```bash
# From repository root
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Phase 4.5 SHAP image explanations also need OpenCV (pulled by requirements.txt):
#   shap + opencv-python-headless  (do not also install opencv-python)

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

Details: [`docs/DATASET.md`](docs/DATASET.md) · [`docs/DATASET_VERSIONING.md`](docs/DATASET_VERSIONING.md)

## Training (Phase 3.2)

```bash
python scripts/train.py --profile debug
python scripts/train.py --profile development
python scripts/train.py --profile production

python -m pytest tests/test_training.py -q
```

Logs: `logs/training.log` · Artefacts: `artifacts/phase3/sprint2/`

## Evaluation (Phase 3.3)

```bash
python scripts/evaluate.py --threshold 0.5
python -m pytest tests/test_evaluation.py -q
```

Artefacts: `artifacts/phase3/sprint3/`

## Inference (Phase 3.4)

```bash
python scripts/predict.py path\to\image.jpg
python scripts/predict.py --folder path\to\images
python -m pytest tests/test_inference.py -q
```

Artefacts: `artifacts/phase3/sprint4/`

## Inference benchmarks (Phase 3.5)

```bash
python scripts/benchmark.py
python scripts/benchmark.py --runs 20
python -m pytest tests/test_benchmark.py -q
```

Artefacts: `artifacts/phase3/benchmark/`

## Explainability (Phase 4)

Plugin-based XAI stack under `ai/explainability/`:

| Sprint | What it does |
|--------|----------------|
| 4.1 | Grad-CAM foundation + explanation artefacts |
| 4.2 | Grad-CAM++, LayerCAM, ScoreCAM, EigenCAM + comparison |
| 4.3 | Focus / localization / quality / trust analytics |
| 4.4 | Explainer benchmark, ranking, recommendations |
| 4.5 | SHAP, faithfulness, counterfactual, fusion, audit |

```python
# Single explanation (Sprint 4.1+)
from ai.explainability import ExplainabilityEngine, ExplainabilityConfig

result = ExplainabilityEngine(
    ExplainabilityConfig(explainer_name="gradcam", device_preference="cpu")
).explain(r"path\to\image.jpg")

# Multi-explainer comparison (Sprint 4.2)
from ai.explainability import ExplainabilityEngine, ExplainabilityConfig

ExplainabilityEngine(ExplainabilityConfig(device_preference="cpu")).compare(
    r"path\to\image.jpg"
)

# Analytics on heatmaps (Sprint 4.3) — does not regenerate CAMs
from ai.explainability.analytics import ExplanationAnalyticsEngine, AnalyticsConfig

ExplanationAnalyticsEngine(AnalyticsConfig()).analyze_from_heatmap_images(
    {"gradcam": r"artifacts\phase4\sprint2\gradcam_heatmap.png"},
    prediction="FAKE",
    model_confidence=80.0,
)

# Rank all registered explainers (Sprint 4.4)
from ai.explainability.benchmark import ExplainabilityBenchmarkSuite

ExplainabilityBenchmarkSuite().run(r"path\to\image.jpg")

# Advanced XAI: SHAP + faithfulness + counterfactual + trust (Sprint 4.5)
from ai.explainability import AdvancedExplainabilityEngine, AdvancedXAIConfig

AdvancedExplainabilityEngine(AdvancedXAIConfig(device_preference="cpu")).analyze(
    r"path\to\image.jpg"
)
```

```bash
python -m pytest tests/test_gradcam.py tests/test_multi_explainer.py `
  tests/test_explanation_analytics.py tests/test_explainability_benchmark.py `
  tests/test_shap.py tests/test_faithfulness.py tests/test_counterfactual.py `
  tests/test_fusion.py tests/test_trust.py tests/test_audit.py -q
```

Artefacts: `artifacts/phase4/sprint{1..5}/`  
Docs: [`PHASE4_SPRINT1.md`](docs/PHASE4_SPRINT1.md) · [`SPRINT2`](docs/PHASE4_SPRINT2.md) · [`SPRINT3`](docs/PHASE4_SPRINT3.md) · [`SPRINT4`](docs/PHASE4_SPRINT4.md) · [`SPRINT5`](docs/PHASE4_SPRINT5.md)

## Product APIs (Phase 3 — Auth / Cases / Evidence)

Flask-Login sessions + SQLAlchemy models under `backend/app/`.

```bash
python backend/run.py
# POST /api/auth/register  /api/auth/login  GET /api/auth/me
# POST /api/cases  …  POST /api/evidence/cases/{id}  …  POST /api/evidence/{id}/analyze

python -m pytest tests/test_auth_api.py tests/test_cases_api.py `
  tests/test_evidence_api.py tests/test_analysis_api.py -q
```

Docs: [`PHASE3_PRODUCT.md`](docs/PHASE3_PRODUCT.md) · [`PHASE3_PRODUCT_ARCHITECTURE.md`](docs/PHASE3_PRODUCT_ARCHITECTURE.md)

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
├── ai/
│   ├── datasets/          # Corpus pipeline + DataLoaders
│   ├── models/            # EfficientNet-B0 + factory
│   ├── training/          # Training CLI / callbacks
│   ├── evaluation/        # Metrics + offline eval
│   ├── inference/         # Investigation prediction
│   ├── benchmark/         # Inference performance suite
│   └── explainability/    # Phase 4 XAI (explainers, analytics, benchmark)
├── backend/               # Flask application shell
├── frontend/              # Templates & static assets
├── dataset/
│   ├── raw/               # Immutable source
│   ├── processed/         # Active train/val/test
│   ├── versions/          # Sealed metadata
│   └── reports/
├── artifacts/
│   ├── checkpoints/       # best.pt / last.pt (local; gitignored)
│   ├── phase3/            # Train / eval / infer / bench outputs
│   └── phase4/            # Explainability sprint artefacts
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

Phase 3 sprint notes: [`PHASE3_SPRINT1`](docs/PHASE3_SPRINT1.md)–[`SPRINT5`](docs/PHASE3_SPRINT5.md)  
Phase 4 sprint notes: [`PHASE4_SPRINT1`](docs/PHASE4_SPRINT1.md)–[`SPRINT4`](docs/PHASE4_SPRINT4.md)

## Architecture

**Presentation → Application → Business Logic → AI Analysis → Storage**

AI code under `ai/` must not import Flask. Explainability is independent of training/eval/benchmark packages and is requested by higher layers when needed.

## License / use

Intended for academic and authorized investigative training contexts. Not a consumer public scanner.
#   M A Y A 
 
 #   M A Y A 
 
 #   M A Y A 
 
 #   M A Y A 
 
 