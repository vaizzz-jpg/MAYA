# MAYA — Media Authenticity Analyzer

AI-powered **digital evidence investigation platform** for authenticity assessment, explainable analysis, and professional investigation workflows.

> **Deepfake detection is one component—not the whole product.**

---

## 📌 Current Status

| Phase                                              | Status     |
| -------------------------------------------------- | ---------- |
| Phase 0 — Design                                   | ✅ Complete |
| Phase 1 — Foundation                               | ✅ Complete |
| Phase 2 — Evidence Data Engineering                | ✅ Complete |
| Phase 2.5 — Dataset Pipeline Review & Optimization | ✅ Complete |
| Phase 3.1 — AI Model Architecture                  | ✅ Complete |
| Phase 3.2 — AI Training Engine                     | ✅ Complete |
| Phase 3.3 — AI Validation & Reporting              | ✅ Complete |
| Phase 3.4 — Investigation Inference                | ✅ Complete |
| Phase 3.5 — AI Performance Benchmarks              | ✅ Complete |
| Phase 4.1 — Explainability / Grad-CAM Foundation   | ✅ Complete |
| Phase 4.2 — Multi-Explainer Framework              | ✅ Complete |
| Phase 4.3 — Explanation Analytics & Trust          | ✅ Complete |
| Phase 4.4 — Explainability Validation & Benchmark  | ✅ Complete |
| Phase 4.5 — Advanced Explainability & Trust Layer  | ✅ Complete |
| Phase 3 Product — Auth / Cases / Evidence / APIs   | ✅ Complete |
| Phase 5 — Product / Reports / Hardening            | ⏳ Pending  |

**Roadmap:** [`docs/ROADMAP.md`](docs/ROADMAP.md)

---

# 🚀 Quick Start

## 1. Create Virtual Environment

From the repository root:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Phase 4.5 SHAP image explanations use:

* `shap`
* `opencv-python-headless`

> Do **not** additionally install `opencv-python`.

## 3. Configure Environment

```powershell
copy .env.example .env
```

## 4. Start the Backend

```bash
python backend/run.py
```

Application:

* Home: `http://127.0.0.1:5000/`
* Health: `http://127.0.0.1:5000/health`

---

# 💻 Hardware Requirements

* Windows 11
* **8 GB RAM**
* CPU-first
* No dedicated GPU required
* Prefer `num_workers=0` DataLoader defaults

---

# 🧠 AI Pipeline

MAYA's AI pipeline is organized into independent stages covering dataset preparation, training, evaluation, inference, benchmarking, and explainability.

---

## Phase 2 — Dataset Workflow

Place the source dataset in `dataset/raw/`.

```bash
# Build balanced 224×224 corpus + reports
python scripts/run_dataset_pipeline.py

# Seal / re-seal version metadata
python scripts/seal_dataset_version.py

# Verify Dataset / DataLoader / versioning
python -m pytest tests/test_dataset.py -q
```

Dataset details:

* [`docs/DATASET.md`](docs/DATASET.md)
* [`docs/DATASET_VERSIONING.md`](docs/DATASET_VERSIONING.md)

Optional external dataset extraction path:

```powershell
$env:MAYA_RAW_DATASET_DIR = "D:\path\to\extract"
```

---

## Phase 3.2 — Training

```bash
python scripts/train.py --profile debug
python scripts/train.py --profile development
python scripts/train.py --profile production

python -m pytest tests/test_training.py -q
```

**Outputs:**

* Logs: `logs/training.log`
* Artifacts: `artifacts/phase3/sprint2/`

---

## Phase 3.3 — Evaluation

```bash
python scripts/evaluate.py --threshold 0.5
python -m pytest tests/test_evaluation.py -q
```

**Artifacts:** `artifacts/phase3/sprint3/`

---

## Phase 3.4 — Investigation Inference

Single image:

```bash
python scripts/predict.py path\to\image.jpg
```

Folder:

```bash
python scripts/predict.py --folder path\to\images
```

Tests:

```bash
python -m pytest tests/test_inference.py -q
```

**Artifacts:** `artifacts/phase3/sprint4/`

---

## Phase 3.5 — Inference Benchmarks

```bash
python scripts/benchmark.py
python scripts/benchmark.py --runs 20

python -m pytest tests/test_benchmark.py -q
```

**Artifacts:** `artifacts/phase3/benchmark/`

---

# 🔍 Explainability — Phase 4

MAYA uses a plugin-based XAI stack located under:

```text
ai/explainability/
```

| Sprint | Capability                                         |
| ------ | -------------------------------------------------- |
| 4.1    | Grad-CAM foundation + explanation artifacts        |
| 4.2    | Grad-CAM++, LayerCAM, ScoreCAM, EigenCAM           |
| 4.3    | Focus, localization, quality & trust analytics     |
| 4.4    | Explainer benchmark, ranking & recommendations     |
| 4.5    | SHAP, faithfulness, counterfactual, fusion & audit |

---

## Phase 4.1 — Single Explanation

```python
from ai.explainability import ExplainabilityEngine, ExplainabilityConfig

result = ExplainabilityEngine(
    ExplainabilityConfig(
        explainer_name="gradcam",
        device_preference="cpu"
    )
).explain(r"path\to\image.jpg")
```

---

## Phase 4.2 — Multi-Explainer Comparison

```python
from ai.explainability import ExplainabilityEngine, ExplainabilityConfig

ExplainabilityEngine(
    ExplainabilityConfig(device_preference="cpu")
).compare(
    r"path\to\image.jpg"
)
```

---

## Phase 4.3 — Explanation Analytics

Analytics operate on existing heatmaps and do not regenerate CAMs.

```python
from ai.explainability.analytics import (
    ExplanationAnalyticsEngine,
    AnalyticsConfig
)

ExplanationAnalyticsEngine(
    AnalyticsConfig()
).analyze_from_heatmap_images(
    {
        "gradcam": r"artifacts\phase4\sprint2\gradcam_heatmap.png"
    },
    prediction="FAKE",
    model_confidence=80.0,
)
```

---

## Phase 4.4 — Explainer Benchmark

```python
from ai.explainability.benchmark import ExplainabilityBenchmarkSuite

ExplainabilityBenchmarkSuite().run(
    r"path\to\image.jpg"
)
```

---

## Phase 4.5 — Advanced XAI

Advanced explainability includes:

* SHAP
* Faithfulness evaluation
* Counterfactual explanations
* Explanation fusion
* Trust analysis
* Audit information

```python
from ai.explainability import (
    AdvancedExplainabilityEngine,
    AdvancedXAIConfig
)

AdvancedExplainabilityEngine(
    AdvancedXAIConfig(device_preference="cpu")
).analyze(
    r"path\to\image.jpg"
)
```

### Phase 4 Test Suite

```powershell
python -m pytest tests/test_gradcam.py tests/test_multi_explainer.py `
  tests/test_explanation_analytics.py tests/test_explainability_benchmark.py `
  tests/test_shap.py tests/test_faithfulness.py tests/test_counterfactual.py `
  tests/test_fusion.py tests/test_trust.py tests/test_audit.py -q
```

**Artifacts:**

```text
artifacts/phase4/sprint1/
artifacts/phase4/sprint2/
artifacts/phase4/sprint3/
artifacts/phase4/sprint4/
artifacts/phase4/sprint5/
```

**Documentation:**

* [`docs/PHASE4_SPRINT1.md`](docs/PHASE4_SPRINT1.md)
* [`docs/PHASE4_SPRINT2.md`](docs/PHASE4_SPRINT2.md)
* [`docs/PHASE4_SPRINT3.md`](docs/PHASE4_SPRINT3.md)
* [`docs/PHASE4_SPRINT4.md`](docs/PHASE4_SPRINT4.md)
* [`docs/PHASE4_SPRINT5.md`](docs/PHASE4_SPRINT5.md)

---

# 🔐 Product APIs — Phase 3

The product backend provides authentication, case management, evidence management, analysis orchestration, and audit functionality.

Backend stack:

```text
Flask
Flask-Login
SQLAlchemy
REST APIs
```

Backend code:

```text
backend/app/
```

## Start Product API

```bash
python backend/run.py
```

### Main API Areas

```text
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me

POST /api/cases
...

POST /api/evidence/cases/{id}
...

POST /api/evidence/{id}/analyze
...
```

### Product API Tests

```powershell
python -m pytest tests/test_auth_api.py tests/test_cases_api.py `
  tests/test_evidence_api.py tests/test_analysis_api.py -q
```

### Documentation

* [`docs/PHASE3_PRODUCT.md`](docs/PHASE3_PRODUCT.md)
* [`docs/PHASE3_PRODUCT_ARCHITECTURE.md`](docs/PHASE3_PRODUCT_ARCHITECTURE.md)

> Authentication currently uses **Flask-Login sessions**. JWT remains deferred by design.

---

# 🔗 Processing Pipeline

The evidence/data processing pipeline follows:

```text
raw/
  ↓
inventory
  ↓
integrity
  ↓
statistics / plots
  ↓
seeded sample
  ↓
preprocess (224 RGB)
  ↓
validate
  ↓
seal version
  ↓
dataset_metadata.json
```

---

# 🐍 PyTorch Dataset Access

```python
from ai.datasets import create_dataloader, build_split_dataset
from ai.datasets.dataset_config import SplitName

loader = create_dataloader(
    SplitName.TRAIN,
    batch_size=16,
    transform_name="train"
)
```

---

# 📦 Dataset Versioning

Active version pointer:

```text
dataset/versions/CURRENT
```

Sealed metadata:

```text
dataset/versions/v1/dataset_metadata.json
```

Future corpora such as **FaceForensics++** and **Celeb-DF** can be integrated through configuration / `MAYA_RAW_DATASET_DIR` without changing the pipeline architecture.

---

# 🗂️ Project Structure

```text
MAYA/
│
├── ai/
│   ├── datasets/          # Corpus pipeline + DataLoaders
│   ├── models/            # EfficientNet-B0 + model factory
│   ├── training/          # Training CLI / callbacks
│   ├── evaluation/        # Metrics + offline evaluation
│   ├── inference/         # Investigation prediction
│   ├── benchmark/         # Inference performance suite
│   └── explainability/    # Phase 4 XAI stack
│
├── backend/
│   └── app/               # Flask application + product APIs
│
├── frontend/               # Templates & static assets
│
├── dataset/
│   ├── raw/               # Immutable source
│   ├── processed/         # Active train / val / test
│   ├── versions/          # Sealed metadata
│   └── reports/           # Dataset reports
│
├── artifacts/
│   ├── checkpoints/       # best.pt / last.pt (local; gitignored)
│   ├── phase3/            # Train / eval / inference / benchmark
│   └── phase4/            # Explainability artifacts
│
├── docs/                  # Project documentation
├── tests/                 # Automated tests
└── scripts/               # Dataset / training / evaluation scripts
```

---

# 🏗️ Architecture

MAYA follows a layered architecture:

```text
Presentation
     ↓
Application
     ↓
Business Logic
     ↓
AI Analysis
     ↓
Storage
```

### Architecture Principles

* AI code under `ai/` must not import Flask.
* Explainability remains independent of training, evaluation, and benchmark packages.
* Higher application layers request AI/XAI functionality when required.
* Product APIs orchestrate authentication, cases, evidence, analysis, and audit workflows.

---

# 🧪 Testing

Individual test suites can be executed with:

```bash
python -m pytest tests/test_dataset.py -q
python -m pytest tests/test_training.py -q
python -m pytest tests/test_evaluation.py -q
python -m pytest tests/test_inference.py -q
python -m pytest tests/test_benchmark.py -q
```

Product API tests:

```powershell
python -m pytest tests/test_auth_api.py tests/test_cases_api.py `
  tests/test_evidence_api.py tests/test_analysis_api.py -q
```

Advanced XAI tests:

```powershell
python -m pytest tests/test_shap.py tests/test_faithfulness.py `
  tests/test_counterfactual.py tests/test_fusion.py `
  tests/test_trust.py tests/test_audit.py -q
```

---

# 📚 Documentation

Start here:

[`docs/00_PHASE0_INDEX.md`](docs/00_PHASE0_INDEX.md)

### Phase 3 Documentation

* [`docs/PHASE3_PRODUCT.md`](docs/PHASE3_PRODUCT.md)
* [`docs/PHASE3_PRODUCT_ARCHITECTURE.md`](docs/PHASE3_PRODUCT_ARCHITECTURE.md)
* [`docs/ROADMAP.md`](docs/ROADMAP.md)

### Phase 4 Documentation

* [`docs/PHASE4_SPRINT1.md`](docs/PHASE4_SPRINT1.md)
* [`docs/PHASE4_SPRINT2.md`](docs/PHASE4_SPRINT2.md)
* [`docs/PHASE4_SPRINT3.md`](docs/PHASE4_SPRINT3.md)
* [`docs/PHASE4_SPRINT4.md`](docs/PHASE4_SPRINT4.md)
* [`docs/PHASE4_SPRINT5.md`](docs/PHASE4_SPRINT5.md)

### Dataset Documentation

* [`docs/DATASET.md`](docs/DATASET.md)
* [`docs/DATASET_VERSIONING.md`](docs/DATASET_VERSIONING.md)

---

# 🔮 Roadmap

The next major stage is:

## Phase 5 — Product / Reports / Hardening

Planned areas include:

* Product web workspace
* Investigation workflows through the browser
* Professional reports
* Production hardening
* Final product integration and polish

See the complete roadmap:

[`docs/ROADMAP.md`](docs/ROADMAP.md)

---

# ⚖️ License / Intended Use

MAYA is intended for **academic and authorized investigative training contexts**.

It is **not a consumer public scanner**.

 
