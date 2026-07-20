# MAYA — Phase 3 Sprint 4

## Investigation Inference Engine

**Status:** Complete  
**Scope:** Single-image (+ folder-ready) inference only — **no Grad-CAM, SHA-256, Flask, frontend, auth, or DB**

---

## Inference architecture

```
CLI / BatchRunner
       ↓
InferencePipeline (orchestration)
  1. validate path
  2. ModelLoader.load()          ← cache, eval mode, device
  3. preprocess_image()          ← resize + ImageNet normalize
  4. predict_probabilities()     ← softmax only
  5. decide_from_probabilities() ← threshold + confidence bands
  6. InvestigationIDGenerator    ← INV-YYYY-NNNNNN
  7. InvestigationResult + artefacts
```

| Module | Responsibility |
|--------|----------------|
| `model_loader.py` | Checkpoint → device → `eval()` (no inference) |
| `preprocessing.py` | Safe image → `(1,3,H,W)` tensor |
| `predictor.py` | Softmax probabilities only |
| `confidence.py` | Class + % + level + threshold decision |
| `investigation_id.py` | Sequential `INV-2026-000001` |
| `result.py` | `InvestigationResult` + JSON |
| `pipeline.py` | Orchestration |
| `batch.py` | One image or folder |
| `cli.py` / `scripts/predict.py` | argparse entry |
| `utils.py` | Timer, path helpers |

## Confidence system

Bands (configurable via `ConfidenceBands`, not hardcoded in predictors):

| Level | Score |
|-------|-------|
| Very High | ≥ 95% |
| High | 85–95% |
| Medium | 70–85% |
| Low | < 70% |

Decision threshold default **0.5** on `P(FAKE)` (`MAYA_INFER_THRESHOLD` / `--threshold`).

## InvestigationResult

Includes investigation id, prediction, confidence, probabilities, threshold, model + dataset version, latency ms, timestamp, image metadata, device, and status. Serializes with `to_json()` / `save_json()`.

## Batch-ready design

`InvestigationBatchRunner.run_one` / `run_folder` share one `InferencePipeline` (and therefore one cached model). Folder mode skips corrupt files and continues.

## Memory optimization (8 GB RAM)

- Model loaded once and reused (`ModelLoader` cache)
- `torch.no_grad()` during forward
- Intermediate tensors deleted after use
- CPU-first device selection via Sprint 3.1 `device.py`

## Integration points (future — not implemented here)

| Concern | Hook |
|---------|------|
| Flask API | Call `InferencePipeline.run(path)` from a service layer |
| Grad-CAM | Consume the same preprocessed tensor + model activations in `ai/explainability/` |
| Evidence DB | Persist `InvestigationResult.to_dict()` via business services |

## How to run

```bash
python scripts/predict.py path\to\image.jpg
python scripts/predict.py --folder path\to\images
python -m pytest tests/test_inference.py -q
```

Artefacts: `artifacts/phase3/sprint4/`

## Explicit non-goals

Grad-CAM, SHA-256, Flask routes, frontend, authentication, database writes.
