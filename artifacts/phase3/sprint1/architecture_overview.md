# MAYA Phase 3 Sprint 1 — Architecture Overview

## Layers

| Layer | Sprint 3.1 components |
|-------|------------------------|
| Configuration | `ai/models/model_config.py` |
| Model builders | `ai/models/efficientnet.py` |
| Factory | `ai/models/model_factory.py` |
| Engine utilities | `ai/engine/device.py` |
| Tests | `tests/test_model.py` |

## Construction flow

1. Load `ModelConfig` (defaults + optional env overrides).
2. Resolve `torch.device` via `get_device(config)` — typically CPU on this workstation.
3. Call `ModelFactory.create(...)` — never instantiate torchvision models elsewhere.
4. EfficientNet-B0 builder freezes the feature extractor and attaches a 2-class head.
5. Forward path verified with a **dummy** tensor `(N, 3, 224, 224) → (N, 2)`.

## Memory posture (8 GB RAM)

* Single model instance in factory / tests (module-scoped fixture).
* Backbone frozen → few trainable parameters.
* No dataset iteration in this sprint.
* Default `batch_size=8`, `num_workers=0` for later training.

## Separation of concerns

| Concern | Location | Status |
|---------|----------|--------|
| Dataset engineering | `ai/datasets/` | Phase 2 complete |
| Model architecture | `ai/models/` | Sprint 3.1 |
| Device selection | `ai/engine/` | Sprint 3.1 |
| Training loop | *(not created)* | Later sprint |
| Flask / UI | `backend/`, `frontend/` | Untouched |
