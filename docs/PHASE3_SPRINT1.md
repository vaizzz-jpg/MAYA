# MAYA — Phase 3 Sprint 1

## AI Foundation & Model Architecture

**Status:** Complete  
**Scope:** Modular model stack only — **no training, evaluation, Grad-CAM, Flask, or UI**

> Note: `PHASE_0_REPORT.md` / `PHASE_1_REPORT.md` / `PHASE_2_REPORT.md` were referenced in the sprint brief but are not present in-repo. This sprint follows the canonical docs: `PROJECT_VISION.md`, `ARCHITECTURE.md`, `CODING_STANDARDS.md`, `DATASET.md`, `DATASET_VERSIONING.md`, and Phase 2.5 review notes.

---

## Why EfficientNet-B0

| Factor | Reason |
|--------|--------|
| Accuracy / cost | Strong ImageNet transfer with fewer parameters than ResNet-50 |
| 8 GB RAM | Smaller backbone + frozen features → feasible CPU fine-tuning later |
| Input size | Native 224×224 matches Phase 2 processed corpus |
| Ecosystem | First-class `torchvision` support and pretrained weights |

## Why Transfer Learning

MAYA’s balanced corpus is intentionally trimmed (5200 images) for memory safety. Training a CNN from scratch would overfit and burn CPU time. ImageNet-pretrained EfficientNet-B0 provides generic visual features; freezing the extractor and replacing the classifier specializes the model for **REAL vs FAKE** with minimal trainable weights.

## Why Factory Pattern

The future training engine must stay agnostic to concrete architectures. `ModelFactory.create(model_name)` is the only construction API. Swapping to EfficientNet-B2, ResNet18, or ViT later means registering a builder — not rewriting the trainer.

## Why Dataclass Config

Hyper-parameters, paths, and device preference live in `ModelConfig`. Environment overrides (`MAYA_MODEL_NAME`, `MAYA_BATCH_SIZE`, …) keep local experiments reproducible without editing code. UPPER_CASE property aliases mirror the sprint brief while runtime code stays PEP 8.

---

## Module map

```
ModelConfig ──► ModelFactory.create() ──► build_efficientnet_b0()
                      │
                      └── (future) resnet / mobilenet / vit builders

get_device(config) ◄── device preference from ModelConfig
```

| Module | Responsibility |
|--------|----------------|
| `ai/models/model_config.py` | Central AI configuration |
| `ai/models/efficientnet.py` | EfficientNet-B0 load / freeze / head |
| `ai/models/model_factory.py` | Registry + `create()` |
| `ai/engine/device.py` | CUDA / MPS / CPU selection |
| `tests/test_model.py` | Config, device, factory, forward shape |

## Future extensibility

Registered **planned** names (raise clear errors until implemented):

- `efficientnet_b1`, `efficientnet_b2`
- `resnet18`, `resnet50`
- `mobilenet_v3`
- `vit`

Add a builder under `ai/models/`, register it in `_REGISTRY`, keep `ModelFactory.create` unchanged.

## How to test

```bash
python -m pytest tests/test_model.py -q
```

## How to regenerate artefacts

```bash
python scripts/generate_sprint1_artifacts.py
```

Outputs land in `artifacts/phase3/sprint1/`.

## Explicit non-goals (this sprint)

- Training loops / optimizers / loss
- Metrics / evaluation
- Dataset loading in model tests (dummy tensors only)
- Flask routes, frontend, Grad-CAM, SHA-256, PDF reports
