"""Generate Phase 3 Sprint 1 artefacts (model summary + architecture overview)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.engine.device import get_device
from ai.models.efficientnet import summarize_model
from ai.models.model_config import ModelConfig
from ai.models.model_factory import ModelFactory
from ai.datasets.logging_setup import configure_dataset_logging

ARCHITECTURE_OVERVIEW = """# MAYA Phase 3 Sprint 1 — Architecture Overview

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
"""


def main() -> int:
    out_dir = ROOT / "artifacts" / "phase3" / "sprint1"
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_dataset_logging(ROOT / "logs", "INFO")
    logger = logging.getLogger("maya.ai.sprint1.artifacts")

    config = ModelConfig(project_root=ROOT, device_preference="cpu")
    device = get_device(config)
    model = ModelFactory.create(config=config)
    model.to(device)

    summary_path = out_dir / "model_summary.txt"
    summary_path.write_text(
        summarize_model(model) + f"\nDevice: {device}\n",
        encoding="utf-8",
    )
    overview_path = out_dir / "architecture_overview.md"
    overview_path.write_text(ARCHITECTURE_OVERVIEW, encoding="utf-8")

    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", overview_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
