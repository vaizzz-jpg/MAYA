"""Model / checkpoint statistics for MAYA benchmarks.

Purpose
-------
Extract model identity, dataset version, checkpoint size, and parameter counts.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai.engine.experiment import read_dataset_version
from ai.inference.inference_config import InferenceConfig
from ai.inference.model_loader import LoadedModel
from ai.models.efficientnet import count_trainable_parameters

logger = logging.getLogger("maya.ai.benchmark.model_stats")


@dataclass
class ModelStats:
    """Static identity + size statistics for the loaded classifier."""

    model_name: str
    model_version: str
    dataset_version: str
    checkpoint_path: str
    checkpoint_size_mb: float
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_model_stats(
    loaded: LoadedModel,
    config: InferenceConfig,
) -> ModelStats:
    """Build ``ModelStats`` from a loaded model handle."""

    trainable, total = count_trainable_parameters(loaded.model)
    frozen = total - trainable
    ckpt = Path(loaded.checkpoint_path)
    size_mb = round(ckpt.stat().st_size / (1024**2), 3) if ckpt.exists() else 0.0
    dataset_version = read_dataset_version(config.project_root)

    stats = ModelStats(
        model_name=loaded.model_name or config.model_name,
        model_version=loaded.model_version or config.model_version,
        dataset_version=dataset_version,
        checkpoint_path=str(ckpt),
        checkpoint_size_mb=size_mb,
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=frozen,
    )
    logger.info(
        "Model %s params total=%s trainable=%s ckpt=%.2f MiB",
        stats.model_name,
        total,
        trainable,
        size_mb,
    )
    return stats
