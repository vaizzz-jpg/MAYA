"""Seeded random sampling for balanced train/validation/test splits."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from ai.datasets.dataset_config import ClassName, DatasetConfig, SplitName
from ai.datasets.inventory import ImageRecord

logger = logging.getLogger("maya.datasets.sampling")


@dataclass(frozen=True)
class SplitAssignment:
    """Paths assigned to a specific split and class."""

    split: SplitName
    class_name: ClassName
    paths: tuple[Path, ...]


def _budget_value(config: DatasetConfig, split: SplitName, cls: ClassName) -> int:
    budget = config.budget_for(split)
    return budget.real if cls == ClassName.REAL else budget.fake


def sample_balanced_splits(
    readable_records: list[ImageRecord],
    config: DatasetConfig,
) -> list[SplitAssignment]:
    """Randomly sample disjoint balanced splits with a fixed seed.

    Raises:
        ValueError: if any class lacks enough readable images.
    """

    rng = random.Random(config.random_seed)
    assignments: list[SplitAssignment] = []

    for cls in ClassName:
        pool = [record.path for record in readable_records if record.class_name == cls]
        rng.shuffle(pool)

        required = sum(_budget_value(config, split, cls) for split in SplitName)
        if len(pool) < required:
            raise ValueError(
                f"Insufficient readable {cls.value} images: have {len(pool)}, need {required}"
            )

        cursor = 0
        for split in SplitName:
            need = _budget_value(config, split, cls)
            chosen = pool[cursor : cursor + need]
            cursor += need
            assignments.append(
                SplitAssignment(split=split, class_name=cls, paths=tuple(chosen))
            )
            logger.info(
                "Sampled %s/%s → %s images (seed=%s)",
                split.value,
                cls.value,
                len(chosen),
                config.random_seed,
            )

    return assignments
