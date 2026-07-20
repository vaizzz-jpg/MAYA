"""Dataset metadata emission for sealed MAYA corpora."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.datasets.dataset_config import ClassName, DatasetConfig, SplitName
from ai.datasets.utils.checksums import hash_path_manifest
from ai.datasets.utils.fs import count_images
from ai.datasets.validation import ValidationReport

logger = logging.getLogger("maya.datasets.metadata")


@dataclass
class DatasetMetadata:
    """Machine-readable identity card for a processed corpus version."""

    dataset_name: str
    dataset_version: str
    project_version: str
    created_at_utc: str
    random_seed: int
    image_resolution: int
    supported_formats: list[str]
    image_counts: dict[str, Any]
    class_counts: dict[str, int]
    dataset_checksum: str
    processed_root: str
    raw_root: str
    future_datasets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _collect_processed_paths(config: DatasetConfig) -> list[Path]:
    paths: list[Path] = []
    for split in SplitName:
        for cls in ClassName:
            folder = config.class_dir(split, cls)
            if not folder.exists():
                continue
            for path in folder.iterdir():
                try:
                    if path.is_file() and path.suffix.lower() in {
                        e.lower() for e in config.supported_extensions
                    }:
                        paths.append(path)
                except OSError:
                    continue
    return paths


def build_dataset_metadata(
    config: DatasetConfig,
    validation: ValidationReport,
    *,
    checksum: str | None = None,
) -> DatasetMetadata:
    """Assemble metadata from config + validation (optionally precomputed checksum)."""

    image_counts: dict[str, Any] = {"splits": {}, "total": validation.total_images}
    class_totals = {ClassName.REAL.value: 0, ClassName.FAKE.value: 0}

    for split in SplitName:
        split_counts: dict[str, int] = {}
        for cls in ClassName:
            folder = config.class_dir(split, cls)
            n = count_images(folder, config.supported_extensions)
            split_counts[cls.value] = n
            class_totals[cls.value] += n
        image_counts["splits"][split.value] = split_counts

    if checksum is None and config.compute_version_checksum:
        logger.info("Computing streaming corpus checksum (one file at a time)")
        paths = _collect_processed_paths(config)
        checksum = hash_path_manifest(paths, root=config.processed_dir)
    elif checksum is None:
        checksum = "skipped"

    return DatasetMetadata(
        dataset_name=config.dataset_name,
        dataset_version=config.dataset_version,
        project_version=config.project_version,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        random_seed=config.random_seed,
        image_resolution=config.image_size,
        supported_formats=list(config.supported_extensions),
        image_counts=image_counts,
        class_counts=class_totals,
        dataset_checksum=checksum,
        processed_root=str(config.processed_dir),
        raw_root=str(config.raw_dir),
        future_datasets=sorted(config.future_datasets.keys()),
    )


def write_dataset_metadata(metadata: DatasetMetadata, output_path: Path) -> Path:
    """Write ``dataset_metadata.json``."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    logger.info("Wrote dataset metadata → %s", output_path)
    return output_path
