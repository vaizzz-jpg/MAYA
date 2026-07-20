"""Dataset inventory / structure discovery.

Purpose:
    Automatically inspect an unknown Kaggle layout and produce counts
    without loading pixel arrays into memory.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ai.datasets.dataset_config import CLASS_ALIASES, ClassName, DatasetConfig
from ai.datasets.utils.fs import iter_files
from ai.datasets.utils.images import probe_size

logger = logging.getLogger("maya.datasets.inventory")


@dataclass
class ImageRecord:
    """Lightweight reference to a single image file."""

    path: Path
    class_name: ClassName
    extension: str
    size_bytes: int


@dataclass
class InventoryResult:
    """Aggregated inventory of a raw dataset tree."""

    root: Path
    folders_scanned: int
    class_folders: dict[str, Path]
    records: list[ImageRecord] = field(default_factory=list)
    images_by_class: Counter = field(default_factory=Counter)
    formats: Counter = field(default_factory=Counter)
    dimension_samples: list[tuple[int, int]] = field(default_factory=list)
    unlabeled_image_folders: list[Path] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return len(self.records)


def iter_image_files(root: Path, extensions: tuple[str, ...]) -> Iterator[Path]:
    """Yield image files under ``root`` one at a time."""

    yield from iter_files(root, extensions=extensions)


def resolve_class_from_parts(parts: tuple[str, ...]) -> ClassName | None:
    """Infer REAL/FAKE from any path component using aliases."""

    for part in reversed(parts):
        alias = CLASS_ALIASES.get(part.lower())
        if alias is not None:
            return alias
    return None


def discover_class_folders(root: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    """Find directories whose names map to REAL/FAKE and contain images."""

    discovered: dict[str, Path] = {}
    if not root.exists():
        return discovered

    for directory in root.rglob("*"):
        try:
            if not directory.is_dir():
                continue
        except OSError:
            continue

        cls = CLASS_ALIASES.get(directory.name.lower())
        if cls is None:
            continue

        has_image = False
        try:
            for child in directory.iterdir():
                if child.is_file() and child.suffix.lower() in {
                    e.lower() for e in extensions
                }:
                    has_image = True
                    break
        except OSError as exc:
            logger.warning("Cannot read folder %s: %s", directory, exc)
            continue

        if has_image:
            # Prefer shallowest unique class folders; keep first seen per canonical class
            key = cls.value
            if key not in discovered:
                discovered[key] = directory
                logger.info("Discovered class folder %s → %s", key, directory)
            else:
                # Additional folders for same class are still scanned via rglob later
                logger.info("Additional %s folder found at %s", key, directory)

    return discovered


def build_inventory(
    config: DatasetConfig,
    *,
    dimension_sample_limit: int = 2000,
) -> InventoryResult:
    """Walk the raw dataset and collect inventory metadata.

    Images are never fully decoded into numpy arrays here. Dimensions are
    sampled up to ``dimension_sample_limit`` for reporting.
    """

    root = config.raw_dir
    folders_scanned = 0
    if root.exists():
        folders_scanned = sum(1 for p in root.rglob("*") if p.is_dir()) + 1

    class_folders = discover_class_folders(root, config.supported_extensions)
    result = InventoryResult(
        root=root,
        folders_scanned=folders_scanned,
        class_folders=class_folders,
    )

    unlabeled_dirs: set[Path] = set()
    dim_count = 0

    for path in iter_image_files(root, config.supported_extensions):
        relative_parts = path.relative_to(root).parts[:-1]
        cls = resolve_class_from_parts(relative_parts)
        if cls is None:
            unlabeled_dirs.add(path.parent)
            logger.warning("Unlabeled image skipped: %s", path)
            continue

        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            logger.warning("Stat failed for %s: %s", path, exc)
            continue

        record = ImageRecord(
            path=path,
            class_name=cls,
            extension=path.suffix.lower(),
            size_bytes=size_bytes,
        )
        result.records.append(record)
        result.images_by_class[cls.value] += 1
        result.formats[record.extension] += 1

        if dim_count < dimension_sample_limit:
            dims = probe_size(path)
            if dims is not None:
                result.dimension_samples.append(dims)
                dim_count += 1

    result.unlabeled_image_folders = sorted(unlabeled_dirs)
    logger.info(
        "Inventory complete: %s images across classes %s",
        result.total_images,
        dict(result.images_by_class),
    )
    return result


def group_paths_by_class(inventory: InventoryResult) -> dict[ClassName, list[Path]]:
    """Group inventory paths by canonical class."""

    grouped: dict[ClassName, list[Path]] = defaultdict(list)
    for record in inventory.records:
        grouped[record.class_name].append(record.path)
    return dict(grouped)
