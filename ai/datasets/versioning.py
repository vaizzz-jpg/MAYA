"""Dataset version sealing for MAYA processed corpora.

Design
------
Images remain in ``dataset/processed/`` (active working set) to avoid
duplicating thousands of JPEGs on disk. Each version under
``dataset/versions/<id>/`` stores:

* ``dataset_metadata.json`` — identity, counts, seed, checksum
* ``manifest.json`` — relative paths + byte sizes (no pixel payloads)
* ``CURRENT`` pointer file at ``dataset/versions/CURRENT``

Future corpora (FaceForensics++, Celeb-DF) re-use the same pipeline and
receive a new version id (``v2``, …) without changing core modules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ai.datasets.dataset_config import ClassName, DatasetConfig, SplitName
from ai.datasets.metadata import DatasetMetadata, build_dataset_metadata, write_dataset_metadata
from ai.datasets.validation import ValidationReport

logger = logging.getLogger("maya.datasets.versioning")


def _build_manifest(config: DatasetConfig) -> dict:
    """Collect relative path + size entries without loading image pixels."""

    entries: list[dict[str, object]] = []
    for split in SplitName:
        for cls in ClassName:
            folder = config.class_dir(split, cls)
            if not folder.exists():
                continue
            try:
                children = sorted(folder.iterdir())
            except OSError as exc:
                logger.warning("Cannot list %s: %s", folder, exc)
                continue
            for path in children:
                try:
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in {e.lower() for e in config.supported_extensions}:
                        continue
                    rel = path.relative_to(config.processed_dir).as_posix()
                    entries.append({"path": rel, "bytes": path.stat().st_size})
                except OSError as exc:
                    logger.warning("Manifest skip %s: %s", path, exc)
    return {
        "dataset_version": config.dataset_version,
        "processed_root": str(config.processed_dir),
        "file_count": len(entries),
        "files": entries,
    }


def seal_dataset_version(
    config: DatasetConfig,
    validation: ValidationReport,
) -> tuple[Path, DatasetMetadata]:
    """Write version artefacts under ``dataset/versions/<version>/``."""

    version_dir = config.version_dir
    version_dir.mkdir(parents=True, exist_ok=True)
    # Placeholder directory for a future sealed v2 tree (documented, empty)
    (config.versions_dir / "future").mkdir(parents=True, exist_ok=True)

    metadata = build_dataset_metadata(config, validation)
    meta_path = write_dataset_metadata(metadata, version_dir / "dataset_metadata.json")

    # Also publish a copy beside QA reports for easy discovery
    write_dataset_metadata(metadata, config.reports_dir / "dataset_metadata.json")

    manifest = _build_manifest(config)
    manifest_path = version_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote version manifest → %s (%s files)", manifest_path, manifest["file_count"])

    current_ptr = config.versions_dir / "CURRENT"
    current_ptr.write_text(config.dataset_version + "\n", encoding="utf-8")
    logger.info("Active dataset version set to %s (%s)", config.dataset_version, meta_path)

    readme = version_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    f"# MAYA Dataset Version `{config.dataset_version}`",
                    "",
                    "This folder seals metadata for the active processed corpus.",
                    "Image pixels live in `dataset/processed/` to avoid duplicated storage.",
                    "See `docs/DATASET_VERSIONING.md`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    return meta_path, metadata
