"""Seal an already-built processed corpus as a dataset version (Phase 2.5)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.datasets.dataset_config import get_dataset_config
from ai.datasets.logging_setup import configure_dataset_logging
from ai.datasets.validation import validate_processed_dataset
from ai.datasets.versioning import seal_dataset_version


def main() -> int:
    """Validate processed folders and write version metadata / manifest."""

    config = get_dataset_config()
    logger = configure_dataset_logging(config.project_root / "logs", "INFO")
    logger.info("Sealing dataset version %s", config.dataset_version)

    validation = validate_processed_dataset(config)
    if not validation.passed:
        logger.error("Cannot seal — validation failed (see counts above)")
        return 2

    meta_path, metadata = seal_dataset_version(config, validation)
    logger.info("Sealed %s checksum=%s", meta_path, metadata.dataset_checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
