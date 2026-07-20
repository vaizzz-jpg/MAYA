"""CLI entrypoint for MAYA Phase 2 / 2.5 dataset engineering."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.datasets.dataset_config import get_dataset_config
from ai.datasets.logging_setup import configure_dataset_logging
from ai.datasets.pipeline import run_dataset_pipeline


def main() -> int:
    """Run the full dataset pipeline and exit with status code."""

    config = get_dataset_config()
    logger = configure_dataset_logging(config.project_root / "logs", "INFO")

    try:
        result = run_dataset_pipeline(config)
    except Exception:
        logger.exception("Dataset pipeline failed")
        return 1

    logger.info("Reports directory: %s", config.reports_dir)
    logger.info("Validation passed: %s", result.validation_passed)
    logger.info("Metadata: %s", result.metadata_path)
    return 0 if result.validation_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
