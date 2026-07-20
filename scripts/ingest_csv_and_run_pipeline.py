"""Ingest FINAL_DATASET.csv URLs into dataset/raw, then run Phase 2 pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.datasets.csv_ingest import DEFAULT_CSV, materialize_raw_from_csv
from ai.datasets.dataset_config import get_dataset_config
from ai.datasets.pipeline import run_dataset_pipeline
from backend.app.utils.logging_config import configure_logging


def main() -> int:
    config = get_dataset_config()
    configure_logging(config.project_root / "logs", "INFO")
    logger = logging.getLogger("maya.datasets.ingest_cli")

    csv_path = Path(
        __import__("os").getenv("MAYA_DATASET_CSV", str(DEFAULT_CSV))
    ).expanduser()
    logger.info("Using catalogue: %s", csv_path)
    logger.info("Raw output dir: %s", config.raw_dir)

    try:
        stats = materialize_raw_from_csv(csv_path, config)
        config.reports_dir.mkdir(parents=True, exist_ok=True)
        ingest_report = config.reports_dir / "ingest_report.json"
        ingest_report.write_text(
            json.dumps(
                {
                    "csv_path": str(csv_path),
                    "raw_dir": str(config.raw_dir),
                    "attempted": stats.attempted,
                    "saved": stats.saved,
                    "skipped_existing": stats.skipped_existing,
                    "failed": stats.failed,
                    "by_class_saved": stats.by_class_saved,
                    "by_class_failed": stats.by_class_failed,
                    "failures_preview": stats.failures[:100],
                    "failure_count": len(stats.failures),
                    "data_quality_note": (
                        "Catalogue labels FAKE_method=StyleGAN3, but URL hosts are "
                        "mostly avatar APIs (DiceBear/Multiavatar/randomuser/pravatar) "
                        "and REAL images are Unsplash URLs. Treat as given catalogue; "
                        "prefer a true deepfake corpus for production detection work."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote ingest report → %s", ingest_report)

        result = run_dataset_pipeline(config)
    except Exception:
        logger.exception("CSV ingest / pipeline failed")
        return 1

    logger.info("Validation passed: %s", result.validation_passed)
    return 0 if result.validation_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
