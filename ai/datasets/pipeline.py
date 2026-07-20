"""End-to-end dataset engineering pipeline orchestrator."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from ai.datasets.dataset_config import DatasetConfig, get_dataset_config
from ai.datasets.integrity import filter_readable_records, verify_integrity
from ai.datasets.inventory import build_inventory
from ai.datasets.preprocessing import materialize_processed_dataset
from ai.datasets.reports import (
    write_dataset_report,
    write_integrity_report,
    write_summary_csv,
    write_validation_report,
)
from ai.datasets.sampling import sample_balanced_splits
from ai.datasets.statistics import compute_statistics
from ai.datasets.validation import validate_processed_dataset
from ai.datasets.versioning import seal_dataset_version
from ai.datasets.visualization import save_class_distribution, save_sample_grid

logger = logging.getLogger("maya.datasets.pipeline")


@dataclass
class PipelineResult:
    """Handles to key artefacts produced by the pipeline."""

    dataset_report: Path
    integrity_report: Path
    validation_report: Path
    summary_csv: Path
    class_distribution_png: Path
    sample_grid_png: Path
    processed_dir: Path
    metadata_path: Path | None
    validation_passed: bool


def _reset_processed_dir(processed_dir: Path) -> None:
    """Remove previous processed outputs so runs are reproducible."""

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        logger.info("Cleared previous processed dataset at %s", processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)


def run_dataset_pipeline(
    config: DatasetConfig | None = None,
    *,
    reset_processed: bool = True,
) -> PipelineResult:
    """Execute inspect → integrity → analyze → sample → preprocess → validate → seal."""

    cfg = config or get_dataset_config()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Pipeline start version=%s seed=%s", cfg.dataset_version, cfg.random_seed)

    logger.info("STEP 1 — Inventory raw dataset at %s", cfg.raw_dir)
    inventory = build_inventory(cfg)
    if inventory.total_images == 0:
        raise FileNotFoundError(
            "No labeled images found under raw dataset root. "
            f"Place the Kaggle REAL/FAKE extract in `{cfg.raw_dir}` "
            "or set environment variable MAYA_RAW_DATASET_DIR."
        )

    logger.info("STEP 2 — Integrity verification")
    integrity = verify_integrity(inventory)
    readable = filter_readable_records(inventory, integrity)
    if not readable:
        raise RuntimeError("No readable images survived integrity checks.")

    logger.info("STEP 3 — Statistics + visualizations")
    stats = compute_statistics(readable, inventory, integrity)
    class_png = save_class_distribution(
        stats, cfg.reports_dir / "class_distribution.png"
    )
    grid_png = save_sample_grid(readable, cfg, cfg.reports_dir / "sample_grid.png")

    dataset_report = write_dataset_report(
        inventory,
        integrity,
        stats,
        cfg,
        cfg.reports_dir / "dataset_report.md",
    )
    integrity_report = write_integrity_report(
        integrity, cfg.reports_dir / "integrity_report.json"
    )
    summary_csv = write_summary_csv(stats, cfg.reports_dir / "dataset_summary.csv")

    logger.info(
        "STEP 4/5 — Sample balanced splits and preprocess to %sx%s RGB",
        cfg.image_size,
        cfg.image_size,
    )
    assignments = sample_balanced_splits(readable, cfg)
    if reset_processed:
        _reset_processed_dir(cfg.processed_dir)
    materialize_processed_dataset(assignments, cfg)

    logger.info("STEP 6 — Validate processed folders")
    validation = validate_processed_dataset(cfg)
    validation_report = write_validation_report(
        validation, cfg.reports_dir / "validation_report.md"
    )

    metadata_path: Path | None = None
    if validation.passed:
        logger.info("STEP 7 — Seal dataset version %s", cfg.dataset_version)
        metadata_path, _ = seal_dataset_version(cfg, validation)
    else:
        logger.warning("Skipping version seal because validation failed")

    result = PipelineResult(
        dataset_report=dataset_report,
        integrity_report=integrity_report,
        validation_report=validation_report,
        summary_csv=summary_csv,
        class_distribution_png=class_png,
        sample_grid_png=grid_png,
        processed_dir=cfg.processed_dir,
        metadata_path=metadata_path,
        validation_passed=validation.passed,
    )
    logger.info(
        "Pipeline completed. validation_passed=%s processed=%s metadata=%s",
        result.validation_passed,
        result.processed_dir,
        result.metadata_path,
    )
    return result
