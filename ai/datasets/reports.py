"""Report writers for Phase 2 dataset engineering outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ai.datasets.dataset_config import DatasetConfig
from ai.datasets.integrity import IntegrityReport
from ai.datasets.inventory import InventoryResult
from ai.datasets.statistics import DatasetStatistics
from ai.datasets.validation import ValidationReport

logger = logging.getLogger("maya.datasets.reports")


def write_dataset_report(
    inventory: InventoryResult,
    integrity: IntegrityReport,
    stats: DatasetStatistics,
    config: DatasetConfig,
    output_path: Path,
) -> Path:
    """Write human-readable dataset inspection report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dim_note = "n/a"
    if stats.widths:
        dim_note = (
            f"width[{min(stats.widths)}–{max(stats.widths)}] "
            f"height[{min(stats.heights)}–{max(stats.heights)}] "
            f"samples={len(stats.widths)}"
        )

    lines = [
        "# MAYA Dataset Report",
        "",
        "## Source",
        "",
        f"- Root: `{inventory.root}`",
        f"- Folders scanned: **{inventory.folders_scanned}**",
        f"- Class folders discovered: **{len(inventory.class_folders)}**",
        "",
        "### Class folders",
        "",
    ]
    if inventory.class_folders:
        for name, path in inventory.class_folders.items():
            lines.append(f"- `{name}` → `{path}`")
    else:
        lines.append("- None discovered (check `MAYA_RAW_DATASET_DIR` / `dataset/raw`)")

    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Total inventoried images: **{inventory.total_images}**",
            f"- Readable images: **{integrity.readable}**",
            f"- Corrupted: **{len(integrity.corrupted)}**",
            f"- Zero-byte: **{len(integrity.zero_byte)}**",
            f"- Unreadable: **{len(integrity.unreadable)}**",
            f"- Duplicate filenames: **{len(integrity.duplicate_filenames)}**",
            "",
            "### Images by class (readable)",
            "",
        ]
    )
    for cls, count in sorted(stats.class_counts.items()):
        lines.append(f"- **{cls}**: {count}")

    lines.extend(
        [
            "",
            "## Formats",
            "",
        ]
    )
    for fmt, count in sorted(stats.format_counts.items()):
        lines.append(f"- `{fmt}`: {count}")

    lines.extend(
        [
            "",
            "## Geometry (sampled)",
            "",
            f"- {dim_note}",
            "",
            "## Target processed corpus",
            "",
            f"- Seed: `{config.random_seed}`",
            f"- Image size: `{config.image_size}×{config.image_size}`",
            f"- Target total images: **{config.total_target_images}**",
            "",
            "| Split | REAL | FAKE |",
            "|-------|------|------|",
            f"| train | {config.train_budget.real} | {config.train_budget.fake} |",
            f"| validation | {config.validation_budget.real} | {config.validation_budget.fake} |",
            f"| test | {config.test_budget.real} | {config.test_budget.fake} |",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote dataset report → %s", output_path)
    return output_path


def write_integrity_report(report: IntegrityReport, output_path: Path) -> Path:
    """Write machine-readable integrity JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Wrote integrity report → %s", output_path)
    return output_path


def write_validation_report(report: ValidationReport, output_path: Path) -> Path:
    """Write processed-folder validation markdown."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MAYA Processed Dataset Validation Report",
        "",
        f"- Passed: **{report.passed}**",
        f"- Balanced REAL/FAKE totals: **{report.balanced}**",
        f"- Total images counted: **{report.total_images}**",
        "",
        "## Folder checks",
        "",
        "| Path | Expected | Actual | OK | Notes |",
        "|------|----------|--------|----|-------|",
    ]
    for check in report.checks:
        lines.append(
            f"| `{check.path}` | {check.expected} | {check.actual} | "
            f"{check.ok} | {check.notes} |"
        )

    if report.missing_folders:
        lines.extend(["", "## Missing folders", ""])
        lines.extend(f"- `{path}`" for path in report.missing_folders)

    if report.empty_folders:
        lines.extend(["", "## Empty folders", ""])
        lines.extend(f"- `{path}`" for path in report.empty_folders)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote validation report → %s", output_path)
    return output_path


def write_summary_csv(stats: DatasetStatistics, output_path: Path) -> Path:
    """Write tabular summary for spreadsheets / appendices."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame: pd.DataFrame = stats.to_frame()
    frame.to_csv(output_path, index=False)
    logger.info("Wrote summary CSV → %s", output_path)
    return output_path
