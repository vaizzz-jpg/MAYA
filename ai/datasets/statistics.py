"""Dataset statistical analysis.

Purpose:
    Produce class distribution and geometry statistics using streaming I/O.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
from PIL import Image, UnidentifiedImageError

from ai.datasets.integrity import IntegrityReport
from ai.datasets.inventory import ImageRecord, InventoryResult

logger = logging.getLogger("maya.datasets.statistics")


@dataclass
class DatasetStatistics:
    """Numeric summaries for reporting and plots."""

    class_counts: Counter = field(default_factory=Counter)
    format_counts: Counter = field(default_factory=Counter)
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    aspect_ratios: list[float] = field(default_factory=list)
    failed_reads: int = 0

    def to_frame(self) -> pd.DataFrame:
        """Build a compact summary table."""

        rows: list[dict] = []
        for cls, count in sorted(self.class_counts.items()):
            rows.append({"metric": "class_count", "key": cls, "value": count})
        for fmt, count in sorted(self.format_counts.items()):
            rows.append({"metric": "format_count", "key": fmt, "value": count})

        if self.widths:
            series_w = pd.Series(self.widths, dtype="float64")
            series_h = pd.Series(self.heights, dtype="float64")
            series_a = pd.Series(self.aspect_ratios, dtype="float64")
            for name, series in (
                ("width", series_w),
                ("height", series_h),
                ("aspect_ratio", series_a),
            ):
                rows.append({"metric": f"{name}_min", "key": name, "value": float(series.min())})
                rows.append({"metric": f"{name}_max", "key": name, "value": float(series.max())})
                rows.append({"metric": f"{name}_mean", "key": name, "value": float(series.mean())})
                rows.append(
                    {"metric": f"{name}_median", "key": name, "value": float(series.median())}
                )

        rows.append(
            {"metric": "dimension_samples", "key": "n", "value": len(self.widths)}
        )
        rows.append(
            {"metric": "failed_dimension_reads", "key": "n", "value": self.failed_reads}
        )
        return pd.DataFrame(rows)


def compute_statistics(
    readable_records: list[ImageRecord],
    inventory: InventoryResult,
    integrity: IntegrityReport,
    *,
    max_dimension_scans: int = 5000,
) -> DatasetStatistics:
    """Compute class/format/geometry statistics incrementally."""

    stats = DatasetStatistics()
    stats.class_counts = Counter(integrity.readable_by_class)
    stats.format_counts = Counter(
        record.extension
        for record in readable_records
    )

    # Prefer previously sampled dims from inventory when available
    for width, height in inventory.dimension_samples:
        stats.widths.append(width)
        stats.heights.append(height)
        if height:
            stats.aspect_ratios.append(width / height)

    remaining = max(0, max_dimension_scans - len(stats.widths))
    if remaining == 0:
        return stats

    scanned = 0
    for record in readable_records:
        if scanned >= remaining:
            break
        try:
            with Image.open(record.path) as img:
                width, height = img.size
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            stats.failed_reads += 1
            logger.debug("Stats read failed for %s: %s", record.path, exc)
            continue

        stats.widths.append(width)
        stats.heights.append(height)
        if height:
            stats.aspect_ratios.append(width / height)
        scanned += 1

    logger.info(
        "Statistics ready: classes=%s dim_samples=%s",
        dict(stats.class_counts),
        len(stats.widths),
    )
    return stats
