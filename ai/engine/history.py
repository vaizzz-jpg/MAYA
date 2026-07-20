"""Epoch-wise metric history for MAYA training runs.

Why this module exists
----------------------
Decouple metric aggregation / CSV export from the training loop.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("maya.ai.engine.history")


@dataclass
class TrainingHistory:
    """Accumulate per-epoch metrics and export them."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def append(self, epoch: int, metrics: dict[str, float]) -> None:
        row = {"epoch": epoch, **metrics}
        self.rows.append(row)
        logger.debug("History epoch=%s metrics=%s", epoch, metrics)

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.rows)

    def best(self, metric: str, mode: str = "min") -> dict[str, Any] | None:
        if not self.rows or metric not in self.rows[0]:
            return None
        key = (lambda r: r[metric])
        return min(self.rows, key=key) if mode == "min" else max(self.rows, key=key)

    def export_csv(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("epoch\n", encoding="utf-8")
            logger.warning("History CSV written empty → %s", path)
            return path

        fieldnames = list(self.rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        logger.info("Wrote training history CSV → %s (%s rows)", path, len(self.rows))
        return path
