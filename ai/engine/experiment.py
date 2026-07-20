"""Sequential experiment JSON tracking for MAYA training runs.

Why this module exists
----------------------
Every training run must leave an auditable record (dataset version, model,
metrics, duration) without coupling the trainer to a specific UI or DB.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("maya.ai.engine.experiment")

_EXPERIMENT_RE = re.compile(r"experiment_(\d+)\.json$", re.IGNORECASE)


@dataclass
class ExperimentTracker:
    """Create ``experiment_XXX.json`` files under an experiment directory."""

    experiment_dir: Path
    records: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.experiment_dir = Path(self.experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def next_path(self) -> Path:
        """Allocate the next sequential experiment file path."""

        highest = 0
        for path in self.experiment_dir.glob("experiment_*.json"):
            match = _EXPERIMENT_RE.search(path.name)
            if match:
                highest = max(highest, int(match.group(1)))
        return self.experiment_dir / f"experiment_{highest + 1:03d}.json"

    def write(
        self,
        *,
        profile: str,
        dataset_version: str,
        model_name: str,
        hyperparameters: dict[str, Any],
        metrics: dict[str, Any],
        duration_seconds: float,
        notes: str = "",
        extras: dict[str, Any] | None = None,
    ) -> Path:
        """Persist one experiment record and return its path."""

        path = self.next_path()
        payload: dict[str, Any] = {
            "experiment_id": path.stem,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
            "dataset_version": dataset_version,
            "model": model_name,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "training_duration_seconds": round(duration_seconds, 3),
            "notes": notes,
        }
        if extras:
            payload["extras"] = extras

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.records.append(path)
        logger.info("Experiment record written → %s", path)
        return path


def read_dataset_version(project_root: Path) -> str:
    """Best-effort read of active dataset version from versions/CURRENT."""

    pointer = project_root / "dataset" / "versions" / "CURRENT"
    if pointer.exists():
        return pointer.read_text(encoding="utf-8").strip() or "unknown"
    return "unknown"
