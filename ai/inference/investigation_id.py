"""Sequential investigation ID generator (INV-YYYY-NNNNNN).

Purpose
-------
Provide a single reusable module for forensic case-style IDs without
scattering counter logic across pipelines.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("maya.ai.inference.investigation_id")


class InvestigationIDGenerator:
    """Allocate sequential IDs persisted on disk for crash-safe continuity."""

    def __init__(self, state_path: Path, *, year: int | None = None) -> None:
        self.state_path = Path(state_path)
        self.year = year or datetime.now().year
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_state(self) -> dict:
        if not self.state_path.exists():
            return {"year": self.year, "counter": 0}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(data.get("year", self.year)) != self.year:
                return {"year": self.year, "counter": 0}
            return {"year": self.year, "counter": int(data.get("counter", 0))}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Corrupt ID state %s (%s) — resetting", self.state_path, exc)
            return {"year": self.year, "counter": 0}

    def _write_state(self, counter: int) -> None:
        payload = {"year": self.year, "counter": counter}
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def next_id(self) -> str:
        """Return the next ID in the form ``INV-2026-000001``."""

        state = self._read_state()
        counter = int(state["counter"]) + 1
        self._write_state(counter)
        investigation_id = f"INV-{self.year}-{counter:06d}"
        logger.info("Allocated investigation id %s", investigation_id)
        return investigation_id
