"""Prediction consistency / reliability benchmarks.

Purpose
-------
Run the same image repeatedly and score prediction stability + confidence drift.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ai.benchmark.core import CoreInferencer

logger = logging.getLogger("maya.ai.benchmark.consistency")


@dataclass
class ConsistencyReport:
    """Reliability metrics for repeated inference on one image."""

    runs: int
    predictions: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    dominant_prediction: str = ""
    consistency_percentage: float = 0.0
    mean_confidence: float = 0.0
    confidence_std_dev: float = 0.0
    reliability_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_consistency(
    image_path: Path,
    inferencer: CoreInferencer,
    *,
    runs: int = 10,
) -> ConsistencyReport:
    """Repeat inference and compute consistency + reliability score."""

    if runs < 1:
        raise ValueError("runs must be >= 1")

    predictions: list[str] = []
    confidences: list[float] = []

    for _ in range(runs):
        outcome = inferencer.infer_once(image_path)
        predictions.append(outcome.decision.predicted_label)
        confidences.append(float(outcome.decision.confidence))

    counts = Counter(predictions)
    dominant, dominant_count = counts.most_common(1)[0]
    consistency_pct = 100.0 * dominant_count / runs
    mean_conf = float(statistics.mean(confidences))
    conf_std = float(statistics.pstdev(confidences)) if len(confidences) > 1 else 0.0

    # Reliability: consistency weighted with inverse confidence volatility
    # (std in percentage points; clamp contribution)
    volatility_penalty = min(conf_std / 50.0, 1.0)
    reliability = max(0.0, (consistency_pct / 100.0) * (1.0 - 0.5 * volatility_penalty))
    reliability_score = round(100.0 * reliability, 2)

    report = ConsistencyReport(
        runs=runs,
        predictions=predictions,
        confidences=confidences,
        dominant_prediction=dominant,
        consistency_percentage=round(consistency_pct, 2),
        mean_confidence=round(mean_conf, 4),
        confidence_std_dev=round(conf_std, 4),
        reliability_score=reliability_score,
    )
    logger.info(
        "Consistency %.1f%% dominant=%s reliability=%.1f",
        report.consistency_percentage,
        report.dominant_prediction,
        report.reliability_score,
    )
    return report
