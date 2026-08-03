"""Cross-explainer agreement for the benchmark — reuses Sprint 4.3 consistency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ai.explainability.analytics.consistency import (
    ConsistencyMetrics,
    compute_consistency_metrics,
)


@dataclass(frozen=True)
class AgreementResult:
    """Suite-level agreement plus a dense similarity matrix."""

    consistency: ConsistencyMetrics
    matrix: dict[str, dict[str, float]]  # cosine similarity name→name
    per_explainer_mean_agreement: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "consistency": self.consistency.to_dict(),
            "matrix": self.matrix,
            "per_explainer_mean_agreement": self.per_explainer_mean_agreement,
        }


def compute_agreement(heatmaps: dict[str, np.ndarray]) -> AgreementResult:
    """Build consistency metrics and a full cosine agreement matrix."""

    consistency = compute_consistency_metrics(heatmaps)
    names = list(consistency.explainer_names) or sorted(heatmaps.keys())
    matrix: dict[str, dict[str, float]] = {n: {} for n in names}

    # Fill from pairwise cosine; diagonal = 1
    pair_lookup = {
        (p.explainer_a, p.explainer_b): p.cosine_similarity
        for p in consistency.pairwise
    }
    for a in names:
        for b in names:
            if a == b:
                matrix[a][b] = 1.0
            elif (a, b) in pair_lookup:
                matrix[a][b] = pair_lookup[(a, b)]
            elif (b, a) in pair_lookup:
                matrix[a][b] = pair_lookup[(b, a)]
            else:
                matrix[a][b] = 0.0

    per_mean: dict[str, float] = {}
    for name in names:
        others = [matrix[name][o] for o in names if o != name]
        per_mean[name] = round(float(np.mean(others)), 6) if others else 1.0

    return AgreementResult(
        consistency=consistency,
        matrix=matrix,
        per_explainer_mean_agreement=per_mean,
    )
