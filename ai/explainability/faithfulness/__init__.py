"""Explanation faithfulness evaluation (Phase 4 Sprint 4.5).

Measures model sensitivity to highlighted regions via controlled occlusion —
not proof that the model 'looked at' those regions.
"""

from __future__ import annotations

from ai.explainability.faithfulness.evaluator import FaithfulnessEngine
from ai.explainability.faithfulness.result import FaithfulnessResult

__all__ = [
    "FaithfulnessEngine",
    "FaithfulnessResult",
]
