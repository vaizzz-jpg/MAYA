"""Model counterfactual perturbation experiments (Phase 4 Sprint 4.5).

Controlled region perturbations (mask / blur / neutral) — not generative
counterfactuals and not claims of real-world causation.
"""

from __future__ import annotations

from ai.explainability.counterfactual.experiments import CounterfactualEngine
from ai.explainability.counterfactual.result import CounterfactualExperiment, CounterfactualResult

__all__ = [
    "CounterfactualEngine",
    "CounterfactualExperiment",
    "CounterfactualResult",
]
