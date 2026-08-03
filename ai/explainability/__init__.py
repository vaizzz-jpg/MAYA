"""MAYA Explainable Intelligence Engine (Phase 4).

Sprint 4.1: Grad-CAM foundation.
Sprint 4.2: Plugin Multi-Explainer Framework.
Sprint 4.3: Explanation Analytics & Trust Engine.
Sprint 4.4: Explainability Validation & Benchmark Suite.
"""

from __future__ import annotations

from ai.explainability.analytics import (
    AnalyticsConfig,
    ExplanationAnalytics,
    ExplanationAnalyticsEngine,
)
from ai.explainability.base import BaseExplainer, Explainer, ExplainerOutput
from ai.explainability.benchmark import (
    ExplainabilityBenchmarkConfig,
    ExplainabilityBenchmarkSuite,
)
from ai.explainability.comparison import ComparisonResult
from ai.explainability.config import ExplainabilityConfig, get_explainability_config
from ai.explainability.engine import ExplainabilityEngine
from ai.explainability.explanation import ExplanationResult
from ai.explainability.explainers import (
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    ScoreCAM,
)
from ai.explainability.registry import (
    DEFAULT_EXPLAINER_NAMES,
    ExplainerRegistry,
    register_default_explainers,
)
from ai.explainability.target_layer import (
    TargetLayerResolver,
    UnsupportedModelForExplainability,
    register_default_target_layers,
)

register_default_target_layers()
register_default_explainers()

__all__ = [
    "AnalyticsConfig",
    "BaseExplainer",
    "ComparisonResult",
    "DEFAULT_EXPLAINER_NAMES",
    "EigenCAM",
    "Explainer",
    "ExplainerOutput",
    "ExplainerRegistry",
    "ExplainabilityBenchmarkConfig",
    "ExplainabilityBenchmarkSuite",
    "ExplainabilityConfig",
    "ExplainabilityEngine",
    "ExplanationAnalytics",
    "ExplanationAnalyticsEngine",
    "ExplanationResult",
    "GradCAM",
    "GradCAMPlusPlus",
    "LayerCAM",
    "ScoreCAM",
    "TargetLayerResolver",
    "UnsupportedModelForExplainability",
    "get_explainability_config",
    "register_default_explainers",
    "register_default_target_layers",
]
