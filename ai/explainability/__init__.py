"""MAYA Explainable Intelligence Engine (Phase 4).

Sprint 4.1: Grad-CAM foundation.
Sprint 4.2: Plugin Multi-Explainer Framework.
Sprint 4.3: Explanation Analytics & Trust Engine.
Sprint 4.4: Explainability Validation & Benchmark Suite.
Sprint 4.5: Advanced Explainability & Trust Layer (SHAP, faithfulness,
            counterfactual, fusion, audit).
"""

from __future__ import annotations

from ai.explainability.advanced_engine import AdvancedExplainabilityEngine
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
from ai.explainability.config import (
    AdvancedXAIConfig,
    CounterfactualSettings,
    ExplainabilityConfig,
    FaithfulnessSettings,
    ShapSettings,
    TrustScoreWeights,
    get_advanced_xai_config,
    get_explainability_config,
)
from ai.explainability.counterfactual import (
    CounterfactualEngine,
    CounterfactualExperiment,
    CounterfactualResult,
)
from ai.explainability.engine import ExplainabilityEngine
from ai.explainability.explanation import ExplanationResult
from ai.explainability.explainers import (
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    ScoreCAM,
)
from ai.explainability.faithfulness import FaithfulnessEngine, FaithfulnessResult
from ai.explainability.fusion import (
    AdvancedInvestigatorSummary,
    ExplanationTrustResult,
    FusedExplanationResult,
    TrustWeights,
    XaiAuditRecord,
    build_audit_record,
    compute_trust_score,
    fuse_explanations,
)
from ai.explainability.registry import (
    DEFAULT_EXPLAINER_NAMES,
    ExplainerRegistry,
    register_default_explainers,
)
from ai.explainability.shap import (
    ShapEngine,
    ShapResult,
    ShapUnavailableError,
)
from ai.explainability.target_layer import (
    TargetLayerResolver,
    UnsupportedModelForExplainability,
    register_default_target_layers,
)

register_default_target_layers()
register_default_explainers()

__all__ = [
    "AdvancedExplainabilityEngine",
    "AdvancedInvestigatorSummary",
    "AdvancedXAIConfig",
    "AnalyticsConfig",
    "BaseExplainer",
    "ComparisonResult",
    "CounterfactualEngine",
    "CounterfactualExperiment",
    "CounterfactualResult",
    "CounterfactualSettings",
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
    "ExplanationTrustResult",
    "FaithfulnessEngine",
    "FaithfulnessResult",
    "FaithfulnessSettings",
    "FusedExplanationResult",
    "GradCAM",
    "GradCAMPlusPlus",
    "LayerCAM",
    "ScoreCAM",
    "ShapEngine",
    "ShapResult",
    "ShapSettings",
    "ShapUnavailableError",
    "TargetLayerResolver",
    "TrustScoreWeights",
    "TrustWeights",
    "UnsupportedModelForExplainability",
    "XaiAuditRecord",
    "build_audit_record",
    "compute_trust_score",
    "fuse_explanations",
    "get_advanced_xai_config",
    "get_explainability_config",
    "register_default_explainers",
    "register_default_target_layers",
]
