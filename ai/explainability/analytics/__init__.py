"""Explanation Analytics & Trust Engine (Phase 4 Sprint 4.3).

Consumes explanation heatmaps; does not generate Grad-CAM or other CAMs.
"""

from __future__ import annotations

from ai.explainability.analytics.analyzer import (
    AnalyticsConfig,
    ExplanationAnalytics,
    ExplanationAnalyticsEngine,
    load_heatmap_image,
)
from ai.explainability.analytics.confidence import ConfidenceBreakdown
from ai.explainability.analytics.consistency import ConsistencyMetrics
from ai.explainability.analytics.explanation_quality import ExplanationQuality
from ai.explainability.analytics.focus_metrics import FocusMetrics
from ai.explainability.analytics.localization import LocalizationMetrics
from ai.explainability.analytics.statistics import ActivationStatistics
from ai.explainability.analytics.summary import InvestigatorSummary

__all__ = [
    "ActivationStatistics",
    "AnalyticsConfig",
    "ConfidenceBreakdown",
    "ConsistencyMetrics",
    "ExplanationAnalytics",
    "ExplanationAnalyticsEngine",
    "ExplanationQuality",
    "FocusMetrics",
    "InvestigatorSummary",
    "LocalizationMetrics",
    "load_heatmap_image",
]
