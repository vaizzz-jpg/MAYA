"""Explanation fusion, trust scoring, and XAI audit (Phase 4 Sprint 4.5)."""

from __future__ import annotations

from ai.explainability.fusion.audit import XaiAuditRecord, build_audit_record
from ai.explainability.fusion.fusion import fuse_explanations
from ai.explainability.fusion.result import (
    AdvancedInvestigatorSummary,
    ExplanationTrustResult,
    FusedExplanationResult,
)
from ai.explainability.fusion.trust import TrustWeights, compute_trust_score

__all__ = [
    "AdvancedInvestigatorSummary",
    "ExplanationTrustResult",
    "FusedExplanationResult",
    "TrustWeights",
    "XaiAuditRecord",
    "build_audit_record",
    "compute_trust_score",
    "fuse_explanations",
]
