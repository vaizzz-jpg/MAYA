"""Confidence / threshold decision layer for MAYA predictions.

Purpose
-------
Convert raw class probabilities into investigator-facing confidence metadata
using configurable bands and decision threshold (never hardcoded call-sites).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from ai.inference.inference_config import ConfidenceBands, InferenceConfig, get_inference_config

logger = logging.getLogger("maya.ai.inference.confidence")


@dataclass(frozen=True)
class ConfidenceDecision:
    """Thresholded class decision plus confidence metadata."""

    predicted_class: int
    predicted_label: str
    confidence: float  # 0–100
    confidence_level: str
    real_probability: float
    fake_probability: float
    threshold: float
    threshold_decision: str  # e.g. "FAKE (>= threshold)" / "REAL (< threshold)"


def confidence_level_from_score(
    confidence_fraction: float,
    bands: ConfidenceBands,
) -> str:
    """Map a 0–1 confidence score to a named level."""

    if confidence_fraction >= bands.very_high:
        return "Very High"
    if confidence_fraction >= bands.high:
        return "High"
    if confidence_fraction >= bands.medium:
        return "Medium"
    return "Low"


def decide_from_probabilities(
    probabilities: torch.Tensor | list[float],
    config: InferenceConfig | None = None,
) -> ConfidenceDecision:
    """Apply threshold on P(positive_class) and compute confidence fields."""

    cfg = config or get_inference_config()
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().flatten().tolist()
    else:
        probs = [float(p) for p in probabilities]

    if len(probs) < 2:
        raise ValueError("Expected at least 2 class probabilities (REAL, FAKE)")

    real_p = float(probs[0])
    fake_p = float(probs[1])
    pos = cfg.positive_class
    pos_p = float(probs[pos])
    threshold = cfg.threshold

    if pos_p >= threshold:
        pred = pos
        decision = f"{cfg.class_names[pos]} (>= threshold {threshold})"
    else:
        pred = 0 if pos == 1 else 1
        decision = f"{cfg.class_names[pred]} (< threshold {threshold})"

    # Confidence = probability of the predicted class
    conf_frac = float(probs[pred])
    level = confidence_level_from_score(conf_frac, cfg.confidence_bands)

    logger.info(
        "Decision=%s confidence=%.2f%% level=%s",
        cfg.class_names[pred],
        conf_frac * 100.0,
        level,
    )
    return ConfidenceDecision(
        predicted_class=pred,
        predicted_label=cfg.class_names[pred],
        confidence=round(conf_frac * 100.0, 4),
        confidence_level=level,
        real_probability=real_p,
        fake_probability=fake_p,
        threshold=threshold,
        threshold_decision=decision,
    )
