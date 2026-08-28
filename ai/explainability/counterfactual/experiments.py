"""Controlled model counterfactual perturbation experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from ai.explainability.config import CounterfactualSettings
from ai.explainability.counterfactual.result import (
    CounterfactualExperiment,
    CounterfactualResult,
)
from ai.explainability.counterfactual.visualization import write_counterfactual_comparison
from ai.explainability.perturbation import important_region_mask, perturb_tensor
from ai.inference.confidence import decide_from_probabilities
from ai.inference.inference_config import InferenceConfig
from ai.inference.predictor import predict_probabilities
from ai.inference.utils import timing_ms

logger = logging.getLogger("maya.ai.explainability.counterfactual.experiments")


def _region_description(mask: np.ndarray) -> str:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return "empty-region"
    coverage = 100.0 * float(mask.mean())
    return (
        f"bbox[r{int(rows.min())}:{int(rows.max())},"
        f"c{int(cols.min())}:{int(cols.max())}] "
        f"coverage={coverage:.1f}%"
    )


class CounterfactualEngine:
    """Run mask / blur / neutral perturbations on important regions."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        settings: CounterfactualSettings | None = None,
        inference_config: InferenceConfig | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.settings = settings or CounterfactualSettings()
        self.inference_config = inference_config or InferenceConfig()
        self.model.eval()

    def run(
        self,
        input_tensor: torch.Tensor,
        importance_map: np.ndarray,
        *,
        investigation_id: str,
        image_name: str = "",
        original_image: Image.Image | None = None,
        target_class: int | None = None,
        artifact_dir: Path | None = None,
        write_visuals: bool = True,
    ) -> CounterfactualResult:
        if not self.settings.enabled:
            raise RuntimeError("Counterfactual analysis is disabled in configuration")
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected input shape (1, C, H, W), got {tuple(input_tensor.shape)}"
            )

        importance = np.asarray(importance_map, dtype=np.float64)
        if importance.ndim != 2:
            raise ValueError(f"importance_map must be 2-D, got {importance.shape}")
        if importance.shape != tuple(input_tensor.shape[-2:]):
            importance = np.asarray(
                Image.fromarray(importance.astype(np.float32), mode="F").resize(
                    (input_tensor.shape[-1], input_tensor.shape[-2]),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float64,
            )

        with timing_ms() as timer:
            probs = predict_probabilities(self.model, input_tensor, self.device)
            decision = decide_from_probabilities(probs[0], self.inference_config)
            class_idx = (
                int(target_class)
                if target_class is not None
                else int(decision.predicted_class)
            )
            original_score = float(probs[0, class_idx].item())
            mask = important_region_mask(
                importance, threshold=self.settings.importance_threshold
            )
            region = _region_description(mask)
            now = CounterfactualResult.utc_now()

            experiments: list[CounterfactualExperiment] = []
            preview_tensors: dict[str, torch.Tensor] = {}

            for method in self.settings.perturbation_methods:
                try:
                    perturbed = perturb_tensor(
                        input_tensor,
                        mask,
                        method=method,
                        blur_radius=self.settings.blur_radius,
                        neutral_value=self.settings.neutral_value,
                    )
                except Exception as exc:
                    logger.exception("Counterfactual perturbation failed (%s)", method)
                    raise RuntimeError(
                        f"Counterfactual perturbation '{method}' failed: {exc}"
                    ) from exc

                pert_probs = predict_probabilities(
                    self.model, perturbed, self.device
                )
                pert_decision = decide_from_probabilities(
                    pert_probs[0], self.inference_config
                )
                pert_score = float(pert_probs[0, class_idx].item())
                conf_delta = round(
                    pert_decision.confidence - decision.confidence, 4
                )
                experiments.append(
                    CounterfactualExperiment(
                        perturbation_strategy=method,
                        affected_region=region,
                        original_prediction=decision.predicted_label,
                        original_confidence=decision.confidence,
                        perturbed_prediction=pert_decision.predicted_label,
                        perturbed_confidence=pert_decision.confidence,
                        confidence_delta=conf_delta,
                        prediction_changed=(
                            pert_decision.predicted_label != decision.predicted_label
                        ),
                        original_score=round(original_score, 8),
                        perturbed_score=round(pert_score, 8),
                        score_delta=round(pert_score - original_score, 8),
                        timestamp=now,
                    )
                )
                preview_tensors[method] = perturbed
                del pert_probs

            # Interpretation from the strongest confidence drop (most sensitive)
            if experiments:
                strongest = min(experiments, key=lambda e: e.confidence_delta)
                interpretation = (
                    f"Model counterfactual perturbation experiments show that "
                    f"'{strongest.perturbation_strategy}' on the important region "
                    f"changed confidence by {strongest.confidence_delta:.2f} "
                    f"percentage points "
                    f"({decision.predicted_label} "
                    f"{decision.confidence:.1f}% → "
                    f"{strongest.perturbed_prediction} "
                    f"{strongest.perturbed_confidence:.1f}%). "
                    "This is evidence of model sensitivity under controlled "
                    "perturbation, not proof of real-world causation."
                )
            else:
                interpretation = "No counterfactual experiments were executed."

            viz_path = ""
            if write_visuals and artifact_dir is not None:
                if original_image is None:
                    raise ValueError(
                        "original_image is required to write "
                        "counterfactual_comparison.png"
                    )
                viz_path = str(
                    write_counterfactual_comparison(
                        original_image=original_image,
                        input_tensor=input_tensor,
                        preview_tensors=preview_tensors,
                        experiments=experiments,
                        artifact_dir=Path(artifact_dir),
                    )
                )

            result = CounterfactualResult(
                investigation_id=investigation_id,
                image_name=image_name,
                original_prediction=decision.predicted_label,
                original_confidence=decision.confidence,
                target_class=class_idx,
                experiments=experiments,
                visualization_path=viz_path,
                generation_time_ms=0.0,
                timestamp=now,
                interpretation=interpretation,
            )
            del preview_tensors

        result.generation_time_ms = round(timer["ms"], 2)
        logger.info(
            "Counterfactual id=%s experiments=%s",
            investigation_id,
            len(result.experiments),
        )
        return result
