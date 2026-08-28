"""Deletion/occlusion faithfulness evaluation against an importance map."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ai.explainability.config import FaithfulnessSettings
from ai.explainability.faithfulness.result import FaithfulnessResult, FaithfulnessStep
from ai.explainability.faithfulness.visualization import write_faithfulness_comparison
from ai.explainability.perturbation import (
    build_cumulative_mask,
    importance_ranking,
    perturb_tensor,
)
from ai.inference.confidence import decide_from_probabilities
from ai.inference.inference_config import InferenceConfig
from ai.inference.predictor import predict_probabilities
from ai.inference.utils import timing_ms

logger = logging.getLogger("maya.ai.explainability.faithfulness.evaluator")


class FaithfulnessEngine:
    """Measure model sensitivity to important vs less-important regions."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        settings: FaithfulnessSettings | None = None,
        inference_config: InferenceConfig | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.settings = settings or FaithfulnessSettings()
        self.inference_config = inference_config or InferenceConfig()
        self.model.eval()

    def evaluate(
        self,
        input_tensor: torch.Tensor,
        importance_map: np.ndarray,
        *,
        investigation_id: str,
        image_name: str = "",
        target_class: int | None = None,
        artifact_dir: Path | None = None,
        write_visuals: bool = True,
    ) -> FaithfulnessResult:
        """Run stepwise important vs unimportant occlusion comparison."""

        if not self.settings.enabled:
            raise RuntimeError("Faithfulness evaluation is disabled in configuration")
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected input shape (1, C, H, W), got {tuple(input_tensor.shape)}"
            )
        importance = np.asarray(importance_map, dtype=np.float64)
        if importance.ndim != 2:
            raise ValueError(f"importance_map must be 2-D, got {importance.shape}")
        if importance.shape != tuple(input_tensor.shape[-2:]):
            # Resize importance to tensor spatial size
            from PIL import Image

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

            ranked_desc = importance_ranking(importance)
            ranked_asc = ranked_desc[::-1]
            h, w = importance.shape
            total = h * w
            frac = float(self.settings.fraction_per_step)
            steps = int(self.settings.steps)

            step_results: list[FaithfulnessStep] = []
            important_drops: list[float] = []
            unimportant_drops: list[float] = []
            last_imp_tensor = input_tensor
            last_unimp_tensor = input_tensor

            for step in range(1, steps + 1):
                count = max(1, int(round(total * frac * step)))
                count = min(count, total)
                imp_mask = build_cumulative_mask(
                    (h, w), ranked_desc, count=count
                )
                unimp_mask = build_cumulative_mask(
                    (h, w), ranked_asc, count=count
                )
                imp_tensor = perturb_tensor(
                    input_tensor,
                    imp_mask,
                    method=self.settings.perturbation_method,
                )
                unimp_tensor = perturb_tensor(
                    input_tensor,
                    unimp_mask,
                    method=self.settings.perturbation_method,
                )
                imp_probs = predict_probabilities(
                    self.model, imp_tensor, self.device
                )
                unimp_probs = predict_probabilities(
                    self.model, unimp_tensor, self.device
                )
                imp_score = float(imp_probs[0, class_idx].item())
                unimp_score = float(unimp_probs[0, class_idx].item())
                imp_drop = original_score - imp_score
                unimp_drop = original_score - unimp_score
                important_drops.append(imp_drop)
                unimportant_drops.append(unimp_drop)
                step_results.append(
                    FaithfulnessStep(
                        step=step,
                        fraction_masked=round(count / float(total), 6),
                        important_score=round(imp_score, 8),
                        unimportant_score=round(unimp_score, 8),
                        important_drop=round(imp_drop, 8),
                        unimportant_drop=round(unimp_drop, 8),
                    )
                )
                last_imp_tensor = imp_tensor
                last_unimp_tensor = unimp_tensor
                del imp_probs, unimp_probs

            # Deletion score: mean drop when removing important regions
            deletion_score = float(np.mean(important_drops)) if important_drops else 0.0
            # Insertion proxy: relative advantage of important vs unimportant deletion
            mean_imp = float(np.mean(important_drops)) if important_drops else 0.0
            mean_unimp = (
                float(np.mean(unimportant_drops)) if unimportant_drops else 0.0
            )
            denom = abs(mean_imp) + abs(mean_unimp) + 1e-8
            insertion_proxy = (mean_imp - mean_unimp) / denom
            # Faithfulness in [0, 1]: higher when important occlusion hurts more
            raw = (mean_imp - mean_unimp + 1.0) / 2.0
            faithfulness = float(max(0.0, min(1.0, raw)))

            conf_drop_imp = round(100.0 * mean_imp, 4)
            conf_drop_unimp = round(100.0 * mean_unimp, 4)

            if faithfulness >= 0.7:
                level = "High measured influence"
            elif faithfulness >= 0.45:
                level = "Moderate measured influence"
            else:
                level = "Low measured influence"

            interpretation = (
                f"{level} of the highlighted regions on the class score. "
                f"Important-region {self.settings.perturbation_method} "
                f"changed the target-class score by about {conf_drop_imp:.2f} "
                f"percentage points on average across {steps} steps, versus "
                f"{conf_drop_unimp:.2f} for less-important regions. "
                "This measures model sensitivity under perturbation — "
                "not proof that the model causally 'looked at' those pixels."
            )

            viz_path = ""
            if write_visuals and artifact_dir is not None:
                viz_path = str(
                    write_faithfulness_comparison(
                        step_results=step_results,
                        artifact_dir=Path(artifact_dir),
                        original_score=original_score,
                    )
                )

            result = FaithfulnessResult(
                investigation_id=investigation_id,
                image_name=image_name,
                prediction=decision.predicted_label,
                original_confidence=decision.confidence,
                original_score=round(original_score, 8),
                target_class=class_idx,
                perturbation_method=self.settings.perturbation_method,
                steps=steps,
                deletion_score=round(deletion_score, 8),
                insertion_proxy_score=round(float(insertion_proxy), 8),
                faithfulness_score=round(faithfulness, 6),
                confidence_drop_important=conf_drop_imp,
                confidence_drop_unimportant=conf_drop_unimp,
                step_results=step_results,
                visualization_path=viz_path,
                generation_time_ms=0.0,
                timestamp=FaithfulnessResult.utc_now(),
                interpretation=interpretation,
            )
            del last_imp_tensor, last_unimp_tensor

        result.generation_time_ms = round(timer["ms"], 2)
        logger.info(
            "Faithfulness id=%s score=%.3f deletion=%.4f",
            investigation_id,
            result.faithfulness_score,
            result.deletion_score,
        )
        return result
