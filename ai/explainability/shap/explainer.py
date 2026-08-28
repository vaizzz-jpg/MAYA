"""SHAP attribution engine for MAYA image classifiers.

Uses the maintained ``shap`` library with an image masker. Designed for
CPU-first EfficientNet-B0 with a hard evaluation budget.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from ai.datasets.dataset_config import IMAGENET_MEAN, IMAGENET_STD
from ai.explainability.config import AdvancedXAIConfig, ColorMap, ShapSettings
from ai.explainability.shap.masker import (
    ShapDependencyError,
    build_image_masker,
    require_shap_image_stack,
)
from ai.explainability.shap.processor import (
    compute_attribution_stats,
    extract_class_attribution,
    split_signed_maps,
)
from ai.explainability.shap.result import ShapResult
from ai.explainability.shap.visualization import write_shap_visualizations
from ai.inference.confidence import decide_from_probabilities
from ai.inference.inference_config import InferenceConfig
from ai.inference.predictor import predict_probabilities
from ai.inference.utils import timing_ms

logger = logging.getLogger("maya.ai.explainability.shap.explainer")

# Re-export for callers / tests
ShapUnavailableError = ShapDependencyError


def _device_label(device: object) -> str:
    name = str(device).upper()
    if name.startswith("CUDA"):
        return "CUDA"
    if name.startswith("CPU"):
        return "CPU"
    return name


def _tensor_to_hwc01(tensor: torch.Tensor) -> np.ndarray:
    """Convert ImageNet-normalized (1,C,H,W) tensor to HWC float in [0, 1]."""

    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"Expected tensor (1,C,H,W), got {tuple(tensor.shape)}")
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(
        1, 3, 1, 1
    )
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(
        1, 3, 1, 1
    )
    denorm = tensor * std + mean
    denorm = denorm.clamp(0.0, 1.0)[0].detach().cpu().permute(1, 2, 0).numpy()
    return np.asarray(denorm, dtype=np.float32)


def _hwc01_batch_to_tensor(
    images_nhwc: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Convert NHWC [0,1] float images to ImageNet-normalized NCHW tensors."""

    arr = np.asarray(images_nhwc, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected NHWC RGB batch, got {arr.shape}")
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(device)


class ShapEngine:
    """Generate SHAP feature-contribution attributions for a loaded model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        settings: ShapSettings | None = None,
        inference_config: InferenceConfig | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.settings = settings or ShapSettings()
        self.inference_config = inference_config or InferenceConfig()
        self.model.eval()

    def explain(
        self,
        input_tensor: torch.Tensor,
        *,
        investigation_id: str,
        image_name: str = "",
        original_image: Image.Image | None = None,
        target_class: int | None = None,
        model_name: str = "",
        model_version: str = "",
        dataset_version: str = "",
        artifact_dir: Path | None = None,
        opacity: float = 0.45,
        color_map: ColorMap = ColorMap.JET,
        write_visuals: bool | None = None,
    ) -> tuple[ShapResult, np.ndarray]:
        """Compute SHAP attributions and optional visualizations.

        Returns:
            (ShapResult, signed_attribution_map HxW)
        """

        if not self.settings.enabled:
            raise RuntimeError("SHAP is disabled in configuration")

        # Validate tensor shape before importing shap so callers get a clear
        # ValueError even when the optional dependency is missing.
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected input shape (1, C, H, W), got {tuple(input_tensor.shape)}"
            )

        require_shap_image_stack()

        with timing_ms() as timer:
            probs = predict_probabilities(self.model, input_tensor, self.device)
            decision = decide_from_probabilities(probs[0], self.inference_config)
            class_idx = (
                int(target_class)
                if target_class is not None
                else int(decision.predicted_class)
            )

            image_hwc = _tensor_to_hwc01(input_tensor)
            attribution = self._compute_shap_values(image_hwc, class_idx)
            abs_map, pos_map, neg_map = split_signed_maps(attribution)
            stats = compute_attribution_stats(attribution)

            viz_paths: dict[str, Path] = {}
            do_viz = (
                self.settings.visualization
                if write_visuals is None
                else bool(write_visuals)
            )
            if do_viz:
                if artifact_dir is None:
                    raise ValueError(
                        "artifact_dir is required when SHAP visualization is enabled"
                    )
                if original_image is None:
                    rgb = (np.clip(image_hwc, 0, 1) * 255.0).astype(np.uint8)
                    original_image = Image.fromarray(rgb, mode="RGB")
                viz_paths = write_shap_visualizations(
                    original=original_image,
                    abs_map=abs_map,
                    positive_map=pos_map,
                    negative_map=neg_map,
                    artifact_dir=Path(artifact_dir),
                    opacity=opacity,
                    color_map=color_map,
                )

            result = ShapResult(
                investigation_id=investigation_id,
                explainer_name="shap",
                model_name=model_name,
                model_version=model_version,
                dataset_version=dataset_version,
                prediction=decision.predicted_label,
                confidence=decision.confidence,
                target_class=class_idx,
                attribution_stats=stats,
                generation_time_ms=0.0,
                device=_device_label(self.device),
                visualization_path=str(viz_paths.get("overlay", "")),
                timestamp=ShapResult.utc_now(),
                heatmap_path=str(viz_paths.get("heatmap", "")),
                overlay_path=str(viz_paths.get("overlay", "")),
                comparison_path=str(viz_paths.get("comparison", "")),
                image_name=image_name,
                real_probability=decision.real_probability,
                fake_probability=decision.fake_probability,
                max_evaluations=self.settings.max_evaluations,
                metadata={
                    "masker": f"blur({self.settings.masker_blur_kernel})",
                    "signed_visualization": str(viz_paths.get("signed", "")),
                    "note": (
                        "SHAP estimates feature contribution; "
                        "it is not a Grad-CAM attention map."
                    ),
                },
            )
            del abs_map, pos_map, neg_map, image_hwc

        result.generation_time_ms = round(timer["ms"], 2)
        logger.info(
            "SHAP complete id=%s class=%s time=%.1fms evals=%s",
            investigation_id,
            class_idx,
            result.generation_time_ms,
            self.settings.max_evaluations,
        )
        return result, attribution

    def _compute_shap_values(
        self,
        image_hwc: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """Run Partition SHAP with a hard evaluation budget."""

        shap = require_shap_image_stack()
        masker = build_image_masker(
            image_hwc, blur_kernel=self.settings.masker_blur_kernel
        )

        model = self.model
        device = self.device

        def predict_fn(batch_nhwc: np.ndarray) -> np.ndarray:
            tensor = _hwc01_batch_to_tensor(batch_nhwc, device)
            with torch.inference_mode():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            del tensor, logits
            return probs

        explainer = shap.Explainer(predict_fn, masker)
        batch = np.expand_dims(image_hwc, axis=0)
        try:
            explanation = explainer(
                batch,
                max_evals=int(self.settings.max_evaluations),
                batch_size=int(self.settings.batch_size),
            )
        except Exception as exc:
            logger.exception("SHAP computation failed")
            raise RuntimeError(f"SHAP attribution failed: {exc}") from exc

        values = explanation.values
        attribution = extract_class_attribution(values, target_class=target_class)
        # Free large explanation payload
        del explanation, values, batch, explainer, masker
        return attribution


def shap_from_config(
    model: nn.Module,
    device: torch.device,
    config: AdvancedXAIConfig,
    inference_config: InferenceConfig | None = None,
) -> ShapEngine:
    """Factory helper binding AdvancedXAIConfig SHAP settings."""

    return ShapEngine(
        model,
        device,
        settings=config.shap,
        inference_config=inference_config,
    )
