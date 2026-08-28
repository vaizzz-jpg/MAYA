"""Tests for MAYA Phase 4 Sprint 4.5 SHAP attribution engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.explainability.config import ShapSettings
from ai.explainability.shap.explainer import ShapEngine
from ai.explainability.shap.masker import (
    ShapDependencyError,
    require_opencv,
    require_shap,
    require_shap_image_stack,
)
from ai.explainability.shap.processor import (
    compute_attribution_stats,
    extract_class_attribution,
    split_signed_maps,
)
from ai.explainability.shap.result import ShapResult
from ai.inference.inference_config import InferenceConfig


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(8 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@pytest.fixture()
def tiny_model() -> TinyCNN:
    model = TinyCNN()
    model.eval()
    return model


@pytest.fixture()
def input_tensor() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)


def test_shap_dependency_importable() -> None:
    shap = require_shap()
    assert hasattr(shap, "Explainer")
    cv2 = require_opencv()
    assert hasattr(cv2, "__version__")
    require_shap_image_stack()


def test_shap_missing_opencv_error_is_actionable() -> None:
    """Dependency error path — does not replace real SHAP execution tests."""

    with patch.dict("sys.modules", {"cv2": None}):
        # Force import failure for cv2
        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "cv2" or name.startswith("cv2."):
                raise ImportError("No module named 'cv2'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import):
            with pytest.raises(ShapDependencyError, match="OpenCV|cv2|opencv-python-headless"):
                require_opencv()


def test_shap_engine_init(tiny_model: TinyCNN) -> None:
    engine = ShapEngine(
        tiny_model,
        torch.device("cpu"),
        settings=ShapSettings(max_evaluations=8, batch_size=2, visualization=False),
    )
    assert engine.settings.max_evaluations == 8
    assert engine.device.type == "cpu"


def test_attribution_processor_shapes() -> None:
    values = np.random.randn(32, 32, 3, 2).astype(np.float32)
    attr = extract_class_attribution(values, target_class=1)
    assert attr.shape == (32, 32)
    abs_m, pos_m, neg_m = split_signed_maps(attr)
    assert abs_m.shape == pos_m.shape == neg_m.shape == (32, 32)
    assert abs_m.min() >= 0.0 and abs_m.max() <= 1.0
    stats = compute_attribution_stats(attr)
    assert stats.mean_abs >= 0.0
    assert 0.0 <= stats.positive_fraction <= 1.0


def test_shap_cpu_execution_and_serialization(
    tiny_model: TinyCNN,
    input_tensor: torch.Tensor,
    tmp_path: Path,
) -> None:
    """Real SHAP Partition attribution on TinyCNN (requires shap + OpenCV)."""

    require_shap_image_stack()
    engine = ShapEngine(
        tiny_model,
        torch.device("cpu"),
        settings=ShapSettings(
            max_evaluations=8,
            batch_size=2,
            masker_blur_kernel=8,
            visualization=True,
        ),
        inference_config=InferenceConfig(project_root=ROOT, image_size=32),
    )
    original = Image.fromarray(
        (np.random.rand(32, 32, 3) * 255).astype(np.uint8), mode="RGB"
    )
    result, attribution = engine.explain(
        input_tensor,
        investigation_id="INV-TEST-000001",
        image_name="sample.png",
        original_image=original,
        target_class=1,
        model_name="tiny",
        model_version="test",
        dataset_version="test-ds",
        artifact_dir=tmp_path / "shap",
    )
    assert isinstance(result, ShapResult)
    assert result.device == "CPU"
    assert result.target_class == 1
    assert attribution.shape == (32, 32)
    assert np.isfinite(attribution).all()
    assert result.attribution_stats.max_abs >= 0.0
    assert Path(result.heatmap_path).is_file()
    assert Path(result.overlay_path).is_file()
    assert Path(result.comparison_path).is_file()

    json_path = result.save_json(tmp_path / "shap_result.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["explainer_name"] == "shap"
    assert payload["investigation_id"] == "INV-TEST-000001"
    assert "attribution_stats" in payload


def test_shap_defaults_to_predicted_class(
    tiny_model: TinyCNN,
    input_tensor: torch.Tensor,
) -> None:
    require_shap_image_stack()
    engine = ShapEngine(
        tiny_model,
        torch.device("cpu"),
        settings=ShapSettings(max_evaluations=8, batch_size=2, visualization=False),
        inference_config=InferenceConfig(project_root=ROOT, image_size=32),
    )
    with torch.inference_mode():
        pred = int(torch.softmax(tiny_model(input_tensor), dim=1)[0].argmax().item())
    result, _attr = engine.explain(
        input_tensor,
        investigation_id="INV-TEST-PRED",
        write_visuals=False,
        target_class=None,
    )
    assert result.target_class == pred


def test_shap_invalid_shape(tiny_model: TinyCNN) -> None:
    engine = ShapEngine(
        tiny_model,
        torch.device("cpu"),
        settings=ShapSettings(max_evaluations=4, visualization=False),
    )
    with pytest.raises(ValueError, match="Expected input shape"):
        engine.explain(
            torch.randn(2, 3, 16, 16),
            investigation_id="INV-TEST-000002",
            write_visuals=False,
        )
