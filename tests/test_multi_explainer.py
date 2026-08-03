"""Pytest suite for MAYA Phase 4 Sprint 4.2 Multi-Explainer Framework."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.explainability.base import BaseExplainer, Explainer
from ai.explainability.config import ColorMap, ExplainabilityConfig
from ai.explainability.engine import ExplainabilityEngine
from ai.explainability.explainers.eigencam import EigenCAM, compute_eigencam_map
from ai.explainability.explainers.gradcam import GradCAM, compute_gradcam_map
from ai.explainability.explainers.gradcam_plus_plus import (
    GradCAMPlusPlus,
    compute_gradcam_pp_map,
)
from ai.explainability.explainers.layercam import LayerCAM, compute_layercam_map
from ai.explainability.explainers.scorecam import ScoreCAM
from ai.explainability.hooks import ActivationGradientHooks
from ai.explainability.registry import (
    DEFAULT_EXPLAINER_NAMES,
    ExplainerRegistry,
    register_default_explainers,
)
from ai.explainability.target_layer import TargetLayerResolver
from ai.inference.inference_config import InferenceConfig
from ai.inference.model_loader import ModelLoader
from ai.inference.preprocessing import preprocess_image


class _TinyCNN(nn.Module):
    """Minimal CNN for fast ScoreCAM / plugin smoke tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "evidence.jpg"
    Image.new("RGB", (96, 96), color=(70, 110, 160)).save(path)
    return path


@pytest.fixture()
def xai_config(tmp_path: Path) -> ExplainabilityConfig:
    ckpt = ROOT / "artifacts" / "checkpoints" / "best.pt"
    return ExplainabilityConfig(
        project_root=ROOT,
        checkpoint_path=ckpt if ckpt.exists() else ROOT / "missing.pt",
        artifact_dir=tmp_path / "artifacts" / "phase4" / "sprint1",
        comparison_artifact_dir=tmp_path / "artifacts" / "phase4" / "sprint2",
        id_state_path=tmp_path / "id_state.json",
        device_preference="cpu",
        opacity=0.4,
        color_map=ColorMap.JET,
        output_resolution=64,
        scorecam_batch_size=4,
        comparison_explainers=(
            "gradcam",
            "gradcam_plus_plus",
            "layercam",
            "eigencam",
        ),
    )


@pytest.fixture()
def loaded_model(xai_config: ExplainabilityConfig):
    infer = InferenceConfig(
        project_root=xai_config.project_root,
        checkpoint_path=xai_config.checkpoint_path,
        device_preference="cpu",
        image_size=224,
    )
    return ModelLoader(infer).load()


def test_registry_lists_all_plugins() -> None:
    ExplainerRegistry.clear()
    register_default_explainers()
    available = ExplainerRegistry.list_available()
    for name in DEFAULT_EXPLAINER_NAMES:
        assert name in available
    assert "gradcam++" in available  # alias


def test_plugin_loading_polymorphism() -> None:
    ExplainerRegistry.clear()
    register_default_explainers()
    model = _TinyCNN()
    device = torch.device("cpu")
    for name in DEFAULT_EXPLAINER_NAMES:
        explainer = ExplainerRegistry.create(name, model, device)
        assert isinstance(explainer, Explainer)
        assert isinstance(explainer, BaseExplainer)


def test_configuration_switching(xai_config: ExplainabilityConfig) -> None:
    xai_config.explainer_name = "layercam"
    engine = ExplainabilityEngine(xai_config)
    assert engine.config.explainer_name == "layercam"
    xai_config.explainer_name = "eigencam"
    assert ExplainabilityEngine(xai_config).config.explainer_name == "eigencam"


def test_explainer_selection_via_config(
    sample_image: Path, xai_config: ExplainabilityConfig
) -> None:
    xai_config.explainer_name = "gradcam_plus_plus"
    xai_config.output_resolution = 64
    result = ExplainabilityEngine(xai_config).explain(sample_image)
    assert result.explainer_name == "gradcam_plus_plus"
    assert Path(result.overlay_path).exists()


def test_algorithm_maps_cpu() -> None:
    acts = torch.rand(1, 3, 5, 5)
    grads = torch.randn(1, 3, 5, 5)
    g = compute_gradcam_map(acts, grads)
    pp = compute_gradcam_pp_map(acts, grads)
    lc = compute_layercam_map(acts, grads)
    ec = compute_eigencam_map(acts)
    assert g.shape == (5, 5)
    assert pp.shape == (5, 5)
    assert lc.shape == (5, 5)
    assert ec.shape == (5, 5)
    assert float(g.min()) >= 0.0
    assert float(pp.min()) >= 0.0
    assert float(lc.min()) >= 0.0


def test_scorecam_on_tiny_cnn_cpu() -> None:
    model = _TinyCNN().eval()
    device = torch.device("cpu")
    x = torch.rand(1, 3, 32, 32)
    layer = model.features[-1]
    explainer = ScoreCAM(model, device, batch_size=2)
    out = explainer.generate(x, target_layer=layer, target_layer_name="features.3")
    assert out.raw_heatmap.ndim == 2
    assert out.probabilities.shape == (2,)


def test_gradient_explainers_on_efficientnet(
    loaded_model, sample_image: Path, xai_config: ExplainabilityConfig
) -> None:
    infer = InferenceConfig(
        project_root=xai_config.project_root,
        checkpoint_path=xai_config.checkpoint_path,
        device_preference="cpu",
    )
    prepared = preprocess_image(sample_image, infer)
    layer, name = TargetLayerResolver.resolve(loaded_model.model, "efficientnet_b0")
    for cls in (GradCAM, GradCAMPlusPlus, LayerCAM, EigenCAM):
        explainer = cls(loaded_model.model, loaded_model.device)
        out = explainer.generate(
            prepared.tensor, target_layer=layer, target_layer_name=name
        )
        assert out.raw_heatmap.ndim == 2
        assert np.isfinite(out.raw_heatmap).all()


def test_comparison_generation(
    sample_image: Path, xai_config: ExplainabilityConfig
) -> None:
    # Exclude ScoreCAM here for runtime; covered separately on TinyCNN.
    engine = ExplainabilityEngine(xai_config)
    result = engine.compare(sample_image)
    out = Path(xai_config.comparison_artifact_dir)
    assert (out / "comparison.png").exists()
    assert (out / "comparison.json").exists()
    assert (out / "comparison_report.md").exists()
    for name in xai_config.comparison_explainers:
        assert (out / f"{name}.png").exists()
    data = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert len(data["explainers"]) == len(xai_config.comparison_explainers)
    assert result.prediction in {"REAL", "FAKE"}


def test_comparison_serialization(tmp_path: Path) -> None:
    from ai.explainability.comparison import (
        ComparisonResult,
        ExplainerComparisonEntry,
        write_comparison_json,
        write_comparison_report,
    )

    result = ComparisonResult(
        image_name="x.jpg",
        prediction="FAKE",
        confidence=90.0,
        model_name="efficientnet_b0",
        model_version="t",
        device="CPU",
        target_layer="features.8",
        explainers=[
            ExplainerComparisonEntry(
                explainer_name="gradcam",
                generation_time_ms=10.0,
                overlay_path="gradcam.png",
                heatmap_path="gradcam_heatmap.png",
                target_class=1,
                target_layer="features.8",
            )
        ],
        comparison_path="comparison.png",
        timestamp="2026-01-01T00:00:00+00:00",
        total_time_ms=10.0,
    )
    j = write_comparison_json(result, tmp_path / "comparison.json")
    m = write_comparison_report(result, tmp_path / "comparison_report.md")
    assert "gradcam" in j.read_text(encoding="utf-8")
    assert "Investigator Notes" in m.read_text(encoding="utf-8")


def test_hook_cleanup_forward_only() -> None:
    model = _TinyCNN()
    layer = model.features[-1]
    with ActivationGradientHooks(layer, capture_gradients=False) as hooks:
        assert len(hooks._handles) == 1
        _ = model(torch.rand(1, 3, 16, 16))
        assert hooks.activations is not None
    assert hooks._handles == []
    assert hooks.activations is None


def test_memory_cleanup_multi_explainer() -> None:
    model = _TinyCNN().eval()
    device = torch.device("cpu")
    x = torch.rand(1, 3, 32, 32)
    layer = model.features[-1]
    for name in ("gradcam", "layercam", "eigencam", "scorecam"):
        explainer = ExplainerRegistry.create(name, model, device)
        if name == "scorecam":
            explainer.batch_size = 2  # type: ignore[attr-defined]
        explainer.generate(x, target_layer=layer, target_layer_name="features.3")
    # No dangling hooks on the layer
    assert getattr(layer, "_forward_hooks", {}) == {} or len(layer._forward_hooks) == 0
