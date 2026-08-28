"""Tests for MAYA Phase 4 Sprint 4.5 counterfactual perturbations."""

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

from ai.explainability.config import CounterfactualSettings
from ai.explainability.counterfactual.experiments import CounterfactualEngine
from ai.explainability.perturbation import important_region_mask, perturb_tensor
from ai.inference.inference_config import InferenceConfig


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


def _blob(size: int = 32) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    heat = np.exp(-((yy - 8) ** 2 + (xx - 8) ** 2) / (2 * 3.0**2))
    return (heat / heat.max()).astype(np.float64)


def test_perturbation_generation() -> None:
    tensor = torch.randn(1, 3, 16, 16)
    mask = important_region_mask(_blob(16), threshold=0.4)
    assert mask.dtype == bool
    assert mask.shape == (16, 16)
    blurred = perturb_tensor(tensor, mask, method="blur", blur_radius=2)
    assert blurred.shape == tensor.shape
    assert not torch.equal(blurred, tensor)


def test_prediction_comparison_and_delta(tmp_path: Path) -> None:
    model = TinyCNN().eval()
    engine = CounterfactualEngine(
        model,
        torch.device("cpu"),
        settings=CounterfactualSettings(
            perturbation_methods=("mask", "blur", "neutral"),
            importance_threshold=0.3,
        ),
        inference_config=InferenceConfig(project_root=ROOT, image_size=32),
    )
    tensor = torch.randn(1, 3, 32, 32)
    original = Image.fromarray(
        (np.random.rand(32, 32, 3) * 255).astype(np.uint8), mode="RGB"
    )
    result = engine.run(
        tensor,
        _blob(32),
        investigation_id="INV-CF-1",
        image_name="cf.png",
        original_image=original,
        artifact_dir=tmp_path,
    )
    assert len(result.experiments) == 3
    for exp in result.experiments:
        assert isinstance(exp.confidence_delta, float)
        assert isinstance(exp.prediction_changed, bool)
        assert exp.original_prediction in {"REAL", "FAKE"}
    assert Path(result.visualization_path).is_file()

    path = result.save_json(tmp_path / "cf.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["investigation_id"] == "INV-CF-1"
    assert len(payload["experiments"]) == 3
