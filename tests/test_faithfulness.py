"""Tests for MAYA Phase 4 Sprint 4.5 faithfulness evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.explainability.config import FaithfulnessSettings
from ai.explainability.faithfulness.evaluator import FaithfulnessEngine
from ai.explainability.perturbation import (
    build_cumulative_mask,
    importance_ranking,
    perturb_tensor,
)
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
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def _blob(size: int = 32) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    heat = np.exp(-((yy - 10) ** 2 + (xx - 10) ** 2) / (2 * 4.0**2))
    return (heat / heat.max()).astype(np.float64)


def test_perturbation_logic() -> None:
    tensor = torch.ones(1, 3, 8, 8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:2, 0:2] = True
    out = perturb_tensor(tensor, mask, method="mask", neutral_value=0.0)
    assert float(out[:, :, 0, 0].mean()) == 0.0
    assert float(out[:, :, 7, 7].mean()) == 1.0
    ranked = importance_ranking(_blob(8))
    cum = build_cumulative_mask((8, 8), ranked, count=4)
    assert int(cum.sum()) == 4


def test_faithfulness_calculation(tmp_path: Path) -> None:
    model = TinyCNN().eval()
    engine = FaithfulnessEngine(
        model,
        torch.device("cpu"),
        settings=FaithfulnessSettings(steps=3, fraction_per_step=0.15),
        inference_config=InferenceConfig(project_root=ROOT, image_size=32),
    )
    tensor = torch.randn(1, 3, 32, 32)
    result = engine.evaluate(
        tensor,
        _blob(32),
        investigation_id="INV-F-1",
        image_name="t.png",
        artifact_dir=tmp_path,
    )
    assert result.steps == 3
    assert len(result.step_results) == 3
    assert 0.0 <= result.faithfulness_score <= 1.0
    assert isinstance(result.confidence_drop_important, float)
    assert Path(result.visualization_path).is_file()
    payload = result.to_dict()
    assert "deletion_score" in payload


def test_faithfulness_invalid_input() -> None:
    model = TinyCNN().eval()
    engine = FaithfulnessEngine(model, torch.device("cpu"))
    with pytest.raises(ValueError, match="Expected input shape"):
        engine.evaluate(
            torch.randn(2, 3, 16, 16),
            _blob(16),
            investigation_id="INV-F-2",
            write_visuals=False,
        )
    with pytest.raises(ValueError, match="2-D"):
        engine.evaluate(
            torch.randn(1, 3, 16, 16),
            np.zeros((16, 16, 3)),
            investigation_id="INV-F-3",
            write_visuals=False,
        )
