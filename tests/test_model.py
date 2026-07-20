"""Unit tests for MAYA Phase 3 Sprint 1 — AI foundation (no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.engine.device import get_device, resolve_device
from ai.models.efficientnet import build_efficientnet_b0, count_trainable_parameters
from ai.models.model_config import ModelConfig, ModelName, get_model_config
from ai.models.model_factory import ModelFactory, UnsupportedModelError


@pytest.fixture(scope="module")
def config() -> ModelConfig:
    """Lightweight config — still uses pretrained weights (once) for realism."""

    return ModelConfig(
        project_root=ROOT,
        model_name=ModelName.EFFICIENTNET_B0.value,
        num_classes=2,
        batch_size=2,
        pretrained_weights=True,
        freeze_backbone=True,
        device_preference="cpu",
    )


@pytest.fixture(scope="module")
def model(config: ModelConfig) -> torch.nn.Module:
    """Build one model instance for the module (avoid duplicate downloads)."""

    return ModelFactory.create(config=config)


def test_config_loads() -> None:
    cfg = get_model_config()
    assert cfg.MODEL_NAME == ModelName.EFFICIENTNET_B0.value
    assert cfg.NUM_CLASSES == 2
    assert cfg.IMAGE_SIZE == 224
    assert cfg.BATCH_SIZE > 0
    assert cfg.CHECKPOINT_DIR.name == "checkpoints"
    assert cfg.PRETRAINED_WEIGHTS is True


def test_device_detection() -> None:
    device = resolve_device("cpu")
    assert device.type == "cpu"
    auto = get_device(ModelConfig(device_preference="auto", project_root=ROOT))
    assert auto.type in {"cpu", "cuda", "mps"}


def test_efficientnet_loads(config: ModelConfig) -> None:
    net = build_efficientnet_b0(config)
    assert net is not None
    trainable, total = count_trainable_parameters(net)
    assert total > trainable > 0


def test_factory_creates_model(model: torch.nn.Module) -> None:
    assert model is not None
    assert ModelFactory.supported_models() == ("efficientnet_b0",)


def test_factory_rejects_unknown() -> None:
    with pytest.raises(UnsupportedModelError, match="Unknown model"):
        ModelFactory.create("not_a_real_model")


def test_factory_rejects_planned_unimplemented() -> None:
    with pytest.raises(UnsupportedModelError, match="planned"):
        ModelFactory.create(ModelName.RESNET18.value)


def test_forward_pass_shape(model: torch.nn.Module, config: ModelConfig) -> None:
    """Dummy tensor only — does not load the MAYA dataset."""

    device = get_device(config)
    model = model.to(device)
    model.eval()
    batch = config.batch_size
    dummy = torch.zeros(batch, 3, config.image_size, config.image_size, device=device)
    with torch.no_grad():
        output = model(dummy)
    assert tuple(output.shape) == (batch, config.num_classes)
    assert output.shape == (batch, 2)
