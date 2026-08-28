"""Stabilization tests: artifact isolation, ID reuse, prediction consistency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.explainability.config import ExplainabilityConfig
from ai.explainability.engine import ExplainabilityEngine
from ai.inference.inference_config import InferenceConfig
from ai.inference.pipeline import InferencePipeline


@pytest.fixture()
def evidence(tmp_path: Path) -> Path:
    path = tmp_path / "evidence.png"
    Image.new("RGB", (180, 140), color=(40, 90, 120)).save(path)
    return path


@pytest.fixture()
def ckpt() -> Path:
    path = ROOT / "artifacts" / "checkpoints" / "best.pt"
    if not path.exists():
        pytest.skip("best.pt checkpoint required")
    return path


def test_artifact_dir_not_derived_from_input_path(
    evidence: Path, ckpt: Path, tmp_path: Path
) -> None:
    """Input under sprint1-like path must not force outputs into that folder."""

    # Place evidence under a folder that looks like sprint1 artefacts
    sprint1_like = tmp_path / "artifacts" / "phase4" / "sprint1"
    sprint1_like.mkdir(parents=True)
    nested = sprint1_like / "original.png"
    Image.open(evidence).save(nested)

    out_dir = tmp_path / "artifacts" / "investigations" / "INV-ISO" / "xai" / "gradcam"
    cfg = ExplainabilityConfig(
        project_root=ROOT,
        checkpoint_path=ckpt,
        artifact_dir=tmp_path / "should_not_use_default",
        id_state_path=tmp_path / "id_state.json",
        device_preference="cpu",
    )
    result = ExplainabilityEngine(cfg).explain(
        nested,
        investigation_id="INV-ISO-000001",
        artifact_dir=out_dir,
    )
    assert Path(result.overlay_path).resolve() == (out_dir / "overlay.png").resolve()
    assert (out_dir / "overlay.png").is_file()
    assert not (sprint1_like / "overlay.png").exists()


def test_investigation_id_reuse(evidence: Path, ckpt: Path, tmp_path: Path) -> None:
    cfg = ExplainabilityConfig(
        project_root=ROOT,
        checkpoint_path=ckpt,
        artifact_dir=tmp_path / "xai",
        id_state_path=tmp_path / "id_state.json",
        device_preference="cpu",
    )
    result = ExplainabilityEngine(cfg).explain(
        evidence, investigation_id="INV-2026-000042"
    )
    assert result.investigation_id == "INV-2026-000042"


def test_prediction_matches_inference_pipeline(
    evidence: Path, ckpt: Path, tmp_path: Path
) -> None:
    infer_cfg = InferenceConfig(
        project_root=ROOT,
        checkpoint_path=ckpt,
        artifact_dir=tmp_path / "infer",
        id_state_path=tmp_path / "shared_ids.json",
        device_preference="cpu",
    )
    inv = InferencePipeline(infer_cfg).run(evidence)

    xai_cfg = ExplainabilityConfig(
        project_root=ROOT,
        checkpoint_path=ckpt,
        artifact_dir=tmp_path / "xai",
        id_state_path=tmp_path / "shared_ids.json",
        device_preference="cpu",
        target_class=None,
    )
    expl = ExplainabilityEngine(xai_cfg).explain(
        evidence,
        investigation_id=inv.investigation_id,
        artifact_dir=tmp_path / "xai" / inv.investigation_id,
    )
    assert expl.investigation_id == inv.investigation_id
    assert expl.prediction == inv.prediction
    assert expl.confidence == pytest.approx(inv.confidence, rel=0, abs=1e-3)
    # target class is predicted class index (0=REAL, 1=FAKE)
    expected_class = 0 if inv.prediction == "REAL" else 1
    assert expl.target_class == expected_class


def test_explicit_target_class_override(
    evidence: Path, ckpt: Path, tmp_path: Path
) -> None:
    cfg = ExplainabilityConfig(
        project_root=ROOT,
        checkpoint_path=ckpt,
        artifact_dir=tmp_path / "xai",
        id_state_path=tmp_path / "id.json",
        device_preference="cpu",
        target_class=0,
    )
    result = ExplainabilityEngine(cfg).explain(evidence)
    assert result.target_class == 0


def test_repeated_explanation_safe(evidence: Path, ckpt: Path, tmp_path: Path) -> None:
    cfg = ExplainabilityConfig(
        project_root=ROOT,
        checkpoint_path=ckpt,
        artifact_dir=tmp_path / "xai_a",
        id_state_path=tmp_path / "id.json",
        device_preference="cpu",
    )
    engine = ExplainabilityEngine(cfg)
    first = engine.explain(evidence, artifact_dir=tmp_path / "run1")
    second = engine.explain(evidence, artifact_dir=tmp_path / "run2")
    assert first.investigation_id != second.investigation_id
    assert Path(first.overlay_path).is_file()
    assert Path(second.overlay_path).is_file()
    assert first.prediction == second.prediction
    # Metadata JSON remains valid after overwrite-free isolated dirs
    payload = json.loads((tmp_path / "run2" / "explanation_result.json").read_text())
    assert payload["investigation_id"] == second.investigation_id
