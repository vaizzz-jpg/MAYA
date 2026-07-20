"""Artefact writers for Sprint 3.4 inference outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ai.inference.result import InvestigationResult
from ai.inference.utils import write_text

logger = logging.getLogger("maya.ai.inference.artifacts")


def write_inference_artifacts(
    result: InvestigationResult,
    artifact_dir: Path,
    *,
    log_lines: list[str] | None = None,
) -> dict[str, Path]:
    """Write prediction JSON / investigation JSON / markdown / log files."""

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    prediction = {
        "investigation_id": result.investigation_id,
        "prediction": result.prediction,
        "confidence": result.confidence,
        "confidence_level": result.confidence_level,
        "real_probability": result.real_probability,
        "fake_probability": result.fake_probability,
        "threshold": result.threshold,
        "image_name": result.image_name,
    }
    paths: dict[str, Path] = {}
    paths["prediction_json"] = write_text(
        out / "prediction.json", json.dumps(prediction, indent=2)
    )
    paths["investigation_result_json"] = result.save_json(out / "investigation_result.json")

    report_md = "\n".join(
        [
            "# MAYA Investigation Prediction Report",
            "",
            f"- Investigation ID: **{result.investigation_id}**",
            f"- Image: `{result.image_name}`",
            f"- Prediction: **{result.prediction}**",
            f"- Confidence: **{result.confidence:.2f}%** ({result.confidence_level})",
            f"- REAL probability: `{result.real_probability:.6f}`",
            f"- FAKE probability: `{result.fake_probability:.6f}`",
            f"- Threshold: `{result.threshold}`",
            f"- Decision: `{result.threshold_decision}`",
            f"- Model: `{result.model_name}` / `{result.model_version}`",
            f"- Dataset version: `{result.dataset_version}`",
            f"- Device: `{result.processing_device}`",
            f"- Latency: `{result.prediction_time_ms:.2f} ms`",
            f"- Status: `{result.processing_status}`",
            f"- Timestamp (UTC): `{result.timestamp}`",
            "",
        ]
    )
    paths["prediction_report_md"] = write_text(out / "prediction_report.md", report_md)

    summary_md = "\n".join(
        [
            "# Pipeline Summary",
            "",
            "1. Validate input image",
            "2. Load / reuse cached model",
            "3. Preprocess (resize + ImageNet normalize)",
            "4. Softmax probabilities",
            "5. Confidence + threshold decision",
            "6. Allocate investigation ID",
            "7. Persist artefacts",
            "",
            f"Last result: `{result.investigation_id}` → {result.prediction}",
            "",
        ]
    )
    paths["pipeline_summary_md"] = write_text(out / "pipeline_summary.md", summary_md)

    log_path = out / "prediction_log.txt"
    lines = log_lines or [
        f"{result.timestamp} | {result.investigation_id} | {result.image_name} | "
        f"{result.prediction} | {result.confidence:.2f}% | {result.processing_status}"
    ]
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    paths["prediction_log_txt"] = write_text(
        log_path, existing + ("\n".join(lines) + "\n")
    )

    logger.info("Inference artefacts written under %s", out)
    return paths
