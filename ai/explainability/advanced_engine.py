"""Advanced Explainability & Trust Layer coordinator (Phase 4 Sprint 4.5).

Orchestrates SHAP, faithfulness, counterfactual analysis, fusion, trust,
audit, and reporting. Does not modify CAM algorithms or the 4.4 benchmark.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ai.explainability.analytics import (
    AnalyticsConfig,
    ExplanationAnalytics,
    ExplanationAnalyticsEngine,
)
from ai.explainability.config import AdvancedXAIConfig, get_advanced_xai_config
from ai.explainability.counterfactual import CounterfactualEngine, CounterfactualResult
from ai.explainability.faithfulness import FaithfulnessEngine, FaithfulnessResult
from ai.explainability.fusion.audit import XaiAuditRecord, build_audit_record
from ai.explainability.fusion.fusion import fuse_explanations
from ai.explainability.fusion.report import write_sprint5_reports
from ai.explainability.fusion.result import FusedExplanationResult
from ai.explainability.fusion.visualization import write_trust_summary_chart
from ai.explainability.heatmap import prepare_heatmap
from ai.explainability.registry import ExplainerRegistry, register_default_explainers
from ai.explainability.shap import ShapEngine, ShapResult, ShapUnavailableError
from ai.explainability.target_layer import TargetLayerResolver
from ai.inference.confidence import decide_from_probabilities
from ai.inference.inference_config import InferenceConfig
from ai.inference.investigation_id import InvestigationIDGenerator
from ai.inference.model_loader import ModelLoader
from ai.inference.predictor import predict_probabilities
from ai.inference.preprocessing import preprocess_image
from ai.inference.utils import ensure_file, timing_ms

logger = logging.getLogger("maya.ai.explainability.advanced_engine")


def _device_label(device: object) -> str:
    name = str(device).upper()
    if name.startswith("CUDA"):
        return "CUDA"
    if name.startswith("CPU"):
        return "CPU"
    return name


def _config_snapshot(config: AdvancedXAIConfig) -> dict[str, Any]:
    return {
        "shap": asdict(config.shap),
        "faithfulness": asdict(config.faithfulness),
        "counterfactual": {
            **{k: v for k, v in asdict(config.counterfactual).items()},
            "perturbation_methods": list(config.counterfactual.perturbation_methods),
        },
        "trust": asdict(config.trust),
        "primary_cam_explainer": config.primary_cam_explainer,
        "device_preference": config.device_preference,
        "model_name": config.model_name,
        "model_version": config.model_version,
        "dataset_version": config.dataset_version,
        "output_resolution": config.output_resolution,
        "image_size": config.image_size,
    }


class AdvancedExplainabilityEngine:
    """End-to-end Sprint 4.5 advanced XAI pipeline."""

    def __init__(
        self,
        config: AdvancedXAIConfig | None = None,
        *,
        model_loader: ModelLoader | None = None,
        id_generator: InvestigationIDGenerator | None = None,
    ) -> None:
        self.config = config or get_advanced_xai_config()
        register_default_explainers()

        infer_cfg = InferenceConfig(
            project_root=self.config.project_root,
            checkpoint_path=self.config.checkpoint_path,
            artifact_dir=self.config.artifact_dir,
            id_state_path=self.config.id_state_path,
            device_preference=self.config.device_preference,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            image_size=self.config.image_size,
        )
        self._infer_cfg = infer_cfg
        self._loader = model_loader or ModelLoader(infer_cfg)
        self._id_gen = id_generator or InvestigationIDGenerator(
            self.config.id_state_path
        )

    def analyze(self, image_path: Path | str) -> FusedExplanationResult:
        """Run enabled advanced XAI stages and write sprint5 artefacts."""

        cfg = self.config
        path = ensure_file(Path(image_path))
        root = Path(cfg.artifact_dir)
        shap_dir = root / "shap"
        faith_dir = root / "faithfulness"
        cf_dir = root / "counterfactual"
        fusion_dir = root / "fusion"
        reports_dir = root / "reports"
        viz_dir = root / "visualizations"
        for d in (shap_dir, faith_dir, cf_dir, fusion_dir, reports_dir, viz_dir):
            d.mkdir(parents=True, exist_ok=True)

        stages: list[str] = []
        artifact_paths: dict[str, str] = {}
        shap_result: ShapResult | None = None
        faith_result: FaithfulnessResult | None = None
        cf_result: CounterfactualResult | None = None
        analytics: ExplanationAnalytics | None = None
        cam_map: np.ndarray | None = None
        target_layer_name: str | None = None
        attribution_map: np.ndarray | None = None

        with timing_ms() as timer:
            loaded = self._loader.load()
            prepared = preprocess_image(path, self._infer_cfg)
            probs = predict_probabilities(
                loaded.model, prepared.tensor, loaded.device
            )
            decision = decide_from_probabilities(probs[0], self._infer_cfg)
            target_class = (
                int(cfg.target_class)
                if cfg.target_class is not None
                else int(decision.predicted_class)
            )
            investigation_id = self._id_gen.next_id()
            out_h = cfg.output_resolution

            with Image.open(path) as img:
                original = img.convert("RGB").resize(
                    (out_h, out_h), Image.Resampling.BILINEAR
                )

            # --- Visual CAM (reuse registry; does not alter algorithms) ---
            try:
                layer, target_layer_name = TargetLayerResolver.resolve(
                    loaded.model, cfg.model_name
                )
                explainer = ExplainerRegistry.create(
                    cfg.primary_cam_explainer, loaded.model, loaded.device
                )
                cam = explainer.generate(
                    prepared.tensor,
                    target_class=target_class,
                    target_layer=layer,
                    target_layer_name=target_layer_name,
                )
                cam_map = prepare_heatmap(
                    cam.raw_heatmap,
                    (out_h, out_h),
                    eps=cfg.heatmap_eps,
                )
                analytics = ExplanationAnalyticsEngine(
                    AnalyticsConfig(
                        project_root=cfg.project_root,
                        artifact_dir=fusion_dir / "analytics_tmp",
                    )
                ).analyze(
                    {cfg.primary_cam_explainer: cam_map},
                    primary_explainer=cfg.primary_cam_explainer,
                    prediction=decision.predicted_label,
                    model_confidence=decision.confidence,
                    image_name=path.name,
                    write_artifacts=False,
                )
                stages.append("cam_analytics")
                del cam
            except Exception:
                logger.exception(
                    "Primary CAM/analytics stage failed — continuing without it"
                )

            importance_for_tests = cam_map

            # --- SHAP ---
            if cfg.shap.enabled:
                try:
                    shap_engine = ShapEngine(
                        loaded.model,
                        loaded.device,
                        settings=cfg.shap,
                        inference_config=self._infer_cfg,
                    )
                    shap_result, attribution_map = shap_engine.explain(
                        prepared.tensor,
                        investigation_id=investigation_id,
                        image_name=path.name,
                        original_image=original,
                        target_class=target_class,
                        model_name=loaded.model_name or cfg.model_name,
                        model_version=loaded.model_version or cfg.model_version,
                        dataset_version=cfg.dataset_version,
                        artifact_dir=shap_dir,
                        opacity=cfg.opacity,
                        color_map=cfg.color_map,
                    )
                    stages.append("shap")
                    for key, val in {
                        "shap_heatmap": shap_result.heatmap_path,
                        "shap_overlay": shap_result.overlay_path,
                        "shap_comparison": shap_result.comparison_path,
                    }.items():
                        if val:
                            artifact_paths[key] = val
                    # Prefer |SHAP| as importance when available
                    importance_for_tests = np.abs(attribution_map)
                    if importance_for_tests.shape != (out_h, out_h):
                        importance_for_tests = prepare_heatmap(
                            importance_for_tests,
                            (out_h, out_h),
                            eps=cfg.heatmap_eps,
                        )
                except ShapUnavailableError:
                    logger.error("SHAP dependency unavailable — skipping SHAP stage")
                except Exception:
                    logger.exception("SHAP stage failed")

            if importance_for_tests is None:
                raise RuntimeError(
                    "Advanced XAI requires either a CAM map or SHAP attribution "
                    "to drive faithfulness/counterfactual stages"
                )

            # Ensure spatial size matches tensor
            if importance_for_tests.shape != tuple(prepared.tensor.shape[-2:]):
                importance_for_tests = prepare_heatmap(
                    importance_for_tests,
                    (prepared.tensor.shape[-2], prepared.tensor.shape[-1]),
                    eps=cfg.heatmap_eps,
                )

            # --- Faithfulness ---
            if cfg.faithfulness.enabled:
                try:
                    faith_engine = FaithfulnessEngine(
                        loaded.model,
                        loaded.device,
                        settings=cfg.faithfulness,
                        inference_config=self._infer_cfg,
                    )
                    faith_result = faith_engine.evaluate(
                        prepared.tensor,
                        importance_for_tests,
                        investigation_id=investigation_id,
                        image_name=path.name,
                        target_class=target_class,
                        artifact_dir=faith_dir,
                    )
                    stages.append("faithfulness")
                    if faith_result.visualization_path:
                        artifact_paths["faithfulness_comparison"] = (
                            faith_result.visualization_path
                        )
                        # Also copy reference under visualizations/
                        artifact_paths["faithfulness_comparison_viz"] = (
                            faith_result.visualization_path
                        )
                except Exception:
                    logger.exception("Faithfulness stage failed")

            # --- Counterfactual ---
            if cfg.counterfactual.enabled:
                try:
                    cf_engine = CounterfactualEngine(
                        loaded.model,
                        loaded.device,
                        settings=cfg.counterfactual,
                        inference_config=self._infer_cfg,
                    )
                    cf_result = cf_engine.run(
                        prepared.tensor,
                        importance_for_tests,
                        investigation_id=investigation_id,
                        image_name=path.name,
                        original_image=original,
                        target_class=target_class,
                        artifact_dir=cf_dir,
                    )
                    stages.append("counterfactual")
                    if cf_result.visualization_path:
                        artifact_paths["counterfactual_comparison"] = (
                            cf_result.visualization_path
                        )
                except Exception:
                    logger.exception("Counterfactual stage failed")

            # --- Fusion + trust ---
            fused = fuse_explanations(
                investigation_id=investigation_id,
                image_name=path.name,
                prediction=decision.predicted_label,
                confidence=decision.confidence,
                cam_name=cfg.primary_cam_explainer,
                analytics=analytics,
                shap=shap_result,
                faithfulness=faith_result,
                counterfactual=cf_result,
                trust_weights=cfg.trust,
                artifact_paths=artifact_paths,
            )
            stages.append("fusion")

            trust_chart = write_trust_summary_chart(fused.trust, viz_dir)
            artifact_paths["trust_summary"] = str(trust_chart)
            # Mirror key viz paths
            if "shap_overlay" in artifact_paths:
                artifact_paths["visualizations_shap_overlay"] = artifact_paths[
                    "shap_overlay"
                ]

            fused.artifact_paths = dict(artifact_paths)

            audit = build_audit_record(
                investigation_id=investigation_id,
                image_identifier=path.name,
                model_name=loaded.model_name or cfg.model_name,
                model_version=loaded.model_version or cfg.model_version,
                dataset_version=cfg.dataset_version,
                configuration=_config_snapshot(cfg),
                device=_device_label(loaded.device),
                generation_time_ms=0.0,
                target_class=target_class,
                target_layer=target_layer_name,
                explainer="advanced_xai",
                output_artifact_paths=artifact_paths,
                stages_completed=stages,
                timestamp=fused.timestamp,
            )

            report_paths = write_sprint5_reports(
                reports_dir=reports_dir,
                shap=shap_result,
                faithfulness=faith_result,
                counterfactual=cf_result,
                fused=fused,
                audit=audit,
            )
            artifact_paths.update(report_paths)
            fused.artifact_paths = dict(artifact_paths)
            audit.output_artifact_paths = dict(artifact_paths)

            # Persist fused JSON under fusion/
            fused.save_json(fusion_dir / "fused_explanation.json")
            artifact_paths["fused_explanation"] = str(
                fusion_dir / "fused_explanation.json"
            )

            del prepared, original, importance_for_tests, attribution_map

        audit.generation_time_ms = round(timer["ms"], 2)
        audit.save_json(reports_dir / "xai_audit.json")

        logger.info(
            "Advanced XAI complete id=%s stages=%s trust=%.1f (%.1f ms)",
            fused.investigation_id,
            stages,
            fused.trust.trust_score,
            timer["ms"],
        )
        return fused
