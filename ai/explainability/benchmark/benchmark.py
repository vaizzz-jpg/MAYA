"""Explainability Validation & Benchmark Suite coordinator (Sprint 4.4).

Benchmarks all registered explainers without modifying CAM algorithms.
Reuses Sprint 4.3 analytics for quality / agreement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.explainability.benchmark.agreement import AgreementResult, compute_agreement
from ai.explainability.benchmark.config import (
    ExplainabilityBenchmarkConfig,
    get_explainability_benchmark_config,
)
from ai.explainability.benchmark.evaluator import ExplainerEvalResult, evaluate_explainer
from ai.explainability.benchmark.ranking import RankableExplainer, rank_explainers
from ai.explainability.benchmark.recommendation import (
    Recommendation,
    build_recommendation,
)
from ai.explainability.benchmark.report import write_benchmark_reports
from ai.explainability.benchmark.statistics import (
    BenchmarkStatistics,
    summarize_ranking,
)
from ai.explainability.benchmark.utils import resolve_sample_image
from ai.explainability.benchmark.visualization import write_benchmark_visualizations
from ai.explainability.registry import ExplainerRegistry, register_default_explainers
from ai.inference.confidence import decide_from_probabilities
from ai.inference.inference_config import InferenceConfig
from ai.inference.model_loader import ModelLoader
from ai.inference.preprocessing import preprocess_image

logger = logging.getLogger("maya.ai.explainability.benchmark.coordinator")


@dataclass
class ExplainabilityBenchmarkResult:
    """Full Sprint 4.4 benchmark package."""

    image_name: str
    prediction: str
    device: str
    explainer_names: list[str]
    evaluations: dict[str, dict[str, Any]]
    agreement: dict[str, Any]
    ranking: dict[str, Any]
    recommendation: dict[str, Any]
    statistics: dict[str, Any]
    artifact_paths: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name,
            "prediction": self.prediction,
            "device": self.device,
            "explainer_names": list(self.explainer_names),
            "evaluations": self.evaluations,
            "agreement": self.agreement,
            "ranking": self.ranking,
            "recommendation": self.recommendation,
            "statistics": self.statistics,
            "artifact_paths": dict(self.artifact_paths),
            "timestamp": self.timestamp,
        }


class ExplainabilityBenchmarkSuite:
    """Run validation/benchmark across registry explainers."""

    def __init__(self, config: ExplainabilityBenchmarkConfig | None = None) -> None:
        self.config = config or get_explainability_benchmark_config()
        register_default_explainers()

    def run(self, image_path: Path | str | None = None) -> ExplainabilityBenchmarkResult:
        cfg = self.config
        path = resolve_sample_image(
            cfg.project_root,
            Path(image_path) if image_path is not None else cfg.sample_image,
        )
        artifact_dir = Path(cfg.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        infer_cfg = InferenceConfig(
            project_root=cfg.project_root,
            checkpoint_path=cfg.checkpoint_path,
            device_preference=cfg.device_preference,
            model_name=cfg.model_name,
            model_version=cfg.model_version,
            image_size=cfg.output_resolution,
        )
        loaded = ModelLoader(infer_cfg).load()
        prepared = preprocess_image(path, infer_cfg)

        # Ensure requested names exist in registry (skip unknown aliases gracefully)
        available = set(ExplainerRegistry.list_available())
        names = [n for n in cfg.explainer_names if n in available]
        if not names:
            raise RuntimeError(
                f"No benchmark explainers available. Requested={cfg.explainer_names}, "
                f"registry={sorted(available)}"
            )

        evaluations: dict[str, ExplainerEvalResult] = {}
        heatmaps = {}
        for name in names:
            logger.info("Benchmarking explainer: %s", name)
            ev = evaluate_explainer(
                name,
                loaded.model,
                loaded.device,
                prepared.tensor,
                model_name=cfg.model_name,
                coverage_threshold=cfg.coverage_threshold,
                output_resolution=cfg.output_resolution,
                warmup_runs=cfg.warmup_runs,
                timed_runs=cfg.timed_runs,
                scorecam_batch_size=cfg.scorecam_batch_size,
            )
            evaluations[name] = ev
            heatmaps[name] = ev.heatmap

        agreement: AgreementResult = compute_agreement(heatmaps)

        rankables = [
            RankableExplainer(
                explainer_name=name,
                quality_score=ev.quality.quality.quality_score,
                focus_score=ev.quality.focus_score,
                localization_score=ev.quality.localization_score,
                coverage=ev.quality.coverage,
                agreement_score=agreement.per_explainer_mean_agreement.get(name, 0.0),
                generation_time_ms=ev.performance.generation_time_ms,
                peak_ram_mb=ev.performance.peak_ram_mb,
            )
            for name, ev in evaluations.items()
        ]
        ranking = rank_explainers(rankables, cfg.weights)
        recommendation: Recommendation = build_recommendation(ranking)
        stats: BenchmarkStatistics = summarize_ranking(ranking.entries)

        # Prediction metadata from first explainer probabilities
        first = next(iter(evaluations.values()))
        decision = decide_from_probabilities(
            first.probabilities.tolist(), infer_cfg
        )

        timestamp = datetime.now(timezone.utc).isoformat()
        result = ExplainabilityBenchmarkResult(
            image_name=path.name,
            prediction=decision.predicted_label,
            device=str(loaded.device).upper(),
            explainer_names=names,
            evaluations={k: v.to_dict() for k, v in evaluations.items()},
            agreement=agreement.to_dict(),
            ranking=ranking.to_dict(),
            recommendation=recommendation.to_dict(),
            statistics=stats.to_dict(),
            timestamp=timestamp,
        )

        viz_paths = write_benchmark_visualizations(ranking, agreement, artifact_dir)
        report_paths = write_benchmark_reports(
            benchmark_payload=result.to_dict(),
            ranking=ranking,
            recommendation=recommendation,
            artifact_dir=artifact_dir,
        )
        result.artifact_paths = {
            **{k: str(v) for k, v in viz_paths.items()},
            **{k: str(v) for k, v in report_paths.items()},
        }
        # Refresh JSON with artifact paths
        write_benchmark_reports(
            benchmark_payload=result.to_dict(),
            ranking=ranking,
            recommendation=recommendation,
            artifact_dir=artifact_dir,
        )

        logger.info(
            "Benchmark complete recommended=%s score=%.1f",
            recommendation.recommended_explainer,
            recommendation.overall_score,
        )
        del prepared, heatmaps
        return result
