"""Report writers for MAYA benchmark artefacts (no benchmarking here)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("maya.ai.benchmark.report")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote JSON → %s", path)
    return path


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote text → %s", path)
    return path


def write_benchmark_reports(
    artifact_dir: Path,
    *,
    package: dict[str, Any],
    latency: dict[str, Any],
    memory: dict[str, Any],
    system: dict[str, Any],
) -> dict[str, Path]:
    """Emit markdown + JSON reports from an already-collected package."""

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["benchmark_json"] = _write_json(out / "benchmark.json", package)
    paths["latency_report_json"] = _write_json(out / "latency_report.json", latency)
    paths["memory_report_json"] = _write_json(out / "memory_report.json", memory)
    paths["system_info_json"] = _write_json(out / "system_info.json", system)

    consistency = package.get("consistency", {})
    throughput = package.get("throughput", {})
    model = package.get("model", {})

    summary = "\n".join(
        [
            "# MAYA Performance Summary",
            "",
            f"- Device: `{system.get('active_device')}`",
            f"- Avg latency: **{latency.get('average_ms', 'n/a')} ms**",
            f"- Peak RSS: **{memory.get('peak_rss_mb', 'n/a')} MiB**",
            f"- Model footprint (est.): **{memory.get('estimated_model_mb', 'n/a')} MiB**",
            f"- Consistency: **{consistency.get('consistency_percentage', 'n/a')}%**",
            f"- Reliability score: **{consistency.get('reliability_score', 'n/a')}**",
            f"- Model: `{model.get('model_name')}` / `{model.get('model_version')}`",
            "",
        ]
    )
    paths["performance_summary_md"] = _write_text(out / "performance_summary.md", summary)

    throughput_lines = []
    for point in throughput.get("points", []):
        throughput_lines.append(
            f"| {point.get('image_count')} | {point.get('images_per_second')} | "
            f"{point.get('elapsed_seconds')} |"
        )

    report = "\n".join(
        [
            "# MAYA Benchmark Report",
            "",
            "## Hardware / Software",
            "",
            f"- OS: {system.get('operating_system')} (`{system.get('os_release')}`)",
            f"- CPU: {system.get('cpu')}",
            f"- RAM: {system.get('available_ram_gb')} / {system.get('total_ram_gb')} GB available",
            f"- Python: {system.get('python_version')}",
            f"- PyTorch: {system.get('pytorch_version')}",
            f"- Torchvision: {system.get('torchvision_version')}",
            f"- Device: {system.get('active_device')}",
            "",
            "## Latency (ms)",
            "",
            f"- Runs: {latency.get('runs')}",
            f"- Average: {latency.get('average_ms')}",
            f"- Median: {latency.get('median_ms')}",
            f"- Min / Max: {latency.get('minimum_ms')} / {latency.get('maximum_ms')}",
            f"- Std Dev: {latency.get('std_dev_ms')}",
            "",
            "## Memory (MiB)",
            "",
            f"- Baseline RSS: {memory.get('baseline_rss_mb')}",
            f"- Peak RSS: {memory.get('peak_rss_mb')}",
            f"- Delta: {memory.get('delta_rss_mb')}",
            f"- Estimated model: {memory.get('estimated_model_mb')}",
            "",
            "## Throughput",
            "",
            "| Images | img/s | Seconds |",
            "|--------|-------|---------|",
            *throughput_lines,
            "",
            "## Consistency",
            "",
            f"- Dominant prediction: `{consistency.get('dominant_prediction')}`",
            f"- Consistency %: {consistency.get('consistency_percentage')}",
            f"- Mean confidence: {consistency.get('mean_confidence')}",
            f"- Confidence σ: {consistency.get('confidence_std_dev')}",
            f"- Reliability score: {consistency.get('reliability_score')}",
            "",
            "## Model",
            "",
            f"- Name / version: `{model.get('model_name')}` / `{model.get('model_version')}`",
            f"- Dataset version: `{model.get('dataset_version')}`",
            f"- Checkpoint: `{model.get('checkpoint_path')}` ({model.get('checkpoint_size_mb')} MiB)",
            f"- Params total / trainable / frozen: "
            f"{model.get('total_parameters')} / {model.get('trainable_parameters')} / "
            f"{model.get('frozen_parameters')}",
            "",
            "## Figures",
            "",
            "- `latency_distribution.png`",
            "- `memory_usage.png`",
            "- `throughput.png`",
            "- `consistency_chart.png`",
            "- `system_summary.png`",
            "",
        ]
    )
    paths["benchmark_report_md"] = _write_text(out / "benchmark_report.md", report)
    return paths
