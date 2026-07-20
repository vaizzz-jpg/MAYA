"""CLI for MAYA Phase 3 Sprint 5 benchmark suite.

Examples
--------
python -m ai.benchmark.cli
python -m ai.benchmark.cli --runs 20
python scripts/benchmark.py --image path/to.jpg --runs 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.benchmark.benchmark import run_benchmark_suite
from ai.inference.inference_config import InferenceConfig
from ai.training.logging_setup import configure_training_logging

logger = logging.getLogger("maya.ai.benchmark.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAYA AI Performance Benchmark Suite")
    parser.add_argument("--runs", type=int, default=10, help="Latency/consistency repetitions")
    parser.add_argument("--image", type=Path, default=None, help="Sample evidence image")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Artefact directory (default: artifacts/phase3/benchmark)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_training_logging(ROOT / "logs")

    cfg = InferenceConfig(project_root=ROOT, device_preference=args.device)
    try:
        package = run_benchmark_suite(
            image_path=args.image,
            runs=max(1, args.runs),
            config=cfg,
            artifact_dir=args.output,
        )
    except Exception:
        logger.exception("Benchmark suite failed")
        return 1

    logger.info(
        "Done avg_latency=%.2fms peak_rss=%.1fMiB reliability=%.1f → %s",
        package.latency.average_ms,
        package.memory.peak_rss_mb,
        package.consistency.reliability_score,
        package.artifact_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
