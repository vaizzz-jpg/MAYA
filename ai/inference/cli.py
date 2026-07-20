"""CLI for MAYA investigation inference.

Usage
-----
python -m ai.inference.cli path/to/image.jpg
python -m ai.inference.cli --folder path/to/images
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.inference.batch import InvestigationBatchRunner
from ai.inference.inference_config import InferenceConfig
from ai.training.logging_setup import configure_training_logging

logger = logging.getLogger("maya.ai.inference.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MAYA Investigation Inference Engine (Sprint 3.4)"
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a single evidence image",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Directory of images for batch inference",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_training_logging(ROOT / "logs")

    if args.image is None and args.folder is None:
        logger.error("Provide an image path or --folder")
        return 2

    cfg = InferenceConfig(project_root=ROOT, device_preference=args.device)
    if args.threshold is not None:
        cfg.threshold = args.threshold
    if args.checkpoint is not None:
        cfg.checkpoint_path = args.checkpoint

    runner = InvestigationBatchRunner(cfg)

    try:
        if args.folder is not None:
            results = runner.run_folder(args.folder)
            logger.info("Batch finished count=%s", len(results))
            return 0 if results else 1
        assert args.image is not None
        result = runner.run_one(args.image)
        logger.info(
            "%s → %s (%.2f%% %s)",
            result.investigation_id,
            result.prediction,
            result.confidence,
            result.confidence_level,
        )
        return 0 if result.processing_status == "success" else 1
    except Exception:
        logger.exception("Inference CLI failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
