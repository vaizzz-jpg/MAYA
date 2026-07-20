"""CLI for MAYA Phase 3 Sprint 3 evaluation.

Usage
-----
python -m ai.evaluation.run_evaluation
python scripts/evaluate.py --threshold 0.5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.datasets.dataloader import create_dataloader
from ai.datasets.dataset_config import DatasetConfig, SplitName
from ai.engine.checkpoint import load_checkpoint
from ai.evaluation.evaluation_config import EvaluationConfig
from ai.evaluation.evaluator import Evaluator
from ai.models.model_config import ModelConfig
from ai.models.model_factory import ModelFactory
from ai.training.logging_setup import configure_training_logging

logger = logging.getLogger("maya.ai.evaluation.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAYA evaluation CLI (Sprint 3.3)")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_training_logging(ROOT / "logs")

    model_cfg = ModelConfig(project_root=ROOT, device_preference=args.device)
    eval_cfg = EvaluationConfig(
        project_root=ROOT,
        device_preference=args.device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        checkpoint_path=args.checkpoint,
    )
    if args.threshold is not None:
        eval_cfg.threshold = args.threshold

    dataset_cfg = DatasetConfig(
        project_root=ROOT,
        batch_size=eval_cfg.batch_size,
        num_workers=0,
        image_size=model_cfg.image_size,
    )
    test_loader = create_dataloader(
        SplitName.TEST,
        config=dataset_cfg,
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        transform_name="eval",
        image_size=model_cfg.image_size,
    )

    model = ModelFactory.create(config=model_cfg)
    ckpt = eval_cfg.checkpoint_path
    if ckpt is not None and ckpt.exists():
        load_checkpoint(ckpt, model=model, map_location="cpu")
        logger.info("Loaded checkpoint %s", ckpt)
    else:
        logger.warning("Checkpoint missing (%s) — evaluating randomly initialized head", ckpt)

    result = Evaluator(
        model=model,
        test_loader=test_loader,
        config=eval_cfg,
        model_config=model_cfg,
    ).evaluate()

    logger.info("Evaluation finished n=%s acc=%s", result.n_samples, result.metrics.get("accuracy"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
