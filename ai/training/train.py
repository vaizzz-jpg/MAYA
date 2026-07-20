"""CLI entry point for MAYA training.

Usage
-----
python -m ai.training.train --profile debug
python -m ai.training.train --profile development
python -m ai.training.train --profile production
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
from ai.engine.trainer import Trainer
from ai.models.model_config import ModelConfig
from ai.training.artifacts import write_sprint2_artifacts
from ai.training.logging_setup import configure_training_logging
from ai.training.training_config import TrainingConfig, apply_profile, list_profiles

logger = logging.getLogger("maya.ai.training.cli")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAYA Phase 3 Sprint 2 training CLI")
    parser.add_argument(
        "--profile",
        default="development",
        choices=list(list_profiles()),
        help="Training intensity profile",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device preference: auto|cpu|cuda|mps",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-text notes stored in experiment JSON",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional per-epoch batch cap (debug / smoke)",
    )
    return parser


def run_training(args: argparse.Namespace) -> int:
    model_cfg = ModelConfig(project_root=ROOT)
    if args.device:
        model_cfg.device_preference = args.device

    config = apply_profile(
        args.profile,
        TrainingConfig(model=model_cfg, project_root=ROOT, notes=args.notes),
    )
    if args.max_batches is not None:
        config.max_batches_per_epoch = args.max_batches

    configure_training_logging(config.log_dir)
    logger.info("Starting training profile=%s", config.profile)

    dataset_cfg = DatasetConfig(
        project_root=ROOT,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.model.image_size,
        random_seed=config.random_seed,
    )

    train_loader = create_dataloader(
        SplitName.TRAIN,
        config=dataset_cfg,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        transform_name="train",
        image_size=config.model.image_size,
    )
    val_loader = create_dataloader(
        SplitName.VALIDATION,
        config=dataset_cfg,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        transform_name="eval",
        image_size=config.model.image_size,
    )

    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=args.resume,
    )
    result = trainer.fit()
    artefacts = write_sprint2_artifacts(result, config)

    logger.info("Training complete epochs_ran=%s", result.epochs_ran)
    logger.info("Artefacts: %s", {k: str(v) for k, v in artefacts.items()})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run_training(args)
    except Exception:
        logging.getLogger("maya.ai.training.cli").exception("Training failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
