"""Training lifecycle callbacks (early stop, checkpoint hooks, LR hooks).

Why this module exists
----------------------
Keep side-effects (stop training, save best weights, log LR) out of the core
epoch loop so the trainer stays a thin coordinator (Open/Closed principle).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("maya.ai.training.callbacks")


@dataclass
class CallbackContext:
    """Per-epoch snapshot passed to callbacks."""

    epoch: int
    metrics: dict[str, float]
    learning_rate: float
    model: Any
    optimizer: Any
    scheduler: Any
    config: Any
    history: Any
    stop_training: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


class Callback(ABC):
    """Base callback interface."""

    def on_train_begin(self, ctx: CallbackContext) -> None:
        return None

    def on_train_end(self, ctx: CallbackContext) -> None:
        return None

    @abstractmethod
    def on_epoch_end(self, ctx: CallbackContext) -> None:
        """Called after train + validate for one epoch."""


@dataclass
class EarlyStopping(Callback):
    """Stop when monitored metric stops improving."""

    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 5
    min_delta: float = 1e-4
    best: float | None = None
    wait: int = 0
    stopped_epoch: int | None = None

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        if self.monitor not in ctx.metrics:
            logger.warning("EarlyStopping: metric %s missing", self.monitor)
            return

        current = ctx.metrics[self.monitor]
        if self.best is None:
            self.best = current
            self.wait = 0
            return

        improved = (
            current < self.best - self.min_delta
            if self.mode == "min"
            else current > self.best + self.min_delta
        )
        if improved:
            self.best = current
            self.wait = 0
            logger.info("EarlyStopping: improvement on %s → %.6f", self.monitor, current)
            return

        self.wait += 1
        logger.info(
            "EarlyStopping: no improve (%s) wait=%s/%s",
            self.monitor,
            self.wait,
            self.patience,
        )
        if self.wait >= self.patience:
            ctx.stop_training = True
            self.stopped_epoch = ctx.epoch
            logger.info("EarlyStopping triggered at epoch %s", ctx.epoch)


@dataclass
class ModelCheckpoint(Callback):
    """Persist best (or every) checkpoint via an injected save callable."""

    save_fn: Callable[[CallbackContext, Path], Path]
    filename: str = "best.pt"
    monitor: str = "val_loss"
    mode: str = "min"
    save_best_only: bool = True
    directory: Path | None = None
    best: float | None = None
    last_path: Path | None = None

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        target_dir = Path(self.directory or ctx.config.checkpoint_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / self.filename

        if self.monitor not in ctx.metrics:
            logger.warning("ModelCheckpoint: metric %s missing — skipping", self.monitor)
            return

        current = ctx.metrics[self.monitor]
        should_save = True
        if self.save_best_only:
            if self.best is None:
                should_save = True
            elif self.mode == "min":
                should_save = current < self.best
            else:
                should_save = current > self.best

        if not should_save:
            return

        self.best = current
        self.last_path = self.save_fn(ctx, path)
        logger.info("ModelCheckpoint saved → %s (%s=%.6f)", self.last_path, self.monitor, current)


@dataclass
class LearningRateSchedulerHook(Callback):
    """Step the LR scheduler once per epoch and record the LR."""

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        if ctx.scheduler is None:
            return
        ctx.scheduler.step()
        if ctx.optimizer is not None and ctx.optimizer.param_groups:
            lr = float(ctx.optimizer.param_groups[0]["lr"])
            ctx.metrics["learning_rate"] = lr
            logger.debug("LR scheduler stepped → %.8f", lr)
