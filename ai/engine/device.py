"""Reusable torch device selection for MAYA (CPU-first, 8 GB safe).

Why this module exists
----------------------
Device selection must live in one place so training, inference, and tests
do not diverge (and so CPU-only Windows hosts are first-class).

Architecture role
-----------------
Utility / Engine Layer (``ai/engine/``). Model builders and a future training
engine call ``get_device()``; they never probe CUDA/MPS themselves.

Design decisions
----------------
* Prefer CUDA → Apple MPS → CPU, overridable via preference string.
* Log the choice once; never print().
* No model or dataset loading here.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

from ai.models.model_config import ModelConfig, get_model_config

logger = logging.getLogger("maya.ai.engine.device")


def resolve_device(preference: str = "auto") -> torch.device:
    """Return a ``torch.device`` based on preference and hardware availability.

    Args:
        preference: ``auto``, ``cuda``, ``mps``, or ``cpu``.
    """

    choice = (preference or "auto").strip().lower()

    if choice == "cpu":
        device = torch.device("cpu")
    elif choice == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable — falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif choice == "mps":
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_ok:
            logger.warning("MPS requested but unavailable — falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    elif choice == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        logger.warning("Unknown device preference %r — using CPU", preference)
        device = torch.device("cpu")

    logger.info("Selected torch device: %s (preference=%s)", device, choice)
    return device


def get_device(config: ModelConfig | None = None) -> torch.device:
    """Resolve device from ``ModelConfig.device_preference`` (or defaults)."""

    cfg = config or get_model_config()
    return resolve_device(cfg.device_preference)


@lru_cache(maxsize=4)
def get_device_cached(preference: str = "auto") -> torch.device:
    """Cached variant to avoid repeated capability probes in hot paths."""

    return resolve_device(preference)


def device_name(device: torch.device | None = None) -> str:
    """Human-readable device string for logs and artefacts."""

    resolved = device if device is not None else get_device()
    return str(resolved)
