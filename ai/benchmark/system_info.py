"""System information collector for MAYA benchmarks.

Purpose
-------
Capture OS / CPU / RAM / Python / torch / device fingerprint as JSON-ready data.
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
import torch

logger = logging.getLogger("maya.ai.benchmark.system_info")


@dataclass
class SystemInfo:
    """Machine + software profile for reproducibility."""

    operating_system: str
    os_release: str
    cpu: str
    cpu_count_logical: int
    total_ram_gb: float
    available_ram_gb: float
    python_version: str
    pytorch_version: str
    torchvision_version: str
    active_device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_system_info(device_preference: str = "auto") -> SystemInfo:
    """Probe the host and return a populated ``SystemInfo``."""

    from ai.engine.device import resolve_device

    try:
        import torchvision

        tv_version = torchvision.__version__
    except Exception:  # noqa: BLE001
        tv_version = "unknown"

    device = resolve_device(device_preference)
    vm = psutil.virtual_memory()
    info = SystemInfo(
        operating_system=platform.system(),
        os_release=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        cpu_count_logical=psutil.cpu_count(logical=True) or 0,
        total_ram_gb=round(vm.total / (1024**3), 2),
        available_ram_gb=round(vm.available / (1024**3), 2),
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__,
        torchvision_version=tv_version,
        active_device=str(device),
    )
    logger.info(
        "System: %s | RAM %.1f/%.1f GB | device=%s",
        info.operating_system,
        info.available_ram_gb,
        info.total_ram_gb,
        info.active_device,
    )
    return info


def save_system_info_json(info: SystemInfo, path: Path) -> Path:
    """Write system_info.json."""

    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
    logger.info("Wrote system info → %s", path)
    return path
