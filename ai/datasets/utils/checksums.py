"""Streaming checksum helpers — never load whole corpora into RAM."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("maya.datasets.utils.checksums")

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def hash_file(path: Path, *, algorithm: str = "sha256") -> str:
    """Hash a single file by streaming chunks."""

    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_path_manifest(
    paths: Iterable[Path],
    *,
    root: Path | None = None,
    algorithm: str = "sha256",
    include_content: bool = False,
) -> str:
    """Build a deterministic corpus checksum from relative paths + sizes.

    By default hashes path|size only (fast, low RAM). Set ``include_content``
    to also stream each file body into the digest when stronger sealing is
    required.
    """

    digest = hashlib.new(algorithm)
    ordered = sorted(paths, key=lambda p: str(p).lower())
    for path in ordered:
        try:
            size = path.stat().st_size
            rel = path.relative_to(root).as_posix() if root is not None else path.as_posix()
            if include_content:
                file_hash = hash_file(path, algorithm=algorithm)
                digest.update(f"{rel}|{size}|{file_hash}\n".encode("utf-8"))
            else:
                digest.update(f"{rel}|{size}\n".encode("utf-8"))
        except OSError as exc:
            logger.warning("Checksum skipped for %s: %s", path, exc)
            continue
    return digest.hexdigest()
