"""CSV/URL ingest for MAYA when the source catalogue is not on-disk images.

Purpose:
    Materialize ``dataset/raw/{REAL,FAKE}`` from ``FINAL_DATASET.csv`` by
    downloading each image sequentially (memory-safe on 8 GB RAM).

Architecture:
    CSV catalogue (immutable) → incremental HTTP fetch → local raw folders
    → existing inventory/integrity/sampling/preprocessing pipeline.
"""

from __future__ import annotations

import csv
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai.datasets.dataset_config import ClassName, DatasetConfig, get_dataset_config

logger = logging.getLogger("maya.datasets.csv_ingest")


def default_catalogue_csv() -> Path:
    """Resolve the default catalogue CSV from configuration."""

    cfg = get_dataset_config()
    return cfg.catalogue_csv or (cfg.project_root / "archive" / "FINAL_DATASET.csv")


DEFAULT_CSV = default_catalogue_csv()


@dataclass
class CsvImageRow:
    """One catalogue row used for download."""

    image_id: str
    url: str
    label: ClassName
    dataset_split: str
    source: str
    fake_method: str


@dataclass
class DownloadStats:
    """Aggregate download counters."""

    attempted: int = 0
    saved: int = 0
    skipped_existing: int = 0
    failed: int = 0
    by_class_saved: dict[str, int] = field(default_factory=dict)
    by_class_failed: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)


def iter_csv_rows(csv_path: Path) -> Iterator[CsvImageRow]:
    """Yield labelled rows from the catalogue without loading all pixels."""

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_raw = (row.get("label") or "").strip().upper()
            if label_raw == ClassName.REAL.value:
                label = ClassName.REAL
            elif label_raw == ClassName.FAKE.value:
                label = ClassName.FAKE
            else:
                logger.warning("Unknown label skipped: %s", label_raw)
                continue

            url = (row.get("image_url") or "").strip()
            if not url:
                continue

            yield CsvImageRow(
                image_id=str(row.get("image_id") or "").strip() or "unknown",
                url=url,
                label=label,
                dataset_split=str(row.get("dataset_split") or "").strip(),
                source=str(row.get("source") or "").strip(),
                fake_method=str(row.get("fake_method") or "").strip(),
            )


def _extension_from_url(url: str) -> str:
    path = url.split("?", 1)[0].lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        if path.endswith(ext):
            return ext
    return ".jpg"


def _destination_for(row: CsvImageRow, raw_dir: Path) -> Path:
    class_dir = raw_dir / row.label.value
    class_dir.mkdir(parents=True, exist_ok=True)
    return class_dir / f"{row.image_id}{_extension_from_url(row.url)}"


def download_one(
    url: str,
    destination: Path,
    *,
    timeout_s: float = 25.0,
    retries: int = 2,
) -> None:
    """Download a single URL to disk with light retries."""

    headers = {
        "User-Agent": (
            "MAYA-DatasetPipeline/1.0 (academic research; "
            "Media Authenticity Analyzer)"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout_s) as response:
                chunk = response.read()
            if not chunk:
                raise OSError("Empty response body")
            destination.write_bytes(chunk)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(str(last_error))


def select_rows_for_budget(
    rows: list[CsvImageRow],
    *,
    per_class_target: int,
    seed: int,
    buffer_ratio: float = 0.2,
) -> list[CsvImageRow]:
    """Seeded selection with extra cushion for download failures.

    Prefers hosts that are more reliably downloadable (skips multiavatar
    first when alternatives exist) while remaining deterministic via seed.
    """

    rng = random.Random(seed)
    selected: list[CsvImageRow] = []
    need = int(per_class_target * (1.0 + buffer_ratio))

    for cls in ClassName:
        pool = [row for row in rows if row.label == cls]
        preferred = [row for row in pool if "multiavatar.com" not in row.url.lower()]
        fallback = [row for row in pool if "multiavatar.com" in row.url.lower()]
        rng.shuffle(preferred)
        rng.shuffle(fallback)
        ordered = preferred + fallback
        take = min(need, len(ordered))
        selected.extend(ordered[:take])
        logger.info(
            "Selected %s/%s catalogue rows for class %s "
            "(target=%s buffer_need=%s preferred=%s fallback=%s)",
            take,
            len(pool),
            cls.value,
            per_class_target,
            need,
            len(preferred),
            len(fallback),
        )
    rng.shuffle(selected)
    return selected


def materialize_raw_from_csv(
    csv_path: Path,
    config: DatasetConfig,
    *,
    per_class_target: int | None = None,
    max_downloads: int | None = None,
    max_workers: int = 6,
) -> DownloadStats:
    """Download images into ``config.raw_dir`` class folders.

    The CSV file itself is never modified. Existing on-disk images are reused.
    Downloads use a small thread pool (I/O-bound) so RAM stays low.
    """

    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV catalogue not found: {csv_path}")

    # Enough for train+val+test per class (2600) by default
    target = per_class_target or (
        config.train_budget.real
        + config.validation_budget.real
        + config.test_budget.real
    )

    all_rows = list(iter_csv_rows(csv_path))
    selected = select_rows_for_budget(
        all_rows,
        per_class_target=target,
        seed=config.random_seed,
    )
    if max_downloads is not None:
        selected = selected[:max_downloads]

    stats = DownloadStats(
        by_class_saved={ClassName.REAL.value: 0, ClassName.FAKE.value: 0},
        by_class_failed={ClassName.REAL.value: 0, ClassName.FAKE.value: 0},
    )
    config.raw_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[CsvImageRow, Path]] = []
    for row in selected:
        stats.attempted += 1
        destination = _destination_for(row, config.raw_dir)
        if destination.exists() and destination.stat().st_size > 0:
            stats.skipped_existing += 1
            stats.saved += 1
            stats.by_class_saved[row.label.value] += 1
            continue
        pending.append((row, destination))

    def _task(item: tuple[CsvImageRow, Path]) -> tuple[CsvImageRow, Path, str | None]:
        row, destination = item
        try:
            download_one(row.url, destination)
            return row, destination, None
        except Exception as exc:  # noqa: BLE001
            return row, destination, str(exc)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_task, item) for item in pending]
        for future in as_completed(futures):
            row, destination, error = future.result()
            completed += 1
            if error is None:
                stats.saved += 1
                stats.by_class_saved[row.label.value] += 1
            else:
                stats.failed += 1
                stats.by_class_failed[row.label.value] += 1
                stats.failures.append(
                    {
                        "image_id": row.image_id,
                        "label": row.label.value,
                        "url": row.url,
                        "error": error,
                    }
                )
                logger.warning(
                    "Download failed id=%s label=%s: %s",
                    row.image_id,
                    row.label.value,
                    error,
                )
                if destination.exists():
                    try:
                        destination.unlink()
                    except OSError:
                        pass

            if completed % 50 == 0 or completed == len(futures):
                logger.info(
                    "Download progress %s/%s saved=%s failed=%s",
                    completed,
                    len(futures),
                    stats.saved,
                    stats.failed,
                )

    logger.info(
        "Ingest complete attempted=%s saved=%s existing=%s failed=%s by_class=%s",
        stats.attempted,
        stats.saved,
        stats.skipped_existing,
        stats.failed,
        stats.by_class_saved,
    )
    return stats
