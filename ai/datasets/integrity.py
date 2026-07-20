"""Dataset integrity verification.

Purpose:
    Detect unreadable, zero-byte, and duplicate-filename issues without
    aborting the pipeline.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ai.datasets.inventory import ImageRecord, InventoryResult

logger = logging.getLogger("maya.datasets.integrity")


@dataclass
class IntegrityIssue:
    """One integrity problem for a path."""

    path: str
    issue_type: str
    detail: str


@dataclass
class IntegrityReport:
    """Structured integrity scan outcome."""

    total_checked: int = 0
    readable: int = 0
    corrupted: list[IntegrityIssue] = field(default_factory=list)
    zero_byte: list[IntegrityIssue] = field(default_factory=list)
    unreadable: list[IntegrityIssue] = field(default_factory=list)
    duplicate_filenames: dict[str, list[str]] = field(default_factory=dict)
    readable_by_class: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        """Serialize for JSON export."""

        return {
            "total_checked": self.total_checked,
            "readable": self.readable,
            "readable_by_class": dict(self.readable_by_class),
            "corrupted_count": len(self.corrupted),
            "zero_byte_count": len(self.zero_byte),
            "unreadable_count": len(self.unreadable),
            "duplicate_filename_keys": len(self.duplicate_filenames),
            "corrupted": [issue.__dict__ for issue in self.corrupted],
            "zero_byte": [issue.__dict__ for issue in self.zero_byte],
            "unreadable": [issue.__dict__ for issue in self.unreadable],
            "duplicate_filenames": self.duplicate_filenames,
        }


def _verify_image(path: Path) -> str | None:
    """Return an error code if the image cannot be verified, else None."""

    try:
        with Image.open(path) as img:
            img.verify()
        # verify() can leave the file in a bad state; reopen for a load smoke test
        with Image.open(path) as img:
            img.load()
        return None
    except UnidentifiedImageError as exc:
        return f"unidentified: {exc}"
    except OSError as exc:
        return f"os_error: {exc}"
    except ValueError as exc:
        return f"value_error: {exc}"
    except Exception as exc:  # noqa: BLE001 — never crash the pipeline
        return f"unexpected: {type(exc).__name__}: {exc}"


def verify_integrity(inventory: InventoryResult) -> IntegrityReport:
    """Scan each inventoried file for integrity problems.

    Processes one image at a time and always continues after failures.
    """

    report = IntegrityReport(total_checked=len(inventory.records))
    name_index: dict[str, list[str]] = defaultdict(list)

    for record in inventory.records:
        path = record.path
        name_index[path.name.lower()].append(str(path))

        if record.size_bytes == 0:
            issue = IntegrityIssue(str(path), "zero_byte", "File size is 0 bytes")
            report.zero_byte.append(issue)
            logger.warning("Zero-byte file: %s", path)
            continue

        error = _verify_image(path)
        if error is None:
            report.readable += 1
            report.readable_by_class[record.class_name.value] += 1
            continue

        if "unidentified" in error:
            issue = IntegrityIssue(str(path), "corrupted", error)
            report.corrupted.append(issue)
        else:
            issue = IntegrityIssue(str(path), "unreadable", error)
            report.unreadable.append(issue)
        logger.warning("Integrity failure (%s): %s", issue.issue_type, path)

    report.duplicate_filenames = {
        name: paths for name, paths in name_index.items() if len(paths) > 1
    }
    if report.duplicate_filenames:
        logger.warning(
            "Found %s duplicate filenames (content may still differ)",
            len(report.duplicate_filenames),
        )

    logger.info(
        "Integrity scan: readable=%s corrupted=%s zero_byte=%s unreadable=%s",
        report.readable,
        len(report.corrupted),
        len(report.zero_byte),
        len(report.unreadable),
    )
    return report


def filter_readable_records(
    inventory: InventoryResult,
    integrity: IntegrityReport,
) -> list[ImageRecord]:
    """Return inventory records that passed integrity checks."""

    bad_paths = {
        issue.path
        for bucket in (integrity.corrupted, integrity.zero_byte, integrity.unreadable)
        for issue in bucket
    }
    return [record for record in inventory.records if str(record.path) not in bad_paths]
