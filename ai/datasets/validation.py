"""Processed-folder validation.

Purpose:
    Confirm required split/class folders exist, are non-empty, and match
    target balanced counts (allowing for skipped corrupt files).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ai.datasets.dataset_config import ClassName, DatasetConfig, SplitName
from ai.datasets.utils.fs import count_images

logger = logging.getLogger("maya.datasets.validation")


@dataclass
class FolderCheck:
    """Validation result for one folder."""

    path: str
    expected: int
    actual: int
    ok: bool
    notes: str = ""


@dataclass
class ValidationReport:
    """Overall processed-corpus validation."""

    checks: list[FolderCheck] = field(default_factory=list)
    missing_folders: list[str] = field(default_factory=list)
    empty_folders: list[str] = field(default_factory=list)
    balanced: bool = False
    passed: bool = False

    @property
    def total_images(self) -> int:
        return sum(check.actual for check in self.checks)


def validate_processed_dataset(config: DatasetConfig) -> ValidationReport:
    """Validate processed train/validation/test class folders."""

    report = ValidationReport()
    class_totals: dict[str, int] = {ClassName.REAL.value: 0, ClassName.FAKE.value: 0}

    for split in SplitName:
        budget = config.budget_for(split)
        expected_map = {
            ClassName.REAL: budget.real,
            ClassName.FAKE: budget.fake,
        }
        for cls, expected in expected_map.items():
            path = config.class_dir(split, cls)
            if not path.exists():
                report.missing_folders.append(str(path))
                report.checks.append(
                    FolderCheck(str(path), expected, 0, False, "missing folder")
                )
                logger.error("Missing folder: %s", path)
                continue

            actual = count_images(path, config.supported_extensions)
            if actual == 0:
                report.empty_folders.append(str(path))

            ok = actual == expected
            note = ""
            if actual != expected:
                note = f"expected {expected}, found {actual}"
                logger.warning("Count mismatch at %s (%s)", path, note)

            report.checks.append(FolderCheck(str(path), expected, actual, ok, note))
            class_totals[cls.value] += actual

    report.balanced = class_totals[ClassName.REAL.value] == class_totals[ClassName.FAKE.value]
    report.passed = (
        not report.missing_folders
        and not report.empty_folders
        and all(check.ok for check in report.checks)
        and report.balanced
    )
    logger.info(
        "Validation passed=%s balanced=%s totals=%s",
        report.passed,
        report.balanced,
        class_totals,
    )
    return report
