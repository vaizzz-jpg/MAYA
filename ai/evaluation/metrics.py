"""Pure metric calculators for MAYA binary authenticity evaluation.

Purpose
-------
Compute scalar classification metrics only — no plotting, no I/O.

Architecture
------------
Leaf utilities used by ``MetricRegistry``. Inputs are NumPy arrays so the
evaluator can stream predictions then hand off a compact summary.

Testing
-------
Unit-test with hand-crafted labels/predictions (see ``tests/test_evaluation.py``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("maya.ai.evaluation.metrics")


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    positive_class: int = 1,
) -> dict[str, int]:
    """Return TP/TN/FP/FN for a binary problem."""

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos = int(positive_class)
    neg = 1 - pos if pos in (0, 1) else 0

    tp = int(np.sum((y_true == pos) & (y_pred == pos)))
    tn = int(np.sum((y_true == neg) & (y_pred == neg)))
    fp = int(np.sum((y_true == neg) & (y_pred == pos)))
    fn = int(np.sum((y_true == pos) & (y_pred == neg)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, **_: Any) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray, *, positive_class: int = 1, **_: Any) -> float:
    c = confusion_counts(y_true, y_pred, positive_class=positive_class)
    return _safe_div(c["tp"], c["tp"] + c["fp"])


def recall(y_true: np.ndarray, y_pred: np.ndarray, *, positive_class: int = 1, **_: Any) -> float:
    c = confusion_counts(y_true, y_pred, positive_class=positive_class)
    return _safe_div(c["tp"], c["tp"] + c["fn"])


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, *, positive_class: int = 1, **_: Any) -> float:
    p = precision(y_true, y_pred, positive_class=positive_class)
    r = recall(y_true, y_pred, positive_class=positive_class)
    return _safe_div(2 * p * r, p + r)


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray, *, positive_class: int = 1, **_: Any) -> float:
    """Alias of recall for the positive (e.g. FAKE) class."""

    return recall(y_true, y_pred, positive_class=positive_class)


def specificity(y_true: np.ndarray, y_pred: np.ndarray, *, positive_class: int = 1, **_: Any) -> float:
    c = confusion_counts(y_true, y_pred, positive_class=positive_class)
    return _safe_div(c["tn"], c["tn"] + c["fp"])


def false_positive_rate(
    y_true: np.ndarray, y_pred: np.ndarray, *, positive_class: int = 1, **_: Any
) -> float:
    c = confusion_counts(y_true, y_pred, positive_class=positive_class)
    return _safe_div(c["fp"], c["fp"] + c["tn"])


def false_negative_rate(
    y_true: np.ndarray, y_pred: np.ndarray, *, positive_class: int = 1, **_: Any
) -> float:
    c = confusion_counts(y_true, y_pred, positive_class=positive_class)
    return _safe_div(c["fn"], c["fn"] + c["tp"])


def balanced_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, *, positive_class: int = 1, **_: Any
) -> float:
    return 0.5 * (
        sensitivity(y_true, y_pred, positive_class=positive_class)
        + specificity(y_true, y_pred, positive_class=positive_class)
    )


def matthews_corrcoef(
    y_true: np.ndarray, y_pred: np.ndarray, *, positive_class: int = 1, **_: Any
) -> float:
    c = confusion_counts(y_true, y_pred, positive_class=positive_class)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return _safe_div(numerator, float(denominator))


def cohen_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, *, positive_class: int = 1, **_: Any
) -> float:
    c = confusion_counts(y_true, y_pred, positive_class=positive_class)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = tp + tn + fp + fn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    pe = (((tp + fp) / n) * ((tp + fn) / n)) + (((tn + fn) / n) * ((tn + fp) / n))
    return _safe_div(po - pe, 1.0 - pe)


def roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    y_score: np.ndarray | None = None,
    positive_class: int = 1,
    **_: Any,
) -> float:
    """ROC AUC via Mann–Whitney ranking (no sklearn).

    ``y_score`` should be P(positive_class). ``y_pred`` is ignored when scores
    are provided (kept for registry signature compatibility).
    """

    del y_pred  # unused when scoring
    if y_score is None:
        logger.warning("roc_auc called without y_score — returning 0.0")
        return 0.0

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    pos_mask = y_true == int(positive_class)
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Rank scores (average ties)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks for ties
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            tie_ranks = ranks[inverse == idx]
            ranks[inverse == idx] = float(np.mean(tie_ranks))

    sum_ranks_pos = float(np.sum(ranks[pos_mask]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_curve_points(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    positive_class: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return FPR, TPR, thresholds for plotting (sorted high→low score)."""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    desc = np.argsort(-scores)
    y_true = y_true[desc]
    scores = scores[desc]

    n_pos = max(int(np.sum(y_true == positive_class)), 1)
    n_neg = max(int(np.sum(y_true != positive_class)), 1)

    tps = np.cumsum(y_true == positive_class)
    fps = np.cumsum(y_true != positive_class)
    tpr = tps / n_pos
    fpr = fps / n_neg
    # Prepend origin
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    thresholds = np.concatenate([[scores[0] + 1e-12], scores])
    return fpr, tpr, thresholds


def precision_recall_curve_points(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    positive_class: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return precision, recall, thresholds for PR curve plotting."""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    desc = np.argsort(-scores)
    y_true = y_true[desc]
    scores = scores[desc]

    n_pos = max(int(np.sum(y_true == positive_class)), 1)
    tps = np.cumsum(y_true == positive_class)
    fps = np.cumsum(y_true != positive_class)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / n_pos
    return precision, recall, scores
