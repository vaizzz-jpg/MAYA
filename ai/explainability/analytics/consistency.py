"""Cross-explainer consistency metrics (agreement / similarity / correlation)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ai.explainability.analytics.statistics import _as_2d_float


@dataclass(frozen=True)
class PairwiseSimilarity:
    """Similarity between two explanation heatmaps."""

    explainer_a: str
    explainer_b: str
    pearson_correlation: float
    spearman_correlation: float
    cosine_similarity: float
    mse: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConsistencyMetrics:
    """Aggregate consistency across a set of explainers."""

    agreement_score: float  # mean pairwise cosine similarity in [0, 1]
    mean_pearson: float
    mean_cosine: float
    mean_mse: float
    pairwise: tuple[PairwiseSimilarity, ...] = field(default_factory=tuple)
    explainer_names: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreement_score": self.agreement_score,
            "mean_pearson": self.mean_pearson,
            "mean_cosine": self.mean_cosine,
            "mean_mse": self.mean_mse,
            "explainer_names": list(self.explainer_names),
            "pairwise": [p.to_dict() for p in self.pairwise],
        }


def compute_consistency_metrics(
    heatmaps: dict[str, np.ndarray],
) -> ConsistencyMetrics:
    """Compare multiple explainer heatmaps for localization agreement."""

    if len(heatmaps) < 2:
        names = tuple(sorted(heatmaps.keys()))
        return ConsistencyMetrics(
            agreement_score=1.0 if heatmaps else 0.0,
            mean_pearson=1.0 if heatmaps else 0.0,
            mean_cosine=1.0 if heatmaps else 0.0,
            mean_mse=0.0,
            pairwise=(),
            explainer_names=names,
        )

    names = tuple(sorted(heatmaps.keys()))
    aligned = {_align_name(n, heatmaps): _as_2d_float(heatmaps[n]) for n in names}
    # Resize all to the first map's shape for fair comparison
    ref_shape = next(iter(aligned.values())).shape
    prepared: dict[str, np.ndarray] = {}
    for name, arr in aligned.items():
        prepared[name] = _resize_if_needed(arr, ref_shape)

    pairs: list[PairwiseSimilarity] = []
    pearsons: list[float] = []
    cosines: list[float] = []
    mses: list[float] = []

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sim = _pairwise(a, b, prepared[a], prepared[b])
            pairs.append(sim)
            pearsons.append(sim.pearson_correlation)
            cosines.append(sim.cosine_similarity)
            mses.append(sim.mse)

    mean_cos = float(np.mean(cosines)) if cosines else 0.0
    # Map cosine from [-1,1] → [0,1] for an investigator-facing agreement score
    agreement = float(np.clip((mean_cos + 1.0) / 2.0, 0.0, 1.0))

    return ConsistencyMetrics(
        agreement_score=round(agreement, 6),
        mean_pearson=round(float(np.mean(pearsons)) if pearsons else 0.0, 6),
        mean_cosine=round(mean_cos, 6),
        mean_mse=round(float(np.mean(mses)) if mses else 0.0, 6),
        pairwise=tuple(pairs),
        explainer_names=names,
    )


def _align_name(name: str, heatmaps: dict[str, np.ndarray]) -> str:
    return name


def _resize_if_needed(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    from PIL import Image

    img = Image.fromarray(arr.astype(np.float32), mode="F")
    resized = img.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64)


def _pairwise(
    name_a: str,
    name_b: str,
    a: np.ndarray,
    b: np.ndarray,
) -> PairwiseSimilarity:
    fa = a.ravel()
    fb = b.ravel()
    pearson = _pearson(fa, fb)
    spearman = _spearman(fa, fb)
    cosine = _cosine(fa, fb)
    mse = float(np.mean((fa - fb) ** 2))
    return PairwiseSimilarity(
        explainer_a=name_a,
        explainer_b=name_b,
        pearson_correlation=round(pearson, 6),
        spearman_correlation=round(spearman, 6),
        cosine_similarity=round(cosine, 6),
        mse=round(mse, 6),
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    return _pearson(ra, rb)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
