"""Faithfulness comparison charts."""

from __future__ import annotations

import logging
from pathlib import Path

from matplotlib import pyplot as plt

from ai.explainability.faithfulness.result import FaithfulnessStep

logger = logging.getLogger("maya.ai.explainability.faithfulness.visualization")


def write_faithfulness_comparison(
    *,
    step_results: list[FaithfulnessStep],
    artifact_dir: Path,
    original_score: float,
) -> Path:
    """Plot target-class score under important vs unimportant occlusion."""

    if not step_results:
        raise ValueError(
            "Cannot generate faithfulness_comparison.png without step results"
        )

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "faithfulness_comparison.png"

    fracs = [0.0] + [s.fraction_masked for s in step_results]
    imp = [original_score] + [s.important_score for s in step_results]
    unimp = [original_score] + [s.unimportant_score for s in step_results]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(fracs, imp, marker="o", label="Occlude important regions")
    ax.plot(fracs, unimp, marker="s", label="Occlude less-important regions")
    ax.set_xlabel("Fraction of pixels perturbed")
    ax.set_ylabel("Target-class score")
    ax.set_title(
        "Faithfulness: measured influence under occlusion\n"
        "(not causal proof of attention)"
    )
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote faithfulness comparison → %s", path)
    return path
