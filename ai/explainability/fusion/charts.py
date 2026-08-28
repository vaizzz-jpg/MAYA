"""Trust / fusion summary visualization."""

from __future__ import annotations

import logging
from pathlib import Path

from matplotlib import pyplot as plt

from ai.explainability.fusion.result import ExplanationTrustResult

logger = logging.getLogger("maya.ai.explainability.fusion.viz")


def write_trust_summary_chart(
    trust: ExplanationTrustResult,
    artifact_dir: Path,
) -> Path:
    """Bar chart of trust components + overall score."""

    if not trust.components:
        raise ValueError("Cannot generate trust_summary.png without components")

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trust_summary.png"

    names = list(trust.components.keys())
    values = [trust.components[n] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].bar(names, values, color="#3b6ea5")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Component score (0–100)")
    axes[0].set_title("Trust components (measured)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(["MAYA Trust Score"], [trust.trust_score], color="#2f9e44")
    axes[1].set_ylim(0, 100)
    axes[1].set_title(f"Overall: {trust.trust_score:.1f}/100 ({trust.grade})")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "MAYA Explanation Trust Score (engineering composite — not a probability)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote trust summary → %s", path)
    return path
