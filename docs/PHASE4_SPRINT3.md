# MAYA — Phase 4 Sprint 4.3: Explanation Analytics & Trust Engine

**Status:** Complete  
**Scope:** Analyze and evaluate existing explanation heatmaps — **does not generate Grad-CAM or other explainers**; **no Flask, frontend, SHA-256, or PDF**

---

## Goal

Help investigators judge whether an explanation is **informative and trustworthy** by quantifying focus, localization, cross-explainer consistency, and producing plain-language summaries.

The analytics engine **only consumes** explanation outputs (normalized heatmaps or saved heatmap images). It stays independent of:

- Training  
- Evaluation  
- Inference  
- Explainability generation (Sprint 4.1 / 4.2)

---

## Architecture

```
Heatmaps (dict[name → ndarray] or PNG paths)
        │
        ▼
ExplanationAnalyticsEngine.analyze()
        │
        ├── focus_metrics
        ├── localization
        ├── statistics
        ├── consistency          (if ≥2 explainers)
        ├── explanation_quality
        ├── confidence           (model vs explanation — separate)
        ├── visualization
        ├── summary
        └── report
        │
        ▼
ExplanationAnalytics (+ artefacts under artifacts/phase4/sprint3/)
```

---

## Explanation quality

A transparent composite score (0–100) with grade **High / Moderate / Low / Insufficient**:

| Component | Weight (multi) | Idea |
|-----------|----------------|------|
| Focus | 40% | Peak, coverage band, sharpness (inverse entropy) |
| Localization | 35% | Connected region + coverage band |
| Consistency | 25% | Cross-explainer agreement (when available) |

Weights redistribute to focus/localization when only one map is provided.

---

## Focus metrics

- **Coverage** — fraction of pixels ≥ threshold  
- **Peak / mean activation**  
- **Activation variance**  
- **Normalized Shannon entropy** — higher → more diffuse  

---

## Localization metrics

- Activated-region **bounding box**  
- **Coverage percentage**  
- **Largest connected region** (4-connected)  
- **Center of attention** (activation-weighted centroid)  

---

## Consistency metrics

Across multiple explainers:

- Pairwise Pearson / Spearman / cosine / MSE  
- **Agreement score** — mean cosine mapped to `[0, 1]`  

---

## Confidence (do not conflate)

| Kind | Source |
|------|--------|
| Model prediction confidence | Classifier probability (0–100) |
| Explanation confidence | Quality + focus (+ agreement) |

The engine never copies model confidence into explanation confidence.

---

## Interpretation guidelines

1. View overlays beside the original evidence.  
2. High model confidence ≠ high explanation confidence.  
3. Low cross-method agreement → treat any single map cautiously.  
4. Low / Insufficient quality → try another explainer or deeper manual review.  
5. Numbers are **indicators**, not legal certainty.

---

## Artefacts

`artifacts/phase4/sprint3/`:

- `activation_histogram.png`  
- `coverage_map.png`  
- `focus_distribution.png`  
- `explanation_analysis.json`  
- `focus_metrics.json`  
- `summary.json` / `summary.md`  
- `analytics_report.md`  

---

## Usage

```python
from ai.explainability.analytics import (
    AnalyticsConfig,
    ExplanationAnalyticsEngine,
)

engine = ExplanationAnalyticsEngine(AnalyticsConfig())
result = engine.analyze(
    {"gradcam": heatmap_a, "layercam": heatmap_b},
    primary_explainer="gradcam",
    prediction="FAKE",
    model_confidence=88.0,
    image_name="evidence.jpg",
)

# Or consume Sprint 4.2 heatmap PNGs:
result = engine.analyze_from_heatmap_images(
    {"gradcam": "artifacts/phase4/sprint2/gradcam_heatmap.png", ...},
    primary_explainer="gradcam",
    prediction="FAKE",
    model_confidence=68.0,
)
```

---

## Current limitations

- Quality weights are heuristic (transparent, not learned).  
- Loading coloured CAM PNGs uses luminance as a proxy for activation intensity.  
- Connected-component analysis is 4-connected only.  
- No backend/UI integration in this sprint.

---

## Future extensions

- Learned / calibrated quality models  
- Persist float `.npy` heatmaps from Sprint 4.2 for lossless analytics  
- Attach analytics to investigation case records  
- UI cards for quality / agreement  

---

## Definition of Done

- [x] Explanation quality quantified  
- [x] Focus + localization metrics  
- [x] Cross-explainer consistency  
- [x] Investigator-friendly summaries  
- [x] Reports + JSON + visualizations  
- [x] Tests  

**STOP after Sprint 4.3.**
