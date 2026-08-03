# MAYA Explanation Analytics Report

- Primary explainer: `gradcam`
- Prediction: **REAL**
- Image: `2793.jpg`
- Timestamp (UTC): `2026-08-03T13:22:17.917730+00:00`

## Quality

- Score: **69.7 / 100** (Moderate)
- Focus component: `46.35`
- Localization component: `89.65`
- Consistency component: `79.12`

## Focus metrics

- Coverage: `0.612643`
- Peak activation: `1.0`
- Mean activation: `0.426501`
- Variance: `0.125713`
- Entropy: `0.961473`

## Localization

- Coverage %: `61.2643`
- Bounding box: `[0, 0, 223, 223]`
- Center of attention: `[148.5106, 117.6779]`
- Largest region area: `30510`

## Confidence

- Model prediction confidence: `68.1%` (Moderate)
- Explanation confidence: `78.11%` (High)
- Notes: Model confidence reflects class probability. Explanation confidence reflects map focus/quality and cross-explainer agreement.

## Consistency

- Agreement score: `0.791158`
- Mean Pearson: `0.381219`
- Mean cosine: `0.582316`
- Explainers: `eigencam, gradcam, gradcam_plus_plus, layercam, scorecam`

## Investigator summary

Explanation for prediction **REAL** using `gradcam` — quality moderate (70/100).

The map is relatively diffuse across many regions. About 61.3% of the image exceeds the activation threshold (0.20). Peak activation is 1.00; mean activation is 0.43.

Highlighted coverage is about 61.3% of the image. The main attention box spans rows 0–223 and columns 0–223. The center of attention is near pixel (row 148.5, col 117.7). The largest connected highlight covers 30510 pixels (60.81% of the map).

The methods largely agree on where to look. Agreement score is 79.1/100 (mean correlation 0.381, mean cosine 0.582).

The model’s prediction confidence is 68.1% (Moderate). Separately, explanation confidence is 78.1% (High). These measure different things and should not be treated as the same.

Overall explanation quality is 69.7/100 (Moderate).

### Guidance

Use the overlay next to the original evidence; do not rely on numbers alone. Explanation confidence is not a legal certainty score.

## Generated files

- `activation_histogram.png`
- `coverage_map.png`
- `focus_distribution.png`
- `explanation_analysis.json`
- `focus_metrics.json`
- `summary.json`
- `summary.md`
