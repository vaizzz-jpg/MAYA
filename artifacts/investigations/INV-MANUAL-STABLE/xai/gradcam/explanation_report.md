# MAYA Explanation Report

- Investigation ID: **INV-MANUAL-STABLE**
- Image: `original.png`
- Prediction: **REAL**
- Confidence: **68.10%**
- REAL probability: `0.681009`
- FAKE probability: `0.318991`
- Explainer: `gradcam`
- Target Layer: `features.8`
- Target Class Index: `0`
- Generation Time: `1379.98 ms`
- Model: `efficientnet_b0` / `sprint3.2-best`
- Dataset Version: `MAYA-Kaggle-Balanced-v1`
- Device: `CPU`
- Timestamp (UTC): `2026-08-26T15:12:35.639804+00:00`

## Generated Files

- Original: `artifacts\investigations\INV-MANUAL-STABLE\xai\gradcam\original.png`
- Heatmap: `artifacts\investigations\INV-MANUAL-STABLE\xai\gradcam\heatmap.png`
- Overlay: `artifacts\investigations\INV-MANUAL-STABLE\xai\gradcam\overlay.png`
- Comparison: `artifacts\investigations\INV-MANUAL-STABLE\xai\gradcam\comparison.png`

## Investigator Notes

Grad-CAM highlights spatial regions that most influenced the classifier
score for the predicted class. Treat the map as an **indicator**, not
legal certainty. Review the overlay beside the original evidence image.
