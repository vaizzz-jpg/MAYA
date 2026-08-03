# MAYA Explanation Report

- Investigation ID: **INV-2026-000001**
- Image: `2793.jpg`
- Prediction: **REAL**
- Confidence: **68.10%**
- REAL probability: `0.681009`
- FAKE probability: `0.318991`
- Explainer: `gradcam`
- Target Layer: `features.8`
- Target Class Index: `0`
- Generation Time: `2543.73 ms`
- Model: `efficientnet_b0` / `sprint3.2-best`
- Dataset Version: `MAYA-Kaggle-Balanced-v1`
- Device: `CPU`
- Timestamp (UTC): `2026-07-20T17:07:29.513019+00:00`

## Generated Files

- Original: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint1\original.png`
- Heatmap: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint1\heatmap.png`
- Overlay: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint1\overlay.png`
- Comparison: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint1\comparison.png`

## Investigator Notes

Grad-CAM highlights spatial regions that most influenced the classifier
score for the predicted class. Treat the map as an **indicator**, not
legal certainty. Review the overlay beside the original evidence image.
