# SHAP Attribution Report

- Investigation ID: `INV-2026-000001`
- Image: `original.png`
- Prediction: **REAL** (68.10%)
- Target class: `0`
- Model: `efficientnet_b0` / `sprint3.2-best`
- Dataset: `MAYA-Kaggle-Balanced-v1`
- Device: `CPU`
- Max evaluations: `16`
- Generation time: `26712.09 ms`
- Timestamp: `2026-08-26T15:21:29.804056+00:00`

## Attribution statistics (measured)

- Mean |SHAP|: `3.44e-06`
- Max |SHAP|: `6.15e-06`
- Mean positive: `3.4e-06`
- Mean negative: `-3.49e-06`
- Positive fraction: `62.50%`
- Negative fraction: `37.50%`

## Interpretation

SHAP estimates **feature contribution** to the predicted class score.
This is **not** a Grad-CAM attention / activation map.
Attribution estimates do not independently prove real-world causation.

## Artefacts

- Heatmap: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint5\shap\shap_heatmap.png`
- Overlay: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint5\shap\shap_overlay.png`
- Comparison: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint5\shap\shap_comparison.png`
