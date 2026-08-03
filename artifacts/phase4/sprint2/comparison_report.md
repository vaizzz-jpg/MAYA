# MAYA Multi-Explainer Comparison Report

- Image: `2793.jpg`
- Prediction: **REAL** (68.10%)
- Model: `efficientnet_b0` / `sprint3.2-best`
- Target Layer: `features.8`
- Device: `CPU`
- Total Time: `77878.45 ms`
- Timestamp (UTC): `2026-07-20T17:24:43.494701+00:00`

## Explainers

### `gradcam`
- Generation time: `587.56 ms`
- Overlay: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint2\gradcam.png`
- Heatmap: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint2\gradcam_heatmap.png`

### `gradcam_plus_plus`
- Generation time: `413.72 ms`
- Overlay: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint2\gradcam_plus_plus.png`
- Heatmap: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint2\gradcam_plus_plus_heatmap.png`

### `layercam`
- Generation time: `371.18 ms`
- Overlay: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint2\layercam.png`
- Heatmap: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint2\layercam_heatmap.png`

### `scorecam`
- Generation time: `76231.53 ms`
- Overlay: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint2\scorecam.png`
- Heatmap: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint2\scorecam_heatmap.png`

### `eigencam`
- Generation time: `274.45 ms`
- Overlay: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint2\eigencam.png`
- Heatmap: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint2\eigencam_heatmap.png`

## Investigator Notes

Compare localization consistency across algorithms. Grad-CAM / Grad-CAM++ /
LayerCAM use gradients; Score-CAM uses forward scores; EigenCAM is
class-agnostic (leading activation PCA). Treat maps as indicators, not certainty.

Combined figure: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint2\comparison.png`
