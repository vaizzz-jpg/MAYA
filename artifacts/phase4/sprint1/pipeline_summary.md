# Explainability Pipeline Summary

1. Validate input image
2. Load / reuse cached model (eval mode)
3. Preprocess (resize + ImageNet normalize)
4. Resolve target layer (or apply override)
5. Load explainer from registry (Grad-CAM)
6. Forward + backward with activation/gradient hooks
7. Normalize / resize heatmap
8. Overlay heatmap onto original
9. Persist visualization artefacts + JSON / Markdown reports
10. Remove hooks and release tensors
