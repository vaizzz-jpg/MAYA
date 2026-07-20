# Pipeline Summary

1. Validate input image
2. Load / reuse cached model
3. Preprocess (resize + ImageNet normalize)
4. Softmax probabilities
5. Confidence + threshold decision
6. Allocate investigation ID
7. Persist artefacts

Last result: `INV-2026-000001` → REAL
