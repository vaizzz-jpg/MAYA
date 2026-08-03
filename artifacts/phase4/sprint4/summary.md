# Explainability Benchmark Summary

**Recommended explainer:** `gradcam` (score 67.1/100)

## Top ranks

1. `gradcam` — 67.1 (quality 68.4, 517 ms)
2. `eigencam` — 64.8 (quality 74.3, 101 ms)
3. `layercam` — 61.6 (quality 62.2, 215 ms)
4. `gradcam_plus_plus` — 61.5 (quality 62.1, 559 ms)
5. `scorecam` — 54.9 (quality 61.8, 50534 ms)

## Why

- `gradcam` ranked #1 with overall score 67.1/100 under the configured weights.
- Explanation quality 68.4/100; focus 49.6; localization 91.5.
- Coverage 58.4%; mean agreement 68.8%; generation 517.2 ms; peak RAM 458.1 MiB.
- Leads `eigencam` by 2.3 overall points.

## Caveats

- Recommendation is relative to this image and current weight configuration.
- Always review overlays beside the original evidence.
- Model confidence and explanation quality are different signals.
