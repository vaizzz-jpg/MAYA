# Faithfulness Evaluation Report

- Investigation ID: `INV-2026-000001`
- Image: `original.png`
- Prediction: **REAL**
- Original confidence: `68.10%`
- Original target-class score: `0.6810087`
- Perturbation method: `mask`
- Steps: `3`
- Deletion score: `0.0534929`
- Insertion proxy score: `0.02105607`
- Faithfulness score: `0.501103`
- Mean confidence drop (important): `5.3493` pp
- Mean confidence drop (less-important): `5.1287` pp
- Generation time: `843.17 ms`

## Step measurements

| Step | Fraction | Important score | Unimportant score | Imp drop | Unimp drop |
|------|----------|-----------------|-------------------|----------|------------|
| 1 | 0.100 | 0.6608 | 0.6516 | 0.0202 | 0.0295 |
| 2 | 0.200 | 0.6143 | 0.6210 | 0.0667 | 0.0600 |
| 3 | 0.300 | 0.6075 | 0.6166 | 0.0735 | 0.0644 |

## Interpretation

Moderate measured influence of the highlighted regions on the class score. Important-region mask changed the target-class score by about 5.35 percentage points on average across 3 steps, versus 5.13 for less-important regions. This measures model sensitivity under perturbation — not proof that the model causally 'looked at' those pixels.

Visualization: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint5\faithfulness\faithfulness_comparison.png`
