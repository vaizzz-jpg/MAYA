# Counterfactual Perturbation Report

- Investigation ID: `INV-2026-000001`
- Image: `original.png`
- Original: **REAL** (68.10%)
- Target class: `0`
- Generation time: `929.17 ms`

## Experiments (measured)

### Strategy: `mask`

- Affected region: `bbox[r0:223,c0:223] coverage=50.0%`
- Original: REAL 68.10%
- Perturbed: REAL 61.64%
- Confidence change: `-6.46` percentage points
- Prediction changed: `False`
- Target-class score: `0.6810087` → `0.61643666` (Δ `-0.0646`)

### Strategy: `blur`

- Affected region: `bbox[r0:223,c0:223] coverage=50.0%`
- Original: REAL 68.10%
- Perturbed: REAL 59.25%
- Confidence change: `-8.85` percentage points
- Prediction changed: `False`
- Target-class score: `0.6810087` → `0.59253591` (Δ `-0.0885`)

### Strategy: `neutral`

- Affected region: `bbox[r0:223,c0:223] coverage=50.0%`
- Original: REAL 68.10%
- Perturbed: REAL 61.64%
- Confidence change: `-6.46` percentage points
- Prediction changed: `False`
- Target-class score: `0.6810087` → `0.61643666` (Δ `-0.0646`)

## Interpretation

Model counterfactual perturbation experiments show that 'blur' on the important region changed confidence by -8.85 percentage points (REAL 68.1% → REAL 59.3%). This is evidence of model sensitivity under controlled perturbation, not proof of real-world causation.

These are **model counterfactual perturbation experiments**, not generative real-world counterfactuals.

Visualization: `C:\Users\Asus\Desktop\MAYA\artifacts\phase4\sprint5\counterfactual\counterfactual_comparison.png`
