# Explanation Trust Report

- Investigation ID: `INV-2026-000001`
- Image: `original.png`
- Prediction: **REAL** (68.10%)
- MAYA Explanation Trust Score: **65.52/100** (Moderate-to-High)

## Components (measured)

| Component | Score (0–100) | Weight used |
|-----------|---------------|-------------|
| quality | 68.43 | 0.3571 |
| faithfulness | 50.11 | 0.4286 |
| localization | 91.48 | 0.2143 |

Missing components: consistency, agreement

## Methodology

MAYA Explanation Trust Score summarizes measured explanation characteristics with configurable weights. It is not a probability that the explanation is scientifically true.

## Investigator summary

# Advanced XAI Investigator Summary

**Prediction:** REAL
**Model confidence:** 68.1%

## Primary visual evidence
Primary visual evidence uses `gradcam` activation localization. Explanation quality is 68.4/100 (Moderate). About 58.4% of the map exceeds the activation threshold; peak activation is 1.00. Center of activation is near (row 155.5, col 107.9) with coverage 58.4%. These describe model activation/localization, not legal certainty.

## SHAP contribution
Mixed positive/negative contribution across the image. Mean |SHAP|=0.0000; positive fraction=62.5%, negative fraction=37.5%. SHAP provides attribution estimates, not Grad-CAM attention.

## Faithfulness
Moderate measured influence of the highlighted regions on the class score. Important-region mask changed the target-class score by about 5.35 percentage points on average across 3 steps, versus 5.13 for less-important regions. This measures model sensitivity under perturbation — not proof that the model causally 'looked at' those pixels.

## Counterfactual sensitivity
Model counterfactual perturbation experiments show that 'blur' on the important region changed confidence by -8.85 percentage points (REAL 68.1% → REAL 59.3%). This is evidence of model sensitivity under controlled perturbation, not proof of real-world causation.

## Explanation assessment
MAYA Explanation Trust Score is 65.5/100 (Moderate-to-High). Components used: quality=68.4 (w=0.36), faithfulness=50.1 (w=0.43), localization=91.5 (w=0.21). Missing: consistency, agreement. This is an engineering summary of measured characteristics, not a probability that the explanation is true.

## Caveats
CAM heatmaps indicate activation/localization. SHAP provides attribution estimates. Faithfulness measures sensitivity to perturbations. Counterfactual experiments measure prediction changes under controlled edits. None of these independently prove real-world causation. Prefer: the model's prediction was strongly associated with highlighted regions — avoid claims of definite manipulation.
