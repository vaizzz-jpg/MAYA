# MAYA — XAI Layer Architecture

## Philosophy

Forensics-grade authenticity assessments must be **explainable, auditable, and trustworthy**. A black-box "REAL / FAKE + 72%” is operationally useless without answering:

- **Why did the model predict this?
- **Where in the image did it look?**
- **How much can we trust the explanation itself?**

MAYA layers five ascending layers of explainability:

1. **Grad-CAM (4.1) – visual spatial heatmap overlay
2. **Multi-explainer agreement (4.2) – 5 CAM variants cross-validated
3. **Explanation analytics + trust metrics (4.3) – focus / coverage / quality scores
4. **Benchmark + ranking (4.4) – explainer recommendations per image
5. **Advanced XAI (4.5): SHAP + faithfulness + counterfactual + fusion + audit trail

## Integration to product layer map

```
POST /api/evidence/{id}/analyze

  ├─ basic (generate_explanation=True, explainer="gradcam"
  │    └─ ExplainabilityEngine.run_explanation()
  │        └─ 4.1-4.2 heatmaps / overlays / comparison PNG
  │
  └─ advanced: advanced_xai: {shap, faithfulness, counterfactual, fusion, trust, multi_explainer: bool, explainer: "gradcam" | list}
       └─ AdvancedExplainabilityEngine (4.5)
            ├─ SHAP                     — pixel-level signed contributions
            ├─ Faithfulness evaluator  — perturb-and-predict fidelity
            ├─ Counterfactual engine   — minimal change needed to flip prediction
            ├─ Fusion engine           — fused multi-source explanation
            └─ Trust + quality scoring — explanation trust grade

Results stored in AnalysisRun:
  ├─ heatmap_path / overlay_path / explanation_json_path     (basic)
  ├─ raw_result_json["advanced_xai_results"]                  (advanced)
  ├─ trust_score / quality_score                            (denormalized columns)
  └─ artifact_dir per investigation_id folder
```

## Configuration per-request cost table

| Feature | Cost (relative) | Opt-in flag | Returns |
|---|---|---|---|
| Basic inference | Baseline — always | Always | prediction + | EfficientNet-B0 base inference |
| Grad-CAM overlay | ~2× inference | `generate_explanation=True` | heatmap + overlay + JSON |
| 5-explainer comparison + comparison | ~8–12× inference | `advanced_xai.multi_explainer=true` | `explainers.* in advanced_xai_results |
| SHAP image | ~30–100× inference | `advanced_xai.shap=true` | shap.* entries |
| Faithfulness | ~5× inference | `advanced_xai.faithfulness=true` | faithfulness.comparison + report |
| Counterfactual | ~20–40× inference | `advanced_xai.counterfactual=true` | counterfactual.comparison + report |
| Fusion | ≥2 methods required | `advanced_xai.fusion=true` | fused explanation + report |
| Trust / quality scores | Low (cheap given above | `advanced_xai.trust=true` | trust + components + grade |

## CPU-first design

- Every explainer engine supports `device_preference="cpu"`.

- SHAP runs with conservative `max_evaluations=64–256 typical image patch.

- Every engine releases tensors and images (no long-lived GPU references; no CUDA used.

- XAI artifacts are written as PNG + JSON under `artifacts/investigations/INV-YYYY-NNNNNN/xai/{explainers,advanced/}` and is traceable to InvestigationReport records for PDF ingestion.

## Why this design?

- **Why EfficientNet-B0?** Mobile-size backbone: smallest ImageNet transfer learning feature extractor generalizes well to unseen face manipulations while staying within 8 GB RAM Windows target — fits CPU-friendly depthwise separable convolutions = low FLOPs budget for forensic CNN feature pyramid semantics.

- **Why Grad-CAM first?** Human interpretable localization without requiring input gradients back-prop through final conv feature maps; no segmentation masks; works for any CNN classifier; widely used as a baseline in deepfake detection XAI literature.

- **Why multiple explainers?** Each CAM family explainer biases (Grad-CAM, Grad-CAM++, LayerCAM, ScoreCAM, EigenCAM) respond to different gradient signal properties (gradient weight computation paths → agreement across 5 explainers → robust consensus, not single-method artifacts.

- **Why SHAP?** Local game-theoretic Shapley values with strict axioms: consistency, efficiency, symmetry, linearity → gives signed attribution signs (anti-forensics beyond just "where" → signed pixels pushed FAKE vs REAL.

- **Why faithfulness?** "If I occlude what the explainer claimed matters → does prediction flip? = true faithfulness metric = Deletion / Insertion AUC curves → validates visualizations actually influenced classification correctness.

- **Why counterfactual?** "What minimal change would flip the decision?" = for human-legible minimal edit for investigators — complements attribution.

- **Why explanation fusion?** Merges 3+ explanations (e.g., Grad-CAM + SHAP + counterfactual → fused visualizations consistently ranked higher by humans = overall trust.

- **Why SHA-256 in evidence?** Cryptographic hash function chain-of-custody requirement forensics.

- **Why audit logs?** Legal-probational investigation.

- **Why Flask monolith?** SRS-mandated, modulaer forensics pipeline modular monolith: Flask matches Phase 0–4 design doc: Python ecosystem torch ← simple; no message broker or separate inference micro-services complexity.

- **Why modular architecture?** services / integrations / ai packages clean separation; future frontend, separate Celery workers, easy swap.

## How an investigation flows through MAYA
Uploads through the complete investigation end-to-end → see docs/INVESTIGATION_WORKFLOW.md.
