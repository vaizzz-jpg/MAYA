# MAYA — Phase 3 Sprint 3

## AI Validation & Intelligence Reporting

**Status:** Complete  
**Scope:** Offline evaluation only — **no trainer/model rewrites, Grad-CAM, Flask, frontend, or SHA**

---

## Purpose

Turn a trained MAYA classifier into investigator-facing evidence: metrics, confidence-tagged predictions, plots, and consolidated Phase 3 documentation.

## Architecture

```
Model (+ checkpoint) + test DataLoader
            ↓
      Evaluator.evaluate()
       ├── collect predictions (thresholded)
       ├── MetricRegistry.calculate_all()
       ├── visualization.generate_all_plots()
       └── reporting.* (txt / json / md)
            ↓
   artifacts/phase3/sprint3/
```

| Module | Responsibility |
|--------|----------------|
| `evaluation_config.py` | Threshold, paths, class names |
| `metrics.py` | Pure metric formulas (no I/O / plots) |
| `metrics_registry.py` | Register / calculate-all pattern |
| `visualization.py` | Matplotlib figures only |
| `reporting.py` | Text/JSON/Markdown only |
| `evaluator.py` | Coordination + `EvaluationResult` |

## Prediction confidence

Every sample in `evaluation_result.json` stores:

- `predicted_class` / `predicted_label`
- `confidence_percentage` (max softmax × 100)
- `raw_probability` (full softmax vector)
- `threshold_used` (from `EvaluationConfig.threshold`, default **0.5**)

Binary decision: predict FAKE (positive class `1`) when `P(FAKE) >= threshold`.

## Metrics (registry)

Accuracy, Precision, Recall, F1, ROC AUC, Balanced Accuracy, MCC, Cohen Kappa, Sensitivity, Specificity, FPR, FNR.

Future metrics: `@MetricRegistry.register("name")` — no evaluator changes.

## Visualizations

| File | Source |
|------|--------|
| `confusion_matrix.png` | Test predictions |
| `roc_curve.png` | Positive-class scores |
| `precision_recall_curve.png` | Positive-class scores |
| `confidence_distribution.png` | Max-softmax histogram + threshold line |
| `accuracy_curve.png` | Sprint 2 `training_history.csv` |
| `loss_curve.png` | Sprint 2 `training_history.csv` |

## Reports

- `classification_report.txt`
- `metrics.json`
- `evaluation_result.json`
- `evaluation_summary.md`
- `phase3_report.md`

## How to run

```bash
python scripts/evaluate.py --threshold 0.5
python -m ai.evaluation.run_evaluation --max-batches 10   # smoke
python -m pytest tests/test_evaluation.py -q
```

Override threshold: `$env:MAYA_EVAL_THRESHOLD = "0.6"`

## Interactions with prior sprints

- Sprint 3.1 `ModelFactory` builds the net; checkpoints via Sprint 3.2 `load_checkpoint`.
- Phase 2 `create_dataloader(SplitName.TEST)` streams the sealed test split.
- Trainer / EfficientNet builders are **not** modified.

## Explicit non-goals

Inference HTTP APIs, Grad-CAM, SHA-256, Flask, frontend.
