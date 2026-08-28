# MAYA — Phase 4 Sprint 4.5: Advanced Explainability & Trust Layer

**Status:** Complete
**Scope:** SHAP attributions, faithfulness testing, counterfactual perturbations, explanation fusion, configurable trust score, investigator summary, and XAI audit trail — **no CAM algorithm changes, no Phase 5, no SHA-256, no Flask/UI/PDF**

---

## 1. Sprint objective

Extend the Explainable Intelligence Engine so investigators can answer:

1. **WHERE** did the model focus? — existing CAM methods (unchanged)
2. **WHAT** contributed to the prediction? — SHAP attributions
3. **IS** the explanation faithful to the model? — deletion/occlusion faithfulness
4. **WHAT** controlled changes alter the score? — model counterfactual perturbation experiments

Results are fused into a structured trust assessment and plain-language summary.

---

## 2. Existing XAI architecture (preserved)

```
Evidence image
      │
      ▼
ExplainabilityEngine (4.1/4.2) → CAM plugins via ExplainerRegistry
      │
      ▼
ExplanationAnalyticsEngine (4.3) → focus / localization / quality / consistency
      │
      ▼
ExplainabilityBenchmarkSuite (4.4) → rank registered CAM explainers
```

Sprint 4.5 adds a **sibling** advanced layer. SHAP / faithfulness / counterfactual are **not** registered as CAM plugins (they do not produce `ExplainerOutput.raw_heatmap` under the Grad-CAM contract). Phase 4.4 benchmarking remains valid.

---

## 3. Why SHAP was added

CAM heatmaps localize **activation**. Investigators also need estimates of **feature contribution** (which regions push the class score up or down). SHAP (Lundberg & Lee, NeurIPS 2017) provides a maintained attribution method appropriate for this distinction.

---

## 4. SHAP methodology

Implementation lives under `ai/explainability/shap/` and uses the maintained **`shap`** library (Partition explainer + `shap.maskers.Image` blur masker).

**Dependencies (required for SHAP image explanations):**

| Package | Role |
|---------|------|
| `shap>=0.44.0` | Attribution engine |
| `opencv-python-headless>=4.8.0` | Required by `shap.maskers.Image` blur/inpaint (import name: `cv2`) |

Install into the **active project `.venv`**:

```bash
python -m pip install "shap>=0.44.0" "opencv-python-headless>=4.8.0"
# or
python -m pip install -r requirements.txt
```

Do **not** install both `opencv-python` and `opencv-python-headless`.
If SHAP/OpenCV is missing when SHAP is requested, MAYA raises `ShapDependencyError` with an actionable install hint — it does **not** silently fall back to Grad-CAM.

Pipeline:

1. Reuse loaded EfficientNet-B0 + Sprint 3.4 preprocessing
2. Convert the normalized tensor to HWC `[0,1]` for the masker
3. Model function re-applies ImageNet normalization and returns class probabilities
4. Run SHAP with a hard `max_evaluations` budget (default **32**, CPU-first)
5. Reduce values to a signed HxW attribution map for the target class
6. Split positive / negative contributions; normalize for display
7. Emit `ShapResult` dataclass + visualizations

Artefacts: `shap_heatmap.png`, `shap_overlay.png`, `shap_comparison.png` (explicitly labeled as **contribution**, not attention).

---

## 5. Difference between CAM and SHAP

| Method | Meaning |
|--------|---------|
| Grad-CAM / Grad-CAM++ / LayerCAM / ScoreCAM / EigenCAM | Model **activation / localization** maps |
| SHAP | **Attribution estimates** of feature contribution to the class score |

Do not treat SHAP overlays as Grad-CAM attention maps.

---

## 6. Faithfulness methodology

Module: `ai/explainability/faithfulness/`

Deletion/occlusion protocol:

1. Record original target-class score
2. Rank pixels by an importance map (prefer `|SHAP|`, else primary CAM)
3. For `steps` (default **5**), progressively perturb the top-k important vs least-important pixels
4. Re-run the **same loaded model** under `inference_mode` / `no_grad`
5. Compare score drops

Reported metrics:

- prediction / confidence drop (important vs less-important)
- deletion score
- insertion proxy (relative advantage of important vs unimportant deletion)
- **faithfulness score** ∈ `[0, 1]` summarizing measured influence

Wording used in reports: **measured influence** / **model sensitivity to highlighted regions** — not causal proof that the model “looked at” a region.

---

## 7. Counterfactual perturbation methodology

Module: `ai/explainability/counterfactual/`

These are **model counterfactual perturbation experiments**, not generative edits:

- Identify an important region (thresholded importance map)
- Apply configurable strategies: `mask`, `blur`, `neutral`
- Compare original vs perturbed prediction and confidence

They measure prediction change under controlled perturbation — **not** real-world causation and **not** photorealistic counterfactuals.

---

## 8. Explanation fusion

Module: `ai/explainability/fusion/`

Fusion **preserves distinct meanings**:

- Visual evidence (CAM analytics)
- SHAP contribution
- Faithfulness
- Counterfactual sensitivity

It does **not** naively average unrelated raw scores into one opaque number without documenting components. Missing stages are omitted; trust weights are renormalized over available components only (no invented values).

Coordinator: `AdvancedExplainabilityEngine` (`ai/explainability/advanced_engine.py`).

---

## 9. Trust score methodology

**MAYA Explanation Trust Score** — an engineering composite in `[0, 100]`, **not** a probability that an explanation is scientifically true.

Default weights (`TrustScoreWeights` in `ai/explainability/config.py`):

| Component | Default weight | Source |
|-----------|----------------|--------|
| Quality | 25% | Sprint 4.3 explanation quality (0–100) |
| Faithfulness | 30% | Faithfulness score (0–1 → ×100) |
| Consistency | 20% | Cross-explainer consistency when available |
| Localization | 15% | Analytics localization component |
| Agreement | 10% | Agreement score when available |

Weights are configurable and always normalized. Grades: High / Moderate-to-High / Moderate / Low / Insufficient.

---

## 10. Audit trail

`XaiAuditRecord` (`ai/explainability/fusion/audit.py`) records:

- investigation_id, image identifier
- model name / version, dataset version
- configuration snapshot
- target class / target layer (when applicable)
- timestamp, device, generation time
- artefact paths
- software/library versions (`torch`, `numpy`, `shap`, …)

Written to `artifacts/phase4/sprint5/reports/xai_audit.json`.
**SHA-256 / digital integrity is intentionally deferred** (later phases).

---

## 11. Performance considerations (8 GB / CPU-first)

- Single cached `ModelLoader` instance
- SHAP `max_evaluations` capped (default 32)
- Faithfulness `steps` capped (default 5)
- Counterfactual limited to configured strategies
- `torch.inference_mode` / `no_grad` for prediction paths
- Intermediate tensors released after stages
- Stages can be disabled via configuration

---

## 12. Limitations

- SHAP Partition estimates with few evaluations are approximate
- Faithfulness and counterfactual use occlusion/blur/neutral fills — synthetic edits
- Trust score weights are heuristic engineering choices
- Primary CAM analytics in the advanced run use a single configured explainer (default Grad-CAM)
- No backend/UI/PDF integration in this sprint

---

## 13. Scientific interpretation

Explicitly:

- **CAM** heatmaps indicate model activation/localization.
- **SHAP** provides attribution estimates.
- **Faithfulness** measures sensitivity of the model to perturbations of highlighted regions.
- **Counterfactual experiments** measure prediction changes under controlled perturbations.

**None of these independently prove real-world causation.** Prefer investigator language such as: *“The model’s prediction was strongly associated with…”* rather than *“The face is definitely manipulated because…”*.

---

## 14. Future improvements

- Persist float attribution maps (`.npy`) for lossless analytics
- Optional calibrated trust models
- Attach advanced XAI packages to Phase 6 integrity + Phase 7 case reports
- Backend APIs (Phase 8) to request advanced explanations

---

## Configuration

```python
from ai.explainability import AdvancedXAIConfig, AdvancedExplainabilityEngine

cfg = AdvancedXAIConfig(
    # shap.max_evaluations, faithfulness.steps, trust weights, …
)
result = AdvancedExplainabilityEngine(cfg).analyze("path/to/evidence.jpg")
print(result.trust.trust_score, result.summary.explanation_assessment)
```

Environment overrides (examples): `MAYA_XAI_SHAP_MAX_EVALS`, `MAYA_XAI_FAITHFULNESS_STEPS`, `MAYA_XAI_SHAP_ENABLED`.

---

## Artefact isolation & investigation linkage

`ExplainabilityEngine.explain()` writes to **`config.artifact_dir`** (or a per-call `artifact_dir=` override). The output directory is **never** inferred from the input image path. The Sprint 4.1 default (`artifacts/phase4/sprint1/`) is intentional for that sprint’s layout — for new investigations, pass an isolated directory:

```python
from ai.explainability import ExplainabilityEngine, ExplainabilityConfig

cfg = ExplainabilityConfig(
    device_preference="cpu",
    # Share ID counter with inference when desired:
    # id_state_path=Path("artifacts/phase3/sprint4/investigation_id_state.json"),
)
engine = ExplainabilityEngine(cfg)
result = engine.explain(
    r"path\to\evidence.jpg",
    investigation_id="INV-2026-000002",  # reuse predict() ID
    artifact_dir=r"artifacts\investigations\INV-2026-000002\xai\gradcam",
)
```

**Target class:** `ExplainabilityConfig.target_class is None` (default) → explain the **predicted** class from the same checkpoint/preprocessing as inference. Set `target_class=0` or `1` to override.

**matplotlib / pyparsing deprecation warnings** seen in pytest come from third-party libraries and are non-blocking; they are not suppressed globally.

---

## Artefacts

Under `artifacts/phase4/sprint5/`:

| Folder | Content |
|--------|---------|
| `shap/` | SHAP heatmaps / overlays / comparison |
| `faithfulness/` | `faithfulness_comparison.png` + metrics |
| `counterfactual/` | `counterfactual_comparison.png` + experiments |
| `fusion/` | Fused JSON |
| `visualizations/` | `trust_summary.png` |
| `reports/` | `shap_report.md`, `faithfulness_report.md`, `counterfactual_report.md`, `explanation_trust_report.md`, `advanced_xai_summary.md`, `advanced_xai_result.json`, `xai_audit.json` |

---

## Tests

| File | Coverage |
|------|----------|
| `tests/test_shap.py` | Init, CPU SHAP, shapes, serialization, visuals |
| `tests/test_faithfulness.py` | Perturbation, deltas, faithfulness score, invalid input |
| `tests/test_counterfactual.py` | Perturbation, prediction compare, JSON |
| `tests/test_fusion.py` | Combination, missing components, no fake averaging |
| `tests/test_trust.py` | Weights, bounds, grades |
| `tests/test_audit.py` | Metadata, timestamp, config, JSON |

Phase 3 / 3.5 / 4.1–4.4 suites must continue to pass.

---

## References

- Selvaraju et al., *Grad-CAM*, ICCV 2017.
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017.

Implementation uses the maintained `shap` library and MAYA’s modular architecture — tutorial code is not copied.

---

## Definition of Done

- [x] SHAP on EfficientNet-B0 pipeline (real attributions)
- [x] Faithfulness measured results
- [x] Counterfactual perturbation experiments
- [x] Fusion + configurable trust score
- [x] Investigator summary + audit trail
- [x] Reports / visualizations from actual computation
- [x] CPU-friendly limits
- [x] CAM algorithms / Phase 4.4 untouched
- [x] Tests + documentation

**STOP after Sprint 4.5.** Do not implement Phase 5, SHA-256, PDF, Flask APIs, or frontend work in this sprint.
