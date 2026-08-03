# MAYA — Phase 4 Sprint 4.1: Explainable Intelligence Engine (Grad-CAM Foundation)

**Status:** Complete  
**Scope:** Reusable Explainability Engine with Grad-CAM as the first registered explainer — **no Grad-CAM++, LayerCAM, ScoreCAM, EigenCAM, Flask, frontend, SHA-256, or PDF**

---

## Goal

Give investigators a trustworthy visual indicator of *where* the EfficientNet-B0 authenticity classifier focused when scoring an evidence image — without coupling XAI logic into training, evaluation, inference, or benchmarking packages.

---

## Explainable AI (why this exists)

MAYA is a digital-evidence investigation platform. A black-box REAL/FAKE score is insufficient for forensic review (SRS **FR-XAI-01/02/03**). Sprint 4.1 delivers the foundation of the Explainable Intelligence Engine so later sprints can embed heatmaps into case reports and UI side-by-side views.

---

## Why Grad-CAM works

Grad-CAM (Selvaraju et al., **ICCV 2017**) is implemented from the paper equations
in native PyTorch (no external Grad-CAM dependency):

1. \( \alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A_{ij}^k} \) — GAP of grads w.r.t. feature maps  
2. \( L^c_{\mathrm{Grad\text{-}CAM}} = \mathrm{ReLU}\big(\sum_k \alpha_k^c A^k\big) \) — weighted maps + ReLU  

where \( y^c \) is the class score **before softmax**. Visualization upsamples
(bilinear) then min-max normalizes for display.

Positive regions correlate with evidence that increased the class score. MAYA
treats the map as an **indicator**, not legal certainty.

---

## Architecture

```
Evidence image
      │
      ▼
ExplainabilityEngine          (orchestrator only)
      │
      ├── ModelLoader + preprocess_image   (reuse Sprint 3.4)
      ├── TargetLayerResolver              (EfficientNet-B0 → features[-1])
      ├── ExplainerRegistry.get("gradcam")
      │         └── GradCAM.generate()     (raw CAM only)
      ├── heatmap.prepare_heatmap()
      ├── overlay.overlay_heatmap()
      ├── visualization.write_visualization_bundle()
      └── explanation.* reports / JSON
```

### Independence rules

| Package | Coupling |
|---------|----------|
| Training / evaluation / benchmark | **None** — never imported |
| Inference | **Composition only** — loader, preprocess, confidence, investigation IDs |
| Flask / UI | **None** |

The inference pipeline may later *request* explanations; it must not own Grad-CAM math.

---

## Module relationships

| Module | Responsibility |
|--------|----------------|
| `base.py` | Abstract `Explainer` + `ExplainerOutput` |
| `registry.py` | Name → factory registry (`register` / `get` / `list_available`) |
| `hooks.py` | Forward + backward hooks + guaranteed cleanup |
| `target_layer.py` | Auto-resolve CAM layer; override path supported |
| `gradcam.py` | Grad-CAM math → raw heatmap only |
| `heatmap.py` | Normalize + resize |
| `overlay.py` | Colormap + opacity blend |
| `visualization.py` | PNG artefacts (original / heatmap / overlay / comparison) |
| `explanation.py` | `ExplanationResult` dataclass + Markdown/JSON writers |
| `engine.py` | End-to-end coordinator |
| `config.py` | Opacity, colormap, resolution, paths |
| `engine.py` | End-to-end coordinator |

Removed: a trivial `utils.py` helper module — small helpers live next to their callers.

---

## Registry pattern

Future explainers require **registration only**:

```python
@ExplainerRegistry.register("scorecam")
def _scorecam_factory(model, device):
    return ScoreCAM(model, device)
```

No `if/elif` chains inside the engine. Default registration: **GradCAM**.

---

## Memory optimization (8 GB / CPU-first)

- Batch size 1; eval mode retained (BN consistency with inference)
- Input `requires_grad` only for the Grad-CAM forward (frozen backbone still yields CAMs)
- Hooks removed via context manager after every explanation
- Intermediate tensors deleted; no duplicate image loads when the original path is available
- Prediction under `torch.no_grad` remains in inference; Grad-CAM uses a separate gradient-enabled path

---

## Integration with the inference engine

Sprint 3.4 documented the hook:

> Consume the same preprocessed tensor + model activations in `ai/explainability/`.

`ExplainabilityEngine` reuses:

- `ModelLoader` / checkpoint `artifacts/checkpoints/best.pt`
- `preprocess_image` (224×224 ImageNet normalize)
- `decide_from_probabilities` for investigator labels
- `InvestigationIDGenerator` for `INV-YYYY-NNNNNN`

It does **not** modify `InferencePipeline` in this sprint.

---

## Artefacts

Written under `artifacts/phase4/sprint1/`:

| File | Content |
|------|---------|
| `original.png` | Resized evidence copy |
| `heatmap.png` | Coloured CAM |
| `overlay.png` | Blended explanation |
| `comparison.png` | Side-by-side strip |
| `explanation.json` | Compact investigator payload |
| `explanation_report.md` | Full report + investigator notes |
| `pipeline_summary.md` | Pipeline steps |
| `explanation_result.json` | Full dataclass dump |

---

## Tests

`tests/test_gradcam.py` covers registry, target-layer detection, hook register/cleanup, heatmap, overlay, serialization, JSON/Markdown, CPU Grad-CAM, engine E2E, and post-run memory cleanup.

---

## Current limitations

- Grad-CAM only (no Grad-CAM++ / LayerCAM / ScoreCAM / EigenCAM yet)
- EfficientNet-B0 target-layer resolver only (other CNNs need a registered resolver or `target_layer_override`)
- No Flask routes, UI embedding, SHA-256 binding, or PDF packaging
- Single-image engine API (batch orchestration deferred)

---

## Future roadmap

1. Register Grad-CAM++ / LayerCAM / ScoreCAM / EigenCAM  
2. Optional attach `explanation_path` onto investigation artefacts  
3. Business-layer + Flask service to request explanations  
4. UI side-by-side original vs overlay (SRS analysis view)  
5. Embed explanation artefacts into PDF case reports  

---

## Definition of Done checklist

- [x] Grad-CAM on EfficientNet-B0  
- [x] Heatmaps + overlays  
- [x] Registry + base abstraction  
- [x] Automatic hook cleanup  
- [x] JSON + Markdown reports  
- [x] Tests  
- [x] This documentation  

**STOP after Sprint 4.1** — do not implement later explainers or backend APIs in this sprint.
