# MAYA — Phase 4 Sprint 4.2: Multi-Explainer Framework

**Status:** Complete  
**Scope:** Plugin-based multi-explainer framework — Grad-CAM, Grad-CAM++, LayerCAM, ScoreCAM, EigenCAM — **no Flask, frontend, SHA-256, or PDF**

---

## Goal

Turn the Sprint 4.1 Explainability Engine into a **plugin framework** where Grad-CAM is one registered explainer among many. New algorithms require only:

1. Implement a class inheriting `Explainer` / `BaseExplainer`
2. Register a factory with `ExplainerRegistry`

**No engine modifications** when adding explainers.

---

## Plugin architecture

```
ExplainabilityConfig.explainer_name / comparison_explainers
                │
                ▼
        ExplainerRegistry.create(name)
                │
                ▼
        Explainer.generate()  →  ExplainerOutput (raw CAM)
                │
                ▼
     heatmap → overlay → artefacts / comparison
```

| Piece | Role |
|-------|------|
| `base.py` | `Explainer` / `BaseExplainer` contract |
| `registry.py` | `register` / `get` / `create` / `list_available` |
| `explainers/*` | Algorithm plugins (paper implementations) |
| `comparison.py` | Multi-explainer side-by-side + JSON/MD |
| `engine.py` | `explain()` (single) + `compare()` (multi) |

Engine never contains `if explainer == ...` algorithm branches.

---

## Registry pattern

```python
@ExplainerRegistry.register("my_cam")
def _factory(model, device):
    return MyCAM(model, device)
```

Built-ins registered on import via `register_default_explainers()`:

| Key | Class | Paper |
|-----|-------|-------|
| `gradcam` | `GradCAM` | Selvaraju et al., ICCV 2017 |
| `gradcam_plus_plus` (`gradcam++`) | `GradCAMPlusPlus` | Chattopadhay et al., WACV 2018 |
| `layercam` | `LayerCAM` | Jiang et al., IEEE TIP 2021 |
| `scorecam` | `ScoreCAM` | Wang et al., CVPRW 2020 |
| `eigencam` | `EigenCAM` | Muhammad & Yeasin, 2020 |

---

## Supported explainers (algorithm comparison)

| Method | Gradients? | Idea | Notes |
|--------|------------|------|-------|
| Grad-CAM | Yes | GAP of `∂y/∂A` as channel weights | Fast baseline |
| Grad-CAM++ | Yes | Pixel-wise higher-order weighting | Better multi-instance focus |
| LayerCAM | Yes | `ReLU(∂y/∂A) ⊙ A` spatial weights | Stronger on finer layers |
| Score-CAM | No | Masked-forward confidence as weights | Faithful but slow (K forwards) |
| EigenCAM | No | 1st PCA component of activations | Class-agnostic overview |

All implemented in **native PyTorch** (no `pytorch-grad-cam` dependency). Engineering details follow mature open-source practice where papers omit implementation specifics.

---

## Configuration

```python
ExplainabilityConfig(
    explainer_name="layercam",                 # single explain()
    comparison_explainers=(...,),              # compare()
    scorecam_batch_size=8,                     # CPU memory bound
    comparison_artifact_dir=.../phase4/sprint2,
)
# Env override: MAYA_XAI_EXPLAINER=gradcam_plus_plus
```

---

## Future extension process

1. Add `ai/explainability/explainers/my_cam.py` implementing `Explainer.generate`
2. Register factory inside `register_default_explainers()` **or** call `ExplainerRegistry.register` from your package
3. Set `explainer_name` / append to `comparison_explainers`
4. Done — do **not** edit `engine.py`

---

## Performance considerations (CPU / 8 GB)

- Gradient CAMs: one forward + one backward; hooks removed after each run
- EigenCAM: one forward + SVD on feature maps
- ScoreCAM: **O(channels)** forwards — batched (`scorecam_batch_size`); dominant cost on EfficientNet-B0 (~1280 channels)
- Comparison reuses one loaded model; releases tensors between explainers

---

## Artefacts

`artifacts/phase4/sprint2/`:

- `gradcam.png`, `gradcam_plus_plus.png`, `layercam.png`, `scorecam.png`, `eigencam.png`
- `comparison.png`, `comparison.json`, `comparison_report.md`

Sprint 4.1 paths (`explain()`, `ai.explainability.gradcam`) remain compatible.

---

## Tests

`tests/test_multi_explainer.py` — registry, plugin loading, config switching, algorithm maps, ScoreCAM on tiny CNN, EfficientNet gradient CAMs, comparison artefacts, serialization, hook/memory cleanup.

`tests/test_gradcam.py` — Sprint 4.1 suite must keep passing.

---

## Definition of Done

- [x] Multiple explainers via one engine  
- [x] New explainers without engine edits  
- [x] Comparison reports  
- [x] Registry functional  
- [x] Backward compatible with Sprint 4.1  

**STOP after Sprint 4.2.**
