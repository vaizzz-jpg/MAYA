# MAYA — Viva / Oral Defence Preparation Guide

## Quick elevator pitch

**MAYA = Media Authenticity Analyzer.** A modular, CPU-first, forensics-grade digital evidence investigation platform. Investigators upload evidence; MAYA runs EfficientNet-B0 authenticity classification + five layers of explainable AI (Grad-CAM through SHAP/Faithfulness/Counterfactual/Fusion/Trust), signs every uploaded file with server-computed SHA-256, writes an append-only audit trail, and produces a signed PDF investigation report with a complete audit timeline.

Built as a production Flask backend. Frontend will be developed separately.

---

## 1. Why EfficientNet-B0?

- Transfer learning backbone with ImageNet-generalizing features
- Depthwise-separable convolutions = small model + **low FLOPs = fits CPU-first / 8 GB Windows 11 target**
- 1280-d feature vector well suited to replace top classifier head (Linear 1280 → 2) for REAL/FAKE
- Outperforms equivalent MobileNet/VGG-size baselines on common deepfake detection benchmarks without retraining a giant backbone.

## 2. Why Grad-CAM?

- Spatial heatmaps without needing segmentation masks or human labels
- Generalizes to ANY CNN classifier via feature-map × gradient back-prop
- Standard baseline in AI forensics + XAI literature
- Produces human-interpretable overlay PNG in **1–2 forward+backward passes** (low CPU cost).

## 3. Why multiple explainers?

- Every CAM family (Grad-CAM, Grad-CAM++, LayerCAM, ScoreCAM, EigenCAM) weights gradients differently.
- Consensus (agreement across 5) rules out single-method artifacts.
- Produces cross-validated XAI for human review = forensics-relevant.

## 4. Why SHAP?

- Only axiomatic local attribution method (consistency, efficiency, linearity, symmetry)
- Signed contributions (pixel groups pushed REAL vs FAKE) — answers "why" not just "where"
- Deeper quantitative insight than heatmaps alone; good for investigator QA.

## 5. Why faithfulness?

- "If I occlude what the explainer claims caused the decision, does prediction flip?"
- Pure validation step; closes the loop around "we showed a heatmap" → "this heatmap actually mattered to classifier output".
- Forensics = adversarial setting; claims MUST be independently validated.

## 6. Why counterfactual explanations?

- Answer "what minimal change would flip the prediction?"
- Human interpretable and legally defensible explanation form: "this is fake because of X; flipping X → real".
- Works in feature space for image edits investigators can reason about.

## 7. Why explanation fusion?

- Merges Grad-CAM + SHAP + counterfactual consensus visualization.
- Human studies consistently prefer fused explanations over any single explainer.
- Improves trust scores (Phase 4.3 analytics) and auditor comprehension.

## 8. Why SHA-256?

- Cryptographic hash function — forensic chain-of-custody standard
- Server-side only (never trust client hash)
- Evidence integrity lifecycle: on upload → on every verify call → every analysis optionally → embedded in PDF + audit trail
- Distinguishes VALID / MODIFIED / MISSING / ERROR — traceable per investigation.

## 9. Why audit logs?

- Append-only immutable trail of every user action required for legal investigations / lab procedures / accreditation (ISO 27001, NIST CSF).
- Structured `details_json` with secrets scrubbed.
- Admins can review the complete investigation trail for any user.

## 10. Why Flask?

- Python-first: seamless torch + torchvision + reportlab integration (key requirement for AI + PDF).
- Application factory pattern + blueprints = clean modular monolith boundaries.
- Flask-Login out-of-the-box secure session cookies.
- Matches existing Phase 0–3 architecture decisions; no need to introduce FastAPI/ASGI complexity while targeting CPU-first sync inference.

## 11. Why modular architecture?

- Service / integration / AI layer separation = swappable implementations:
  - Swap SQLite → Postgres without touching ai/ code.
  - Replace EfficientNet → Vision Transformer without touching the API routes.
  - Build React/Svelte frontend later without backend rewrites.
- Separation of concerns for academic viva: traceable packages.

## 12. How an investigation flows through MAYA

1. Analyst registers/login (Flask session cookie).
2. Creates a Case → `CASE-YYYY-NNNNNN`.
3. Uploads evidence → server generates UUID filename, validates size/MIME/ext, computes server SHA-256 → `EVIDENCE_UPLOADED` audit.
4. (Optional) Integrity verify → VALID/MODIFIED/MISSING/ERROR.
5. Runs analysis → backend calls `InferencePipeline.run(image)`:
   - cached EfficientNet checkpoint
   - confidence + prediction label
   - `INV-YYYY-NNNNNN` investigation ID
   - optional basic Grad-CAM
   - optional advanced XAI (SHAP/fusion/faithfulness/counterfactual/trust) on opt-in.
6. Analysis row + artifacts persisted, evidence updated, `ANALYSIS_STARTED/XAI_GENERATED/ANALYSIS_COMPLETED` audit.
7. Generate PDF investigation report → `RPT-YYYY-NNNNNN` with report SHA-256 + audit timeline + investigator notes + limitations.
8. Download report (ownership-checked, path-traversal safe).
9. Close case → `CASE_CLOSED`.
10. Logout → `USER_LOGOUT` audit.

## 13. Current limitations (honest answer = best in viva)

- **CPU-first**: SHAP + fusion still ~30–100× inference per call; opt-in by design.
- **Backend-only**: No frontend yet. UI/UX is future work.
- **SQLite default**: Production deployments should switch to Postgres; SQLite is appropriate for lab/demo.
- **No mTLS/S3/deduplication**: Minimal reproducible deployment; enterprise features left as future.
- **Model weights**: Trained on Phase 2 dataset — transfer-generalized, not a government-grade certified detector.
- **No rate-limiting/LDAP/OIDC**: Intentionally omitted for scope; document as future.

## 14. Future frontend architecture (briefly, as promised)

- SPA: React + TypeScript + Vite.
- Routes: Login, Register, Cases (table/grid), Case Detail → Evidence List → Evidence Detail, Investigation Viewer with:
  - Prediction + confidence card
  - Side-by-side Original / Grad-CAM / SHAP overlay viewer with tabs per explainer
  - Faithfulness chart (Deletion/Insertion AUC)
  - Counterfactual comparison
  - Trust / quality gauge
  - SHA-256 integrity badge
  - Full audit timeline
  - Report download + notes edit
- API: Fetch JSON using session cookie-based auth.
- Hosted statically behind same TLS reverse proxy as backend.
