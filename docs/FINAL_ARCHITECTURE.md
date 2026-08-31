# MAYA — Final Architecture Reference

## Layered Design (Binding rule)

```
 ┌──────────────────────────────────────────────────────┐
 │ PRESENTATION  — Jinja2 HTML shell (frontend/); No UI │
 │  logic — ONLY render minimal shell.                  │
 ├──────────────────────────────────────────────────────┤
 │ JSON API      — routes under backend/app/api/*.py    │
 │  Flask-Login sessions; JSON envelope ok/error.       │
 ├──────────────────────────────────────────────────────┤
 │ SERVICES      — backend/app/services/*.py            │
 │  Business rules, ownership, lifecycle orchestration  │
 ├──────────────────────────────────────────────────────┤
 │ INTEGRATIONS  — backend/app/integrations/ai_bridge.py│
 │  Calls ai.inference + ai.explainability — ZERO math │
 │  Reuses: ModelLoader cache, Investigation IDs.      │
 ├──────────────────────────────────────────────────────┤
 │ AI ANALYSIS   — ai/ (Flask-free; pure torch/numpy)   │
 │  datasets │ models │ training │ evaluation │ infer.  │
 │  explainability (4.1 Grad-CAM → 4.5 Advanced XAI)   │
 ├──────────────────────────────────────────────────────┤
 │ STORAGE       — SQLAlchemy 2.x (SQLite default) +    │
 │  uploads/ evidence uploads; artifacts/ AI/XAI runs;  │
 │  reports/ PDF investigation reports                  │
 └──────────────────────────────────────────────────────┘
```

## Module Responsibilities

| Module | Owner |
|---|---|
| `backend/app/__init__.py` | Application factory, directory setup, error handlers, blueprints |
| `backend/app/config/config.py` | Environment-based settings; `.env` via dotenv |
| `backend/app/models/entities.py` | 6 ORM entities: User, Case, Evidence, AnalysisRun, AuditLog, InvestigationReport |
| `backend/app/models/enums.py` | Role, status, event, integrity enums |
| `backend/app/api/*.py` | 5 blueprint groups: auth, cases, evidence, analysis (incl. reports), audit |
| `backend/app/services/*.py` | auth, case, evidence, analysis, report business logic |
| `backend/app/audit/service.py` | Append-only `record_audit()` writer |
| `backend/app/storage/` | UUID-safe filenames + path traversal guardrails |
| `backend/app/security/` | Werkzeug password hashing + `@login_required_api` + role check |
| `backend/app/schemas/` | `api_success` / `api_error` / ORM → dict |
| `backend/app/exceptions.py` | Typed MayaProductError hierarchy |
| `ai/inference/` | InferencePipeline, ModelLoader, Investigation IDs, confidence |
| `ai/explainability/` | Grad-CAM family + SHAP + faithfulness + counterfactual + fusion + trust + audit |
| `ai/datasets/` | Dataset pipeline — independent of product backend |

## Data Flow: End-to-End Investigation

```
POST /api/auth/login → Flask session cookie
  ↓
POST /api/cases → Case (CASE-YYYY-NNNNNN)
  ↓
POST /api/evidence/cases/{id} (multipart file)
  → storage.py: UUID filename, ext/MIME validation
  → hash_file → sha256_hash stored in Evidence
  → record_audit(EVIDENCE_UPLOADED)
  ↓
POST /api/evidence/{id}/verify-integrity
  → re-hash on disk vs stored hash → VALID / MODIFIED / MISSING / ERROR
  ↓
POST /api/evidence/{id}/analyze
  → services.analysis_service
    → verify integrity (optional)
    → record_audit(ANALYSIS_STARTED)
    → integrations.run_inference(image)
        → InferencePipeline
          → ModelLoader(cached EfficientNet-B0 best.pt)
          → preprocess_image
          → predict_probabilities
          → decide_from_probabilities → confidence / label
          → INV-YYYY-NNNNNN id
    → (optional) run_explanation → ExplainabilityEngine
        → ExplainerRegistry.{gradcam,gradcam++,layercam,scorecam,eigencam}
    → (optional) run_advanced_xai → AdvancedExplainabilityEngine
        → SHAP + faithfulness + counterfactual + fusion + trust
    → persist AnalysisRun + raw_result_json
    → evidence.analysis_status = COMPLETED / FAILED
    → record_audit(ANALYSIS_COMPLETED | ANALYSIS_FAILED, XAI_GENERATED)
  ↓
POST /api/analysis/{id}/report
  → report_service.generate_investigation_report → reportlab PDF
    → InvestigationReport entity (RPT-YYYY-NNNNNN, sha256, size, path)
    → record_audit(REPORT_GENERATED)
```

## Authentication Model

- **Mechanism:** Flask-Login `UserMixin` + Werkzeug session cookies
- **Secrets:** `SECRET_KEY` from `.env` (defaults to dev-only; `ALLOW_PUBLIC_REGISTRATION` gates signup)
- **Roles:** `INVESTIGATOR` (default, owns their cases/evidence/analyses/reports) + `ADMIN` (list all cases + all audit rows)

## Security Boundaries (verified)

- Server-computed SHA-256 (never trust client hash)
- Filenames: `uuid4().hex + .ext` under `UPLOAD_DIR/cases/{case_id}/`; resolved then root prefix-checked
- Artifacts under `ROOT_DIR/artifacts/investigations/{investigation_id}/` — download endpoint re-resolves and refuses escape
- DB parameterized only (SQLAlchemy ORM)
- No stack traces in JSON (MayaProductError hierarchy → safe message + error_code fields)
- Audit filter: non-admins only own user_id rows
- Ownership checks: case/evidence/analysis/services return 403 on cross-user access

## Performance (CPU-first, 8 GB RAM Windows 11)

- **Inference**: Process-scoped `InferencePipeline` → cached ModelLoader via `get_inference_pipeline()` in integrations
- **Image preprocessing**: Deleted tensor promptly after predict (`del prepared.tensor, probs` in `pipeline.run`)
- **XAI is opt-in**: `advanced_xai.*` booleans gate SHAP/fusion/faithfulness/counterfactual/trust stages
- **Response**: Persist `raw_result_json` once; return via schemas (no repeated serialization)
- **PDF report**: reportlab Platypus flowables with capped image scale (≤ 15×10 cm visual)
