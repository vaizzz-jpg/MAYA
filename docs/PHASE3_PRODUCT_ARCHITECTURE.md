# MAYA — Phase 3 Product Architecture

**Status:** Design / inspection complete — implementation gated on auth decision  
**Scope:** Authentication, Case Management, Evidence Management, Backend APIs, AI ↔ Backend Integration  
**Constraint:** Reuse existing Flask + SQLAlchemy + AI/XAI subsystems; do not rebuild them

---

## 1. Existing architecture (inspected)

### 1.1 Layers already in place

```
frontend/          Presentation (Jinja + static) — Phase 1 shell only
backend/app/       Flask application factory, config, DB wiring, routes stub
ai/                AI Analysis layer (NO Flask imports) — Phase 3.1–4.5 COMPLETE
uploads/           Evidence upload root (created by factory; unused by product yet)
reports/           Report root (unused by product yet)
artifacts/         AI/XAI computation artefacts (phase3/phase4 sprint folders)
dataset/           Training corpus (immutable raw + processed) — separate from case evidence
logs/              Rotating maya.log
```

### 1.2 Backend foundation (Phase 1)

| Component | Path | State |
|-----------|------|--------|
| Entrypoint | `backend/run.py` | Runs `create_app()` on `127.0.0.1:5000` |
| Factory | `backend/app/__init__.py` | Config, dirs, logging, DB, blueprints, HTML errors |
| Extensions | `backend/app/extensions.py` | `db = SQLAlchemy()` only |
| Config | `backend/app/config/config.py` | Dev/Test/Prod; `.env` via dotenv |
| DB init | `backend/app/database/init_db.py` | `create_all()` — **no business models yet** |
| Routes | `backend/app/routes/` | `GET /health`, `GET /` shell only |
| Models | `backend/app/models/` | **Empty stub** |
| Services | `backend/app/services/` | **Empty stub** |
| Auth package | `backend/app/auth/` | **Empty stub** |

**Framework decision (locked by SRS):** Flask monolith — **not FastAPI**.

### 1.3 Database technology

- **ORM:** Flask-SQLAlchemy + SQLAlchemy 2.x  
- **Default DB:** SQLite at `backend/instance/maya.db`  
- **Test DB:** `sqlite:///:memory:` via `TestingConfig`  
- **Migrations:** Not yet (Flask-Migrate deferred; `create_all` acceptable for this product sprint if documented)  
- **Design source of truth:** `docs/04_DATABASE_DESIGN.md`

### 1.4 AI / XAI public integration APIs (reuse only)

| Capability | Call site | Notes |
|------------|-----------|--------|
| Inference | `InferencePipeline(config).run(path) → InvestigationResult` | Cached `ModelLoader` |
| Investigation ID | `InvestigationIDGenerator` / pipeline-allocated `INV-YYYY-NNNNNN` | Persist in analysis row |
| Grad-CAM / CAM | `ExplainabilityEngine(cfg).explain(path, investigation_id=..., artifact_dir=...)` | Isolated artefact dirs |
| Advanced XAI | `AdvancedExplainabilityEngine(cfg).analyze(path)` | Optional / config-gated |
| SHA-256 | `ai.datasets.utils.checksums.hash_file(path)` | Reuse — do **not** duplicate |

**Boundary rule:** Routes and services **orchestrate**; `ai/` **computes**. Never import Flask from `ai/`.

### 1.5 Logging

`backend/app/utils/logging_config.py` → `logs/maya.log` + console. AI modules use `maya.ai.*` loggers. Product services must use `maya.*` / `maya.backend.*` — never log passwords or tokens.

### 1.6 What is missing (product)

- User / Case / Evidence / Analysis / Audit ORM models  
- Auth (login, roles, protected APIs)  
- Case / Evidence / Analysis REST (or JSON) APIs  
- Evidence upload + integrity verify endpoints  
- `AnalysisService` bridge to inference + XAI  
- Product audit log persistence  
- Backend pytest coverage (currently **zero** Flask tests)

---

## 2. Product architecture (target)

Extend the **existing** `backend/app/` layout — do **not** create a parallel `app/` tree at repo root.

```
backend/app/
├── __init__.py              # create_app (extend: register API blueprints, auth)
├── extensions.py            # db (+ login_manager or jwt extension — see §Auth decision)
├── config/                  # extend BaseConfig (JWT/session, upload limits, roles)
├── models/                  # User, Case, Evidence, AnalysisRun, AuditLog
├── schemas/                 # NEW: request/response dict builders / validation helpers
├── repositories/            # NEW thin data-access (optional if services stay small)
├── services/                # auth, case, evidence, analysis, audit
├── security/                # NEW: password hashing, auth decorators, ownership checks
├── storage/                 # NEW: safe evidence file storage under UPLOAD_DIR
├── integrations/            # NEW: AI bridge (calls InferencePipeline / ExplainabilityEngine)
├── audit/                   # NEW or under services/: append-only audit writer
├── api/                     # NEW JSON blueprints: auth, cases, evidence, analysis, audit
├── routes/                  # keep health + shell; HTML UI can come later
└── utils/                   # logging (existing)
```

### Responsibility split

| Layer | Owns |
|-------|------|
| `api/` | HTTP validation, status codes, auth context, response shaping |
| `services/` | Business rules, ownership, lifecycle |
| `integrations/` | Calls into `ai.inference` / `ai.explainability` only |
| `storage/` | Safe filenames, paths under `UPLOAD_DIR/cases/{case_id}/` |
| `security/` | Hashing, current user, role checks |
| `models/` | Persistence schema |
| `ai/` | Unchanged computation engines |

---

## 3. Database architecture (proposed entities)

Aligned with `docs/04_DATABASE_DESIGN.md`, extended for analysis status and evidence analysis state.

### 3.1 `users`

| Field | Type | Notes |
|-------|------|--------|
| id | Integer PK | |
| email | String unique | Login identity |
| username | String unique | |
| password_hash | String | Werkzeug (or project-standard) — never plaintext |
| full_name | String | |
| role | Enum | `INVESTIGATOR`, `ADMIN` |
| is_active | Boolean | |
| created_at | DateTime | |
| last_login_at | DateTime nullable | |

### 3.2 `cases`

| Field | Type | Notes |
|-------|------|--------|
| id | Integer PK | Internal |
| case_number | String unique | Human-facing stable ID (e.g. `CASE-2026-000001`) |
| title | String | |
| description | Text | |
| status | Enum | `OPEN`, `IN_PROGRESS`, `CLOSED`, `ARCHIVED` |
| priority | Enum | `LOW`, `MEDIUM`, `HIGH` (optional default MEDIUM) |
| created_by_user_id | FK users | Owner |
| created_at / updated_at / closed_at | DateTime | |

**Authorization:** Investigator sees own cases; Admin may list/access all (explicit).

### 3.3 `evidence`

| Field | Type | Notes |
|-------|------|--------|
| id | Integer PK | |
| case_id | FK cases | Required |
| original_filename | String | Display only |
| stored_filename | String | Server-generated safe name |
| storage_path | String | Relative under `UPLOAD_DIR` |
| media_type | String/Enum | Match AI: image types supported by inference |
| mime_type | String | Validated server-side |
| file_size_bytes | Integer | |
| sha256_hash | CHAR(64) | From `hash_file` |
| status | Enum | `UPLOADED`, `PROCESSING`, `ANALYZED`, `FAILED`, `ARCHIVED` |
| analysis_status | Enum | Mirror / denormalize last analysis: `NONE`, `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED` |
| uploaded_by_user_id | FK users | |
| uploaded_at | DateTime | |
| notes | Text nullable | |

### 3.4 `analysis_runs` (investigations)

| Field | Type | Notes |
|-------|------|--------|
| id | Integer PK | `analysis_id` |
| evidence_id | FK evidence | |
| case_id | FK cases | Denormalized for query speed |
| investigation_id | String | `INV-YYYY-NNNNNN` from AI pipeline |
| status | Enum | `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED` |
| prediction | String nullable | REAL/FAKE |
| confidence | Float nullable | 0–100 |
| model_name / model_version / dataset_version | String | From InvestigationResult |
| explainer_name | String nullable | e.g. gradcam |
| generate_explanation | Boolean | Request flag |
| artifact_dir | String nullable | Root for this run’s artefacts |
| heatmap_path / overlay_path | String nullable | Relative or absolute refs |
| explanation_json_path | String nullable | |
| raw_result_json | Text | Serialized InvestigationResult (+ XAI summary) |
| error_message | Text nullable | |
| started_at / completed_at | DateTime | |
| created_by_user_id | FK users | |

### 3.5 `audit_logs`

| Field | Type | Notes |
|-------|------|--------|
| id | Integer PK | |
| user_id | FK users nullable | |
| case_id | FK nullable | |
| evidence_id | FK nullable | |
| analysis_id | FK nullable | |
| event_type | String/Enum | See §Audit model |
| timestamp | DateTime | |
| details_json | Text | Non-sensitive metadata only |

**Append-oriented:** no update/delete API for audit rows in normal operation.

---

## 4. Authentication flow

### 4.1 Architectural conflict (MUST RESOLVE BEFORE CODE)

| Source | Auth model |
|--------|------------|
| Existing SRS (`docs/02_SRS.md`) + Architecture | **Flask-Login + Werkzeug** session cookies |
| This product sprint prompt | **JWT / access-token** (`POST /auth/login`, bearer tokens) |
| `requirements.txt` | Flask-Login **commented out**; **no PyJWT** |

**STOP rule:** Do not implement both. Choose one before STEP 4.

**Recommendation (consistent with existing MAYA design):**

1. Prefer **Flask-Login + Werkzeug** sessions for the investigator web app (SRS).  
2. If REST clients need tokens later, add JWT as an **optional** API auth mode without removing sessions.  
3. If product owners insist on JWT-only for this sprint, document the intentional SRS deviation and add `PyJWT` / Flask-JWT-Extended carefully.

Password hashing: **Werkzeug** `generate_password_hash` / `check_password_hash` (already a dependency).

### 4.2 Planned endpoints (auth)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/api/auth/register` | Public (or admin-only if locked down) | Create user |
| POST | `/api/auth/login` | Public | Session or JWT |
| POST | `/api/auth/logout` | Required | Invalidate session / client discard token |
| GET | `/api/auth/me` | Required | Current user profile + role |

Roles: `INVESTIGATOR`, `ADMIN` (extensible Enum).

---

## 5. Case lifecycle

```
OPEN → IN_PROGRESS → CLOSED → ARCHIVED
         ↑______________|
```

| Action | Rule |
|--------|------|
| Create | Owner = current user; status OPEN; allocate `case_number` |
| Update | Owner or ADMIN; not when ARCHIVED |
| Close | Sets CLOSED + `closed_at` |
| List | Owner’s cases; ADMIN all |
| Get | Ownership check |

API: `POST/GET /api/cases`, `GET/PATCH /api/cases/{case_id}`, `POST /api/cases/{case_id}/close`

---

## 6. Evidence lifecycle

```
Upload → UPLOADED → (analyze) PROCESSING → ANALYZED | FAILED → ARCHIVED
```

1. Validate extension / size / MIME against inference-supported image set.  
2. Generate safe `stored_filename` (UUID-based); never trust client filename for paths.  
3. Store under `UPLOAD_DIR / cases / {case_id} / {stored_filename}`.  
4. Compute SHA-256 via existing `hash_file`.  
5. Persist metadata + hash.  
6. Audit `EVIDENCE_UPLOADED`.

Integrity verify: re-hash file bytes → compare to `sha256_hash` → `VALID` | `TAMPERED` | `ERROR`.

API: upload under case; `GET` metadata; list by case; `POST /api/evidence/{id}/verify-integrity`.

---

## 7. AI integration flow

```
POST /api/evidence/{id}/analyze
        ↓
AnalysisService (backend)
        ↓ verify ownership + integrity (optional gate)
        ↓ status = PROCESSING
        ↓
integrations.ai_bridge
        ↓ InferencePipeline.run(storage_path)
        ↓ InvestigationResult (prediction, confidence, investigation_id, …)
        ↓ optional ExplainabilityEngine.explain(
              path,
              investigation_id=...,
              artifact_dir=artifacts/investigations/{inv}/xai/gradcam
          )
        ↓
Persist AnalysisRun + Evidence status
        ↓
Audit ANALYSIS_COMPLETED | ANALYSIS_FAILED
        ↓
JSON response (schemas — no ORM leak)
```

**Non-negotiable:** No model construction, Grad-CAM math, or preprocessing inside routes.

**Performance:** Reuse cached `ModelLoader` / singleton pipeline helper per process — do not reload EfficientNet per request.

**Tests:** Mock `InferencePipeline.run` / explain in unit tests; one integration test may call real AI if checkpoint exists (or mock with verified call args).

---

## 8. API architecture

JSON API under `/api/*` prefix (keeps Phase 1 HTML shell at `/`).

| Group | Prefix |
|-------|--------|
| Auth | `/api/auth` |
| Cases | `/api/cases` |
| Evidence | `/api/evidence` |
| Analysis / Investigations | `/api/analysis` or nested `/api/evidence/{id}/analyze` |
| Explainability metadata | via analysis response + `GET /api/analysis/{id}` |
| Audit | `/api/audit` (ADMIN or scoped) |
| Reports | Stub/deferred if PDF not in scope — return artefact refs only |

OpenAPI: Flask does not auto-generate like FastAPI; document endpoints in `docs/PHASE3_PRODUCT.md` and optionally add `flasgger` later — **not required for first implementation**.

---

## 9. Security model

- Authenticated identity from session/JWT context — **never** trust client `user_id`.  
- Case/evidence access via ownership (+ ADMIN).  
- Path traversal blocked in storage layer.  
- Upload limits via `MAX_CONTENT_LENGTH`.  
- Parameterized ORM queries only.  
- No stack traces in JSON error bodies.  
- Secrets only from env (`SECRET_KEY`, future JWT secret).

---

## 10. Artifact storage model

| Kind | Location |
|------|----------|
| Case evidence bytes | `uploads/cases/{case_id}/{stored_filename}` |
| Inference artefacts | Prefer `artifacts/investigations/{investigation_id}/inference/` |
| XAI artefacts | `artifacts/investigations/{investigation_id}/xai/{explainer}/` |
| Sprint demo folders | Unchanged for offline AI sprints; product runs use investigation-scoped dirs |

DB stores **paths/references**, not image blobs.

---

## 11. Audit model

Event types (minimum):

`USER_REGISTERED`, `USER_LOGIN`, `CASE_CREATED`, `CASE_UPDATED`, `CASE_CLOSED`,  
`EVIDENCE_UPLOADED`, `EVIDENCE_VERIFIED`, `ANALYSIS_STARTED`, `ANALYSIS_COMPLETED`,  
`ANALYSIS_FAILED`, `REPORT_GENERATED` (when applicable)

Never store passwords or tokens in `details_json`.

---

## 12. Integration boundaries

| Allowed | Forbidden |
|---------|-----------|
| `services` → `integrations` → `ai.inference` / `ai.explainability` | Routes → `torch` / CAM algorithms |
| Reuse `hash_file` | Second hashing implementation |
| Extend `BaseConfig` | Parallel config system |
| Flask-SQLAlchemy models | Second ORM |
| New blueprints under `backend/app` | Second web framework |

---

## 13. Testing strategy (planned)

| Suite | Focus |
|-------|--------|
| `tests/test_auth_api.py` | Register/login/me/roles |
| `tests/test_cases_api.py` | CRUD, close, ownership |
| `tests/test_evidence_api.py` | Upload, reject bad files, SHA-256, verify |
| `tests/test_analysis_api.py` | Orchestration with mocked AI |
| `tests/test_audit_api.py` | Event persistence |
| `tests/test_product_integration.py` | Auth → case → upload → analyze (mocked AI) |
| Existing `tests/test_*.py` | **Must remain green** (Phase 3–4 AI/XAI) |

Use `create_app("testing")` + in-memory SQLite.

---

## 14. Implementation order (after auth decision)

1. Models + DB create_all  
2. Auth  
3. Cases  
4. Evidence + SHA-256  
5. Audit service  
6. Analysis integration  
7. Full regression  

---

## 15. Open decision log

| ID | Decision needed | Options | Default proposal |
|----|-----------------|---------|------------------|
| D1 | Auth mechanism | Flask-Login sessions vs JWT | **Flask-Login** (SRS) |
| D2 | Register open vs admin-gated | Public register vs admin creates users | Public register for lab; disable in prod config flag |
| D3 | Sync vs async analysis | Sync in-request vs background | **Sync** first (no Celery); service boundary allows future queue |

---

**Next step:** Confirm **D1 (Auth)** — then begin STEP 4 (Authentication implementation).
