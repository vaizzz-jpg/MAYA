# MAYA — Phase 3 Product Layer

**Status:** Complete (Flask-Login sessions; JWT deferred by design)  
**Scope:** Authentication, cases, evidence, integrity, audit, REST APIs, AI orchestration  
**Auth decision:** Flask-Login + Werkzeug (SRS-aligned) — not JWT

Architecture detail: [`PHASE3_PRODUCT_ARCHITECTURE.md`](PHASE3_PRODUCT_ARCHITECTURE.md)

---

## 1. Product architecture

```
Client (JSON)
    ↓
Flask /api/* blueprints          (backend/app/api/)
    ↓
Services                         (auth, case, evidence, analysis)
    ↓
Integrations + Storage + Audit
    ↓
AI engines (unchanged)           InferencePipeline / ExplainabilityEngine
    ↓
SQLite (Flask-SQLAlchemy) + uploads/ + artifacts/investigations/
```

Backend **orchestrates**; `ai/` **computes**. Routes never construct models or Grad-CAM.

---

## 2. Authentication

- **Sessions** via Flask-Login (`login_user` / `logout_user`)
- Passwords: Werkzeug `generate_password_hash` / `check_password_hash`
- Roles: `INVESTIGATOR`, `ADMIN`
- Identity always from session — never from client `user_id`

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/api/auth/register` | Public | Create user |
| POST | `/api/auth/login` | Public | Start session |
| POST | `/api/auth/logout` | Required | End session |
| GET | `/api/auth/me` | Required | Current user |

---

## 3. Case lifecycle

Statuses: `OPEN` → `IN_PROGRESS` → `CLOSED` → `ARCHIVED`  
Stable ID: `CASE-YYYY-NNNNNN`

| Method | Path | Notes |
|--------|------|--------|
| POST | `/api/cases` | Create (owner = current user) |
| GET | `/api/cases` | Own cases (ADMIN: all) |
| GET | `/api/cases/{id}` | Ownership enforced |
| PATCH | `/api/cases/{id}` | Update fields/status |
| POST | `/api/cases/{id}/close` | Set CLOSED + `closed_at` |

---

## 4. Evidence lifecycle

Statuses: `UPLOADED` → `PROCESSING` → `ANALYZED` | `FAILED` → `ARCHIVED`

Storage: `uploads/cases/{case_id}/{uuid}{ext}`  
SHA-256 via existing `ai.datasets.utils.checksums.hash_file`.

| Method | Path | Notes |
|--------|------|--------|
| POST | `/api/evidence/cases/{case_id}` | multipart `file` |
| GET | `/api/evidence/cases/{case_id}` | List |
| GET | `/api/evidence/{id}` | Metadata |
| POST | `/api/evidence/{id}/verify-integrity` | VALID / TAMPERED / ERROR |

Allowed extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` (match inference).

---

## 5. Database model

Tables: `users`, `cases`, `evidence`, `analysis_runs`, `audit_logs`  
ORM: `backend/app/models/entities.py`  
Default DB: `backend/instance/maya.db` (SQLite)

---

## 6. API architecture

All product JSON under `/api/*`. Envelope:

```json
{ "ok": true, "data": { ... } }
{ "ok": false, "error": "error_code", "message": "..." }
```

Phase 1 HTML shell remains at `/` and `/health`.

---

## 7. AI integration

`AnalysisService.analyze_evidence`:

1. Ownership + optional integrity check  
2. `run_inference(path)` → `InferencePipeline`  
3. Optional `run_explanation(...)` → `ExplainabilityEngine`  
4. Persist `analysis_runs` + artefact paths under `artifacts/investigations/{INV-...}/`  
5. Audit `ANALYSIS_*`

| Method | Path |
|--------|------|
| POST | `/api/evidence/{id}/analyze` |
| GET | `/api/analysis/{id}` |
| GET | `/api/investigations/{id}` |

Example request:

```json
{ "generate_explanation": true, "explainer": "gradcam" }
```

---

## 8. Security

- Session cookies HttpOnly  
- Case/evidence ownership (+ ADMIN)  
- Safe filenames / path confinement under `UPLOAD_DIR`  
- Upload size via `MAX_CONTENT_LENGTH`  
- ORM parameterized queries  
- No passwords/tokens in audit or API responses  

---

## 9. Audit

Append-only `audit_logs`. Events include registration, login, case/evidence/analysis lifecycle.  
`GET /api/audit` — own events (ADMIN: all, capped).

---

## 10. Error handling

`MayaProductError` hierarchy → JSON status codes (`backend/app/exceptions.py`).  
Unhandled analysis failures → `500 analysis_failed` after persisting FAILED status.

---

## 11. Testing

| File | Coverage |
|------|----------|
| `tests/test_auth_api.py` | Register, login, logout, 401, duplicates |
| `tests/test_cases_api.py` | CRUD, close, unauthorized |
| `tests/test_evidence_api.py` | Upload, reject, SHA-256, integrity |
| `tests/test_analysis_api.py` | Mocked AI flow + failure persistence |

AI calls are **mocked** in unit/integration product tests. Full AI suite remains separate.

```bash
python -m pytest tests/test_auth_api.py tests/test_cases_api.py \
  tests/test_evidence_api.py tests/test_analysis_api.py -q
python -m pytest tests/ -q
```

---

## 12. Deployment considerations

```bash
copy .env.example .env   # set SECRET_KEY
python -m pip install -r requirements.txt
python backend/run.py    # http://127.0.0.1:5000
```

Env: `SECRET_KEY`, `DATABASE_URL`, `LOG_LEVEL`, `ALLOW_PUBLIC_REGISTRATION`, `MAX_CONTENT_LENGTH`.

For production: strong `SECRET_KEY`, HTTPS, disable public registration if needed, backed-up `uploads/` + DB.

---

## 13. Future improvements

- JWT optional for external API clients  
- Flask-Migrate / Alembic  
- Evidence archive/delete endpoints  
- Async analysis queue  
- OpenAPI (e.g. flasgger)  
- PDF reports (Phase later)  
- Wire Advanced XAI (Sprint 4.5) into analyze options  

---

## Endpoint summary

| Group | Endpoints |
|-------|-----------|
| Auth | `POST /api/auth/register`, `login`, `logout`, `GET /api/auth/me` |
| Cases | `POST/GET /api/cases`, `GET/PATCH /api/cases/{id}`, `POST .../close` |
| Evidence | `POST/GET /api/evidence/cases/{case_id}`, `GET /api/evidence/{id}`, `POST .../verify-integrity` |
| Analysis | `POST /api/evidence/{id}/analyze`, `GET /api/analysis/{id}`, `GET /api/investigations/{id}` |
| Audit | `GET /api/audit` |
| Health | `GET /health` |
