# MAYA — High-Level Architecture & Module Breakdown

## 1. Why architecture comes before code

Without explicit layers, Flask projects accumulate “route files that do everything”: upload, hash, call model, write DB, render HTML. That becomes untestable and blocks AI iteration. MAYA separates **HTTP concerns**, **use-cases**, **domain rules**, **AI engines**, and **persistence**.

---

## 2. Selected architecture: Layered Architecture

```
Presentation Layer          frontend templates/static + route handlers (thin)
        ↓
Application Layer           Flask app factory, blueprints, request orchestration
        ↓
Business Logic Layer        services (cases, evidence, scoring, reports, audit)
        ↓
AI Analysis Layer           inference, preprocessing, explainability, forensics signals
        ↓
Storage Layer               SQLAlchemy models, SQLite/files (uploads, reports, artifacts)
```

### Why this architecture was selected

| Criterion | Layered fit |
|-----------|-------------|
| College project timeline | Clear, teachable, examiner-friendly |
| Flask templates (not SPA) | Presentation sits naturally on top |
| Swap SQLite → Postgres later | Storage isolation via ORM |
| Swap/improve AI models | AI layer can change without rewriting cases |
| 8 GB RAM | No microservice orchestration overhead |
| SOLID-friendly | Services = SRP units; routes stay thin |

### Dependency rule

Upper layers may call lower layers.  
**AI and storage must never import Flask request/response objects.**  
Routes must not contain SQL or model inference code.

---

## 3. Suggested alternatives (decision gate)

The following were considered and **not selected as default**. Changing requires your explicit approval.

| Alternative | Pros | Cons for MAYA | Recommendation |
|-------------|------|---------------|----------------|
| Hexagonal (ports/adapters) | Excellent test seams | Overkill early; more boilerplate | Revisit if AI plugins explode |
| Microservices (API + AI worker) | Scale AI separately | Heavy for 8 GB; ops complexity | Future scope only |
| SPA (React) + API | Richer UI | Larger surface, more RAM/tooling | Avoid for major project core |
| FastAPI + async | Modern API DX | Less ideal for server-rendered investigator UI | Optional later for AI worker API |

**Phase 1 proceeds with layered Flask monolith + factory pattern.**

---

## 4. Technology stack justification

Already fixed by product constraints; architectural notes:

- **Flask Blueprints** → modular Application Layer  
- **Service classes** → Business Logic Layer  
- **`ai/` package outside request cycle** → AI Analysis Layer remains reusable from scripts/tests  
- **SQLite via SQLAlchemy** → Storage Layer with portable URL  
- **Bootstrap 5 / Chart.js / AOS** → Presentation productivity without custom frameworks  

---

## 5. Module breakdown

### 5.1 Presentation (`frontend/`)

| Module | Responsibility |
|--------|----------------|
| `templates/` | Jinja pages extending base layout |
| `static/css` | Minimal overrides only when Bootstrap is insufficient |
| `static/js` | Page behavior, Chart.js init, AOS init |
| `static/images` | Branding / UI assets (not evidence storage) |

### 5.2 Application (`backend/app/`)

| Module | Responsibility |
|--------|----------------|
| `__init__.py` | Application factory `create_app()` |
| `config/` | Environment-based settings classes |
| `extensions.py` | Shared Flask extensions (db, later login) |
| `routes/` | Thin HTTP adapters / blueprints |
| `utils/` | Logging setup, path helpers, validators |
| `auth/` | Login managers / decorators (Phase 2+) |

### 5.3 Business (`backend/app/services/`)

| Future service | Responsibility |
|----------------|----------------|
| `case_service` | Case CRUD and status transitions |
| `evidence_service` | Upload orchestration + integrity hash |
| `analysis_service` | Orchestrate AI + persist results |
| `report_service` | PDF generation and filing |
| `audit_service` | Append audit events |
| `settings_service` | Runtime settings access |

Phase 1 creates the **package**, not the business services.

### 5.4 Domain models (`backend/app/models/`)

SQLAlchemy entities: User, Case, Evidence, Report, AuditLog, Setting (implemented in later phases).

### 5.5 AI (`ai/`)

| Package | Responsibility |
|---------|----------------|
| `preprocessing/` | Resize/normalize without mutating originals |
| `inference/` | Model load + predict |
| `explainability/` | Grad-CAM artifacts |
| `forensics/` | Non-ML forensic signals |
| `training/` | Offline training scripts |
| `datasets/` | Dataset helpers / docs pointers |

### 5.6 Storage artifacts (filesystem)

| Path | Responsibility |
|------|----------------|
| `uploads/` | Original evidence blobs |
| `reports/` | Generated PDFs |
| `logs/` | Application logs |
| `dataset/` | Training data (not served by web) |

---

## 6. Folder structure (authoritative)

```
MAYA/
├── backend/
│   ├── app/
│   │   ├── routes/
│   │   ├── services/
│   │   ├── models/
│   │   ├── database/
│   │   ├── auth/
│   │   ├── utils/
│   │   ├── config/
│   │   ├── extensions.py
│   │   └── __init__.py          # application factory
│   └── run.py                   # entrypoint
├── frontend/
│   ├── templates/
│   └── static/{css,js,icons,images}/
├── ai/{training,inference,preprocessing,explainability,forensics,datasets}/
├── reports/
├── uploads/
├── dataset/
├── docs/
├── tests/
├── scripts/
├── logs/
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

### Phase 1 adaptation note

Your original sketch listed `backend/app/app.py`. Flask convention uses **`create_app()` in `backend/app/__init__.py`** plus **`backend/run.py`**. Same layered architecture; cleaner imports and testing.

---

## 7. Request flow (future — illustrative)

1. Investigator submits upload form (Presentation)  
2. Evidence blueprint validates request (Application)  
3. `EvidenceService` hashes file, stores metadata (Business)  
4. Optional later: `AnalysisService` calls `ai.inference` (AI)  
5. Results persisted via SQLAlchemy + files (Storage)  
6. Template renders scores + explanation image (Presentation)  

---

## 8. Clean Architecture / SOLID mapping (practical)

| Principle | MAYA practice |
|-----------|---------------|
| SRP | One service per domain concern |
| OCP | New analysis engines via AI adapters, not route edits |
| LSP/ISP | Keep interfaces small (later protocols if needed) |
| DIP | Services depend on repositories/AI abstractions, not Flask |

We apply these **where practical**—not as ceremony that blocks delivery.
