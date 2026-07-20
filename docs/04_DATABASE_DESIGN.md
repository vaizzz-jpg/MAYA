# MAYA — Database Planning (Design Only)

## 1. Why this exists

Database planning locks entities and relationships before SQLAlchemy models are written. Phase 1 wires SQLAlchemy and creates the database file **without** implementing business tables yet (or with empty metadata only). Full ORM models arrive when authentication/cases begin.

**Phase 0 status:** Design complete. **No table implementation in Phase 1 beyond infrastructure.**

---

## 2. Design goals

- Case-centric investigation data model  
- Evidence integrity fields first-class (hash, size, mime)  
- Analysis runs and reports as first-class history (not overwritten blobs)  
- Audit trail for accountability  
- Settings as key-value for operational knobs without redeploying code for every flag  

---

## 3. ER diagram description

```
users 1───* cases
users 1───* audit_logs
users 1───* reports          (generated_by)
cases 1───* evidence
cases 1───* reports
evidence 1───* analysis_runs   (planned; may live in Evidence JSON early, preferred as table)
users *───* cases              (optional later: case_assignments)
settings                        (standalone key-value)
```

### Narrative ER

- A **User** creates and owns **Cases**.  
- A **Case** contains many **Evidence** items.  
- Each **Evidence** item may have many **Analysis Run** records over time (re-analysis).  
- A **Report** belongs to a **Case** (and optionally references specific evidence/analysis).  
- **AuditLog** records who did what, optionally linked to case/evidence IDs.  
- **Settings** stores deployment configuration entries.

---

## 4. Table designs

### 4.1 `users`

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| email | VARCHAR UNIQUE | Login identity |
| username | VARCHAR UNIQUE | Display / alternate login |
| password_hash | VARCHAR | Werkzeug hash only |
| full_name | VARCHAR | |
| role | VARCHAR | `investigator`, `admin`, … |
| is_active | BOOLEAN | Soft disable |
| created_at | DATETIME | |
| last_login_at | DATETIME NULL | |

**Relationships:** one-to-many → cases, audit_logs, reports.

---

### 4.2 `cases`

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| case_number | VARCHAR UNIQUE | Human-facing reference |
| title | VARCHAR | |
| description | TEXT | |
| status | VARCHAR | `open`, `under_review`, `closed`, `archived` |
| created_by_user_id | FK → users.id | |
| created_at | DATETIME | |
| updated_at | DATETIME | |
| closed_at | DATETIME NULL | |

**Relationships:** many-to-one → users; one-to-many → evidence, reports.

---

### 4.3 `evidence`

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| case_id | FK → cases.id | Required |
| original_filename | VARCHAR | Display name |
| stored_filename | VARCHAR | Safe stored name |
| storage_path | VARCHAR | Relative path under uploads |
| mime_type | VARCHAR | |
| file_size_bytes | INTEGER | |
| sha256_hash | CHAR(64) | Integrity |
| media_type | VARCHAR | `image`, `video`, … |
| uploaded_by_user_id | FK → users.id | |
| uploaded_at | DATETIME | |
| notes | TEXT NULL | Investigator notes |

**Relationships:** many-to-one → cases, users; one-to-many → analysis_runs (planned).

**Integrity rule:** Never overwrite `storage_path` bytes; derivatives go elsewhere.

---

### 4.4 `analysis_runs` (planned for AI phase)

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| evidence_id | FK → evidence.id | |
| model_name | VARCHAR | |
| model_version | VARCHAR | |
| authenticity_score | FLOAT | Aggregated 0–100 or 0–1 (document unit) |
| deepfake_probability | FLOAT | |
| label | VARCHAR | e.g., `likely_authentic`, `suspicious`, `likely_manipulated` |
| explanation_path | VARCHAR NULL | Heatmap/derivative path |
| raw_result_json | TEXT | Structured details |
| started_at | DATETIME | |
| completed_at | DATETIME NULL | |
| status | VARCHAR | `queued`, `running`, `completed`, `failed` |
| error_message | TEXT NULL | |

Included in Phase 0 ER so storage design anticipates AI without forcing Phase 1 models.

---

### 4.5 `reports`

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| case_id | FK → cases.id | |
| generated_by_user_id | FK → users.id | |
| title | VARCHAR | |
| storage_path | VARCHAR | Under `reports/` |
| format | VARCHAR | `pdf` |
| sha256_hash | CHAR(64) | Report integrity |
| generated_at | DATETIME | |
| parameters_json | TEXT | What was included |

---

### 4.6 `audit_logs`

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| user_id | FK → users.id NULL | Null for system events |
| action | VARCHAR | e.g., `evidence.upload` |
| entity_type | VARCHAR NULL | `case`, `evidence`, … |
| entity_id | INTEGER NULL | |
| ip_address | VARCHAR NULL | |
| user_agent | VARCHAR NULL | |
| details_json | TEXT | Non-sensitive context |
| created_at | DATETIME | |

**Design note:** Prefer append-only usage; no update/delete APIs for investigators.

---

### 4.7 `settings`

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| key | VARCHAR UNIQUE | e.g., `max_upload_mb` |
| value | TEXT | Stored as string; cast in service |
| value_type | VARCHAR | `int`, `str`, `bool`, `json` |
| updated_at | DATETIME | |
| updated_by_user_id | FK → users.id NULL | |

---

## 5. Relationship summary

| From | To | Cardinality | On delete (planned) |
|------|----|-------------|---------------------|
| users | cases | 1:N | Restrict (keep history) |
| cases | evidence | 1:N | Cascade or restrict—prefer soft-delete later |
| evidence | analysis_runs | 1:N | Cascade with evidence |
| cases | reports | 1:N | Restrict |
| users | audit_logs | 1:N | Restrict |
| users | settings updates | 1:N | Set null |

---

## 6. Indexing plan (later implementation)

- `users.email`, `users.username` unique  
- `cases.case_number` unique  
- `evidence.case_id`, `evidence.sha256_hash`  
- `audit_logs.created_at`, (`entity_type`, `entity_id`)  
- `analysis_runs.evidence_id`, `analysis_runs.created/completed`  

---

## 7. Explicit non-goals for Phase 1

- No ORM entity classes for the tables above  
- No migrations framework required yet (can add Flask-Migrate in Phase 2)  
- `db.create_all()` may run against empty metadata until models exist  

This keeps Phase 1 a **foundation**, not a premature schema freeze in code.
