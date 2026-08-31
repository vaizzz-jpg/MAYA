# MAYA — Investigation Workflow

## Role-based view

## 1. Onboarding & session

1. Admin creates account OR user registers (if `ALLOW_PUBLIC_REGISTRATION=true`)
2. `POST /api/auth/login` → session cookie set
3. `GET /api/auth/me` returns role `INVESTIGATOR` or `ADMIN`

## 2. Open a case

```
POST /api/cases
{title, description, priority: LOW|MEDIUM|HIGH}
→ case_number = CASE-YYYY-NNNNNN
→ status: OPEN
→ audit: CASE_CREATED
```

## 3. Upload evidence

```
POST /api/evidence/cases/{case_id}   (multipart: file, notes?)
→ storage_path: uploads/cases/{case_id}/{uuid}.{ext}
→ sha256_hash: server-computed SHA-256
→ status: UPLOADED, analysis_status: NONE
→ audit: EVIDENCE_UPLOADED
```

Upload is path-traversal free (UUID stored filenames, ext/MIME validated, max size 16 MB, post-save re-checked size).

## 4. Integrity check (optional, recommended before analysis)

```
POST /api/evidence/{id}/verify-integrity
→ integrity_status: VALID | MODIFIED | MISSING | ERROR
→ audit: EVIDENCE_VERIFIED
```

## 5. AI analysis

### 5a. Basic inference + single explainer

```
POST /api/evidence/{id}/analyze
{generate_explanation: true, explainer: "gradcam", verify_before_analyze: true}
→ run = AnalysisRun
  → status: PROCESSING → COMPLETED
  → investigation_id INV-YYYY-NNNNNN (from InvestigationIDGenerator
  → prediction REAL/FAKE
  → confidence 0-100
  → model/model_version/dataset_version
  → artifact_dir: artifacts/investigations/INV-...
  → heatmap_path overlay_path explanation_json_path
  → trust_score quality_score (if advanced enabled)
→ Evidence.analysis_status = COMPLETED
→ audit: ANALYSIS_STARTED, XAI_GENERATED, ANALYSIS_COMPLETED
```

### 5b. Full analysis with advanced XAI

```json
{
  "generate_explanation": true,
  "explainer": "gradcam",
  "explainers": ["gradcam","gradcam_plus_plus","layercam","scorecam","eigencam"],
  "advanced_xai": {
    "shap": true,
    "faithfulness": true,
    "counterfactual": true,
    "fusion": true,
    "trust": true,
    "multi_explainer": true
  }
}
```

Response includes:
```json
{
  "advanced_xai_results": {
    "methods_run": ["gradcam","shap","faithfulness","counterfactual","fusion","trust"],
    "explainers": {"gradcam": {...}, "layercam": {...}, ...},
    "trust": {"trust_score": 0.82, "grade": "B", "components": {...}},
    "artifact_paths": { "shap_overlay":"...", "faithfulness_comparison":"...", ...}
  }
}
```

## 6. Retrieve results

```
GET /api/analysis/{id}
→ prediction, confidence, investigation_id, artifact_dir, XAI artifacts
```

## 7. Generate investigation PDF report

```
POST /api/analysis/{id}/report
{investigator_notes: "Subject's claim of original provenance refuted.", format: "pdf"}
→ report_number: RPT-YYYY-NNNNNN
→ InvestigationReport entity with sha256_hash of PDF itself
→ audit: REPORT_GENERATED
GET /api/reports/{id}/download → binary PDF (ownership-check protected)
```

Report contains: Identifiers, assessment (M, Evidence metadata + SHA-256, prediction + confidence, model info, XAI methods used, visualization references where present SHAP / faithfulness where computed, audit timeline table, limitations/disclaimer/investigator notes.

## 8. Close case

```
POST /api/cases/{id}/close
→ status CLOSED + closed_at
→ audit: CASE_CLOSED
```

## 9. Audit trail (ADMIN)

```
GET /api/audit → 200 rows desc
→ USER_REGISTERED … REPORT_GENERATED with structured details_json
```

## 10. Logout

```
POST /api/auth/logout
→ Flask-Login logout_user() + audit USER_LOGOUT
```

## Summary — invariants enforced in MAYA

- Every evidence file on server compute its evidence file has a server-computed SHA-256, integrity-checkable at any time.
- Every analysis = 1:1 analysis inves investigation ID (INV-.) that persists both in AnalysisRun.investigation_id AND appears in audit ANALYSIS_COMPLETED details.
- Artifacts are written under artifacts/investigations/INV-* and never moved/deleted by unrelated operations.
