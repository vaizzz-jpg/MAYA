# MAYA — Software Requirement Specification (SRS)

## Document control

| Field | Value |
|-------|-------|
| Product | MAYA — Media Authenticity Analyzer |
| Phase | 0 (Design) |
| Implementation status | Requirements only; no application logic yet |

### Why an SRS exists

The SRS is the contract between product intent and engineering. Every Phase 2+ feature should map to a requirement ID so scope stays controlled and testable.

---

## 1. Introduction

### 1.1 Purpose

Specify functional and non-functional requirements for MAYA so investigators can manage cases, upload evidence, run authenticity analysis, view explanations, and export reports.

### 1.2 Product perspective

MAYA is a **monolithic web application** (Flask + server-rendered templates) with clear internal layers. It is designed for internal / lab / agency deployment—not a public marketplace app.

### 1.3 User classes

| ID | Class | Description |
|----|-------|-------------|
| U-INV | Investigator | Primary user: creates cases, uploads evidence, runs analysis, views reports |
| U-SUP | Supervisor (optional later) | Reviews cases / reports |
| U-ADM | Administrator | Manages users, settings, retention policies |

Phase 1 implements **no authentication**. Roles appear from Phase 2 onward.

---

## 2. Functional Requirements

### Why FRs exist

Functional requirements define **what** the system must do. IDs enable traceability into tests and demos.

### 2.1 Authentication & access (Phase 2+)

| ID | Requirement |
|----|-------------|
| FR-AUTH-01 | System shall allow registered users to log in with email/username and password. |
| FR-AUTH-02 | System shall hash passwords using Werkzeug secure password hashing. |
| FR-AUTH-03 | System shall protect case/evidence routes from anonymous access. |
| FR-AUTH-04 | System shall support logout and session termination. |
| FR-AUTH-05 | System shall restrict administrative settings to admin role. |

### 2.2 Case management

| ID | Requirement |
|----|-------------|
| FR-CASE-01 | Investigator shall create a case with title, description, and optional reference number. |
| FR-CASE-02 | Investigator shall list and search their accessible cases. |
| FR-CASE-03 | Investigator shall open a case detail view showing linked evidence and status. |
| FR-CASE-04 | Investigator shall update case status (e.g., open, under review, closed). |
| FR-CASE-05 | System shall record case lifecycle events in audit logs. |

### 2.3 Evidence intake & integrity

| ID | Requirement |
|----|-------------|
| FR-EVD-01 | Investigator shall upload media files into a selected case. |
| FR-EVD-02 | System shall reject disallowed file types and enforce size limits. |
| FR-EVD-03 | System shall compute and store a cryptographic hash (SHA-256) of uploaded bytes. |
| FR-EVD-04 | System shall store original filename, stored path, MIME type, size, and upload timestamp. |
| FR-EVD-05 | Investigator shall view evidence metadata and integrity hash on the evidence viewer. |
| FR-EVD-06 | System shall prevent path traversal and unsafe filenames on upload. |

### 2.4 Analysis & authenticity

| ID | Requirement |
|----|-------------|
| FR-AI-01 | Investigator shall trigger authenticity analysis on eligible evidence. |
| FR-AI-02 | System shall produce a structured analysis result (scores, labels, model version). |
| FR-AI-03 | System shall support deepfake / manipulation probability scoring. |
| FR-AI-04 | System shall aggregate an authenticity score using documented weighted inputs. |
| FR-AI-05 | System shall persist analysis runs linked to evidence (re-runnable history). |
| FR-AI-06 | System shall record model identifier and preprocessing version for reproducibility. |

### 2.5 Explainability

| ID | Requirement |
|----|-------------|
| FR-XAI-01 | System shall generate an explanation artifact (e.g., Grad-CAM heatmap overlay) for image analysis. |
| FR-XAI-02 | Investigator shall view explanation alongside model prediction. |
| FR-XAI-03 | Reports shall reference or embed explanation outputs where available. |

### 2.6 Forensics & metadata (progressive)

| ID | Requirement |
|----|-------------|
| FR-FOR-01 | System shall extract available basic image metadata where present. |
| FR-FOR-02 | System shall surface forensic indicators as separate signals (not silently merged into AI score). |
| FR-FOR-03 | Future: ExifTool-backed extraction without breaking current APIs. |

### 2.7 Reporting

| ID | Requirement |
|----|-------------|
| FR-RPT-01 | Investigator shall generate a PDF report for a case or selected evidence analysis. |
| FR-RPT-02 | Report shall include case identifiers, evidence hashes, scores, timestamps, and analyst identity. |
| FR-RPT-03 | System shall store report generation records. |

### 2.8 Audit & settings

| ID | Requirement |
|----|-------------|
| FR-AUD-01 | System shall append immutable-style audit entries for login, upload, analyze, report, status change. |
| FR-SET-01 | Administrator shall configure upload limits and retention-related settings. |
| FR-SET-02 | System shall load secrets and environment-specific settings from environment variables. |

### 2.9 Platform health (Phase 1)

| ID | Requirement |
|----|-------------|
| FR-PLT-01 | System shall expose a health endpoint confirming the application is running. |
| FR-PLT-02 | System shall initialize database schema through controlled startup/init flow. |

---

## 3. Non-Functional Requirements

### Why NFRs exist

NFRs define quality attributes. For MAYA, auditability, integrity, and low-memory operation are as important as prediction accuracy.

### 3.1 Performance & resource use

| ID | Requirement |
|----|-------------|
| NFR-PERF-01 | Application idle footprint shall remain usable on **8 GB RAM** Windows hosts alongside OS. |
| NFR-PERF-02 | Image inference shall target CPU-friendly models; GPU shall not be required. |
| NFR-PERF-03 | Upload and analysis shall provide clear progress/status feedback (no silent long hangs). |
| NFR-PERF-04 | Dashboard list pages shall paginate; avoid loading entire evidence blobs into memory. |

### 3.2 Security

| ID | Requirement |
|----|-------------|
| NFR-SEC-01 | Secrets shall not be committed to version control. |
| NFR-SEC-02 | CSRF protections shall be applied to state-changing forms (Phase 2+). |
| NFR-SEC-03 | Uploaded files shall be stored outside public URL trees where practical. |
| NFR-SEC-04 | Sessions shall use secure cookie flags appropriate to deployment mode. |
| NFR-SEC-05 | Authorization checks shall be enforced server-side (never UI-only). |

### 3.3 Reliability & maintainability

| ID | Requirement |
|----|-------------|
| NFR-REL-01 | Structured logging shall replace `print()` for operational events. |
| NFR-REL-02 | Business logic shall live in services, not route handlers. |
| NFR-REL-03 | Modules shall follow single-responsibility boundaries. |
| NFR-REL-04 | Configuration shall be environment-driven (`python-dotenv`). |

### 3.4 Usability

| ID | Requirement |
|----|-------------|
| NFR-UX-01 | UI shall follow investigator workflow: Cases → Evidence → Analysis → Report. |
| NFR-UX-02 | UI shall use Bootstrap 5 + Bootstrap Icons for consistency and speed. |
| NFR-UX-03 | Error pages shall be professional and non-revealing in production. |

### 3.5 Auditability & evidence integrity

| ID | Requirement |
|----|-------------|
| NFR-FOR-01 | Original evidence bytes shall not be overwritten by analysis previews. |
| NFR-FOR-02 | Analysis derivatives (heatmaps, thumbnails) shall be stored as separate artifacts. |
| NFR-FOR-03 | Critical actions shall be attributable to a user identity (post-auth). |

### 3.6 Portability

| ID | Requirement |
|----|-------------|
| NFR-PORT-01 | Local development shall run on Windows 11 with SQLite. |
| NFR-PORT-02 | Folder layout shall allow later PostgreSQL swap via SQLAlchemy URLs only. |

---

## 4. Technology Stack (Requirement-aligned)

| Layer | Choice | Requirement rationale |
|-------|--------|----------------------|
| Backend | Python + Flask | Lightweight, blueprint-friendly, suitable for 8 GB RAM |
| ORM | SQLAlchemy | Maintainable schema, avoid raw SQL |
| Auth | Flask-Login + Werkzeug | Standard, secure enough for project scope |
| DB | SQLite | Zero ops for lab; file-backed |
| Frontend | HTML/CSS/JS + Bootstrap 5 | Fast investigator UI without SPA overhead |
| Charts | Chart.js | Required visualizations without custom chart code |
| Motion | AOS.js | Light scroll animations without custom engines |
| AI | PyTorch + Torchvision | Transfer learning; CPU-capable models |
| XAI | Grad-CAM | Visual explanations for investigators |
| Reports | ReportLab | Server-side PDF generation |
| Imaging | OpenCV + Pillow | Controlled preprocessing |
| Config | python-dotenv | Environment separation |

---

## 5. Requirement priority for phased delivery

| Phase | Focus | Primary IDs |
|-------|-------|-------------|
| 0 | Design | This document |
| 1 | Foundation | FR-PLT-* |
| 2 | Auth + cases + evidence intake | FR-AUTH-*, FR-CASE-*, FR-EVD-* |
| 3 | AI + XAI + scoring | FR-AI-*, FR-XAI-* |
| 4 | Reports + forensics deepen | FR-RPT-*, FR-FOR-* |
| 5 | Hardening + polish | NFR-* completion |

---

## 6. Acceptance mapping (high level)

A Phase is complete when:

1. Mapped FR/NFR IDs are implemented or explicitly deferred with justification  
2. Manual test checklist for those IDs passes on the development machine  
3. No circular dependency between Presentation and AI layers  
