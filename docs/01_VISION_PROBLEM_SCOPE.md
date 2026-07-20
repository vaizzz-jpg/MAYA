# MAYA — Vision, Problem Statement & Scope

## 1. Project Vision

### Why this exists

A vision statement prevents the team from collapsing MAYA into “upload an image, get real/fake.” Investigators need a **case-centric evidence workflow** with integrity, auditability, explainability, and reporting.

### Vision statement

**MAYA (Media Authenticity Analyzer)** is an AI-powered **digital evidence investigation platform** that helps authorized investigators assess whether uploaded media evidence is authentic, document how that conclusion was reached, and produce professional investigation artifacts suitable for case records.

### What MAYA is

| MAYA is | MAYA is not |
|---------|-------------|
| Case-managed evidence workspace | A public consumer deepfake toy |
| Authenticity + forensics assistant | A single binary classifier UI |
| Explainable investigation support | A black-box “AI says fake” tool |
| Audit-friendly workflow software | An entertainment / social media scanner |

### Target users

- Digital forensic investigators  
- Cyber crime investigation units  
- Law enforcement digital evidence teams  
- Cybersecurity / SOC analysts handling media evidence  
- Academic / training use under controlled accounts  

### Product positioning

Deepfake detection is **one analysis engine** inside a larger investigation platform that also covers evidence intake, integrity checks, metadata/forensic cues, authenticity scoring, explanation artifacts, and reports.

---

## 2. Problem Statement

### Why this exists

Investigators increasingly face AI-generated and manipulated media. Existing tools are often either:

1. **Research demos** — high accuracy claims, weak case workflow, poor audit trail  
2. **Consumer scanners** — not designed for chain-of-custody style thinking  
3. **Heavy enterprise suites** — costly, complex, and poorly suited to constrained college/lab hardware  

### Problem statement (formal)

Law enforcement and digital forensic practitioners need a lightweight, auditable platform to upload, organize, analyze, and report on digital media evidence for authenticity indicators—including deepfake detection with explainability—without requiring high-end GPU infrastructure or opaque commercial black boxes.

### Consequences if unsolved

- Ad-hoc analyses with no case linkage  
- Irreproducible results (“I ran a model yesterday”)  
- Unexplained AI verdicts that cannot be defended  
- Weak evidence integrity practices during analysis  

---

## 3. Project Scope

### Why this exists

Scope defines what will be built in the major project timeline versus deferred. Clear scope protects delivery quality on 8 GB RAM hardware.

### In scope (major project core)

1. Secure investigator authentication and role-aware access (Phase 2+)  
2. Case management (create, list, open, close/archive)  
3. Evidence upload with integrity hashing (e.g., SHA-256)  
4. Evidence viewer for images (video later if time permits)  
5. AI deepfake / manipulation analysis pipeline (efficient transfer learning)  
6. Explainability (Grad-CAM / attention visualization)  
7. Basic forensic / metadata signals (progressively)  
8. Authenticity scoring aggregation (transparent formula)  
9. PDF investigation reports  
10. Audit logging of critical investigator actions  
11. Professional investigator-oriented UI  

### Out of scope (initial major project)

- Real-time CCTV stream analysis  
- Mobile native apps  
- Multi-tenant SaaS billing  
- Courtroom e-discovery integrations  
- Full video temporal deepfake pipeline (unless Phase N stretch)  
- Distributed GPU cluster inference  
- Public anonymous uploads  
- Blockchain “notarization” as a core dependency  

### Assumptions

- Users are authenticated investigators on a trusted network or campus lab  
- Primary media type for v1: **images**  
- Inference runs on **CPU** with optimized models  
- SQLite is sufficient for single-deployment / lab use  

### Constraints

- Windows 11 development machine, **8 GB RAM**, no high-end GPU  
- Prefer Flask + SQLite + lightweight frontend stack  
- Prefer maintainable Clean Architecture / layered boundaries over feature quantity  

---

## 4. Future Scope

### Why this exists

Future scope shows examiners and stakeholders that the architecture anticipates growth without claiming undeliverable features now.

### Planned extensions

| Area | Future capability |
|------|-------------------|
| Media | Video segment analysis, audio deepfake cues |
| Forensics | ExifTool integration, ELA, noise residual analysis |
| AI | Ensemble models, uncertainty estimation, continual evaluation |
| Security | SSO/LDAP, hardware security module for keys, stricter CoC |
| Ops | PostgreSQL migration, container deployment, reverse proxy |
| Collaboration | Multi-investigator case notes, peer review workflow |
| Reporting | Court-oriented export packs, signed report hashes |

---

## Architecture note

Scope is delivered through layered modules (see `03_ARCHITECTURE.md`) so deepfake detection can evolve without rewriting case/evidence storage.
