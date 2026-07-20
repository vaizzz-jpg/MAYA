# MAYA — Development Roadmap, Risks & Performance Constraints

## 1. Development Roadmap

### Why this exists

A phased roadmap keeps the team from jumping to AI before foundation, auth, and evidence integrity exist. Each phase has a clear exit gate.

### Phase 0 — Design (current documentation)

**Deliverables:** Vision, SRS, architecture, DB design, UI plan, risks, constraints  
**Exit gate:** Documents reviewed; architecture accepted  

### Phase 1 — Foundation (implementation now)

**Deliverables:**

- Folder structure  
- `requirements.txt`, `.gitignore`, `README.md`  
- Config + `.env` pattern  
- Flask application factory  
- Logging  
- SQLAlchemy initialization  
- Blueprint registration  
- Bootstrap base template + nav + error pages  
- Health check route only  

**Exit gate:** App starts; `/health` returns OK; templates render  

### Phase 2 — Identity, Cases, Evidence

- Flask-Login authentication  
- User model + password hashing  
- Case CRUD  
- Evidence upload + SHA-256  
- Audit log for critical actions  
- Dashboard / case / evidence UI (from UI plan)  

**Exit gate:** Investigator can create case, upload image, see hash  

### Phase 3 — AI Analysis + Explainability

- Lightweight transfer-learning model (CPU)  
- Preprocessing pipeline  
- Inference service  
- Grad-CAM explanation artifacts  
- Authenticity score aggregation (documented weights)  
- Analysis screen  

**Exit gate:** Analysis run persists with score + explanation path  

### Phase 4 — Reports + Forensics depth

- ReportLab PDF reports  
- Metadata / basic forensic signals  
- Report screen + history  

**Exit gate:** PDF generated with case IDs, hashes, scores  

### Phase 5 — Hardening & Demo Polish

- CSRF, stricter upload validation  
- Performance pass on 8 GB RAM  
- Test suite expansion  
- Examiner demo script / docs  

**Exit gate:** Stable demo path Cases → Upload → Analyze → Report  

---

## 2. Risk Analysis

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | 8 GB RAM OOM during PyTorch load | High | Medium | Tiny/efficient models; lazy load; limit batch size=1; avoid concurrent heavy jobs |
| R2 | Scope creep into full video deepfake suite | High | High | Lock image-first scope; video = future |
| R3 | Black-box AI unacceptable to examiners | High | Medium | Mandatory explanation artifacts + model version logging |
| R4 | Evidence overwrite / integrity loss | Critical | Low–Med | Separate originals vs derivatives; store SHA-256; never mutate originals |
| R5 | SQLite concurrency limits | Medium | Medium | Single worker for lab; document Postgres migration path |
| R6 | Insecure uploads (path traversal, malware EXE as image) | High | Medium | MIME/extension allowlist; safe filenames; size caps; store outside static |
| R7 | Tutorial-style tangled code | Medium | High | Services + blueprints; code review against architecture doc |
| R8 | Model accuracy oversold | High | Medium | UI language: “indicators / likelihood”, not legal certainty |
| R9 | Dependency bloat | Medium | Medium | Pin versions; add heavy libs only when phase needs them |
| R10 | Time overrun on UI polish | Medium | Medium | Bootstrap-first; defer custom CSS |

---

## 3. Performance Constraints (Hardware)

### Machine profile

- Windows 11  
- **8 GB RAM**  
- No high-end GPU  

### Engineering rules

1. Prefer **transfer learning** on compact backbones over training large models from scratch on this machine.  
2. Default inference **batch size = 1**.  
3. Unload/load models deliberately; avoid keeping multiple large models resident.  
4. Do not use Electron or heavy SPA frameworks for the core product.  
5. Stream/write uploads to disk; do not hold entire large files in memory when avoidable.  
6. Paginate list endpoints/views.  
7. Use OpenCV/Pillow carefully; release arrays; prefer resizing copies for inference.  
8. Logging to file with rotation; avoid huge debug dumps in hot paths.  
9. Chart.js / AOS via CDN in templates to keep repo lean (or vendor later if offline required).  
10. Training jobs run offline via `scripts/` / `ai/training`, not inside web request threads.

### Performance acceptance (practical)

- App cold start on dev machine: interactive within a reasonable few seconds  
- Health endpoint: near-instant  
- Image analysis (Phase 3): minutes possible on CPU—UI must show status, never pretend real-time if not  

---

## 4. Future Scope (recap)

See also `01_VISION_PROBLEM_SCOPE.md`. High-value futures:

- Video pipelines  
- ExifTool depth  
- Ensemble + uncertainty  
- PostgreSQL + reverse proxy deployment  
- SSO for agencies  
- Multi-investigator review  

---

## 5. Phase 0 / Phase 1 verification checklist

### Phase 0

- [x] Vision distinct from deepfake-only apps  
- [x] SRS with FR/NFR IDs  
- [x] Layered architecture documented  
- [x] Module + folder structure defined  
- [x] DB ER + tables designed  
- [x] UI screens designed  
- [x] Roadmap, risks, performance constraints written  

### Phase 1

- [x] Repository foundation files present  
- [x] Flask app starts via documented command  
- [x] `/health` OK  
- [x] Base layout renders Bootstrap  
- [x] Error pages registered  
- [x] Logging writes to configured location  
- [x] DB engine initializes without business models  

---

## 6. Mentorship note

If hardware struggles once AI lands, the correct order of defense is: **smaller model → fewer simultaneous jobs → externalize training → optional later worker process**—not rewriting the case management platform.
