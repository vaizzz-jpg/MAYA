# MAYA — Phase 0 Documentation Index

**Project:** MAYA (Media Authenticity Analyzer)  
**Phase:** 0 — Project Design (No Implementation)  
**Audience:** Investigators, academic reviewers, development team

## Purpose of Phase 0

Phase 0 exists to lock product vision, scope, architecture, and constraints **before** writing application logic, AI pipelines, or UI code. This prevents costly rework and ensures MAYA is designed as a forensic investigation platform—not a demo deepfake classifier.

## Document Map

| # | Document | Contents |
|---|----------|----------|
| 1 | [01_VISION_PROBLEM_SCOPE.md](01_VISION_PROBLEM_SCOPE.md) | Vision, problem statement, scope, future scope |
| 2 | [02_SRS.md](02_SRS.md) | Software Requirements Specification (FR / NFR) |
| 3 | [03_ARCHITECTURE.md](03_ARCHITECTURE.md) | Layered architecture, modules, tech stack justification |
| 4 | [04_DATABASE_DESIGN.md](04_DATABASE_DESIGN.md) | ER description, tables, relationships (design only) |
| 5 | [05_UI_DESIGN.md](05_UI_DESIGN.md) | Screen-by-screen investigator UI plan |
| 6 | [06_ROADMAP_RISKS_PERFORMANCE.md](06_ROADMAP_RISKS_PERFORMANCE.md) | Roadmap, risks, hardware/performance constraints |

## Architecture Stance (Decision Gate)

**Selected:** Layered Architecture (Presentation → Application → Business → AI → Storage).

**Minor refinement applied in Phase 1 (standard Flask practice, not a competing architecture):**

- Application factory (`create_app`) instead of a single global app object
- Central `extensions.py` for SQLAlchemy (and later Flask-Login) to avoid circular imports
- `frontend/` kept as the presentation root; Flask is configured to load templates/static from there

If you prefer a different architecture (e.g., hexagonal/ports-and-adapters, microservices), decide **before** Phase 2. Changing after service layers exist increases refactor cost.

## Exit Criteria for Phase 0

- [x] Vision and problem clearly distinct from “deepfake demo”
- [x] Functional and non-functional requirements documented
- [x] Architecture and module boundaries defined
- [x] Database entities and relationships designed (not implemented)
- [x] UI screens planned for investigator workflow
- [x] Risks and 8 GB RAM constraints documented
- [x] Roadmap agreed through Phase 1+

## Next Phase

**Phase 1** builds only the runnable foundation: folders, config, Flask factory, logging, SQLAlchemy wiring, base UI shell, health check. No auth, no AI, no case/evidence business logic.

Implementation notes: [`07_PHASE1_FOUNDATION.md`](07_PHASE1_FOUNDATION.md)
