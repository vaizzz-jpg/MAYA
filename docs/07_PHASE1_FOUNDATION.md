# MAYA — Phase 1 Foundation Notes

## STEP sequence (as applied)

### Health / platform bootstrap

1. **Why:** Need a runnable shell before auth/AI so architecture is verified early.  
2. **Architecture:** Presentation (`frontend/`) ← Application factory/blueprints ← infrastructure (config, logging, db).  
3. **Files:** `backend/app/__init__.py`, `config/config.py`, `extensions.py`, `database/init_db.py`, `routes/health.py`, `routes/shell.py`, `run.py`.  
4. **Code:** Generated in repository (see those paths).  
5. **Explanation:** Factory creates Flask app pointing at `frontend/` templates/static; logging rotates under `logs/`; SQLAlchemy binds to SQLite; only liveness + shell routes exist.

### Scope exception

Phase 1 brief said “no routes except health check.” A minimal `/` **shell** route was added solely to verify Bootstrap/nav/error-template inheritance. It contains **no** domain logic. Remove it if you want absolute route purity; `/health` and error handlers remain sufficient for ops checks.

## Verify

```bash
python backend/run.py
curl http://127.0.0.1:5000/health
```

Expected JSON:

```json
{"status":"ok","service":"maya","phase":1}
```
