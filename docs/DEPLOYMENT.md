# MAYA — Deployment Guide

## Environment variables

Copy `.env.example` → `.env`:

| Key | Default | Purpose |
|-----|---------|---------|
| `FLASK_ENV` | `development` | `development` / `testing` / `production` |
| `SECRET_KEY` | `dev-only-change-me` | Session signing — **change in production** |
| `DATABASE_URL` | `sqlite:///backend/instance/maya.db` | Any SQLAlchemy URL |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `MAX_CONTENT_LENGTH` | `16777216` (16 MB) | Upload size cap in bytes |
| `ALLOW_PUBLIC_REGISTRATION` | `true` | Open registration? Disable in prod. |
| `MAYA_HOST` | `127.0.0.1` | Bind address in run.py |
| `MAYA_PORT` | `5000` | Bind port |
| `MAYA_DEBUG` | `1` | Flask debug mode in run.py (0 to disable) |

## Local development (No Docker)

```powershell
# Repository root:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
copy .env.example .env
# edit SECRET_KEY, ALLOW_PUBLIC_REGISTRATION=false for lab use

python backend/run.py
# http://127.0.0.1:5000/health  → {"ok":true,...}
```

### API quick test with curl/PowerShell:

```powershell
$body = @{ email="a@maya.test"; username="alice"; password="securepass1" } | ConvertTo-Json
Invoke-RestMethod -Method Post http://127.0.0.1:5000/api/auth/register -Body $body -ContentType "application/json"
```

## Docker deployment

### Build

```bash
docker build -t maya-backend:latest .
```

### Run plain container (production defaults):

```bash
docker run --name maya-backend \
  -e SECRET_KEY=a-long-random-secret \
  -e ALLOW_PUBLIC_REGISTRATION=false \
  -p 5000:5000 \
  -v maya-data:/app/backend/instance \
  -v maya-data:/app/uploads \
  -v maya-data:/app/reports \
  -v maya-data:/app/artifacts/investigations \
  --restart unless-stopped \
  maya-backend:latest
```

### docker-compose (recommended)

```bash
# set SECRET_KEY in .env or override
docker compose up -d --build
docker compose logs -f maya-backend
docker compose down
```

`docker-compose.yml` mounts the `maya-data` named volume for database + uploads + reports + investigation artifacts across container restarts.

### Healthcheck

Both Dockerfile and compose define HTTP health probes on `/health` every 60 s.

```bash
curl http://127.0.0.1:5000/health
```

### Checkpoint placement

Dockerfile intentionally does **not** copy `artifacts/checkpoints/best.pt` (size; may be secret per dataset license). Mount it:

```bash
-v "$(pwd)/artifacts/checkpoints:/app/artifacts/checkpoints:ro"
```

### Volume persistence locations

| Path in container | Purpose |
|---|---|
| `/app/backend/instance/maya.db` | SQLite database |
| `/app/uploads/` | Uploaded evidence (UUID-named) |
| `/app/reports/` | Generated PDF reports |
| `/app/artifacts/investigations/` | Per-investigation AI/XAI artifacts |
| `/app/artifacts/checkpoints/` | `best.pt` model weights (read-only) |

## Switching DB to Postgres (optional)

Install driver and set `DATABASE_URL`:

```
pip install psycopg2-binary
DATABASE_URL=postgresql+psycopg2://user:pwd@host:5432/maya
```

SQLAlchemy models are portable; no raw SQL in product code.

## Production readiness checklist

- `ALLOW_PUBLIC_REGISTRATION=false`
- `SECRET_KEY` ≥ 32 bytes random (store in Vault / secret manager)
- Behind TLS-terminating reverse proxy (nginx / caddy) setting `X-Forwarded-*` headers
- Flask dev server → Waitress: `waitress-serve --port 5000 --call 'backend.app:create_app'`
- Rotate logs (`logs/maya.log` via RotatingFileHandler in logging_config.py)
- Backup: `maya.db` + uploads/ + reports/ + artifacts/investigations/

## Scaling notes

- AI inference is **CPU-first sync per request**. For 2+ parallel analysts, use multiple workers (`waitress-serve --threads=4`). Heavy SHAP / advanced XAI should remain explicitly opt-in (`advanced_xai.*` booleans).
- Artifact directories are append-only. Safe for network storage (SMB/NFS) as long as POSIX rename is atomic.
