# Skin Analyzer Backend (FastAPI)

Live scoring runs through `calibrate.py` (calibration.json + clinical
attenuation). `model.py` and `predict.py` have been retired.

## Endpoints
- `GET /` and `GET /status` -> `{"status":"ok"}`
- `POST /analyze` (multipart): `file` (image), `edad` (int, form), `sex` (str, form, optional)
  -> `{"diagnosis": str, "results": {param: {score, severity, emoji, age_context}}}`

## Run locally
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn main:app --reload
    # docs: http://127.0.0.1:8000/docs
    python test_app.py   # sanity checks

## Deploy on Render (Docker)
- Service type: Web Service, environment: Docker
- The container binds `$PORT` automatically.
- Free tier sleeps after ~15 min idle; first request cold-starts (30-60s).
  Keep warm with an external uptime ping to `/status` every ~10 min,
  or use a paid instance.

## Notes
- CORS is pinned to known origins in `main.py` (`ALLOWED_ORIGINS`); add your
  production frontend domain there.
- Uploads are capped at 10 MB and never persisted.
