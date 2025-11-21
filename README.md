# Aging Analyzer Backend (FastAPI)

## Ejecutar localmente
- python -m venv .venv
- source .venv/bin/activate  # En Windows: .venv\Scripts\activate
- pip install -r requirements.txt
- uvicorn main:app --reload

Abrir: http://127.0.0.1:8000/docs

## Despliegue en Render
- Subir repo con `main.py`, `model.py`, `requirements.txt`
- Service type: Web Service
- Runtime: Python 3.11
- Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
