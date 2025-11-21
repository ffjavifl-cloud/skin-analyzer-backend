from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

app = FastAPI()

# Habilitar CORS para permitir conexión desde el frontend en GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes reemplazar "*" por tu dominio exacto si lo prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar calibración desde archivo
try:
    with open("calibration.json", "r") as f:
        calibration = json.load(f)
except Exception as e:
    calibration = {}
    print(f"⚠️ Error al cargar calibration.json: {e}")

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    report = {}
    for param, values in calibration.items():
        score = (values.get("severe", 0) * 10) / (values.get("mild", 1) + 1)
        report[param] = round(score, 1)

    return {
        "scores": report,
        "diagnosis": "Informe clínico generado con calibración"
    }
