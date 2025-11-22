from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

# Importa la función calibrada desde model.py
from model import predict_scores

app = FastAPI(title="Skin Analyzer Training API")

# CORS para conexión desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a tu dominio si lo deseas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def status():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Skin Analyzer API activa"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        print(f"📥 Imagen recibida: {file.filename}, tamaño: {len(raw)} bytes")

        image = Image.open(io.BytesIO(raw)).convert("RGB")

        scores = predict_scores(image)
        print(f"✅ Scores generados: {scores}")

        top_param = max(scores, key=lambda k: scores[k])
        diagnosis_map =
