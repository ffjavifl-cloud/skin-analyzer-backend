from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

# Importa la función calibrada desde model.py
from model import predict_scores

# Inicializa la aplicación FastAPI
app = FastAPI(title="Skin Analyzer Training API")

# Configuración de CORS: permite conexión desde tu frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Puedes restringir a ["https://ffjavifl-cloud.github.io"] para mayor seguridad
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Endpoint rápido para verificación de estado
@app.get("/status")
def status():
    return {"status": "ok"}

# Endpoint raíz (opcional)
@app.get("/")
def root():
    return {"message": "Skin Analyzer API activa"}

# Endpoint principal de análisis
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Leer imagen enviada
        raw = await file.read()
        print(f"📥 Imagen recibida: {file.filename}, tamaño: {len(raw)} bytes")

        image = Image.open(io.BytesIO(raw)).convert("RGB")

        # Analizar imagen con modelo calibrado
        scores = predict_scores(image)
        print(f"✅ Scores generados: {scores}")

        # Diagnóstico basado en el parámetro más alto
        top_param = max(scores, key=lambda k: scores[k])
        diagnosis_map = {
            "dryness": "Signos de sequedad prominentes.",
            "pigmentation": "Pigmentación destacada.",
            "wrinkles": "Arrugas marcadas.",
            "lines": "Líneas visibles.",
            "texture-pores": "Textura/poros acentuados.",
            "brightness": "Brillo bajo (posible iluminación subóptima)."
        }
        diagnosis = diagnosis_map.get(top_param, "Evaluación clínica general.")

        return {
            "diagnosis": diagnosis,
            "scores": scores
        }

    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "No se pudo procesar la imagen",
                "details": str(e)
            }
        )
