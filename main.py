from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# Asegúrate de que model.py esté en la misma carpeta
from model import predict_scores

app = FastAPI(title="Aging Analyzer API")

# CORS: permite conexión desde tu frontend en GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a ["https://ffjavifl-cloud.github.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Leer imagen enviada
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")

        # Analizar imagen con tu modelo clínico
        scores = predict_scores(image)

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
        # Manejo de errores para que Swagger y el frontend lo vean
        return {
            "error": "No se pudo procesar la imagen",
            "details": str(e)
        }
