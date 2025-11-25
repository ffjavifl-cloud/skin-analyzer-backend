from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

# Importa funciones clínicas desde model.py
from model import predict_scores, classify_severity

# Inicializa la aplicación FastAPI
app = FastAPI(title="Skin Analyzer Training API")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Puedes restringir a ["https://ffjavifl-cloud.github.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Emojis por severidad
EMOJIS = {
    "Mild": "🟢",
    "Moderate": "🟠",
    "Severe": "🔴"
}

# Interpretación contextual por edad
def interpret_by_age(param: str, score: float, edad: int) -> str:
    if edad < 30:
        if param in ["wrinkles", "lines"] and score > 6.5:
            return "Signos prematuros para edad joven."
        if param == "brightness" and score < 3.0:
            return "Brillo bajo para piel joven."
        if param == "dryness" and score > 6.5:
            return "Sequedad inusual en piel joven."
    elif edad >= 60:
        if param in ["wrinkles", "lines"] and score < 4.0:
            return "Piel notablemente conservada para edad avanzada."
        if param == "brightness" and score > 7.0:
            return "Brillo elevado para edad madura."
    return "Interpretación acorde a edad."

# Endpoint raíz
@app.get("/")
def root():
    return {"status": "ok"}

# Endpoint principal de análisis
@app.post("/analyze")
async def analyze(file: UploadFile = File(...), edad: int = 40):
    try:
        # Leer imagen enviada
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")

        # Analizar imagen
        scores = predict_scores(image)

        # Clasificar cada parámetro con severidad, emoji y contexto por edad
        classified = {
            param: {
                "score": round(score, 2),
                "severity": classify_severity(score),
                "emoji": EMOJIS[classify_severity(score)],
                "age_context": interpret_by_age(param, score, edad)
            }
            for param, score in scores.items()
        }

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

        return JSONResponse(content={
            "diagnosis": diagnosis,
            "results": classified
        })

    except Exception as e:
        return JSONResponse(content={
            "error": "No se pudo procesar la imagen",
            "details": str(e)
        }, status_code=500)
